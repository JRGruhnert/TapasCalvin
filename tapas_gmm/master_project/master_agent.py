from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch import nn
import numpy as np
from tapas_gmm.master_project.master_ppo_gnn import Master_GNN_PPO
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.master_gnn import HRL_GNN, HRL_GNN2
from tapas_gmm.master_project.master_graph import Graph
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.master_project.master_baseline import ActorCriticBase, PPOActorCritic
from tapas_gmm.master_project.master_converter import NodeConverter
from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    State,
    StateSpace,
    StateType,
    Task,
    RewardMode,
)


@dataclass
class RLConfig:
    action_space: ActionSpace = ActionSpace.STATIC
    state_space: StateSpace = StateSpace.DYNAMIC
    reward_mode: RewardMode = RewardMode.SPARSE
    batch_size: int = (
        2048  # 2048 1024 How many steps to collect before updating the policy
    )
    mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
    n_epochs: int = 50  # How many passes over the collected batch per update
    lr_actor: float = 0.0003  # Step size for actor optimizer
    lr_critic: float = 0.0003  # Step size for critic optimizer
    gamma: float = 0.99  # How much future rewards are worth today
    gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
    eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
    entropy_coef: float = 0.01  # Weight on the entropy bonus to encourage exploration
    value_coef: float = 0.5  # Weight on the critic (value) loss vs. the policy loss
    max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
    target_kl: float = 0.02  # (Optional) early stopping if KL divergence gets too large
    success_threshold: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 0.05,
            StateType.Quat: 0.1,
            StateType.Scalar: 0.05,
        }
    )
    reward_scale: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 2.0,
            StateType.Quat: 1.0,
            StateType.Scalar: 2.0,
        }
    )


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.obs_dict = {
            k: [] for k in [StateType.Transform, StateType.Quat, StateType.Scalar]
        }

        self.goal_dict = {
            k: [] for k in [StateType.Transform, StateType.Quat, StateType.Scalar]
        }

        self.c = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
        self.c.clear()

        for k in self.obs_dict:
            self.obs_dict[k].clear()
        for k in self.goal_dict:
            self.goal_dict[k].clear()

    def store_dicts(
        self, obs: dict[StateType, torch.Tensor], goal: dict[StateType, torch.Tensor]
    ):
        for k, v in obs.items():
            self.obs_dict[k].append(v)
        for k, v in goal.items():
            self.goal_dict[k].append(v)

    def save_as_npy(self, path: str, batch_num: int):
        file_path = path + f"rollout_buffer_{batch_num}.npy"
        data = {}

        for k, v_list in self.obs_dict.items():
            data[f"obs_{k.name}"] = torch.stack(v_list).cpu().numpy()

        for k, v_list in self.goal_dict.items():
            data[f"goal_{k.name}"] = torch.stack(v_list).cpu().numpy()

        data["actions"] = (
            torch.stack(self.actions).cpu().numpy()
            if isinstance(self.actions[0], torch.Tensor)
            else np.array(self.actions)
        )
        data["c"] = torch.stack(self.c).cpu().numpy()
        data["logprobs"] = torch.tensor(self.logprobs).cpu().numpy()
        data["rewards"] = np.array(self.rewards)
        data["state_values"] = torch.tensor(self.state_values).cpu().numpy()
        data["is_terminals"] = np.array(self.is_terminals)

        np.save(file_path, data)


class Agent(ABC):

    def __init__(
        self,
        parameters: RLConfig,
    ):
        # Hyperparameters
        self.parameters = parameters

        # Initialize the agent
        self.policy_new: ActorCriticBase = None
        self.policy_old: ActorCriticBase = None
        self.optimizer: torch.optim.Adam = None
        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.state_dim = State.count_by_state_space(self.parameters.state_space)
        self.action_dim = Task.count_by_action_space(self.parameters.action_space)
        self.converter = NodeConverter(
            State.list_by_state_space(parameters.state_space),
            Task.list_by_action_space(parameters.action_space),
            normalized=True,
        )
        total = sum(parameters.reward_scale.values())
        self.normalized_reward_scale = {
            key: value / total for key, value in parameters.reward_scale.items()
        }

    @abstractmethod
    def act(self, current: HRLPolicyObservation, goal: HRLPolicyObservation) -> int:
        pass

    def save_buffer(self, path: str, batch_num: int):
        """
        Save the current buffer to a .npz file.
        """
        self.buffer.save_as_npy(path, batch_num)

    def compute_gae(
        self,
        rewards: list[float],
        values: list[torch.Tensor],
        is_terminals: list[float],
    ):
        advantages = []
        gae = 0
        values = values + [0]  # add dummy for V(s_{T+1})
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.parameters.gamma * values[step + 1] * (1 - is_terminals[step])
                - values[step]
            )
            gae = (
                delta
                + self.parameters.gamma
                * self.parameters.gae_lambda
                * (1 - is_terminals[step])
                * gae
            )
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    @abstractmethod
    def call_evaluate(
        self,
        obs: dict[StateType, torch.Tensor],
        goal: dict[StateType, torch.Tensor],
        action: torch.Tensor,
        c: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def update(self):
        advantages, rewards = self.compute_gae(
            self.buffer.rewards, self.buffer.state_values, self.buffer.is_terminals
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert list to tensor
        old_obs = {
            k: torch.stack(v, dim=0).detach().to(device)
            for k, v in self.buffer.obs_dict.items()
        }
        old_goal = {
            k: torch.stack(v, dim=0).detach().to(device)
            for k, v in self.buffer.goal_dict.items()
        }

        if len(self.buffer.c) == 0:
            # No C‑nodes in this batch—create an “empty batch” of shape [0, D]
            old_c = None
        else:
            old_c = torch.stack(self.buffer.c, dim=0).detach().to(device)

        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )

        early_stop = False
        # Optimize policy for K epochs
        for epoch in range(self.parameters.n_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.parameters.batch_size)

            for start in range(
                0, self.parameters.batch_size, self.parameters.mini_batch_size
            ):
                end = start + self.parameters.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = {k: v[mb_idx] for k, v in old_obs.items()}
                mb_goal = {k: v[mb_idx] for k, v in old_goal.items()}
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_rewards = rewards[mb_idx]

                if old_c is None:
                    # Evaluate policy
                    logprobs, state_values, dist_entropy = self.call_evaluate(
                        mb_obs, mb_goal, mb_actions
                    )
                else:
                    # For now I loop over the minibatch
                    # Here is optimization potential but for this project it shouldnt matter that much
                    logprobs_list = []
                    state_values_list = []
                    dist_entropy_list = []

                    for i in range(mb_idx.shape[0]):
                        obs_i = {k: v[mb_idx[i]] for k, v in old_obs.items()}
                        goal_i = {k: v[mb_idx[i]] for k, v in old_goal.items()}

                        logprob_i, value_i, entropy_i = self.call_evaluate(
                            obs_i, goal_i, mb_actions[i], old_c[mb_idx[i]]
                        )

                        logprobs_list.append(logprob_i.unsqueeze(0))
                        state_values_list.append(value_i.unsqueeze(0))
                        dist_entropy_list.append(entropy_i)

                    # Concatenate tensors to simulate a batch
                    logprobs = torch.cat(logprobs_list, dim=0)
                    state_values = torch.cat(state_values_list, dim=0)
                    dist_entropy = torch.stack(dist_entropy_list).mean()

                state_values = torch.squeeze(state_values)

                # Ratios
                ratios = torch.exp(logprobs - mb_logprobs.detach())

                # Surrogate loss
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.parameters.eps_clip,
                        1 + self.parameters.eps_clip,
                    )
                    * mb_advantages
                )

                # PPO loss
                loss: torch.Tensor = (
                    -torch.min(surr1, surr2)
                    + self.parameters.value_coef
                    * self.mse_loss(state_values, mb_rewards)
                    - self.parameters.entropy_coef * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(
                    self.policy_new.parameters(), self.parameters.max_grad_norm
                )
                self.optimizer.step()

                # Optional KL early stopping
                # if self.parameters.target_kl is not None:
                #    with torch.no_grad():
                #        kl = (mb_logprobs - logprobs).mean()
                #        if kl > self.parameters.target_kl:
                #            print(f"Early stopping at epoch {epoch} due to KL={kl:.4f}")
                #            early_stop = True
                #            break  # break minibatch loop
            if early_stop:
                break

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, path: str):
        """
        Save the model to the specified path.
        """
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path: str):
        """
        Load the model from the specified path.
        """
        self.policy_old.load_state_dict(
            torch.load(path, map_location=lambda storage, _: storage)
        )
        self.policy_new.load_state_dict(
            torch.load(path, map_location=lambda storage, _: storage)
        )

    def difference_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.state_type]
            # Multiply and accumulate
            total += scale * diff

        return total / len(prev)

    def on_off_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.state_type]
            if diff < 0:
                total -= 1.0 * scale
            elif diff > 0:
                total += 1.0 * scale
        return total / len(prev)

    def failed_step(self, reward: float = 0.0) -> tuple[float, bool]:
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(True)
        return reward, True

    def success_step(self, reward: float = 10.0) -> tuple[float, bool]:
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(True)
        return reward, True

    def normal_step(self, reward: float = 0.0) -> tuple[float, bool]:
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(False)
        return reward, False

    def step(
        self,
        prev_obs: HRLPolicyObservation,
        next_obs: HRLPolicyObservation,
        goal_obs: HRLPolicyObservation,
        last_step: bool = False,
    ) -> tuple[float, bool]:
        """Compute reward based on progress toward the goal state.

        Args:
            prev_obs: Previous observation.
            next_obs: Current observation after taking an action.
            goal_obs: Desired goal observation.

        Returns:
            Reward (positive if closer to goal, negative if moving away).
        """
        # Compute distances to goal
        prev_dist = self.converter.dict_distance(prev_obs, goal_obs)
        next_dist = self.converter.dict_distance(next_obs, goal_obs)

        ##### Checking if goal is reached
        goal_reached = True
        for key, value in next_dist.items():
            if value > self.parameters.success_threshold[key.value.state_type]:
                goal_reached = False
                # print(
                #    f"Goal not reached for {key.name}: {value}"
                # )
                # break

        # If goal is reached, give a large reward and mark as terminal
        if goal_reached:
            return self.success_step(reward=100.0)
        elif last_step:
            if self.parameters.reward_mode == RewardMode.SPARSE:
                return self.failed_step(reward=0.0)
            return self.failed_step(reward=-10.0)
        else:
            if self.parameters.reward_mode == RewardMode.SPARSE:
                return self.normal_step(reward=0.0)
            elif self.parameters.reward_mode == RewardMode.RANGE:
                step_reward = self.difference_reward(prev_dist, next_dist)
            else:
                step_reward = self.on_off_reward(prev_dist, next_dist)
            step_reward *= 10  # The reward is between -1 and 1 -> -10 and 10
            return self.normal_step(reward=step_reward)


class PPOAgent(Agent):
    def __init__(self, parameters: RLConfig):
        super().__init__(parameters)
        self.policy_new: PPOActorCritic = PPOActorCritic(
            self.state_dim, self.action_dim
        )
        self.policy_old: PPOActorCritic = PPOActorCritic(
            self.state_dim, self.action_dim
        )
        self.policy_old.load_state_dict(self.policy_new.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.policy_new.actor.parameters(),
                    "lr": self.parameters.lr_actor,
                },
                {
                    "params": self.policy_new.critic.parameters(),
                    "lr": self.parameters.lr_critic,
                },
            ]
        )

    def act(self, obs: HRLPolicyObservation, goal: HRLPolicyObservation) -> int:
        obs_dict = self.converter.tensor_dict_values(obs)
        goal_dict = self.converter.tensor_dict_values(goal)

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(obs_dict, goal_dict)

        self.buffer.store_dicts(obs_dict, goal_dict)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def call_evaluate(
        self,
        obs: dict[StateType, torch.Tensor],
        goal: dict[StateType, torch.Tensor],
        action: torch.Tensor,
        c: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy_new.evaluate(obs, goal, action)


class GNNAgent(Agent):
    def __init__(self, parameters: RLConfig):
        super().__init__(parameters)
        self.graph = Graph(
            action_space=self.parameters.action_space,
            state_space=self.parameters.state_space,
        )
        self.policy_new: Master_GNN_PPO = Master_GNN_PPO(
            self.state_dim, self.action_dim
        )
        self.policy_old: Master_GNN_PPO = Master_GNN_PPO(
            self.state_dim, self.action_dim
        )
        self.policy_old.load_state_dict(self.policy_new.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.policy_new.actor.parameters(),
                    "lr": self.parameters.lr_actor,
                },
                {
                    "params": self.policy_old.critic.parameters(),
                    "lr": self.parameters.lr_critic,
                },
            ]
        )

    def act(self, current: HRLPolicyObservation, goal: HRLPolicyObservation) -> int:
        self.graph.update(current, goal)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(self.graph)

        self.buffer.c.append(self.graph.c)
        self.buffer.store_dicts(self.graph.b, self.graph.a)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def call_evaluate(
        self,
        obs: dict[StateType, torch.Tensor],
        goal: dict[StateType, torch.Tensor],
        action: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.graph.overwrite(goal, obs, c)
        return self.policy_new.evaluate(self.graph, action)
