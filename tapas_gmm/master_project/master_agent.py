from typing import Dict, Tuple
from loguru import logger
from abc import ABC, abstractmethod
import torch
from torch import nn
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.master_baseline import ActorCriticBase, PPOActorCritic
from tapas_gmm.master_project.master_converter import P_A_Converter, P_B_Converter
from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    ObservationState,
    StateSpace,
    StateType,
    Task,
)
from tapas_gmm.master_project.master_gnn import HRL_GNN, HRL_GNN2
from tapas_gmm.master_project.master_graph import Graph
from tapas_gmm.master_project.master_helper import HRLHelper
from tapas_gmm.master_project.master_observation import HRLPolicyObservation


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.obs_dict = {
            k: [] for k in [StateType.Euler, StateType.Quat, StateType.Scalar]
        }

        self.goal_dict = {
            k: [] for k in [StateType.Euler, StateType.Quat, StateType.Scalar]
        }
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


class Agent(ABC):

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        goal: HRLPolicyObservation,
        threshold: float,
        precision: float = 0.01,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        epochs: int = 80,
    ):

        self.buffer = RolloutBuffer()
        self.action_space = action_space
        self.state_space = state_space
        self.goal = goal
        self.threshold = threshold
        self.precision = precision
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.policy_new: ActorCriticBase = None
        self.policy_old: ActorCriticBase = None
        self.optimizer: torch.optim.Adam = None
        self.mse_loss = nn.MSELoss()

    @abstractmethod
    def reset(self, goal: HRLPolicyObservation):
        pass

    @abstractmethod
    def act(self, obs: HRLPolicyObservation):
        pass

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_obs = {
            k: torch.stack(v, dim=0).detach().to(device)
            for k, v in self.buffer.obs_dict.items()
        }
        old_goal = {
            k: torch.stack(v, dim=0).detach().to(device)
            for k, v in self.buffer.goal_dict.items()
        }

        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_new.evaluate(
                old_obs, old_goal, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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
            torch.load(path, map_location=lambda storage, loc: storage)
        )
        self.policy_new.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage)
        )

    def get_reward_step(
        self,
        prev_obs: HRLPolicyObservation,
        next_obs: HRLPolicyObservation,
    ) -> tuple[float, bool]:
        """Compute reward based on progress toward the goal state.

        Args:
            prev_obs: Previous observation.
            next_obs: Current observation after taking an action.
            goal_obs: Desired goal observation.

        Returns:
            Reward (positive if closer to goal, negative if moving away).
        """
        # Extract relevant features (e.g., state difference to goal)
        goal_converter = P_A_Converter(self.goal, state_space=self.state_space)

        # Compute distances to goal
        prev_dist = goal_converter.partition_features(prev_obs).mean().item()
        next_dist = goal_converter.partition_features(next_obs).mean().item()

        # Reward = Improvement in distance (positive if closer, negative if farther)
        reward = prev_dist - next_dist  # If next_dist < prev_dist → positive reward

        # Optional (maybe i am not using it) Clip or scale rewards for stability
        reward = max(min(reward, 1.0), -1.0)  # Bound between [-1, 1]
        if next_dist <= self.precision:
            self.buffer.rewards.append(10.0)
            self.buffer.is_terminals.append(True)
            return 10.0, True

        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(False)
        return reward, False


class GNNAgent(Agent):

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        goal: HRLPolicyObservation,
        threshold: float,
        precision: float = 0.01,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
    ):
        # Initialize parent class
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            goal=goal,
            threshold=threshold,
            precision=precision,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
        )
        self.graph = Graph(goal, action_space, state_space)

        # self.gnn = HRL_GNN(num_states=self.graph.b_converter.num_states(goal))
        self.policy_new = HRL_GNN2(
            num_states=self.graph.b_converter.num_states(goal),
            dim_a=self.graph.a_converter.num_features(goal),
            dim_c=len(HRLHelper.c_states()),
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy_new.actor.parameters(), "lr": lr_actor},
                {"params": self.policy_old.critic.parameters(), "lr": lr_critic},
            ]
        )

    def reset(self, goal: HRLPolicyObservation):
        """Reset the agent with a new goal."""
        self.goal = goal
        self.graph.goal_update(goal)

    def act(self, observation: HRLPolicyObservation) -> int:
        self.graph.state_update(observation)
        self.action, self.logprob = self.policy_new.forward(self.graph.data)
        return self.action

    def update(
        self,
        current: HRLPolicyObservation,
        zero_reward: bool = False,
    ) -> tuple[float, bool]:
        """
        Update the agent's internal state based on the current and goal observations.
        This could involve updating the graph or the GNN parameters.
        """
        if zero_reward:
            reward = -1.0
            done = False
        else:
            reward, done = self.get_reward_step(self.last, current)

        # Update previous observation
        self.last = current

        # Compute loss
        loss = -self.logprob * reward

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()  # valid!
        self.optimizer.step()
        return reward, done


class PPOAgent(Agent):

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        goal: HRLPolicyObservation,
        threshold: float,
        precision: float = 0.01,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
    ):
        # Initialize parent class
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            goal=goal,
            threshold=threshold,
            precision=precision,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
        )
        self.goal = goal
        self.precision = precision
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.state_dim = ObservationState.count_by_state_space(state_space)
        self.action_dim = Task.count_by_action_space(action_space)

        self.policy_new = PPOActorCritic(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy_new.actor.parameters(), "lr": lr_actor},
                {"params": self.policy_new.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = PPOActorCritic(self.state_dim, self.action_dim)
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        self.converter = P_B_Converter()

    def reset(self, goal: HRLPolicyObservation):
        self.goal = goal

    def preprocess_state(
        self, obs: HRLPolicyObservation, goal: HRLPolicyObservation
    ) -> Dict[str, torch.Tensor]:
        obs_dict: Dict[str, torch.Tensor] = self.converter.partition_features(obs)
        goal_dict: Dict[str, torch.Tensor] = self.converter.partition_features(goal)
        combined_dict = {
            key: torch.cat([obs_dict[key], goal_dict[key]], dim=0) for key in obs_dict
        }
        return combined_dict

    def preprocess_state_seperate(
        self, obs: HRLPolicyObservation, goal: HRLPolicyObservation
    ) -> Tuple[Dict[StateType, torch.Tensor], Dict[StateType, torch.Tensor]]:
        obs_dict: Dict[StateType, torch.Tensor] = self.converter.partition_features(obs)
        goal_dict: Dict[StateType, torch.Tensor] = self.converter.partition_features(
            goal
        )
        return obs_dict, goal_dict

    def act(self, obs: HRLPolicyObservation) -> int:
        obs_dict, goal_dict = self.preprocess_state_seperate(obs, self.goal)

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(obs_dict, goal_dict)

        self.buffer.store_dicts(obs_dict, goal_dict)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
        obs_tensor = obs.to_tensor()
        action, logprob, value = self.policy.act(obs_tensor)
        self.logprob = logprob
        self.last_value = value
        return action

    # TODO: Finish PPO agent
    # TODO: Finish PPO network
    # TODO: Train PPO agent
    # TODO: Copy test file and run
    # ----------------------------------------------
    # TODO: Copy Plot graph and update and plot
    # TODO: Update GNN agent
    # TODO: Update GNN network
    # TODO: Train GNN agent
