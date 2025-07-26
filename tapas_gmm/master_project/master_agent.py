from dataclasses import dataclass
import os
import re
from typing import Any, Type
import torch
from torch import nn
import numpy as np
from tapas_gmm.master_project.networks import Network, import_network
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.master_observation import Observation
from tapas_gmm.master_project.networks.base import ActorCriticBase
from tapas_gmm.master_project.master_definitions import State, Task


@dataclass
class AgentConfig:
    name: str
    network: Network
    # Default values
    early_stop: int = 10  # None if no early stop else the epoch number
    max_epoch: int = 50
    saving_freq: int = 5  # Saving frequence of trained model
    saving_path: str = "results/"

    save_stats = True
    batch_size: int = 2048
    mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
    n_epochs: int = 50  # How many passes over the collected batch per update
    lr_actor: float = 0.0006  # Step size for actor optimizer
    lr_critic: float = 0.0003  # Step size for critic optimizer
    gamma: float = 0.99  # How much future rewards are worth today
    gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
    eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
    entropy_coef: float = 0.01  # Weight on the entropy bonus to encourage exploration
    value_coef: float = 0.5  # Weight on the critic (value) loss vs. the policy loss
    max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
    target_kl: float = 0.02  # (Optional) early stopping if KL divergence gets too large


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.obs = []
        self.goal = []
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
        self.obs.clear()
        self.goal.clear()

    def health(self):
        lengths = [
            len(self.actions),
            len(self.obs),
            len(self.goal),
            len(self.logprobs),
            len(self.rewards),
            len(self.state_values),
            len(self.is_terminals),
        ]
        return all(l == lengths[0] for l in lengths)

    def has_batch(self, batch_size: int):
        return len(self.actions) == batch_size

    def save(self, path: str, epoch: int):
        file_path = path + f"stats_epoch{epoch}.npy"
        data = {}

        data["actions"] = (
            torch.stack(self.actions).cpu().numpy()
            if isinstance(self.actions[0], torch.Tensor)
            else np.array(self.actions)
        )
        data["logprobs"] = torch.tensor(self.logprobs).cpu().numpy()
        data["rewards"] = np.array(self.rewards)
        data["state_values"] = torch.tensor(self.state_values).cpu().numpy()
        data["is_terminals"] = np.array(self.is_terminals)

        np.save(file_path, data)

    def stats(self) -> tuple[float, float, float]:
        # Calculate episode statistics
        episode_rewards = []
        episode_lengths_batch = []
        episode_success = []

        current_episode_reward = 0
        current_episode_length = 0
        for _, (reward, terminal) in enumerate(zip(self.rewards, self.is_terminals)):
            current_episode_reward += reward
            current_episode_length += 1

            if terminal:
                episode_rewards.append(current_episode_reward)
                episode_lengths_batch.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0

                # TODO: Remove hardcoded 50 as success threshold
                episode_success.append(1 if reward >= 50 else 0)

        return (
            np.sum(episode_rewards),
            np.mean(episode_lengths_batch),
            np.sum(episode_success) / len(episode_success),
        )


class Agent:

    def __init__(
        self,
        config: AgentConfig,
        tasks: list[Task],
        states: list[State],
        task_parameter: dict[Task, dict[State, np.ndarray]],
    ):
        # Hyperparameters
        self.config = config
        Net = import_network(config.network)
        # Initialize the agent
        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.policy_new: ActorCriticBase = Net(tasks, states, task_parameter)
        self.policy_old: ActorCriticBase = Net(tasks, states, task_parameter)
        self.optimizer = torch.optim.Adam(
            self.policy_new.parameters(),
            lr=self.config.lr_actor,
        )

        # Internal flags and counter
        self.waiting_feedback = False
        self.current_epoch = 0

        ### Directory path for the agent (specified by name)
        self.directory_path = config.saving_path + config.name + "/"
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

    def act(self, obs: Observation, goal: Observation) -> int:
        if self.waiting_feedback:
            raise UserWarning(
                "The agent hasn't recieved any feedback of previous action yet. "
                "Learning will not work without feedback. Please call feedback() fafter every act() call."
            )

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(obs, goal)

        self.buffer.obs.append(obs)
        self.buffer.goal.append(goal)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.waiting_feedback = True
        return action.item()

    def feedback(self, reward: float, terminal: bool):
        if not self.waiting_feedback:
            raise UserWarning(
                "The agent is recieving feedback without any action taken. "
                "Learning will not work without actions. Is this correct?"
            )

        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(terminal)
        self.waiting_feedback = False
        return self.buffer.has_batch(self.config.batch_size)

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
                + self.config.gamma * values[step + 1] * (1 - is_terminals[step])
                - values[step]
            )
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * (1 - is_terminals[step])
                * gae
            )
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    def learn(self, verbose: bool = False) -> bool:
        if not self.buffer.health():
            raise UserWarning("RolloutBuffer is not in sync. Agent can't learn.")
        if verbose:
            total_reward, episode_legth, success_rate = self.buffer.stats()
            print(
                f"Total Reward: {total_reward} \t Episode Length: {episode_legth} \t Success Rate \t {success_rate}"
            )
            print("Called learning on new batch. Updating gradients of the agent!")

        advantages, rewards = self.compute_gae(
            self.buffer.rewards, self.buffer.state_values, self.buffer.is_terminals
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_obs = self.buffer.obs
        old_goal = self.buffer.goal

        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )

        early_stop = False
        # Optimize policy for K epochs
        for epoch in range(self.config.n_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.config.batch_size)

            for start in range(0, self.config.batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = old_obs[mb_idx]
                mb_goal = old_goal[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_rewards = rewards[mb_idx]

                # Evaluate policy
                logprobs, state_values, dist_entropy = self.policy_new.evaluate(
                    mb_obs, mb_goal, mb_actions
                )

                state_values = torch.squeeze(state_values)

                # Ratios
                ratios = torch.exp(logprobs - mb_logprobs.detach())

                # Surrogate loss
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.config.eps_clip,
                        1 + self.config.eps_clip,
                    )
                    * mb_advantages
                )

                # PPO loss
                loss: torch.Tensor = (
                    -torch.min(surr1, surr2)
                    + self.config.value_coef * self.mse_loss(state_values, mb_rewards)
                    - self.config.entropy_coef * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(
                    self.policy_new.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Optional KL early stopping
                if self.config.target_kl is not None:
                    with torch.no_grad():
                        kl = (mb_logprobs - logprobs).mean()
                        if kl > self.config.target_kl:
                            print(f"Early stopping at epoch {epoch} due to KL={kl:.4f}")
                            early_stop = True
                            break  # break minibatch loop
            if early_stop:
                break

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        if self.config.save_stats:
            self.buffer.save(self.config.saving_path + "logs/", self.current_epoch)
        # clear buffer
        self.buffer.clear()

        # Update Epoch
        self.current_epoch += 1

        if self.current_epoch == self.config.max_epoch:
            return True

        if (
            self.config.early_stop is not None
            and self.current_epoch >= self.config.early_stop
        ):
            # TODO: Currently just False cause early stop is not implemented
            return False

        return False  # No early stopping

    def save(self, verbose: bool = False):
        """
        Save the model to the specified path.
        """
        saving_in = self.current_epoch % self.config.saving_freq
        if saving_in == 0:
            if verbose:
                print("Saving Checkpoint!")
            checkpoint_path = self.directory_path + "model_cp_epoch{}.pth".format(
                self.current_epoch,
            )
            torch.save(self.policy_old.state_dict(), checkpoint_path)
        else:
            if verbose:
                print(f"Next saving of Checkpoint in {saving_in} epochs.")

    def load(self, external_path: str = None):
        """
        Load the model from the specified path.
        """
        if external_path is not None:
            # Load from provided path
            checkpoint_path = external_path
            match = re.search(r"epoch(\d+)", os.path.basename(checkpoint_path))
            self.current_epoch = int(match.group(1)) if match else 0
        else:
            # Search for the latest checkpoint in the directory
            files = os.listdir(self.directory_path)
            checkpoints = [
                f for f in files if re.match(r"model_cp_epoch(\d+)\.pth$", f)
            ]
            if not checkpoints:
                raise FileNotFoundError("No checkpoint found in directory.")

            # Get highest-numbered checkpoint
            latest_checkpoint = max(
                checkpoints, key=lambda f: int(re.search(r"epoch(\d+)", f).group(1))
            )
            self.current_epoch = int(
                re.search(r"epoch(\d+)", latest_checkpoint).group(1)
            )
            checkpoint_path = os.path.join(self.directory_path, latest_checkpoint)

        print(f"Loading checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, _: storage)
        )
        self.policy_new.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, _: storage)
        )
