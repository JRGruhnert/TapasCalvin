import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod

from tapas_gmm.master_project.master_data_def import StateType
from tapas_gmm.master_project.master_encoder import (
    EulerEncoder,
    QuaternionEncoder,
    ScalarEncoder,
)


class ActorCriticBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, obs: dict[StateType, torch.Tensor], goal: dict[StateType, torch.Tensor]
    ):
        pass

    @abstractmethod
    def act(
        self, obs: dict[StateType, torch.Tensor], goal: dict[StateType, torch.Tensor]
    ):
        pass

    @abstractmethod
    def evaluate(
        self,
        obs: dict[StateType, torch.Tensor],
        goal: dict[StateType, torch.Tensor],
        action: torch.Tensor,
    ):
        pass


class PPOActorCritic(ActorCriticBase):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        h_dim_encoder: int = 32,
        h_dim1: int = 256,
        h_dim2: int = 64,
    ):
        super().__init__()

        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Euler.name: EulerEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Euler.name: EulerEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.combined_feature_dim = h_dim_encoder * 2 * state_dim  # encoder count

        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.Tanh(),
            nn.Linear(h_dim1, h_dim2),
            nn.Tanh(),
            nn.Linear(h_dim2, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.Tanh(),
            nn.Linear(h_dim1, h_dim2),
            nn.Tanh(),
            nn.Linear(h_dim2, 1),
        )

    def forward(
        self, obs: dict[StateType, torch.Tensor], goal: dict[StateType, torch.Tensor]
    ):
        # obs and goal are dicts with keys 'euler', 'quat', 'scalar'

        obs_encoded = [self.encoder_obs[k.name](v) for k, v in obs.items()]
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in goal.items()]
        # Flatten each encoded component
        obs_flat = torch.cat(
            [
                (v.unsqueeze(0) if v.dim() == 2 else v).flatten(start_dim=1)
                for v in obs_encoded
            ],
            dim=1,
        )
        goal_flat = torch.cat(
            [
                (v.unsqueeze(0) if v.dim() == 2 else v).flatten(start_dim=1)
                for v in goal_encoded
            ],
            dim=1,
        )
        # obs_flat = torch.cat([v.flatten(start_dim=1) for v in obs_encoded], dim=1)
        # goal_flat = torch.cat([v.flatten(start_dim=1) for v in goal_encoded], dim=1)
        # Final combined input to the network
        x = torch.cat([obs_flat, goal_flat], dim=1)  # .unsqueeze(
        #    0
        # )  # shape: [1, combined_feature_dim]
        # x = torch.cat(
        #    obs_encoded + goal_encoded, dim=-1
        # )  # shape: [B, combined_feature_dim]
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)  # shape: [B]
        return logits, value

    def act(
        self, obs: dict[StateType, torch.Tensor], goal: dict[StateType, torch.Tensor]
    ):
        prob, value = self.forward(obs, goal)
        dist = Categorical(prob)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action, logprob, value

    def evaluate(
        self,
        obs: dict[StateType, torch.Tensor],
        goal: dict[StateType, torch.Tensor],
        action: int,
    ):
        prob, value = self.forward(obs, goal)
        dist = Categorical(prob)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy
