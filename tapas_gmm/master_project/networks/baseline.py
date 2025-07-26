import torch
import torch.nn as nn
from tapas_gmm.master_project.networks.base import BaselineBase
from tapas_gmm.master_project.observation import Observation


class BaselineV1(BaselineBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.combined_feature_dim = self.dim_encoder * self.dim_state * 2

        h_dim1 = self.combined_feature_dim // 2
        h_dim2 = h_dim1 // 2
        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.Tanh(),
            nn.Linear(h_dim1, h_dim2),
            nn.Tanh(),
            nn.Linear(h_dim2, self.dim_tasks),
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
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # obs and goal are dicts with keys 'euler', 'quat', 'scalar'
        obs_dict, goal_dict = self.to_batch(obs, goal)

        obs_encoded = [self.encoder_obs[k.name](v) for k, v in obs_dict.items()]
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in goal_dict.items()]

        # Flatten each encoded component
        obs_flat = torch.cat([v.flatten(start_dim=1) for v in obs_encoded], dim=1)
        goal_flat = torch.cat([v.flatten(start_dim=1) for v in goal_encoded], dim=1)

        x = torch.cat([obs_flat, goal_flat], dim=1)  # .unsqueeze(
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)  # shape: [B]
        return logits, value
