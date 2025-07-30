import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, LayerNorm, GINConv, GINEConv
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.networks.layers.master_modules import (
    GinUnactivatedMlp,
    GinStandardMLP,
)


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.dim_encoder,
                out_dim=self.dim_state,
            ),
        )

        self.action_gin = GINConv(
            nn=GinUnactivatedMlp(self.dim_state),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes

        x1 = self.state_gin(
            (x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
        )

        x2 = self.action_gin(
            (x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
        )

        max_pool = global_max_pool(x2, task_batch_idx)  # [B, D]
        mean_pool = global_mean_pool(x2, task_batch_idx)  # [B, D]
        # reduce to a (B, ‑) vector for your critic head:
        pooled = torch.cat(
            [
                max_pool.max(dim=1).values.unsqueeze(-1),
                mean_pool.mean(dim=1).unsqueeze(-1),
            ],
            dim=1,
        )  # [B,2]

        value = self.critic_head(pooled).squeeze(-1)  # [B]
        logits = x2.view(-1, self.dim_tasks)  # [B, dim_tasks]
        return logits, value

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        obs_dict = self.cnv.tensor_state_dict_values(obs)
        goal_dict = self.cnv.tensor_state_dict_values(goal)
        obs_encoded = [
            self.encoder_obs[k.value.type.name](v.to(device))
            for k, v in obs_dict.items()
        ]
        goal_encoded = [
            self.encoder_goal[k.value.type.name](v.to(device))
            for k, v in goal_dict.items()
        ]
        obs_tensor = torch.stack(obs_encoded, dim=0)  # [num_states, feature_size]
        goal_tensor = torch.stack(goal_encoded, dim=0)  # [num_states, feature_size]
        task_tensor = self.cnv.tensor_task_distance(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.cnv.state_state_full
        data[("obs", "obs-task", "task")].edge_index = self.cnv.state_task_full
        return data.to(device)
