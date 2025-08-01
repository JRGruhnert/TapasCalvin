import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GINEConv, GINConv
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase, PPOType
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.networks.layers.mlp import (
    GinUnactivatedMLP,
    GinStandardMLP,
)


class MeanMaxPoolNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_task: int,
        dim_state: int,
        ppo_type: PPOType,
    ):
        super().__init__()
        self.ppo_type = ppo_type
        self.dim_tasks = dim_task
        self.dim_state = dim_state
        self.dim_features = dim_features

        self.state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_state,
            ),
            # edge_dim=1,
        )

        self.action_gin = GINEConv(
            nn=GinUnactivatedMLP(self.dim_state),
            edge_dim=1,
        )

        self.critic_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = batch.batch_dict
        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes

        x1 = self.state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            # edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.action_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        if self.ppo_type is PPOType.ACTOR:
            return x2.view(-1, self.dim_tasks)  # [B, dim_tasks]
        else:
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

            return self.critic_head(pooled).squeeze(-1)  # [B]


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.actor = MeanMaxPoolNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_state,
            dim_task=self.dim_tasks,
            ppo_type=PPOType.ACTOR,
        )
        self.critic = MeanMaxPoolNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_state,
            dim_task=self.dim_tasks,
            ppo_type=PPOType.CRITIC,
        )

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        logits = self.actor(batch)
        value = self.critic(batch)
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

        data[("goal", "goal-obs", "obs")].edge_index = self.cnv.state_state_sparse
        data[("obs", "obs-task", "task")].edge_index = self.cnv.state_task_full

        # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr
        return data.to(device)
