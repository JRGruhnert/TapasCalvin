import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, LayerNorm, GINConv, GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.networks.master_modules import (
    GinActionMlp,
    GinStateMlp,
)


class GnnV1(GnnBase):

    def __init__(
        self,
        *args,
        dim_gat_out: int = 64,
        dim_head: int = 32,
        attention_heads: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gat1 = GATv2Conv(
            (self.dim_encoder, self.dim_encoder),
            self.dim_state,
            heads=attention_heads,
            concat=False,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            (self.dim_state, self.dim_state),
            dim_gat_out,
            heads=attention_heads,
            concat=False,
            # edge_dim=1,
            add_self_loops=False,
        )

        self.norm1 = LayerNorm(self.dim_state)
        self.norm2 = LayerNorm(dim_gat_out)
        self.actor = nn.Sequential(
            # nn.BatchNorm1d(gat_out),
            nn.Linear(dim_gat_out, dim_head),
            nn.Tanh(),
            nn.Linear(dim_head, 1),
        )

        self.critic = nn.Sequential(
            # nn.BatchNorm1d(gat_out),
            nn.Linear(dim_gat_out, dim_head),
            nn.Tanh(),
            nn.Linear(dim_head, 1),
        )

        self.critic_readout = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(dim_gat_out, 1), nn.Sigmoid())
        )
        self.critic_head = nn.Linear(dim_gat_out, 1)

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = batch.batch_dict

        x1: torch.Tensor = self.gat1(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
            return_attention_weights=None,
        )

        x2 = self.gat2(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=None,
            return_attention_weights=None,
        )

        logits = self.actor(x2).squeeze(-1)
        v_feat = self.critic_readout(x2)
        value = self.critic_head(v_feat).squeeze(-1)
        return logits, value

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.converter.tensor_state_dict_values(goal)
        obs_dict = self.converter.tensor_state_dict_values(obs)
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
        task_tensor = self.converter.tensor_task_distance(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.converter.state_state_edges(
            full=True
        )

        data[("obs", "obs-task", "task")].edge_index = self.converter.state_task_edges(
            full=False
        )

        data[("goal", "goal-obs", "obs")].edge_attr = self.converter.state_state_attr()
        data[("obs", "obs-task", "task")].edge_attr = self.converter.state_task_attr()
        return data


class GnnV2(GnnBase):

    def __init__(
        self,
        *args,
        dim_mlp: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_gin = GINConv(
            nn=GinStateMlp(self.dim_encoder, dim_mlp, self.dim_state),
        )

        self.action_gin = GINConv(
            nn=GinActionMlp(self.dim_state),
        )

        self.critic_head = GinActionMlp(2)

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict

        x_obs_updated = self.state_gin(
            (x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
        )

        x_task_updated = self.action_gin(
            (x_obs_updated, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
        )

        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes
        max_pool = global_max_pool(x_task_updated, task_batch_idx)  # [B, D]
        mean_pool = global_mean_pool(x_task_updated, task_batch_idx)  # [B, D]
        # reduce to a (B, ‑) vector for your critic head:
        pooled = torch.cat(
            [
                max_pool.max(dim=1).values.unsqueeze(-1),
                mean_pool.mean(dim=1).values.unsqueeze(-1),
            ],
            dim=1,
        )  # [B,2]

        logits = x_task_updated.squeeze(-1)
        value = self.critic_head(pooled).squeeze(-1)  # [B]

        return logits, value

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.converter.tensor_state_dict_values(goal)
        obs_dict = self.converter.tensor_state_dict_values(obs)
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
        task_tensor = self.converter.tensor_task_distance(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.converter.state_state_edges(
            full=False
        )
        data[("obs", "obs-task", "task")].edge_index = self.converter.state_task_edges(
            full=False
        )

        data[("goal", "goal-obs", "obs")].edge_attr = self.converter.state_state_attr()
        data[("obs", "obs-task", "task")].edge_attr = self.converter.state_task_attr()
        return data


class GnnV3(GnnBase):

    def __init__(
        self,
        *args,
        dim_mlp: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_gin = GINEConv(
            nn=nn.Sequential(
                nn.Linear(self.dim_encoder, dim_mlp),
                nn.Tanh(),
                nn.Linear(dim_mlp, self.dim_state),
                nn.Tanh(),
            ),
            train_eps=True,
            edge_dim=1,
        )

        self.action_gin = GINEConv(
            nn=nn.Sequential(
                nn.Linear(self.dim_state, self.dim_state // 2),
                nn.Tanh(),
                nn.Linear(self.dim_state // 2, 1),
            ),
            train_eps=True,
            edge_dim=1,
        )

        self.critic_head = GinActionMlp(2)

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = batch.batch_dict

        x1: torch.Tensor = self.state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.action_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes
        max_pool = global_max_pool(x2, task_batch_idx)  # [B, D]
        mean_pool = global_mean_pool(x2, task_batch_idx)  # [B, D]
        # reduce to a (B, ‑) vector for your critic head:
        pooled = torch.cat(
            [
                max_pool.max(dim=1).values.unsqueeze(-1),
                mean_pool.mean(dim=1).values.unsqueeze(-1),
            ],
            dim=1,
        )  # [B,2]

        logits = x2.squeeze(-1)
        value = self.critic_head(pooled).squeeze(-1)  # [B]

        return logits, value

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.converter.tensor_state_dict_values(goal)
        obs_dict = self.converter.tensor_state_dict_values(obs)
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
        task_tensor = self.converter.tensor_task_distance(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.converter.state_state_edges(
            full=True
        )
        data[("obs", "obs-task", "task")].edge_index = self.converter.state_task_edges(
            full=True
        )

        data[("goal", "goal-obs", "obs")].edge_attr = self.converter.state_state_attr()
        data[("obs", "obs-task", "task")].edge_attr = self.converter.state_task_attr()
        return data
