import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, LayerNorm, GINConv, GINEConv
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

        task_actor_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        task_critic_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

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

        self.actor_gin = GINConv(
            nn=task_actor_mlp,
        )

        self.critic_gin = GINConv(
            nn=task_critic_mlp,
        )

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
        x2 = F.leaky_relu(x1, negative_slope=0.01)  # Apply LeakyReLU after GNN

        x3 = self.gat2(
            x=(x2, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=None,
            return_attention_weights=None,
        )
        x4 = F.leaky_relu(x3, negative_slope=0.01)  # Apply LeakyReLU after GNN

        logits = self.actor(x2).squeeze(-1)
        task_batch_idx = batch_dict["task"]
        v_feat = self.critic_readout(x2, task_batch_idx)
        value = self.critic_head(v_feat).squeeze(-1)
        return logits, value

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.cnv.tensor_state_dict_values(goal)
        obs_dict = self.cnv.tensor_state_dict_values(obs)
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
        task_tensor = self.cnv.tensor_task_distance(obs).to(device)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.cnv.state_state_edges(
            full=True
        ).to(device)

        data[("obs", "obs-task", "task")].edge_index = self.cnv.state_task_edges(
            full=False
        ).to(device)

        data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr().to(
            device
        )
        data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr().to(
            device
        )
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
                nn.LeakyReLU(),
                nn.Linear(dim_mlp, self.dim_state),
                nn.LeakyReLU(),
            ),
            train_eps=True,
            edge_dim=1,
        )

        self.action_gin = GINEConv(
            nn=nn.Sequential(
                nn.Linear(self.dim_state, self.dim_state // 2),
                nn.LeakyReLU(),
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
        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes

        x1 = self.state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.action_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
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

        data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr
        return data.to(device)


class GnnV4(GnnBase):

    def __init__(
        self,
        *args,
        dim_mlp: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        state_state_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.LeakyReLU(),
        )

        state_task_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.LeakyReLU(),
        )

        task_actor_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        task_critic_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        self.state_state_gin = GINEConv(
            nn=state_state_mlp,
            train_eps=True,
            edge_dim=1,
        )

        self.state_task_gin = GINEConv(
            nn=state_task_mlp,
            edge_dim=2,
        )

        self.actor_gin = GINConv(
            nn=task_actor_mlp,
        )

        self.critic_gin = GINConv(
            nn=task_critic_mlp,
        )

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

        x1 = self.state_state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.state_task_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        logits = self.actor_gin(
            x=(x2, x_dict["actor"]),
            edge_index=edge_index_dict[("task", "task-actor", "actor")],
        )
        value = self.critic_gin(
            x=(x2, x_dict["critic"]),
            edge_index=edge_index_dict[("task", "task-critic", "critic")],
        )
        task_batch = batch.batch_dict["actor"]  # shape: [num_task_nodes]
        task_node_values = logits[
            task_batch == 0
        ]  # x2 is e.g. shape [num_task_nodes, D]
        print(task_node_values)
        print(logits.view(-1, self.dim_tasks))
        return logits.view(-1, self.dim_tasks), value.squeeze(-1)

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.cnv.tensor_state_dict_values(goal)
        obs_dict = self.cnv.tensor_state_dict_values(obs)
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

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = torch.zeros(self.dim_tasks, self.dim_encoder)
        data["actor"].x = torch.zeros(self.dim_tasks, 1)
        data["critic"].x = torch.zeros(1, 1)

        data[("goal", "goal-obs", "obs")].edge_index = self.cnv.state_state_full
        data[("obs", "obs-task", "task")].edge_index = self.cnv.state_task_full
        data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr_weighted(
            obs
        )
        data[("task", "task-actor", "actor")].edge_index = self.cnv.task_task_sparse
        data[("task", "task-critic", "critic")].edge_index = self.cnv.task_single
        return data.to(device)


class GnnV5(GnnBase):

    def __init__(
        self,
        *args,
        dim_mlp: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder * 2, self.dim_encoder),
            nn.Tanh(),
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.Tanh(),
        )

        state_task_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.Tanh(),
            nn.Linear(self.dim_encoder, self.dim_encoder),
            nn.Tanh(),
        )

        task_actor_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.Tanh(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        task_critic_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.Tanh(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        self.state_task_gin = GINEConv(
            nn=state_task_mlp,
            edge_dim=2,
        )

        self.actor_gin = GINConv(
            nn=task_actor_mlp,
        )

        self.critic_gin = GINConv(
            nn=task_critic_mlp,
        )

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

        x1 = self.state_task_gin(
            x=(x_dict["state"], x_dict["task"]),
            edge_index=edge_index_dict[("state", "state-task", "task")],
            edge_attr=edge_attr_dict[("state", "state-task", "task")],
        )

        logits = self.actor_gin(
            x=(x1, x_dict["actor"]),
            edge_index=edge_index_dict[("task", "task-actor", "actor")],
        )
        value = self.critic_gin(
            x=(x1, x_dict["critic"]),
            edge_index=edge_index_dict[("task", "task-critic", "critic")],
        )

        return logits.view(-1, self.dim_tasks), value.squeeze(-1)

    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        goal_dict = self.cnv.tensor_state_dict_values(goal)
        obs_dict = self.cnv.tensor_state_dict_values(obs)
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
        state_tensor1 = torch.cat(
            [obs_tensor, goal_tensor], dim=1
        )  # shape: [num_states, feature_size * 2]
        state_tensor = self.state_mlp(state_tensor1)
        data = HeteroData()
        data["state"].x = state_tensor
        data["task"].x = torch.zeros(self.dim_tasks, self.dim_encoder)
        data["actor"].x = torch.zeros(self.dim_tasks, 1)
        data["critic"].x = torch.zeros(1, 1)

        data[("task", "task-actor", "actor")].edge_index = self.cnv.task_task_sparse
        data[("task", "task-critic", "critic")].edge_index = self.cnv.task_single
        data[("state", "state-task", "task")].edge_index = self.cnv.state_task_full
        data[("state", "state-task", "task")].edge_attr = (
            self.cnv.state_task_attr_weighted(obs)
        )
        return data.to(device)
