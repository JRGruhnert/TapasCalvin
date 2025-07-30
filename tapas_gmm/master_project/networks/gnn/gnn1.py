import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, LayerNorm, GINConv, GINEConv
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase
from tapas_gmm.utils.select_gpu import device


class Gnn(GnnBase):

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
