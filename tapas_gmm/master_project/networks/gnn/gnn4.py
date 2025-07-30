import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.networks.layers.master_modules import (
    GinUnactivatedMLP,
    GinStandardMLP,
)


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_state_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.dim_encoder,
                out_dim=self.dim_encoder,
                hidden_dim=self.dim_encoder,
            ),
            edge_dim=1,
        )

        self.state_task_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.dim_encoder,
                out_dim=self.dim_encoder,
                hidden_dim=self.dim_encoder,
            ),
            edge_dim=2,
        )

        self.actor_gin = GINConv(
            nn=GinUnactivatedMLP(self.dim_encoder),
        )

        self.critic_gin = GINConv(
            nn=GinUnactivatedMLP(self.dim_encoder),
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
        data[("task", "task-critic", "critic")].edge_index = self.cnv.task_to_single
        return data.to(device)
