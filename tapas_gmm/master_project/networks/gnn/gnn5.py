import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from tapas_gmm.master_project.networks.layers.master_modules import (
    GinStandardMLP,
    StandardMLP,
    UnactivatedMLP,
)
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase, PPOType
from tapas_gmm.utils.select_gpu import device


class CombinedGoalObsNetwork(nn.Module):
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

        self.state_task_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_features,
                hidden_dim=self.dim_features,
            ),
            edge_dim=2,
        )

        self.actor_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

        self.critic_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x1 = self.state_task_gin(
            x=(x_dict["state"], x_dict["task"]),
            edge_index=edge_index_dict[("state", "state-task", "task")],
            edge_attr=edge_attr_dict[("state", "state-task", "task")],
        )

        if self.ppo_type is PPOType.ACTOR:
            logits = self.actor_gin(
                x=(x1, x_dict["actor"]),
                edge_index=edge_index_dict[("task", "task-actor", "actor")],
            )

            return logits.view(-1, self.dim_tasks)

        else:
            value = self.critic_gin(
                x=(x1, x_dict["critic"]),
                edge_index=edge_index_dict[("task", "task-critic", "critic")],
            )

            return value.squeeze(-1)


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.state_mlp = StandardMLP(
            in_dim=self.dim_encoder * 2,
            out_dim=self.dim_encoder,
        )

        self.actor = CombinedGoalObsNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_state,
            dim_task=self.dim_tasks,
            ppo_type=PPOType.ACTOR,
        )
        self.critic = CombinedGoalObsNetwork(
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
        data[("task", "task-critic", "critic")].edge_index = self.cnv.task_to_single
        data[("state", "state-task", "task")].edge_index = self.cnv.state_task_full
        data[("state", "state-task", "task")].edge_attr = (
            self.cnv.state_task_attr_weighted(obs)
        )
        return data.to(device)
