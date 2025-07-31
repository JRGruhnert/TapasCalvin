import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINEConv, GINConv, GIN
from torch_geometric.nn import GATv2Conv
from tapas_gmm.master_project.observation import Observation
from tapas_gmm.master_project.networks.base import GnnBase, PPOType
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.networks.layers.master_modules import (
    GinStandardMLP,
    StandardMLP,
)


class GatReadoutNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_action: int,
        ppo_type: PPOType,
    ):
        super().__init__()
        self.ppo_type = ppo_type
        self.dim_action = dim_action
        self.dim_features = dim_features

        self.state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=dim_features,
                out_dim=dim_features,
                hidden_dim=dim_features,
            ),
            # edge_dim=1,
        )

        self.task_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=dim_features,
                out_dim=dim_features,
                hidden_dim=dim_features,
            ),
            edge_dim=2,
        )
        self.gat = GATv2Conv(
            dim_features,
            dim_features,
            concat=False,
            add_self_loops=False,
        )

        self.critic_mlp = StandardMLP(
            in_dim=dim_features,
            out_dim=1,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x1 = self.state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            # edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.task_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        x3, (edge_index, att_weights) = self.gat(
            x=(x2, x_dict["value"]),
            edge_index=edge_index_dict[("task", "task-value", "value")],
            return_attention_weights=True,
        )
        if self.ppo_type is PPOType.ACTOR:
            return att_weights.view(-1, self.dim_action)  # [B, dim_tasks]
        else:
            x3 = x3.view(-1, self.dim_features)  # [B, dim_encoder]
            return self.critic_mlp(x3).squeeze(-1)  # [B]


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.actor = GatReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_action=self.dim_tasks,
            ppo_type=PPOType.ACTOR,
        )

        self.critic = GatReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_action=self.dim_tasks,
            ppo_type=PPOType.CRITIC,
        )

    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        probs = self.actor(batch)
        value = self.critic(batch)
        return probs, value

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
        # task_tensor = self.cnv.tensor_task_distance(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = torch.zeros(self.dim_tasks, self.dim_encoder)
        data["value"].x = torch.zeros(1, self.dim_encoder)

        data[("goal", "goal-obs", "obs")].edge_index = self.cnv.state_state_sparse
        data[("obs", "obs-task", "task")].edge_index = self.cnv.state_task_full
        # data[("task", "task-value", "value")].edge_index = self.cnv.task_to_single
        data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        # data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr
        data[("obs", "obs-task", "task")].edge_attr = self.cnv.state_task_attr_weighted(
            obs
        )
        return data.to(device)

    def act(
        self,
        obs: Observation,
        goal: Observation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward([obs], [goal])
        assert probs.shape == (
            1,
            self.dim_tasks,
        ), f"Expected logits shape ({1}, {self.dim_tasks}), got {probs.shape}"
        assert value.shape == (1,), f"Expected value shape ({1},), got {value.shape}"
        dist = Categorical(probs=probs)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(
        self,
        obs: list[Observation],
        goal: list[Observation],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(obs) == len(goal), "Observation and Goal lists have different sizes."
        probs, value = self.forward(obs, goal)
        assert probs.shape == (
            len(obs),
            self.dim_tasks,
        ), f"Expected logits shape ({len(obs)}, {self.dim_tasks}), got {probs.shape}"
        assert value.shape == (
            len(obs),
        ), f"Expected value shape ({len(obs)},), got {value.shape}"
        dist = Categorical(probs=probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy
