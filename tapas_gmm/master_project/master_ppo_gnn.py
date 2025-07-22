import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, LayerNorm, GINConv, GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
from tapas_gmm.master_project.master_baseline import ActorCriticBase
from tapas_gmm.master_project.master_data_def import StateType
from tapas_gmm.master_project.master_encoder import (
    GinActionMlp,
    GinStateMlp,
    TransformEncoder,
    QuaternionEncoder,
    ScalarEncoder,
)
from tapas_gmm.master_project.master_graph import Graph


class GNN_PPO(ActorCriticBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        h_dim_encoder: int = 32,
        gat_out: int = 64,
        head_hidden: int = 32,
        attention_heads: int = 1,
    ):
        super().__init__()

        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.gat1 = GATv2Conv(
            (h_dim_encoder, h_dim_encoder),
            state_dim,
            heads=attention_heads,
            concat=False,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            (state_dim, state_dim),
            gat_out,
            heads=attention_heads,
            concat=False,
            # edge_dim=1,
            add_self_loops=False,
        )

        self.norm1 = LayerNorm(state_dim)
        self.norm2 = LayerNorm(gat_out)
        self.actor = nn.Sequential(
            # nn.BatchNorm1d(gat_out),
            nn.Linear(gat_out, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )

        self.critic = nn.Sequential(
            # nn.BatchNorm1d(gat_out),
            nn.Linear(gat_out, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )

        self.critic_readout = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(gat_out, 1), nn.Sigmoid())
        )
        self.critic_head = nn.Linear(gat_out, 1)

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in graph.a.items()]
        obs_encoded = [self.encoder_obs[k.name](v) for k, v in graph.b.items()]
        a_tensor = torch.cat(goal_encoded, dim=0)
        b_tensor = torch.cat(obs_encoded, dim=0)
        # a_tensor = torch.cat([a_encoded, self.one_hot], dim=1)
        # b_tensor = torch.cat([b_encoded, self.one_hot], dim=1)
        # print(f"A\t Mean: {a_tensor.mean().item()}\t Std: {a_tensor.std().item()}")
        # print(f"B\t Mean: {b_tensor.mean().item()} \t Std: {b_tensor.std().item()}")
        # print(f"C\t Mean: {graph.c.mean().item()} \t Std: {graph.c.std().item()}")

        x1: torch.Tensor = self.gat1(
            x=(a_tensor, b_tensor),
            edge_index=graph.state_state_edges_full,
            edge_attr=graph.state_state_attr,
            return_attention_weights=None,
        )
        # print(f"X1: {x1}")
        x2 = self.norm1(x1)
        # print(f"X2: {x2}")
        x3, (edge_idx_bc, attn_bc) = self.gat2(
            x=(x2, graph.c),
            edge_index=graph.state_task_edges_full,
            edge_attr=None,
            return_attention_weights=True,
        )
        # print(f"X3: {x3}")
        x4 = self.norm2(x3)
        # print(f"X4: {x4}")
        # print(edge_idx_bc)
        # print(attn_bc)

        print(f"X1 Tensor \t Mean: {x1.mean().item()} \t Std: {x1.std().item()}")
        print(f"X2 Tensor \t Mean: {x2.mean().item()} \t Std: {x2.std().item()}")
        print(f"X3 Tensor \t Mean: {x3.mean().item()} \t Std: {x3.std().item()}")
        print(f"X4 Tensor \t Mean: {x4.mean().item()} \t Std: {x4.std().item()}")
        # print(f"a_tensor.shape: {a_tensor.shape}")
        # print(f"b_tensor.shape: {b_tensor.shape}")
        # print(f"graph.c.shape: {graph.c.shape}")
        # print(f"ab_edges.shape: {graph.ab_edges.shape}")
        # print(f"bc_edges.shape: {graph.bc_edges.shape}")
        # print(f"x1.shape: {x1.shape}")
        # print(f"x2.shape: {x2.shape}")

        logits = self.actor(x4).squeeze(-1)
        # print("Logits")
        print(
            f"LO Tensor \t Mean: {logits.mean().item()} \t Std: {logits.std().item()}"
        )
        # v_feat = x2.mean(dim=0, keepdim=True)
        # value = self.critic(v_feat).squeeze(-1)
        v_feat = self.critic_readout(x4)  # → [1, gat_out]
        value = self.critic_head(v_feat).squeeze(-1)  # → scalar
        return logits, value

    def act(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action, logprob, value

    def evaluate(
        self,
        graph: Graph,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        # print(dist.probs)
        action_logprobs = dist.log_prob(action)
        # print(action_logprobs)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy


class GNN_PPO2(ActorCriticBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        h_dim_encoder: int = 32,
        gat_out: int = 64,
        head_hidden: int = 32,
        attention_heads: int = 1,
        hidden_mlp_dim1: int = 32,
        mlp_out: int = 16,
    ):
        super().__init__()

        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.state_gin = GINConv(
            nn=GinStateMlp(h_dim_encoder, hidden_mlp_dim1, state_dim),
        )

        self.action_gin = GINConv(
            nn=GinActionMlp(state_dim),
        )

        self.critic_head = GinActionMlp(2)

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in graph.a.items()]
        obs_encoded = [self.encoder_obs[k.name](v) for k, v in graph.b.items()]
        a_tensor = torch.cat(goal_encoded, dim=0)
        b_tensor = torch.cat(obs_encoded, dim=0)

        x1: torch.Tensor = self.state_gin(
            x=(a_tensor, b_tensor),
            edge_index=graph.state_state_edges_full,
        )
        x2 = self.action_gin(
            x=(x1, graph.c),
            edge_index=graph.state_task_edges_full,
        )

        logits = x2.squeeze(-1)
        max = global_max_pool(x2, None)
        mean = global_mean_pool(x2, None)
        max_val = max.max()
        mean_val = mean.mean()
        combined = torch.tensor([max_val, mean_val])
        value = self.critic_head(combined)
        return logits, value

    def act(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action, logprob, value

    def evaluate(
        self,
        graph: Graph,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        # print(dist.probs)
        action_logprobs = dist.log_prob(action)
        # print(action_logprobs)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy


class GNN_PPO3(ActorCriticBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        h_dim_encoder: int = 32,
        gat_out: int = 64,
        head_hidden: int = 32,
        attention_heads: int = 1,
        hidden_mlp_dim1: int = 32,
        mlp_out: int = 16,
    ):
        super().__init__()

        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quat.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.state_gin = GINEConv(
            nn=GinStateMlp(h_dim_encoder, hidden_mlp_dim1, state_dim),
            train_eps=True,
            edge_dim=1,
        )

        self.action_gin = GINEConv(
            nn=GinActionMlp(state_dim),
            train_eps=True,
            edge_dim=1,
        )

        self.critic_head = GinActionMlp(2)

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in graph.a.items()]
        obs_encoded = [self.encoder_obs[k.name](v) for k, v in graph.b.items()]
        a_tensor = torch.cat(goal_encoded, dim=0)
        b_tensor = torch.cat(obs_encoded, dim=0)

        x1: torch.Tensor = self.state_gin(
            x=(a_tensor, b_tensor),
            edge_index=graph.state_state_edges_full,
            edge_attr=graph.state_state_attr,
        )
        x2 = self.action_gin(
            x=(x1, graph.c),
            edge_index=graph.state_task_edges_full,
            edge_attr=graph.state_task_attr,
        )

        logits = x2.squeeze(-1)
        max = global_max_pool(x2, None)
        mean = global_mean_pool(x2, None)
        max_val = max.max()
        mean_val = mean.mean()
        combined = torch.tensor([max_val, mean_val])
        value = self.critic_head(combined)
        return logits, value

    def act(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action, logprob, value

    def evaluate(
        self,
        graph: Graph,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph)
        dist = Categorical(logits=logits)
        # print(dist.probs)
        action_logprobs = dist.log_prob(action)
        # print(action_logprobs)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy
