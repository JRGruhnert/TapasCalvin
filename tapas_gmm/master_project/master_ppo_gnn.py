import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from tapas_gmm.master_project.master_baseline import ActorCriticBase
from tapas_gmm.master_project.master_data_def import StateType
from tapas_gmm.master_project.master_encoder import (
    TransformEncoder,
    QuaternionEncoder,
    ScalarEncoder,
)
from tapas_gmm.master_project.master_graph import Graph


class Master_GNN_PPO(ActorCriticBase):
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
            add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            (state_dim, state_dim),
            gat_out,
            heads=attention_heads,
            concat=False,
            # edge_dim=1,
            add_self_loops=False,
        )

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
            edge_index=graph.ab_edges,
            edge_attr=graph.ab_edge_attr,
            return_attention_weights=None,
        )
        # print(x1)
        x2, (edge_idx_bc, attn_bc) = self.gat2(
            x=(x1, graph.c),
            edge_index=graph.bc_edges,
            edge_attr=None,
            return_attention_weights=True,
        )
        print(edge_idx_bc)
        print(attn_bc)
        # print(f"X1 Tensor \t Mean: {x1.mean().item()} \t Std: {x1.std().item()}")
        # print(f"X2 Tensor \t Mean: {x2.mean().item()} \t Std: {x2.std().item()}")
        # print(f"a_tensor.shape: {a_tensor.shape}")
        # print(f"b_tensor.shape: {b_tensor.shape}")
        # print(f"graph.c.shape: {graph.c.shape}")
        # print(f"ab_edges.shape: {graph.ab_edges.shape}")
        # print(f"bc_edges.shape: {graph.bc_edges.shape}")
        # print(f"x1.shape: {x1.shape}")
        # print(f"x2.shape: {x2.shape}")

        logits = self.actor(x2).squeeze(-1)
        # print("Logits")
        # print(logits)
        # v_feat = x2.mean(dim=0, keepdim=True)
        # value = self.critic(v_feat).squeeze(-1)
        v_feat = self.critic_readout(x2)  # → [1, gat_out]
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
