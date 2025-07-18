import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv
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
        gat_out: int = 32,
        head_hidden: int = 16,
        attention_heads: int = 3,
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
        )
        self.gat2 = GATv2Conv(
            (state_dim, state_dim), gat_out, heads=attention_heads, concat=False
        )

        self.actor = nn.Sequential(
            nn.BatchNorm1d(gat_out),
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

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        goal_encoded = [self.encoder_goal[k.name](v) for k, v in graph.a.items()]
        obs_encoded = [self.encoder_obs[k.name](v) for k, v in graph.b.items()]
        a_tensor = torch.cat(goal_encoded, dim=0)
        b_tensor = torch.cat(obs_encoded, dim=0)
        # print(f"A\t Mean: {a_tensor.mean().item()}\t Std: {a_tensor.std().item()}")
        # print(f"B\t Mean: {b_tensor.mean().item()} \t Std: {b_tensor.std().item()}")
        # print(f"C\t Mean: {graph.c.mean().item()} \t Std: {graph.c.std().item()}")

        x1: torch.Tensor = self.gat1((a_tensor, b_tensor), graph.ab_edges)
        x2: torch.Tensor = self.gat2((x1, graph.c), graph.bc_edges)
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
        v_feat = x2.mean(dim=0, keepdim=True)
        value = self.critic(v_feat).squeeze(-1)
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
