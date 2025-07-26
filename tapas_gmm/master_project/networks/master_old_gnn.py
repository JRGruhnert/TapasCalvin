from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tapas_gmm.master_project.networks.baseline import ActorCriticBase
from tapas_gmm.master_project.definitions import StateType
from tapas_gmm.master_project.networks.master_modules import (
    TransformEncoder,
    QuaternionEncoder,
    ScalarEncoder,
)
from tapas_gmm.master_project.old.master_graph import GraphData
from torch_geometric.nn import GATv2Conv


class HRL_GNN(ActorCriticBase):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        h_dim_encoder: int = 32,
        h_dim1: int = 256,
        h_dim2: int = 64,
    ):
        super().__init__()

        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(h_dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(h_dim_encoder),
                StateType.Scalar.name: ScalarEncoder(h_dim_encoder),
            }
        )

        # self.one_hot = torch.eye(num_states)
        self.register_buffer("one_hot", torch.eye(state_dim))  # Ensures gradient flow

        # 5) GAT A→B: in_channels=(d_A, d_B) → out_channels = d_B
        self.gat_ab = GATv2Conv(
            in_channels=h_dim_encoder,
            out_channels=state_dim * 2,
        )
        # 6) GAT B→C: in_channels=(d_B, d_C) → out_channels = d_C
        self.gat_bc = GATv2Conv(
            in_channels=state_dim * 2,
            out_channels=h_dim2,
        )

        # 7) Finaler Kopf: d_C → 1 Logit pro Policy-Knoten
        self.policy_head = nn.Linear(h_dim2, 1)

    def forward(self, data: GraphData) -> tuple[int, torch.Tensor]:
        b_scalar = self.encoder_scalar(data.b_scalar)
        b_euler = self.encoder_euler(data.b_euler)
        b_quat = self.encoder_quat(data.b_quat)
        # concateniere alle B-Knoten-Features:
        x_b = torch.cat(
            [
                b_euler,
                b_quat,
                b_scalar,
            ],
            dim=0,
        )
        # TODO: states have to stay in place so that oh vector makes sense
        # base_oh = torch.eye(x_b, device=x_b.device)
        x_oh = torch.cat([x_b, self.one_hot], dim=1)
        # x_oh = self.one_hot[: len(x_b)]  # Example usage
        x_b_updated = self.gat_ab((data.a, x_oh), data.ab_edges)  # → [num_states, d_B]

        x_c_updated = self.gat_bc(
            (x_b_updated, data.c), data.bc_edges
        )  # → [num_policies, d_C]
        print(data.ab_edges)
        print(data.a)
        print(x_oh)
        print(x_b_updated)
        print(x_c_updated)
        # 6) Finaler Kopf: [num_policies, d_C] → [num_policies, 1], dann squeeze → [num_policies]
        logits = self.policy_head(x_c_updated).squeeze(-1)

        print(data.bc_edges)
        print(logits)

        # 7) Log-Softmax über alle Policy-Logits
        # log_probs = F.log_softmax(logits, dim=0)
        dist = torch.distributions.Categorical(logits=logits)
        if self.training:
            action = dist.sample()
        else:
            action = dist.probs.argmax(dim=0)
        log_prob = dist.log_prob(action)
        # Checks wether parameter requires_grad status
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        return action.item(), log_prob


class HRL_GNN2(nn.Module):

    def __init__(
        self,
        state_dim: int,
        dim_a: int,
        action_dim: int,
        d_linear: int = 32,
    ):
        super().__init__()
        # Wir hängen an den d_linear-Vektor ein One-Hot der Länge num_states an → ergibt d_B:
        self.feature_dim = d_linear + state_dim  # +1 für One-Hot

        self.num_states = state_dim

        # self.one_hot = torch.eye(num_states)
        self.register_buffer("one_hot", torch.eye(state_dim))  # Ensures gradient flow

        self.encoder_a = nn.Linear(dim_a, self.feature_dim)

        self.encoder_b_euler = TransformEncoder(d_linear)
        self.encoder_b_quat = QuaternionEncoder(d_linear)
        self.encoder_b_scalar = ScalarEncoder(d_linear)

        self.encoder_c = nn.Linear(action_dim, self.feature_dim)

        # self.d_C = dim_c_out  # Policy-Knoten-Embedding-Dimension
        # 5) GAT A→B: in_channels=(d_A, d_B) → out_channels = d_B
        self.gat_ab = GATv2Conv(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
        )
        # 6) GAT B→C: in_channels=(d_B, d_C) → out_channels = d_C
        self.gat_bc = GATv2Conv(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
        )

        # 7) Finaler Kopf: d_C → 1 Logit pro Policy-Knoten
        self.policy_head = nn.Linear(self.feature_dim, 1)

    def forward(self, data: GraphData) -> tuple[int, torch.Tensor]:
        a_features = self.encoder_a(data.a)
        b_features_scalar = self.encoder_b_scalar(data.b_scalar)
        b_features_euler = self.encoder_b_euler(data.b_euler)
        b_features_quat = self.encoder_b_quat(data.b_quat)
        c_features = self.encoder_c(data.c)
        # concateniere alle B-Knoten-Features:
        b_features_temp = torch.cat(
            [
                b_features_euler,
                b_features_quat,
                b_features_scalar,
            ],
            dim=0,
        )

        b_features = torch.cat([b_features_temp, self.one_hot], dim=1)
        ab_features = torch.cat([a_features, b_features], dim=0)

        # AB Edges updated to represent collapsed a and b partitions
        ab_src = data.ab_edges[0]  # in [0…num_A−1]
        ab_dst = data.ab_edges[1] + a_features.size(0)  # shift B for the single a node
        ab_edges = torch.stack([ab_src, ab_dst], dim=0)

        agg_ab_features = self.gat_ab(ab_features, ab_edges)

        _, agg_b_features = agg_ab_features.split(
            [
                a_features.size(0),
                b_features.size(0),
            ],
            dim=0,
        )
        bc_features = torch.cat([agg_b_features, c_features], dim=0)

        # BC Edges updated to represent collapsed b and c partitions
        bc_src = data.bc_edges[0]
        bc_dst = data.bc_edges[1] + b_features.size(0)

        bc_edges = torch.stack([bc_src, bc_dst], dim=0)
        agg_bc_features = self.gat_bc(bc_features, bc_edges)

        # Split into B and C parts along the node dimension (dim=0):
        _, agg_c_features = agg_bc_features.split(
            [
                b_features.size(0),
                c_features.size(0),
            ],
            dim=0,
        )

        # Finaler Kopf: [num_policies, d_C] → [num_policies, 1], dann squeeze → [num_policies]
        logits = self.policy_head(agg_c_features).squeeze(-1)

        # Log-Softmax über alle Policy-Logits
        # log_probs = F.log_softmax(logits, dim=0)
        dist = torch.distributions.Categorical(logits=logits)
        # print(f"Logits: {dist.logits}")
        # print(f"Probs: {dist.probs}")
        if self.training:
            action = dist.sample()
        else:
            action = dist.probs.argmax(dim=0)
        log_prob = dist.log_prob(action)
        # print(log_prob)
        return action.item(), log_prob
