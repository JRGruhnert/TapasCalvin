import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import MessagePassing


class QuaternionEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, q):
        return self.fc(q)


class TransformEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, e):
        return self.fc(e)


class ScalarEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim),
            nn.ReLU(),
        )

    def forward(self, s):
        return self.fc(s)


class GinStateMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, s):
        return self.fc(s)


class GinActionMlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, s):
        return self.fc(s)


class SrcOnlyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean")  # or 'add', 'max'

    def message(self, x_j):  # only uses src features
        return x_j

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


class PureSourceGINConv(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr="add")  # GIN uses sum aggregation
        self.mlp = GinActionMlp(in_channels)

    def message(self, x_j):
        return x_j  # Only source node features

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)  # no x added
        return self.mlp(out)  # No destination feature at all
