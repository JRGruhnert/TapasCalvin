import torch.nn as nn
from torch_geometric.nn import MessagePassing


class TwoLayerMLP(nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )


class GinStandardMLP(nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )


class GinUnactivatedMLP(nn.Sequential):
    def __init__(self, dim_in):
        super().__init__(
            nn.Linear(dim_in, dim_in // 2),
            nn.BatchNorm1d(dim_in // 2),
            nn.ReLU(),
            nn.Linear(dim_in // 2, 1),
        )
