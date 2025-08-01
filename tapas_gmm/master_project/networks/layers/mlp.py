import torch.nn as nn


class StandardMLP(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )


class UnactivatedMLP(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )


class GinStandardMLP(nn.Sequential):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )


class GinUnactivatedMLP(nn.Sequential):
    def __init__(self, dim_in: int):
        super().__init__(
            nn.Linear(dim_in, dim_in // 2),
            nn.LayerNorm(dim_in // 2),
            nn.ReLU(),
            nn.Linear(dim_in // 2, 1),
        )
