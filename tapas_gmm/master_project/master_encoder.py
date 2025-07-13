import torch.nn as nn
import torch.nn.functional as F


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
