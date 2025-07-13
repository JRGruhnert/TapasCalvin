from typing import Dict
import torch
from torch_geometric.data import HeteroData

from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    State,
    StateSpace,
    Task,
)
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.master_project.master_converter import (
    NodeConverter,
    P_C_Converter,
)


class GraphData(HeteroData):
    def __init__(self):
        super().__init__()

        # Initialize empty node features (can be set later)
        self["partition_a"].x = torch.tensor([], dtype=torch.float32)
        self["partition_b_euler"].x = torch.tensor([], dtype=torch.float32)
        self["partition_b_quat"].x = torch.tensor([], dtype=torch.float32)
        self["partition_b_scalar"].x = torch.tensor([], dtype=torch.float32)
        self["partition_c"].x = torch.tensor([], dtype=torch.float32)

        self._b_size: int = (
            self.b_euler.size(0) + self.b_quat.size(0) + self.b_scalar.size(0)
        )

        self._ab_edges: torch.Tensor = torch.empty((2, 0), dtype=torch.int)

        self._bc_edges: torch.Tensor = torch.empty((2, 0), dtype=torch.int)

    # AB Edges
    @property
    def ab_edges(self) -> torch.Tensor:
        return self._ab_edges

    def set_ab_edges(self):
        self._ab_edges = torch.cartesian_prod(
            torch.arange(len(self["partition_a"].x)),
            torch.arange(self._b_size),
        ).T

    # BC Edges
    @property
    def bc_edges(self) -> torch.Tensor:
        return self._bc_edges

    def set_bc_edges(self, edges: torch.Tensor):
        self._bc_edges = edges

    # Partition A
    @property
    def a(self) -> torch.Tensor:
        return self["partition_a"].x

    def set_a(self, features: torch.Tensor):
        self["partition_a"].x = features

    def set_b(self, features: Dict[str, torch.Tensor]):
        self.set_b_euler(features["euler"])
        self.set_b_quat(features["quat"])
        self.set_b_scalar(features["scalar"])
        self._b_size = (
            self.b_euler.size(0) + self.b_quat.size(0) + self.b_scalar.size(0)
        )
        self.set_ab_edges()

    # Partition B - Euler
    @property
    def b_euler(self) -> torch.Tensor:
        return self["partition_b_euler"].x

    def set_b_euler(self, features: torch.Tensor):
        self["partition_b_euler"].x = features

    # Partition B - Quaternion
    @property
    def b_quat(self) -> torch.Tensor:
        return self["partition_b_quat"].x

    def set_b_quat(self, features: torch.Tensor):
        self["partition_b_quat"].x = features

    # Partition B - Scalar
    @property
    def b_scalar(self) -> torch.Tensor:
        return self["partition_b_scalar"].x

    def set_b_scalar(self, features: torch.Tensor):
        self["partition_b_scalar"].x = features

    # Partition C
    @property
    def c(self) -> torch.Tensor:
        return self["partition_c"].x

    def set_c(self, features: torch.Tensor):
        self["partition_c"].x = features


class Graph:
    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
    ):
        state_list = State.list_by_state_space(state_space)
        task_list = Task.list_by_action_space(action_space)
        self.converter = NodeConverter(state_list, task_list, True)
        self.data = GraphData()
        self.data.set_bc_edges(self.c_converter.bc_edges)

    def update(
        self, current: HRLPolicyObservation, goal: HRLPolicyObservation
    ) -> GraphData:
        self.data.set_a(self.converter.tensor(current))
        self.data.set_b(self.converter.tensor_dict(current))
        self.data.set_c(self.converter.partition_features(current))
        return self.data
