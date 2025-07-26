from abc import ABC, abstractmethod
import numpy as np
import torch
import numpy as np

from tapas_gmm.master_project.definitions import (
    StateType,
    Task,
    State,
)
from tapas_gmm.master_project.observation import Observation


class StateConverter(ABC):
    def __init__(self, state: State, normalized: bool = True):
        self.min = state.value.min
        self.max = state.value.max
        self.normalized = normalized

    def clamp(self, x: np.ndarray | float) -> np.ndarray:
        return np.clip(x, self.min, self.max)

    @abstractmethod
    def value(self, current: np.ndarray | float) -> float | np.ndarray:
        """Return the (optionally normalized) raw feature value."""
        pass

    @abstractmethod
    def distance(self, current: np.ndarray | float, goal: np.ndarray | float) -> float:
        """Return the (optionally normalized) difference to the goal."""
        pass

    def normalize(self, x: np.ndarray | float) -> float:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.min) / (self.max - self.min)


class ScalarConverter(StateConverter):

    def value(self, current: float) -> float:
        clamped = self.clamp(current)
        if not self.normalized:
            return clamped
        else:
            return self.normalize(clamped)

    def distance(self, current: float, goal: float) -> float:
        cl_curr = self.clamp(current)
        cl_goal = self.clamp(goal)
        if not self.normalized:
            return abs(cl_goal - cl_curr)
        else:
            norm_curr = self.normalize(cl_curr)
            norm_goal = self.normalize(cl_goal)
            return abs(norm_goal - norm_curr)


class TransformConverter(StateConverter):
    def value(self, current: np.ndarray) -> np.ndarray:
        clamped = self.clamp(current)
        if not self.normalized:
            return clamped
        else:
            return self.normalize(clamped)

    def distance(self, current: np.ndarray, goal: np.ndarray) -> float:
        """
        Returns the Euclidean distance between current and goal
        (after clamping current), optionally normalized by max span.
        """
        cl_curr = self.clamp(current)
        cl_goal = self.clamp(goal)
        if not self.normalized:
            return np.linalg.norm(cl_curr - cl_goal)
        else:
            norm_curr = self.normalize(cl_curr)
            norm_goal = self.normalize(cl_goal)
            return np.linalg.norm(norm_curr - norm_goal)


class QuaternionConverter(StateConverter):
    def __init__(self, state, normalized: bool = True):
        """
        Quaternions are assumed unit‑length; we don't clamp components.
        """
        super().__init__(state, normalized)
        self.ident = np.array([0.0, 0.0, 0.0, 1.0])

    def normalize_quat(self, q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)

    def angular_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Smallest rotation angle between two quaternions, in radians [0, π].
        """
        q1u = self.normalize_quat(q1)
        q2u = self.normalize_quat(q2)
        dot = np.clip(np.abs(np.dot(q1u, q2u)), -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def canonicalize(self, q: np.ndarray) -> np.ndarray:
        """
        Enforce qw >= 0 by flipping sign if needed.
        q is assumed unit‑length.
        """
        # quaternion format [qx, qy, qz, qw]
        if q[3] < 0:
            return -q
        return q

    def value(self, current: np.ndarray) -> np.ndarray:
        """
        Returns the normalized quaternion, or the identity quaternion if current is None.
        """
        if not self.normalized:
            return self.canonicalize(current)
        else:
            q_unit = self.normalize_quat(current)
            return self.canonicalize(q_unit)

    def distance(self, current: np.ndarray, goal: np.ndarray) -> float:
        """
        Returns the angular distance between current and goal quaternions,
        optionally normalized by π to lie in [0,1].
        """
        angle = self.angular_distance(current, goal)
        if not self.normalized:
            return float(angle)
        return float(angle / np.pi)


class IgnoreConverter(StateConverter):
    def __init__(self):
        """
        Quaternions are assumed unit‑length; we don't clamp components.
        """

    def value(self, current: np.ndarray | float) -> np.ndarray | float:
        """
        Returns zeros as in current size, indicating no contribution to the feature.
        """
        if isinstance(current, np.ndarray):
            value = np.zeros_like(current)
            if current.shape[0] == 3:
                return value
            elif current.shape[0] == 4 or current.shape[0] == 7:
                value[-1] = 1.0  # last element is 1.0 is for unit quaternion
                return value

        else:
            return 0.0

    def distance(self, current: np.ndarray | float, goal: np.ndarray | float) -> float:
        """
        Returns a constant value of 0.0, indicating no contribution to the feature.
        """
        return 0.0


class Converter:

    def __init__(
        self,
        tasks: list[Task],
        states: list[State],
        task_parameter: dict[Task, dict[State, np.ndarray]] = None,
        normalized: bool = True,
    ):
        """
        Initialize the converter with a goal observation.
        """
        self.tasks = tasks
        self.states = states
        self.tps = task_parameter
        self.converter: dict[State, StateConverter] = {}
        self.ignore_converter = IgnoreConverter()
        for state in states:
            if state.value.type == StateType.Transform:
                self.converter[state] = TransformConverter(state, normalized)
            elif state.value.type == StateType.Quaternion:
                self.converter[state] = QuaternionConverter(state, normalized)
            elif state.value.type == StateType.Scalar:
                self.converter[state] = ScalarConverter(state, normalized)
            else:
                raise ValueError(f"Unsupported state type: {state.value.type}")

    def dict_distance(
        self,
        current: Observation,
        goal: Observation,
    ) -> dict[State, torch.Tensor]:
        """
        Compute the distance for each state in the observation.
        Returns a dictionary mapping each state to its distance tensor.
        """
        distances = {}
        for key, converter in self.converter.items():
            dist = converter.distance(current.states[key], goal.states[key])
            distances[key] = torch.tensor(dist).float()
        return distances

    def tensor_dict_distance(
        self,
        current: Observation,
        goal: Observation,
    ) -> dict[StateType, torch.Tensor]:
        # Initialize groups
        transform_values = []
        quaternion_values = []
        scalar_values = []
        for key, converter in self.converter.items():
            val = converter.distance(current.states[key], goal.states[key])
            if key.value.type == StateType.Transform:
                transform_values.append(np.array([val]))
            elif key.value.type == StateType.Quaternion:
                quaternion_values.append(np.array([val]))
            elif key.value.type == StateType.Scalar:
                scalar_values.append(np.array([val]))

        # Stack each group (assumes all shapes in group match!)
        return {
            StateType.Transform: torch.from_numpy(np.stack(transform_values)).float(),
            StateType.Quaternion: torch.from_numpy(np.stack(quaternion_values)).float(),
            StateType.Scalar: torch.from_numpy(np.stack(scalar_values)).float(),
        }

    def tensor_combined_distance(
        self, current: Observation, goal: Observation
    ) -> torch.Tensor:
        # Convert all values to numpy arrays and concatenate
        values = []
        for key, converter in self.converter.items():
            val = converter.distance(current.states[key], goal.states[key])
            val = np.asarray(val).flatten()  # Ensures it's an array and flattens it
            values.append(val)

        # Concatenate all into a single flat array
        flat_array = np.concatenate(values)
        result_1d = torch.from_numpy(flat_array).float()
        return result_1d.unsqueeze(0)

    def tensor_state_dict_values(
        self,
        obs: Observation,
    ) -> dict[State, torch.Tensor]:
        """
        Compute the value for each state in the observation.
        Returns a dictionary mapping each state to its value.
        """
        values = {}
        for key, converter in self.converter.items():
            val = converter.value(obs.states[key])
            values[key] = torch.from_numpy(val).float()
        return values

    def tensor_type_dict_values(
        self,
        obs: Observation,
    ) -> dict[StateType, torch.Tensor]:
        grouped = {t: [] for t in StateType}

        for key, converter in self.converter.items():
            value = converter.value(obs.states[key])
            grouped[key.value.type].append(value)

        return {
            t: torch.from_numpy(np.stack(vals)).float()
            for t, vals in grouped.items()
            if vals  # only include non-empty
        }

    def tensor_combined_values(self, current: Observation) -> torch.Tensor:
        # Convert all values to numpy arrays and concatenate
        values = []
        for key, converter in self.converter.items():
            val = converter.value(current.states[key])
            val = np.asarray(val).flatten()  # Ensures it's an array and flattens it
            values.append(val)

        # Concatenate all into a single flat array
        flat_array = np.concatenate(values)
        result_1d = torch.from_numpy(flat_array).float()
        return result_1d.unsqueeze(0)

    def tensor_task_distance(
        self,
        current: Observation,
    ) -> torch.Tensor:
        features: list[np.ndarray] = []
        for task in self.tasks:
            task_features: list[float] = []
            task_tps = self.tps[task]
            for key, converter in self.converter.items():
                if key in task_tps:
                    # Use the task-specific value if available
                    task_value = converter.distance(current.states[key], task_tps[key])
                else:
                    # Empty value if not specified
                    task_value = self.ignore_converter.distance(
                        current.states[key], current.states[key]
                    )

                task_features.append(task_value)
            features.append(np.array(task_features))
        return torch.from_numpy(np.stack(features, axis=0)).float()

    def state_state_edges(self, full: bool) -> torch.Tensor:
        num_states = len(self.states)
        if not full:
            # Only connects edges with same index
            # (therefore being the same state but with different value)
            src = torch.arange(num_states)
            dst = torch.arange(num_states)
        else:
            # Fully connected Graph
            src = torch.arange(num_states).unsqueeze(1).repeat(1, num_states).flatten()
            dst = torch.arange(num_states).repeat(num_states)
        return torch.stack([src, dst], dim=0)

    def state_task_edges(self, full: bool) -> torch.Tensor:
        if not full:
            edge_list = []
            for task_idx, task in enumerate(self.tasks):
                task_tps = self.tps[task]
                for state_idx, state in enumerate(self.states):
                    if state in task_tps:
                        # connect B-node b_idx to C-node c_idx
                        edge_list.append((state_idx, task_idx))
            return torch.tensor(edge_list, dtype=torch.long).t()
        else:
            src = (
                torch.arange(len(self.states))
                .unsqueeze(1)
                .repeat(1, len(self.tasks))
                .flatten()
            )
            dst = torch.arange(len(self.tasks)).repeat(len(self.states))
            return torch.stack([src, dst], dim=0)

    def state_state_attr(self) -> torch.Tensor:
        # Build edge_index using ab_edges()
        edge_index = self.state_state_edges(True)  # shape [2, E]
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Set attribute to 1 if src == dst, else 0
        edge_attr = (src == dst).to(torch.float).unsqueeze(-1)  # shape [E, 1]
        return edge_attr

    def state_task_attr(self, weighted: bool = False) -> torch.Tensor:
        # Fully connected edge index
        # TODO: Implement weighted
        full_edge_index = self.state_task_edges(full=True)  # shape [2, E]

        # Sparse (actual) edge index
        sparse_edge_index = self.state_task_edges(full=False)  # shape [2, E_sparse]

        # Build set of valid (state_idx, task_idx) from sparse edges
        sparse_edge_set = set((s.item(), t.item()) for s, t in sparse_edge_index.t())

        # Create attribute tensor: 1 if (s, t) in sparse, else 0
        attrs = [
            1.0 if (s.item(), t.item()) in sparse_edge_set else 0.0
            for s, t in full_edge_index.t()
        ]
        edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(-1)  # shape [E, 1]

        return edge_attr
