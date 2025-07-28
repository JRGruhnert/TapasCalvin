from abc import ABC, abstractmethod
from functools import cached_property
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

    def clamp(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.min, self.max)

    @abstractmethod
    def value(self, current: np.ndarray) -> float | np.ndarray:
        """Return the (optionally normalized) raw feature value."""
        pass

    @abstractmethod
    def distance(self, current: np.ndarray, goal: np.ndarray) -> float:
        """Return the (optionally normalized) difference to the goal."""
        pass

    def normalize(self, x: np.ndarray) -> float | np.ndarray:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.min) / (self.max - self.min)


class ScalarConverter(StateConverter):

    def value(self, current: np.ndarray) -> np.ndarray:
        clamped = self.clamp(current)
        if not self.normalized:
            return clamped
        else:
            return self.normalize(clamped)

    def distance(self, current: np.ndarray, goal: np.ndarray) -> float:
        cl_curr = self.clamp(current)
        cl_goal = self.clamp(goal)
        if not self.normalized:
            return abs(cl_goal - cl_curr).item()
        else:
            norm_curr = self.normalize(cl_curr)
            norm_goal = self.normalize(cl_goal)
            return abs(norm_goal - norm_curr).item()


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
        pad: bool = False,
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

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        num_states = len(self.states)
        indices = torch.arange(num_states)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_state_full(self) -> torch.Tensor:
        num_states = len(self.states)
        src = torch.arange(num_states).unsqueeze(1).repeat(1, num_states).flatten()
        dst = torch.arange(num_states).repeat(num_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_task_sparse(self) -> torch.Tensor:
        edge_list = []
        for task_idx, task in enumerate(self.tasks):
            task_tps = self.tps[task]
            for state_idx, state in enumerate(self.states):
                if state in task_tps:
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def state_task_full(self) -> torch.Tensor:
        num_states = len(self.states)
        num_tasks = len(self.tasks)
        src = torch.arange(num_states).unsqueeze(1).repeat(1, num_tasks).flatten()
        dst = torch.arange(num_tasks).repeat(num_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def task_task_sparse(self) -> torch.Tensor:
        num_tasks = len(self.tasks)
        indices = torch.arange(num_tasks)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def task_single(self) -> torch.Tensor:
        num_tasks = len(self.tasks)
        indices = torch.arange(num_tasks)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def state_state_attr(self) -> torch.Tensor:
        # Build edge_index using ab_edges()
        edge_index = self.state_state_full  # shape [2, E]
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Set attribute to 1 if src == dst, else 0
        edge_attr = (src == dst).to(torch.float).unsqueeze(-1)  # shape [E, 1]
        return edge_attr

    @cached_property
    def state_task_attr(self) -> torch.Tensor:
        full = self.state_task_full  # [2, E]
        sparse = self.state_task_sparse  # [2, E_sparse]

        # Convert sparse edges to a flat index for fast lookup
        num_tasks = len(self.tasks)
        sparse_flat = sparse[0] * num_tasks + sparse[1]  # shape [E_sparse]
        full_flat = full[0] * num_tasks + full[1]  # shape [E]

        # Check which full edges are in sparse
        is_in_sparse = torch.isin(full_flat, sparse_flat)

        edge_attr = is_in_sparse.float().unsqueeze(-1)  # shape [E, 1]
        return edge_attr

    def tensor_task_distance(
        self,
        current: Observation,
        pad: bool = False,
    ) -> torch.Tensor:
        features: list[np.ndarray] = []
        for task in self.tasks:
            task_features: list[np.ndarray] = []
            task_tps = self.tps[task]
            for key, converter in self.converter.items():
                if key in task_tps:
                    val = converter.distance(current.states[key], task_tps[key])
                    task_value = np.array([val, 1.0]) if pad else np.array(val)
                else:
                    val = self.ignore_converter.distance(
                        current.states[key], current.states[key]
                    )
                    task_value = np.array([val, 0.0]) if pad else np.array(val)

                task_features.append(task_value)
            # Ensure consistent 2D shape: [num_states, feature_dim]
            task_features = np.stack(task_features, axis=0)  # shape: [num_states, 2]
            features.append(task_features)

        features = np.stack(features, axis=0)  # shape: [num_tasks, num_states, 2]
        return torch.from_numpy(features).float()

    def state_task_attr_weighted(self, current: Observation) -> torch.Tensor:
        full = self.state_task_full  # [2, E]
        state_indices = full[0]  # [E]
        task_indices = full[1]  # [E]

        dist_matrix = self.tensor_task_distance(current, pad=True)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[task_indices, state_indices]  # [E, 2]
        return edge_attr
