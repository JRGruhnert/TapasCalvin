from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict
from loguru import logger
import numpy as np
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    StateInfo,
    StateSpace,
    StateType,
    Task,
    State,
)
from tapas_gmm.master_project.master_helper import HRLHelper
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.policy.models.tpgmm import Gaussian
from scipy.stats import chi2


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


class NodeConverter:
    def __init__(
        self,
        state_list: list[State],
        task_list: list[Task],
        normalized: bool = True,
    ):
        """
        Initialize the converter with a goal observation.
        """
        self.tasks = task_list
        self.states = state_list
        self.converter: Dict[State, StateConverter] = {}
        self.ignore_converter = IgnoreConverter()
        for state in state_list:
            if state.value.state_type == StateType.Transform:
                self.converter[state] = TransformConverter(state, normalized)
            elif state.value.state_type == StateType.Quat:
                self.converter[state] = QuaternionConverter(state, normalized)
            elif state.value.state_type == StateType.Scalar:
                self.converter[state] = ScalarConverter(state, normalized)
            else:
                raise ValueError(f"Unsupported state type: {state.value.state_type}")

    def dict_distance(
        self,
        current: HRLPolicyObservation,
        goal: HRLPolicyObservation,
    ) -> Dict[State, torch.Tensor]:
        """
        Compute the distance for each state in the observation.
        Returns a dictionary mapping each state to its distance tensor.
        """
        distances = {}
        for key, converter in self.converter.items():
            dist = converter.distance(current.split_states[key], goal.split_states[key])
            distances[key] = torch.tensor(dist).float()
        return distances

    def tensor_dict_distance(
        self,
        current: HRLPolicyObservation,
        goal: HRLPolicyObservation,
    ) -> Dict[StateType, torch.Tensor]:
        # Initialize groups
        transform_values = []
        quaternion_values = []
        scalar_values = []
        for key, converter in self.converter.items():
            val = converter.distance(current.split_states[key], goal.split_states[key])
            if key.value.state_type == StateType.Transform:
                transform_values.append(np.array([val]))
            elif key.value.state_type == StateType.Quat:
                quaternion_values.append(np.array([val]))
            elif key.value.state_type == StateType.Scalar:
                scalar_values.append(np.array([val]))

        # Stack each group (assumes all shapes in group match!)
        return {
            StateType.Transform: torch.from_numpy(np.stack(transform_values)).float(),
            StateType.Quat: torch.from_numpy(np.stack(quaternion_values)).float(),
            StateType.Scalar: torch.from_numpy(np.stack(scalar_values)).float(),
        }

    def tensor_combined_distance(
        self, current: HRLPolicyObservation, goal: HRLPolicyObservation
    ) -> torch.Tensor:
        # Convert all values to numpy arrays and concatenate
        values = []
        for key, converter in self.converter.items():
            val = converter.distance(current.split_states[key], goal.split_states[key])
            val = np.asarray(val).flatten()  # Ensures it's an array and flattens it
            values.append(val)

        # Concatenate all into a single flat array
        flat_array = np.concatenate(values)
        result_1d = torch.from_numpy(flat_array).float()
        return result_1d.unsqueeze(0)

    def dict_values(
        self,
        obs: HRLPolicyObservation,
    ) -> Dict[State, float | np.ndarray]:
        """
        Compute the value for each state in the observation.
        Returns a dictionary mapping each state to its value.
        """
        values = {}
        for key, converter in self.converter.items():
            val = converter.value(obs.split_states[key])
            values[key] = val
        return values

    def tensor_dict_values(
        self,
        obs: HRLPolicyObservation,
    ) -> Dict[StateType, torch.Tensor]:
        # Initialize groups
        transform_values = []
        quaternion_values = []
        scalar_values = []
        for key, converter in self.converter.items():
            val = converter.value(obs.split_states[key])
            if key.value.state_type == StateType.Transform:
                transform_values.append(val)
            elif key.value.state_type == StateType.Quat:
                quaternion_values.append(val)
            elif key.value.state_type == StateType.Scalar:
                scalar_values.append(np.array([val]))

        # Stack each group (assumes all shapes in group match!)
        return {
            StateType.Transform: torch.from_numpy(np.stack(transform_values)).float(),
            StateType.Quat: torch.from_numpy(np.stack(quaternion_values)).float(),
            StateType.Scalar: torch.from_numpy(np.stack(scalar_values)).float(),
        }

    def tensor_combined_values(self, current: HRLPolicyObservation) -> torch.Tensor:
        # Convert all values to numpy arrays and concatenate
        values = []
        for key, converter in self.converter.items():
            val = converter.value(current.split_states[key])
            val = np.asarray(val).flatten()  # Ensures it's an array and flattens it
            values.append(val)

        # Concatenate all into a single flat array
        flat_array = np.concatenate(values)
        result_1d = torch.from_numpy(flat_array).float()
        return result_1d.unsqueeze(0)

    def tensor_task_distance(
        self,
        current: HRLPolicyObservation,
    ) -> torch.Tensor:
        features: list[np.ndarray] = []
        for task in self.tasks:
            task_features: list[float] = []
            tp_dict = HRLHelper.get_tp_from_task(
                task, split_pose=True, active_states=self.states
            )
            for key, converter in self.converter.items():
                if key in tp_dict:
                    # Use the task-specific value if available
                    task_value = converter.distance(
                        current.split_states[key], tp_dict[key]
                    )
                else:
                    # Empty value if not specified
                    task_value = self.ignore_converter.distance(
                        current.split_states[key], current.split_states[key]
                    )

                task_features.append(task_value)
            features.append(np.array(task_features))
        return torch.from_numpy(np.stack(features, axis=0)).float()


class EdgeConverter:
    def __init__(self, active_states: list[State], active_tasks: list[Task]):
        """
        Initialize the edge converter with a list of active states and tasks.
        """
        self.active_states = active_states
        self.active_tasks = active_tasks

    def ab_edges(self, gin_based: bool) -> torch.Tensor:
        num_states = len(self.active_states)
        if gin_based:
            src = torch.arange(num_states)
            dst = torch.arange(num_states)
        else:
            src = torch.arange(num_states).unsqueeze(1).repeat(1, num_states).flatten()
            dst = torch.arange(num_states).repeat(num_states)
        return torch.stack([src, dst], dim=0)

    def bc_edges(self, gin_based: bool) -> torch.Tensor:
        edge_list = []
        for task_idx, task in enumerate(self.active_tasks):
            tp_dict = HRLHelper.get_tp_from_task(task, True, self.active_states)
            for state_idx, state in enumerate(self.active_states):
                if state in tp_dict:
                    # connect B-node b_idx to C-node c_idx
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    def ab_attr(self) -> torch.Tensor:
        # Build edge_index using ab_edges()
        edge_index = self.ab_edges(False)  # shape [2, E]
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Set attribute to 1 if src == dst, else 0
        edge_attr = (src == dst).to(torch.float).unsqueeze(-1)  # shape [E, 1]
        return edge_attr


class P_C_GaussianConverter(StateConverter):
    def __init__(self, g: Gaussian):
        """
        Represents a single Gaussian prior.

        Parameters:
        - mu (np.ndarray): Mean vector of shape (D,)
        - sigma (np.ndarray): Covariance matrix of shape (D, D)
        """
        self._g = g
        self._mu, self._sigma = g.get_mu_sigma(mu_on_tangent=True, as_np=True)
        self._dof = self._mu.shape[0]
        self._det_sigma = np.linalg.det(self._sigma)

    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Compute the Mahalanobis distance between x and the stored mean/cov.

        Parameters:
        - x (np.ndarray): Input vector of shape (D,)

        Returns:
        - float: Mahalanobis distance
        """

        # Swap third (index 2) and last (index -1)
        x[3], x[-1] = x[-1], x[3]

        padd_x = np.concatenate([x, np.array([1.0])])  # 1.0 for gripper

        mu_test, _ = self._g.get_mu_sigma(mu_on_tangent=False, as_np=True)

        logp = self._g.prob_from_np(mu_test, log=True)

        # recover Mahalanobis distance
        norm_term = 0.5 * (self._dof * np.log(2 * np.pi) + np.log(self._det_sigma))

        d2 = max(-2 * (logp + norm_term), 0.0)
        return np.sqrt(d2)

    def _mahalanobis_score(self, x: np.ndarray) -> float:
        """
        Compute a normalized Mahalanobis alignment score in [0, 1].

        Parameters:
        - x (np.ndarray): Input vector of shape (D,)

        Returns:
        - float: A score in [0, 1], higher is better alignment (1 is perfect match)
        """
        d2 = self._mahalanobis_distance(x) ** 2
        return 1.0 - chi2.cdf(d2, df=self._dof)

    def distance(self, current: np.ndarray) -> float:
        """
        Compute the Mahalanobis distance between current and the stored mean/cov.

        Parameters:
        - current (np.ndarray): Current observation vector of shape (D,)

        Returns:
        - float: Mahalanobis distance
        """
        result = self._mahalanobis_score(current)
        return result


class P_C_StartPoseConverter:
    def __init__(self, reference_pose: np.ndarray, weights: np.ndarray = None):
        """
        Compare poses using a weighted distance.

        Parameters:
        - reference_pose (np.ndarray): [x, y, z, qx, qy, qz, qw] reference pose
        - weights (np.ndarray): length-6 vector for weighting pos + rot diffs
                                If None, defaults to [1,1,1,1,1,1]
        """
        assert reference_pose.shape == (7,), "Pose must be [x, y, z, qx, qy, qz, qw]"
        self._ref_pose = reference_pose
        # weights = np.ndarray([2, 2, 2, 1, 1, 1])
        self._weights = weights if weights is not None else np.ones(6)

    def _quat_log_map(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Return rotation difference log(q2⁻¹ * q1) as 3D rotvec"""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r_rel = r2.inv() * r1
        return r_rel.as_rotvec()  # shape (3,)

    def _distance(self, query_pose: np.ndarray) -> float:
        """
        Compute weighted Mahalanobis-like distance between two poses.

        Parameters:
        - query_pose (np.ndarray): [x, y, z, qx, qy, qz, qw]

        Returns:
        - float: distance
        """
        assert query_pose.shape == (7,), "Pose must be [x, y, z, qx, qy, qz, qw]"
        pos_diff = query_pose[:3] - self._ref_pose[:3]
        rot_diff = self._quat_log_map(query_pose[3:], self._ref_pose[3:])
        delta = np.concatenate([pos_diff, rot_diff])  # shape (6,)
        weighted = self._weights * delta**2
        return np.sqrt(np.sum(weighted))

    def _similarity_score(self, query_pose: np.ndarray) -> float:
        """
        Return a similarity score between 0 and 1 (1 = exact match)

        Parameters:
        - query_pose (np.ndarray): [x, y, z, qx, qy, qz, qw]

        Returns:
        - float: similarity score
        """
        return self._distance(query_pose)

    def feature(self, current: np.ndarray) -> float:
        return self._similarity_score(current)


class P_C_IgnoreConverter(StateConverter):
    def distance(self, _: np.ndarray) -> float:
        """
        Returns a constant value of 0.0, indicating no contribution to the feature.
        """
        return 0.0


class P_C_NodeConverter:
    def __init__(self, model: Task):
        self._feature_converter: dict[State, StateConverter] = {
            obs_state: P_C_IgnoreConverter() for obs_state in HRLHelper.c_states()
        }
        for key, value in HRLHelper.get_tp_from_task(model).items():
            self._feature_converter[key] = P_C_StartPoseConverter(value)

        for key, value in model.value.precondition.items():
            if isinstance(value, float):
                self._feature_converter[key] = ScalarConverter(key, value)
            else:
                self._feature_converter[key] = P_C_StartPoseConverter(value)

    def node_features(
        self,
        obs: HRLPolicyObservation,
        only_precondition: bool = False,
    ) -> np.ndarray:
        if not only_precondition:
            features: list[float] = [
                self._feature_converter[key].distance(value)
                for key, value in obs.normal_states.items()
            ]

        else:
            features: list[float] = [
                self._feature_converter[key].distance(value)
                for key, value in obs.normal_states.items()
                if not isinstance(self._feature_converter[key], P_C_IgnoreConverter)
            ]
        return np.array(features)

    def get_ee_score(self, obs: HRLPolicyObservation) -> float:
        return self._feature_converter[State.EE_Pose].distance(
            obs.normal_states[State.EE_Pose]
        )

    @cached_property
    def index_list(self) -> list[int]:
        # TODO: Fix
        indices: list[int] = []
        for idx, feature_converter in enumerate(self._feature_converter.values()):
            if not isinstance(feature_converter, P_C_IgnoreConverter):
                indices.append(idx)
        return indices


class P_C_Converter:
    def __init__(self, dataset: ActionSpace, states: list[State]):
        self._node_converter: Dict[Task, P_C_NodeConverter] = {
            model: P_C_NodeConverter(model) for model in HRLHelper.models(dataset)
        }
        self.states: list[State] = states

    def partition_features(self, obs: HRLPolicyObservation) -> torch.Tensor:
        """Convert the observation into a feature vector for Partition C using the node converters."""
        features: list[np.ndarray] = []
        for _, converter in self._node_converter.items():
            features.append(converter.node_features(obs))

        return torch.from_numpy(np.stack(features, axis=0)).float()

    def evaluate_ee_pose(self, obs: HRLPolicyObservation) -> float:
        raise NotImplementedError()

    @cached_property
    def bc_edges(self) -> torch.Tensor:
        edge_list = []
        # TODO: Object poses hardcoded because all the same, but have to change later
        for i, node_converter in enumerate(self._node_converter.items()):
            for state in node_converter[0].value.precondition:
                if state.value.state_type == StateType.Pose:
                    sub_states: list[State] = State.from_pose_string(
                        state.value.identifier
                    )
                    for sub_state in sub_states:
                        index = self.states.index(sub_state)
                        edge_list.append([index, i])
                else:
                    index = self.states.index(state)
                    edge_list.append([index, i])  # edge from j to i

        # Converts to tensor with shape [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # transposes to [2, N]
        logger.warning(f"BC Edges {edge_index}")
        return edge_index
