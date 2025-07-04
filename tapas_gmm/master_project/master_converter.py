from abc import abstractmethod
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
    ObservationState,
)
from tapas_gmm.master_project.master_helper import HRLHelper
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.policy.models.tpgmm import Gaussian
from scipy.stats import chi2


class FeatureConverter:
    @abstractmethod
    def feature(self) -> float:
        raise NotImplementedError("Subclasses must implement the feature method.")


class P_A_ScalarConverter:
    def node_features(self, current: float, goal: float) -> float:
        """
        Compute the absolute difference between the current and goal scalar values.

        Parameters:
        - current (float): Current scalar value.
        - goal (float): Goal scalar value.

        Returns:
        - float: Absolute difference between current and goal.
        """
        return abs(current - goal)


class P_A_EulerConverter:
    def node_features(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Compute the absolute difference between the current and goal Euler angles.

        Parameters:
        - current (np.ndarray): Current Euler angles as a numpy array.
        - goal (np.ndarray): Goal Euler angles as a numpy array.

        Returns:
        - float: Absolute difference between current and goal Euler angles.
        """
        if current.shape != goal.shape:
            raise ValueError("Current and goal must have the same shape.")
        return np.abs(current - goal)


class P_A_QuatConverter:
    def node_features(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Compute normalized angular distance between two quaternions (range [0, 1])."""
        # Absolute dot product to handle sign ambiguity
        dot = np.abs(np.dot(current, goal))
        angular_distance = 2 * np.arccos(
            np.clip(dot, -1.0, 1.0)
        )  # Clip for numerical stability
        return angular_distance / np.pi  # Scale to [0, 1]


class P_A_Converter:
    def __init__(self, goal: HRLPolicyObservation, state_space: StateSpace):
        """
        Initialize the converter with a goal observation.
        """
        self._goal = goal
        self.scalar_converter = P_A_ScalarConverter()
        self.euler_converter = P_A_EulerConverter()
        self.quat_converter = P_A_QuatConverter()

    def partition_features(self, current: HRLPolicyObservation) -> torch.Tensor:
        feature_dict: Dict[ObservationState, float] = {}
        for key, goal_value in self._goal.euler_states.items():
            feature_dict[key] = self.euler_converter.node_features(
                current.euler_states[key], goal_value
            )
        for key, goal_value in self._goal.quat_states.items():
            feature_dict[key] = self.quat_converter.node_features(
                current.quat_states[key], goal_value
            )
        for key, goal_value in self._goal.scalar_states.items():
            feature_dict[key] = self.scalar_converter.node_features(
                current.scalar_states[key], goal_value
            )

        # Convert all values to numpy arrays and concatenate
        values = []
        for key in self._goal.split_keys:
            val = feature_dict[key]
            val = np.asarray(val).flatten()  # Ensures it's an array and flattens it
            values.append(val)

        # Concatenate all into a single flat array
        flat_array = np.concatenate(values)
        result_1d = torch.from_numpy(flat_array).float()
        return result_1d.unsqueeze(0)

    def num_features(self, obs: HRLPolicyObservation) -> int:
        print(f"Dim A {self.partition_features(obs).size()}")
        return self.partition_features(obs).size(1)


class P_B_Converter:

    def partition_features(
        self,
        obs: HRLPolicyObservation,
        state_space: StateSpace = None,
    ) -> Dict[StateType, torch.Tensor]:
        # Initialize groups
        euler_values = [obs.euler_states[key] for key in obs.euler_states.keys()]
        quaternion_values = [obs.quat_states[key] for key in obs.quat_states.keys()]
        scalar_values = [
            np.array([obs.scalar_states[key]]) for key in obs.scalar_states.keys()
        ]

        # Stack each group (assumes all shapes in group match!)

        return {
            StateType.Euler: torch.from_numpy(np.stack(euler_values)).float(),
            StateType.Quat: torch.from_numpy(np.stack(quaternion_values)).float(),
            StateType.Scalar: torch.from_numpy(np.stack(scalar_values)).float(),
        }

    def num_states(self, obs: HRLPolicyObservation):
        return len(obs.euler_states) + len(obs.scalar_states) + len(obs.quat_states)

    def get_state_list(self, obs: HRLPolicyObservation) -> list[ObservationState]:
        return (
            list(obs.euler_states.keys())
            + list(obs.quat_states.keys())
            + list(obs.scalar_states.keys())
        )


class P_C_ScalarConverter(FeatureConverter):
    def __init__(self, state: ObservationState, target: float):
        self.target = target
        self.min = state.value.min
        self.max = state.value.max
        self.state = state

    def feature(self, current: float) -> float:
        if not isinstance(current, (int, float)):
            raise TypeError(
                f"Expected a numeric type, got {type(current)} with value {current}"
            )

        value = min(max(self.min, current), self.max)

        diff = abs(self.target - value)
        normalized = diff / (self.max - self.min)

        # print(f"Name: {self.state.value.identifier}")
        # print(f"Min: {self.state.value.min}")
        # print(f"Max: {self.state.value.max}")
        # print(f"Target: {self.target}")
        # print(f"Current: {current}")
        # print(f"Clipped: {value}")
        # normalize
        return normalized


class P_C_GaussianConverter(FeatureConverter):
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

    def feature(self, current: np.ndarray) -> float:
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
        print(f"Distance: {dist}")
        dist2 = 1 - np.exp(-dist)
        print(f"Distance2: {dist2}")
        return dist2

    def feature(self, current: np.ndarray) -> float:
        return self._similarity_score(current)


class P_C_IgnoreConverter(FeatureConverter):
    def feature(self, _: np.ndarray) -> float:
        """
        Returns a constant value of 0.0, indicating no contribution to the feature.
        """
        return 0.0


class P_C_NodeConverter:
    def __init__(self, model: Task):
        self._feature_converter: dict[ObservationState, FeatureConverter] = {
            obs_state: P_C_IgnoreConverter() for obs_state in HRLHelper.c_states()
        }
        for key, value in HRLHelper.get_tp_indices_from_model(model).items():
            self._feature_converter[key] = P_C_StartPoseConverter(value)

        for key, value in model.value.precondition.items():
            if isinstance(value, float):
                self._feature_converter[key] = P_C_ScalarConverter(key, value)
            else:
                self._feature_converter[key] = P_C_StartPoseConverter(value)

    def node_features(
        self,
        obs: HRLPolicyObservation,
        only_precondition: bool = False,
    ) -> np.ndarray:
        if not only_precondition:
            features: list[float] = [
                self._feature_converter[key].feature(value)
                for key, value in obs.normal_states.items()
            ]

        else:
            features: list[float] = [
                self._feature_converter[key].feature(value)
                for key, value in obs.normal_states.items()
                if not isinstance(self._feature_converter[key], P_C_IgnoreConverter)
            ]
        return np.array(features)

    def get_ee_score(self, obs: HRLPolicyObservation) -> float:
        return self._feature_converter[ObservationState.EE_Pose].feature(
            obs.normal_states[ObservationState.EE_Pose]
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
    def __init__(self, dataset: ActionSpace, states: list[ObservationState]):
        self._node_converter: Dict[Task, P_C_NodeConverter] = {
            model: P_C_NodeConverter(model) for model in HRLHelper.models(dataset)
        }
        self.states: list[ObservationState] = states

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
                    sub_states: list[ObservationState] = (
                        ObservationState.from_pose_string(state.value.identifier)
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
