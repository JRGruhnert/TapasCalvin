from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as R
import torch
from loguru import logger

from tapas_gmm.encoder.encoder import ObservationEncoder, ObservationEncoderConfig
from tapas_gmm.env.environment import BaseEnvironment

from tapas_gmm.policy.models.master_project.hrl_gnn import HRL_GNN
from tapas_gmm.policy.models.master_project.hrl_model import (
    HierarchicalLearner,
    HierarchicalLearnerConfig,
)
from tapas_gmm.policy.models.tpgmm import TPGMM, AutoTPGMM, AutoTPGMMConfig, TPGMMConfig

# from tapas_gmm.policy.motion_planner import MotionPlannerPolicy
from tapas_gmm.policy.policy import Policy, PolicyConfig
from utils.robot_trajectory import RobotTrajectory

zero_pos = np.array([0, 0, 0])
zero_quat = np.array([1, 0, 0, 0])

close_gripper = torch.tensor([-0.9])
open_gripper = torch.tensor([0.9])


@dataclass
class MasterPolicyConfig(PolicyConfig):
    model: HierarchicalLearnerConfig

    time_based: bool | None = None

    predict_dx_in_xdx_models: bool = False

    batch_predict_in_t_models: bool = True
    batch_t_max: float = 1
    topp_in_t_models: bool = True
    return_full_batch: bool = False

    time_scale: float = 1  # 0.25
    pos_lag_thresh: float | None = 0.02
    quat_lag_thresh: float | None = 0.1
    pos_change_thresh: float | None = 0.002
    quat_change_thresh: float | None = 0.002

    topp_supersampling: float = 0.15

    dbg_prediction: bool = False

    binary_gripper_action: bool = False
    binary_gripper_closed_threshold: float = 0

    encoder: Any = None
    obs_encoder: ObservationEncoderConfig = ObservationEncoderConfig()
    encoder_path: str = "demos_vit_keypoints_encoder"

    postprocess_prediction: bool = True

    invert_prediction_batch: bool = False

    # visual_embedding_dim: int | None
    # proprio_dim: int
    # action_dim: int
    # lstm_layers: int

    # use_ee_pose: bool
    # add_gripper_state: bool

    # training: PolicyTrainingConfig | None


class HRLPolicy(Policy):
    # TODO: Change
    def __init__(
        self, config: MasterPolicyConfig, skip_module_init: bool = False, **kwargs
    ):
        super().__init__(config, skip_module_init, **kwargs)

        self.config = config

        self._gnn = None
        self._graph = None
        self._partition_A = None
        self._partition_B = None
        self._partition_C = None
        self._edgesAB = None
        self._edgesBC = None

    def _init_gnn(self, num_states=5, num_policies=3, d_linear=32, d_A=16, d_C=20):

        self._gnn = HRL_GNN(
            num_states=num_states,
            num_policies=num_policies,
            d_linear=d_linear,
            d_A=d_A,
            dim_c_out=d_C,
            num_heads=1,
        )

        self._graph = None  # TODO: add edges already?

    def _update_gnn(
        self,
    ):
        raise NotImplementedError

    def _get_partition_A(self):
        raise NotImplementedError

    def _get_partition_B(self):
        raise NotImplementedError

    def _get_partition_C(self):
        raise NotImplementedError

    def _get_partition_B(self):
        raise NotImplementedError

    def _get_edges_AB(self):
        raise NotImplementedError

    def _get_edges_BC(self):
        raise NotImplementedError

    # TODO: Change
    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self._t_curr = np.array([0.0])

        self.model.reset_episode()

        self._env = env

        if self.config.batch_predict_in_t_models:
            self._prediction_batch = None

        self._last_prediction = None
        self._last_pose = None

    # TODO: Change
    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray | RobotTrajectory, dict]:
        info = {}

        frame_trans, frame_quats, viz_encoding = self._get_frame_trans(obs)

        info["viz_encoding"] = viz_encoding

        if self.config.batch_predict_in_t_models and self._time_based:
            if self._prediction_batch is None:
                self._prediction_batch = self._create_prediction_batch(
                    obs=obs, frame_trans=frame_trans, frame_quats=frame_quats
                )
                self._env.publish_path(self._prediction_batch)

                if self.config.return_full_batch:
                    info["done"] = True

                    if self.config.binary_gripper_action:
                        self._prediction_batch.gripper = self._binary_gripper_action(
                            self._prediction_batch.gripper
                        )

                    return self._prediction_batch, info

            if self._prediction_batch.is_finished:
                # action = self._make_noop_plan(obs.cpu(), duration=1)[0]
                # info = {"done": True}
                prediction = (
                    self._last_prediction
                    if self._last_prediction is not None
                    else self._make_noop_plan(obs.cpu(), duration=1)[0]
                )
                info["done"] = self._last_prediction is None
            else:
                prediction = self._prediction_batch.step()
                info["done"] = False

            action = (
                self._postprocess_prediction(obs.ee_pose.numpy(), prediction.ee)
                if self.config.postprocess_prediction
                else prediction
            )

            self._last_prediction = prediction

        else:
            _, action, _ = self._predict_and_step(
                obs=obs,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                postprocess=self.config.postprocess_prediction,
            )
            info["done"] = False

        if self.config.binary_gripper_action:
            action[-1] = self._binary_gripper_action(action[-1])

        return action, info

    # TODO: Change
    def from_disk(self, chekpoint_path: str) -> None:
        self.model.from_disk(
            chekpoint_path,
            force_config_overwrite=self._force_overwrite_checkpoint_config,
        )

        self._add_init_ee_pose_as_frame = self.model._demos.meta_data[
            "add_init_ee_pose_as_frame"
        ]
        self._add_world_frame = self.model._demos.meta_data["add_world_frame"]

        self._model_contains_rotation = self.model.add_rotation_component
        self._model_contains_time_dim = self.model.add_time_component
        self._model_contains_action_dim = self.model.add_action_component
        self._model_contains_gripper_action = self.model.add_gripper_action
        self._model_factorizes_action = self.model.action_as_orientation
        if self._model_factorizes_action:
            assert self._model_contains_action_dim
            assert self.model.action_with_magnitude, "Need magnitude for factorization."

        if self._time_based is None:
            self._time_based = self._model_contains_time_dim
            logger.info(
                "Detected time-based model: {}. Using time-driven policy. "
                "Set time_based in config to overwrite.",
                self._time_based,
            )

        # assert self._model_contains_time_dim == self._time_based, (
        #     "Model contains time dimension, but policy is not time-based, or vice "
        #     "versa. TODO: implement case where model is time-based, but policy is "
        #     "not."
        # )

        if self._time_based and self._model_contains_action_dim:
            logger.info(
                "Got TXDX model, flag to be time-driven and flag to "
                + "prediction action directly (TDX)."
                if self._predict_dx_in_xdx_models
                else "predict action from state (TX)."
            )

        logger.info("Creating local marginals")
        self._local_marginals = self.model.get_frame_marginals(
            time_based=self._time_based
        )

    def _mahalanobis_distance(
        self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> float:
        """
        Compute the Mahalanobis distance between a vector and a Gaussian distribution.

        Parameters:
        - x (np.ndarray): The input vector of shape (n,).
        - mean (np.ndarray): The mean vector of the distribution of shape (n,).
        - cov (np.ndarray): The covariance matrix of the distribution of shape (n, n).

        Returns:
        - float: The Mahalanobis distance.
        """
        x, mean, cov = np.asarray(x), np.asarray(mean), np.asarray(cov)
        cov_inv = np.linalg.inv(cov)
        diff = x - mean
        return np.sqrt(diff.T @ cov_inv @ diff)

    def _normalized_mahalanobis_score(self, d_squared: float, dof: int) -> float:
        """
        Normalize a Mahalanobis distance using the Chi-square CDF.

        Parameters:
        - d_squared (float): The squared Mahalanobis distance (D^2).
        - dof (int): Degrees of freedom, typically the dimension of the vector.

        Returns:
        - float: A score in [0, 1], where higher is better alignment.
        """
        return 1 - chi2.cdf(d_squared, dof)

    def _relative_quaternion_diff(
        self, q_goal: np.ndarray, q_current: np.ndarray
    ) -> np.ndarray:
        """
        Compute the relative rotation between two quaternions via Hamilton product.

        Parameters:
        - q_goal (np.ndarray): The goal quaternion as [x, y, z, w].
        - q_current (np.ndarray): The current quaternion as [x, y, z, w].

        Returns:
        - np.ndarray: The relative quaternion (goal * current⁻¹), shape (4,).
        """
        qg = R.from_quat(q_goal)
        qc_inv = R.from_quat(q_current).inv()
        return (qg * qc_inv).as_quat()

    def _pose_to_tangent_vector(
        self, quat: np.ndarray, pos: np.ndarray, mu_quat: np.ndarray
    ) -> np.ndarray:
        """
        Project a quaternion and position into the tangent space at a reference quaternion.

        Parameters:
        - quat (np.ndarray): The input quaternion [x, y, z, w].
        - pos (np.ndarray): The position vector [x, y, z].
        - mu_quat (np.ndarray): The reference mean quaternion [x, y, z, w].

        Returns:
        - np.ndarray: Concatenated tangent vector, typically shape (7,) or (6,) if Log is used.
        """
        q_rel = self._relative_quaternion_diff(quat, mu_quat)
        return np.concatenate([q_rel, pos])

    def _flatten_goal_diff(self, x: list, y: list, types: list[str]) -> np.ndarray:
        """
        Compute the flattened absolute difference vector between current and goal state.

        Parameters:
        - x (list): List of current state components (e.g., position, orientation, scalars).
        - y (list): List of goal state components, same structure as `x`.
        - types (list of str): Type of each component in `x`/`y`, one of ['euler', 'quat', 'scalar'].

        Returns:
        - np.ndarray: A flattened vector of absolute differences.
        """
        diffs = []
        for xi, yi, t in zip(x, y, types):
            if t == "scalar" or t == "euler":
                diffs.append(np.abs(xi - yi))
            elif t == "quat":
                diffs.append(np.abs(self._relative_quaternion_diff(yi, xi)))
        return np.concatenate(diffs)

    def _project_state_component(
        self, xi: np.ndarray, state_type: str, W: dict, b: dict
    ) -> np.ndarray:
        """
        Apply type-specific linear projection to a state component.

        Parameters:
        - xi (np.ndarray): Input feature vector (e.g., 1D for scalar, 3D for Euler, 4D for quaternion).
        - state_type (str): Type of the input, one of ['scalar', 'euler', 'quat'].
        - W (dict): Dictionary mapping state_type to weight matrix np.ndarray.
        - b (dict): Dictionary mapping state_type to bias vector np.ndarray.

        Returns:
        - np.ndarray: Projected feature vector.
        """
        return W[state_type] @ xi + b[state_type]

    def _stable_softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of a vector in a numerically stable way.

        Parameters:
        - x (np.ndarray): Input vector.

        Returns:
        - np.ndarray: Softmax probabilities.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
