import os
from dataclasses import dataclass
from functools import wraps
from typing import Any
import pybullet as p
import numpy as np
import torch
from loguru import logger


from calvin_env.envs.observation import CalvinObservation
from calvin_env.envs.calvin_env import get_env_from_cfg, CalvinEnvironment
from tapas_gmm.env import Environment
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.utils.geometry_np import (
    axis_angle_to_quaternion,
    conjugate_quat,
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    quat_real_last_to_real_first,
    quat_real_first_to_real_last,
    quaternion_diff,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_pose_diff,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


@dataclass(kw_only=True)
class CalvinConfig(BaseEnvironmentConfig):
    action_mode: str | None = None  # TODO evaluate if this is needed
    env_type: Environment = Environment.CALVIN

    planning_action_mode: bool = False  # TODO evaluate if this is needed
    absolute_action_mode: bool = False  # TODO evaluate if this is needed
    action_frame: str = "end effector"  # TODO evaluate if this is needed

    postprocess_actions: bool = True  # TODO evaluate if this is needed
    background: str | None = None  # TODO evaluate if this is needed
    model_ids: tuple[str, ...] | None = None  # TODO evaluate if this is needed
    cameras: tuple[str, ...] = ("front", "wrist")


class Calvin(BaseEnvironment):
    def __init__(self, config=None, eval=False, **kwargs):
        if config is None:
            config = CalvinConfig(
                task="Undefined",
                cameras=("wrist", "front"),
                camera_pose={},
                image_size=(256, 256),
                static=False,
                headless=False,
                scale_action=False,
                delay_gripper=False,
                gripper_plot=False,
                postprocess_actions=False,
            )
        super().__init__(config)

        self.cameras = config.cameras

        self.env: CalvinEnvironment = get_env_from_cfg(
            eval
        )  # Give the config to the env so that i can connect both config systems and remove the pain
        if self.env is None:
            raise RuntimeError("Could not create environment.")
        self.env.reset()

    def close(self):
        self.env.close()

    def reset(
        self, scene_obs=None, static=True
    ) -> tuple[CalvinObservation, float, bool, dict]:
        return self.env.reset(scene_obs=scene_obs, static=static)

    def reset_to_demo(self, path: str) -> CalvinObservation:
        raise NotImplementedError("Not implemented yet")

    def update_prediction_marker(self, points: list):
        self.env.update_prediction_marker(points)

    def step(self, action: np.ndarray, info: dict = None) -> tuple[CalvinObservation, float, bool, dict]:  # type: ignore
        """
        Postprocess the action and execute it in the environment.
        Simple wrapper around _step, that provides the kwargs for
        postprocessing from self.config.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.

        Returns
        -------
        tuple[CalvinObservation, float, bool, dict]
            The observation, reward, done flag and info dict.
        """

        return self._step(
            action,
            postprocess=self.do_postprocess_actions,
            delay_gripper=self.do_delay_gripper,
            scale_action=self.do_scale_action,
            policy_info=info,
        )

    def _step(
        self,
        action: np.ndarray,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
        policy_info: dict = None,
    ) -> tuple[CalvinObservation, float, bool, dict]:  # type: ignore
        """
        Postprocess the action and execute it in the environment.
        Catches invalid actions and executes a zero action instead.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.
        postprocess : bool, optional
            Whether to postprocess the action at all, by default True
        delay_gripper : bool, optional
            Whether to delay the gripper action. Usually needed for ML
            policies, by default True
        scale_action : bool, optional
            Whether to scale the action. Usually needed for ML policies,
            by default True

        Returns
        -------
        CalvinObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        RuntimeError
            If raised by the environment.
        """
        prediction_is_quat = action.shape[0] == 8
        prediction_is_euler_or_aa = action.shape[0] == 7
        if prediction_is_euler_or_aa:
            logger.warning("Prediction rotation in interpreted as euler angles.")
        action_mode = "quat_abs"

        if postprocess:
            action_delayed = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
                prediction_is_euler=prediction_is_euler_or_aa,
            )
            action_mode = "quat_rel"
        else:
            action_delayed = action

        # NOTE: Quaternion in Calvin is also real-last.
        gripper = 0.0 if np.isnan(action_delayed[-1]) else action_delayed[-1]
        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper]

        if np.isnan(action_delayed).any():
            logger.warning("NaN action, skipping")
            action_delayed = zero_action

        # action_delayed[:3] *= 0.05

        obs, reward, done, info = self.env.step(action_delayed, action_mode)
        self.env.render(obs, policy_info)
        return obs, reward, done, info

    @staticmethod
    def compute_ee_delta(
        current_obs: CalvinObservation, next_obs: CalvinObservation
    ) -> np.ndarray:
        """
        Computes the relative end effector pose change between two observations.
        Returns as position delta and axis-angle rotation delta.
        The gripper action is also included in the action vector."""

        gripper_action = np.array(
            [2 * next_obs.ee_state - 1]  # map from [0, 1] to [-1, 1]
        )

        curr_b = current_obs.ee_pose[:3]
        curr_q = quat_real_last_to_real_first(current_obs.ee_pose[3:])
        curr_A = quaternion_to_matrix(curr_q)

        next_b = next_obs.ee_pose[:3]
        next_q = quat_real_last_to_real_first(next_obs.ee_pose[3:])
        next_A = quaternion_to_matrix(next_q)
        next_hom = homogenous_transform_from_rot_shift(next_A, next_b)

        # Transform from world into EE frame. In EE frame target pose and delta pose
        # are the same thing.
        world2ee = invert_homogenous_transform(
            homogenous_transform_from_rot_shift(curr_A, curr_b)
        )
        rot_delta = quaternion_to_axis_angle(quaternion_pose_diff(curr_q, next_q))

        pred_local = world2ee @ next_hom
        pos_delta = pred_local[:3, 3]

        return np.concatenate([pos_delta, rot_delta, gripper_action])

    def get_inverse_kinematics(
        self, target_pose: np.ndarray, reference_qpos: np.ndarray, max_configs: int = 20
    ) -> np.ndarray:
        # TODO: Evaluate if this is correct
        return self.env.robot.mixed_ik.ik_fast.get_ik_solution(
            target_pose[:3], target_pose[3:]
        )

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Postprocess the quaternion action.
        NOTE: quaternion is real first! Real last is only used internally in the
        franka environment. All interfaces with the rest of the codebase should
        use real-first quaternions.
        """
        return p.getEulerFromQuaternion(quat_real_first_to_real_last(quaternion))  # type: ignore
