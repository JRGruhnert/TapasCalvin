import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch
from loguru import logger

from calvin_env.envs.master_tasks.task import PressButton
from tapas_gmm.env import Environment
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.utils.geometry_np import (
    conjugate_quat,
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    quat_real_first_to_real_last,
    quat_real_last_to_real_first,
    quaternion_diff,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_pose_diff,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)

from calvin_env.envs.calvin_env import (
    CalvinObservation,
    CalvinEnv,
    get_env_from_cfg,
)


@dataclass(kw_only=True)
class CalvinEnvironmentConfig(BaseEnvironmentConfig):
    action_mode: Any = None  # TODO evaluate if this is needed
    env_type: Environment = Environment.CALVINBENCH

    planning_action_mode: bool = False  # TODO evaluate if this is needed
    absolute_action_mode: bool = False  # TODO evaluate if this is needed
    action_frame: str = "end effector"  # TODO evaluate if this is needed

    postprocess_actions: bool = True  # TODO evaluate if this is needed
    background: str | None = None  # TODO evaluate if this is needed
    model_ids: tuple[str, ...] | None = None  # TODO evaluate if this is needed
    cameras: tuple[str, ...] = ("front", "wrist")


class CalvinEnvironment(BaseEnvironment):
    def __init__(self, config: CalvinEnvironmentConfig, **kwargs):
        super().__init__(config)

        self.cameras = config.cameras

        self.calvin_env: CalvinEnv = get_env_from_cfg(
            config.task
        )  # Give the config to the env so that i can connect both config systems and remove the pain
        if self.calvin_env is None:
            raise RuntimeError("Could not create environment.")
        self.calvin_env.reset()

    def close(self):
        self.calvin_env.close()

    def reset(self) -> tuple[CalvinObservation, float, bool, dict]:
        return self.calvin_env.reset()

    def reset_to_demo(self, path: str) -> CalvinObservation:
        raise NotImplementedError("Not implemented yet")

    def update_prediction_marker(self, points: list):
        self.calvin_env.update_prediction_marker(points)

    def _step(
        self,
        action: np.ndarray,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
        policy_info: dict = None,
        rel: bool = False,
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
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        RuntimeError
            If raised by the environment.
        """
        prediction_is_quat = action.shape[0] == 8

        if postprocess:
            action_delayed = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
            )
        else:
            action_delayed = action

        gripper = 0.0 if np.isnan(action_delayed[-1]) else action_delayed[-1]
        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper]

        if np.isnan(action_delayed).any():
            logger.warning("NaN action, skipping")
            action_delayed = zero_action

        obs, reward, done, info = self.calvin_env.step(action_delayed, rel)
        self.calvin_env.render(obs, policy_info)
        return obs, reward, done, info

    @staticmethod
    def _get_action(
        current_obs: CalvinObservation, next_obs: CalvinObservation  # type: ignore
    ) -> np.ndarray:
        gripper_action = np.array(
            [2 * next_obs.gripper_state - 1]  # map from [0, 1] to [-1, 1]
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
        self.calvin_env.robot.mixed_ik.ik_fast.get_ik_solution(
            target_pose[:3], target_pose[3:]
        )

        arm = self.env._robot.arm  # .copy()
        arm.set_joint_positions(reference_qpos[:7], disable_dynamics=True)
        arm.set_joint_target_velocities([0] * len(arm.joints))

        return arm.solve_ik_via_sampling(
            position=target_pose[:3],
            quaternion=quat_real_first_to_real_last(target_pose[3:7]),
            relative_to=None,
            ignore_collisions=True,
            max_configs=max_configs,  # samples this many configs, then ranks them
        )[
            0
        ]  # return the closest one

        # return arm.solve_ik_via_jacobian(
        #     position=target_pose[:3],
        #     quaternion=quat_real_first_to_real_last(target_pose[3:7]),
        #     relative_to=None,
        # )
