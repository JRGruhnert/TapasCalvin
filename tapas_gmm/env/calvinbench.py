import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch
from loguru import logger

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
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


from calvin_env.envs.calvin_env import (
    CalvinObservation,
    CalvinEnv,
    get_env_from_cfg,
)

task_switch = {
    "PickRedCube": None,
}


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
    cameras: tuple[str, ...] = ("static", "gripper")


class CalvinEnvironment(BaseEnvironment):
    def __init__(self, config: CalvinEnvironmentConfig, **kwargs):
        super().__init__(config)

        self.cameras = config.cameras

        self.calvin_env: CalvinEnv = (
            get_env_from_cfg()
        )  # Give the config to the env so that i can connect both config systems and remove the pain
        if self.calvin_env is None:
            raise RuntimeError("Could not create environment.")
        self.calvin_env.reset()

    def close(self):
        self.calvin_env.close()

    def reset(self):
        obs, info = self.calvin_env.reset()
        obs = self.process_observation(obs)
        return obs, info

    def reset_to_demo(self, path: str):
        super().reset()

        obs = self.calvin_env.reset_from_storage(path)
        obs = self.process_observation(obs)

        return obs

    def _step(
        self,
        action: np.ndarray,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
    ) -> tuple[SceneObservation, float, bool, dict]:  # type: ignore
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
        assert len(action) == 7, f"Action has wrong length: {len(action)}"

        if postprocess:
            action_delayed = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=False,
                prediction_is_euler=True,
            )
        else:
            action_delayed = action
        print(f"Here")
        obs, _, _, _ = self.calvin_env.step(action_delayed)
        print(f"Here2")
        self.calvin_env.render()
        print(f"Here3")

        obs = None if obs is None else self.process_observation(obs)
        print(f"Here4")
        return obs

    def process_observation(self, obs: CalvinObservation) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation.

        Parameters
        ----------
        obs : CalvinObservation
            Observation as Calvins's Observation class.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        camera_obs = {}

        for cam in self.cameras:
            rgb = getattr(obs, cam + "_rgb").transpose((2, 0, 1)) / 255
            depth = getattr(obs, cam + "_depth")
            mask = getattr(obs, cam + "_mask").astype(int)
            extr = obs.misc[cam + "_camera_extrinsics"]
            intr = obs.misc[cam + "_camera_intrinsics"]

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(depth),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(extr),
                    "intr": torch.Tensor(intr),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.cameras)} | camera_obs
        )

        joint_pos = torch.Tensor(obs.joint_positions)
        joint_vel = torch.Tensor(obs.joint_velocities)

        ee_pose = torch.Tensor(obs.gripper_pose)
        """
        ee_pose = torch.Tensor(
            np.concatenate(
                [
                    obs.gripper_pose[:3],
                    obs.gripper_pose[3:],
                ]
            )
        )
        """
        logger.info(f"EE Pose original {obs.gripper_pose}")
        logger.info(f"EE Pose {ee_pose}")
        gripper_open = torch.Tensor([obs.gripper_open])

        flat_object_poses = obs.task_low_dim_state
        """
        n_objs = int(len(flat_object_poses) // 7)  # poses are 7 dim and stacked

        object_poses = tuple(
            np.concatenate((pose[:3], quat_real_last_to_real_first(pose[3:])))
            for pose in np.split(flat_object_poses, n_objs)
        )

        object_poses = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses)}
        )
        """
        print(f"Flat Object Poses: {flat_object_poses}")
        # object_poses = torch.Tensor(flat_object_poses)
        obs = SceneObservation(
            action=None,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            object_poses=flat_object_poses,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=gripper_open,
            batch_size=empty_batchsize,
        )

        return obs

    @staticmethod
    def _get_action(
        current_obs: SceneObservation, next_obs: SceneObservation  # type: ignore
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
