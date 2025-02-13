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
    ObservationConfig,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


from calvin_env.envs.custom_env import (
    CalvinSimEnv,
    get_env_from_cfg,
)


@dataclass(kw_only=True)
class CalvinTapasBridgeEnvironmentConfig(BaseEnvironmentConfig):
    action_mode: Any = None
    env_type: Environment = Environment.CALVINBENCH

    absolute_action_mode: bool = False
    action_frame: str = "end effector"

    demo_path: str | None = None
    generate: bool = False
    postprocess_actions: bool = True
    background: str | None = None
    model_ids: tuple[str, ...] | None = None
    cameras: tuple[str, ...] = ("static", "gripper")

class CalvinTapasBridgeEnvironment(BaseEnvironment):
    def __init__(self, config: CalvinTapasBridgeEnvironmentConfig, **kwargs):
        super().__init__(config)

        self.cameras = config.cameras

        self.calvin_env: CalvinSimEnv = get_env_from_cfg()
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
    ) -> tuple[SceneObservation, float, bool, dict]: # type: ignore
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
        """
        if postprocess:
            action_delayed = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
            )
        else:
        """
        obs, reward, done, info = self.calvin_env.step(action)

        obs = None if obs is None else self.process_observation(obs)

        return obs, reward, done, info

    def get_camera_pose(self) -> dict[str, np.ndarray]:
        return {name: cam.get_pose() for name, cam in self.camera_map.items()}

    def set_camera_pose(self, pos_dict: dict[str, np.ndarray]) -> None:
        for camera_name, pos in pos_dict.items():
            if camera_name in self.camera_map:
                camera = self.camera_map[camera_name]
                camera.set_pose(pos)

    def _get_obj_poses(self) -> np.ndarray:
        """
        Low dim state of the task can contain more than just object poses, eg.
        force sensor readings, joint positions, etc., which makes it hard to parse.

        This is a fallback method to get the object poses from the task state.
        """
        info = self.calvin_env.get_info()
        scene_info = info["scene_info"]
        fixed_objects = scene_info["fixed_objects"]
        movable_objects = scene_info["movable_objects"]
        #print(f"Fixed objects: {fixed_objects}")
        #print(f"Movable objects: {movable_objects}")

        state = []
        '''
        for obj, objtype in self.task_env._task._initial_objs_in_scene:
            if not obj.still_exists():
                # It has been deleted
                empty_len = 7
                # if objtype == ObjectType.JOINT:
                #     empty_len += 1
                # elif objtype == ObjectType.FORCE_SENSOR:
                #     empty_len += 6
                state.extend(np.zeros((empty_len,)).tolist())
            else:
                state.extend(np.array(obj.get_pose()))
                # if obj.get_type() == ObjectType.JOINT:
                #     state.extend([Joint(obj.get_handle()).get_joint_position()])
                # elif obj.get_type() == ObjectType.FORCE_SENSOR:
                #     forces, torques = ForceSensor(obj.get_handle()).read()
                #     state.extend(forces + torques)
                '''
        return np.array(state).flatten()

    def process_observation(self, obs: dict[str, dict]) -> SceneObservation: # type: ignore
        """
        Convert the observation from the environment to a SceneObservation.

        Parameters
        ----------
        obs : Observation
            The observation from the environment.
            rgb_obs: dict[str, np.ndarray]
                The RGB images from the cameras.
            depth_obs: dict[str, np.ndarray]
                The depth images from the cameras.
            mask_obs: dict[str, np.ndarray]

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        rgb_obs = obs["rgb_obs"]
        depth_obs = obs["depth_obs"]
        robot_obs = obs["robot_obs"]
        scene_obs = obs["scene_obs"]
        mask_obs = obs["mask_obs"]
        extr_obs = obs["extr_obs"]
        intr_obs = obs["intr_obs"]
        robot_info = obs["robot_info"]

        if len(rgb_obs) == 0:
            logger.warning("RGB observation is None.")
            return None

        camera_obs = {}
        for cam in self.cameras:
            rgb = rgb_obs.get(f"rgb_{cam}").transpose((2, 0, 1)) / 255 # Normalize to [0, 1]
            depth = depth_obs.get(f"depth_{cam}")
            mask = None#mask_obs.get(f"mask_{cam}").astype(int)
            extr = extr_obs.get(f"extr_{cam}")
            intr = intr_obs.get(f"intr_{cam}").astype(float)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(depth),
                    "mask": None,#torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(extr),
                    "intr": torch.Tensor(intr),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.cameras)} | camera_obs
        )

        joint_pos = torch.Tensor(robot_info["arm_joint_positions"])
        joint_vel = torch.Tensor(robot_info["arm_joint_velocities"])
        ee_pose = torch.Tensor(robot_info["ee_pose"])
        gripper_open = torch.Tensor([robot_info["gripper_opening_width"]])

        object_poses = self._get_obj_poses()
        object_poses = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses)}
        )

        obs = SceneObservation(
            action=None,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            object_poses=object_poses,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=gripper_open,
            batch_size=empty_batchsize,
        )

        return obs

    @staticmethod
    def _get_action(
        current_obs: dict, next_obs: dict
    ) -> np.ndarray:
        gripper_action = np.array(
            [2 * next_obs.gripper_open - 1]  # map from [0, 1] to [-1, 1]
        )

        curr_b = current_obs.gripper_pose[:3]
        curr_q = quat_real_last_to_real_first(current_obs.gripper_pose[3:])
        curr_A = quaternion_to_matrix(curr_q)

        next_b = next_obs.gripper_pose[:3]
        next_q = quat_real_last_to_real_first(next_obs.gripper_pose[3:])
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

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        return quat_real_first_to_real_last(quaternion)

    def get_inverse_kinematics(
        self, target_pose: np.ndarray, reference_qpos: np.ndarray, max_configs: int = 20
    ) -> np.ndarray:
        arm = self.calvin_env._robot.arm
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
        ]
