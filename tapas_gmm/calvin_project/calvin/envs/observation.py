import numpy as np
import torch
import pybullet as p
from tapas_gmm.policy.models.master_project.hrl_observation import HRLPolicyObservation
from tapas_gmm.utils.geometry_np import (
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    quat_real_last_to_real_first,
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


class CalvinObservation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(
        self,
        camera_names: list[str],
        rgb: dict[str, np.ndarray],
        depth: dict[str, np.ndarray],
        pcd: dict[str, np.ndarray],
        mask: dict[str, np.ndarray],
        extr: dict[str, np.ndarray],
        intr: dict[str, np.ndarray],
        object_poses: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        low_dim_object_poses: np.ndarray,
        low_dim_object_states: np.ndarray,
        # Already available in the observation
        # but not used in the code
        joint_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_forces: np.ndarray,
        gripper_state: float,
        gripper_pose: np.ndarray,
        tcp_pose: np.ndarray,
        tcp_state: float,
        gripper_matrix: np.ndarray,
        gripper_joint_positions: np.ndarray,
        gripper_touch_forces: np.ndarray,
    ):
        self._camera_names = camera_names
        self._rgb = rgb
        self._depth = depth
        self._pcd = pcd
        self._mask = mask
        self._extr = extr
        self._intr = intr
        self._object_poses = object_poses
        self._object_states = object_states
        self._low_dim_object_poses = low_dim_object_poses
        self._low_dim_object_states = low_dim_object_states
        self._joint_vel = joint_vel
        self._joint_pos = joint_pos
        self._joint_forces = joint_forces
        self._gripper_state = gripper_state
        self._gripper_pose = gripper_pose
        self._tcp_pose = tcp_pose
        self._tcp_state = tcp_state
        self._gripper_matrix = gripper_matrix
        self._gripper_joint_positions = gripper_joint_positions
        self._gripper_touch_forces = gripper_touch_forces

        self._action = None
        self._reward = None
        self._done = None

    @property
    def camera_names(self) -> list[str]:
        """
        Get the camera names.

        Returns
        -------
        list[str]
            The camera names.
        """
        return self._camera_names

    @property
    def rgb(self) -> dict[str, np.ndarray]:
        """
        Get the RGB images.

        Returns
        -------
        dict[str, np.ndarray]
            The RGB images.
        """
        return self._rgb

    @property
    def depth(self) -> dict[str, np.ndarray]:
        """
        Get the depth images.

        Returns
        -------
        dict[str, np.ndarray]
            The depth images.
        """
        return self._depth

    @property
    def pcd(self) -> dict[str, np.ndarray]:
        """
        Get the point cloud data.

        Returns
        -------
        dict[str, np.ndarray]
            The point cloud data.
        """
        return self._pcd

    @property
    def mask(self) -> dict[str, np.ndarray]:
        """
        Get the masks.

        Returns
        -------
        dict[str, np.ndarray]
            The masks.
        """
        return self._mask

    @property
    def extr(self) -> dict[str, np.ndarray]:
        """
        Get the extrinsic camera parameters.

        Returns
        -------
        dict[str, np.ndarray]
            The extrinsic camera parameters.
        """
        return self._extr

    @property
    def intr(self) -> dict[str, np.ndarray]:
        """
        Get the intrinsic camera parameters.

        Returns
        -------
        dict[str, np.ndarray]
            The intrinsic camera parameters.
        """
        return self._intr

    @property
    def low_dim_object_poses(self) -> np.ndarray:
        """
        Get the low-dimensional object poses.

        Returns
        -------
        np.ndarray
            The low-dimensional object poses.
        """
        return self._low_dim_object_poses

    @property
    def low_dim_object_states(self) -> np.ndarray:
        """
        Get the low-dimensional object states.

        Returns
        -------
        np.ndarray
            The low-dimensional object states.
        """
        return self._low_dim_object_states

    @property
    def joint_vel(self) -> np.ndarray:
        """
        Get the joint velocities.

        Returns
        -------
        np.ndarray
            The joint velocities.
        """
        return self._joint_vel

    @property
    def joint_pos(self) -> np.ndarray:
        """
        Get the joint positions.

        Returns
        -------
        np.ndarray
            The joint positions.
        """
        return self._joint_pos

    @property
    def joint_forces(self) -> np.ndarray:
        """
        Get the joint forces.

        Returns
        -------
        np.ndarray
            The joint forces.
        """
        return self._joint_forces

    @property
    def object_poses(self) -> dict[str, np.ndarray]:
        """
        Get the object poses.

        Returns
        -------
        dict[str, np.ndarray]
            The object poses.
        """
        return self._object_poses

    @property
    def object_states(self) -> dict[str, np.ndarray]:
        """
        Get the object states.

        Returns
        -------
        dict[str, np.ndarray]
            The object states.
        """
        return self._object_states

    @property
    def ee_pose(self) -> np.ndarray:
        """
        Get the end effector pose.

        Returns
        -------
        np.ndarray
            The end effector pose.
        """
        return self._tcp_pose

    @property
    def ee_state(self) -> np.ndarray:
        """
        Get the end effector state.

        Returns
        -------
        np.ndarray
            The end effector state.
        """
        return self._tcp_state

    @property
    def action(self) -> np.ndarray:
        """
        Get the action for the observation.

        Returns
        -------
        np.ndarray
            The action.
        """
        return self._action

    @action.setter
    def action(self, action: np.ndarray) -> None:
        """
        Set the action for the observation.

        Parameters
        ----------
        action : np.ndarray
            The action to set.
        """
        self._action = action

    @property
    def reward(self) -> float:
        """
        Get the reward for the observation.

        Returns
        -------
        float
            The reward.
        """
        return self._reward

    @reward.setter
    def reward(self, reward: float) -> None:
        """
        Set the reward for the observation.

        Parameters
        ----------
        reward : float
            The reward to set.
        """
        self._reward = reward

    @property
    def done(self) -> bool:
        """
        Get the done flag for the observation.

        Returns
        -------
        bool
            The done flag.
        """
        return self._done

    @done.setter
    def done(self, done: bool) -> None:
        """
        Set the done flag for the observation.

        Parameters
        ----------
        done : bool
            The done flag to set.
        """
        self._done = done

    def to_rlbench_format(self) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        # TODO: IS conversionb correct?
        if self.action is None:
            action = None
        else:
            action = torch.Tensor(self.action)
        if self.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([self.reward])

        camera_obs = {}

        for cam in self._camera_names:
            self._rgb[cam] = self._rgb[cam].transpose((2, 0, 1)) / 255
            self._mask[cam] = self._mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(self._rgb[cam]),
                    "depth": torch.Tensor(self._depth[cam]),
                    "mask": torch.Tensor(self._mask[cam]).to(torch.uint8),
                    "extr": torch.Tensor(self._extr[cam]),
                    "intr": torch.Tensor(self._intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict({"_order": CameraOrder._create(self._camera_names)} | camera_obs)

        joint_pos = torch.Tensor(self._joint_pos)
        joint_vel = torch.Tensor(self._joint_vel)
        ee_pose = torch.Tensor(self.ee_pose)
        ee_state = torch.Tensor([self.ee_state])

        object_pose_len = 7
        object_poses_list = self._low_dim_object_poses.reshape(-1, object_pose_len)

        object_poses = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses_list)},
        )

        object_state_len = 1
        object_states_list = self._low_dim_object_states.reshape(-1, object_state_len)

        object_states = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(state) for i, state in enumerate(object_states_list)},
        )

        obs = SceneObservation(
            feedback=reward,
            action=action,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_batchsize,
        )
        return obs

    def to_hrl_policy_format(self) -> HRLPolicyObservation:
        """
        Convert the observation from the environment to a HRLPolicyObservation.

        Returns
        -------
        HRLPolicyObservation
            The observation in common format as HRLPolicyObservation.
        """

        object_poses = self._object_poses
        object_states = self._object_states
        tapas_observation = self.to_rlbench_format()

        obs = HRLPolicyObservation(
            object_poses=object_poses,
            object_states=object_states,
            tapas_observation=tapas_observation,
        )
        return obs
