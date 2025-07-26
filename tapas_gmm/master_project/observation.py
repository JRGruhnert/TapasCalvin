import numpy as np
import torch

from tapas_gmm.master_project.definitions import (
    State,
    Task,
    _origin_ee_tp_pose,
)
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)

from calvin_env.envs.observation import CalvinObservation


def _to_rlbench_format(obs: CalvinObservation) -> SceneObservation:  # type: ignore
    """
    Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

    Returns
    -------
    SceneObservation
        The observation in common format as SceneObservation.
    """
    if obs.action is None:
        action = None
    else:
        action = torch.Tensor(obs.action)
    if obs.reward is None:
        reward = torch.Tensor([0.0])
    else:
        reward = torch.Tensor([obs.reward])

    camera_obs = {}

    for cam in obs._camera_names:
        obs._rgb[cam] = obs._rgb[cam].transpose((2, 0, 1)) / 255
        obs._mask[cam] = obs._mask[cam].astype(int)

        camera_obs[cam] = SingleCamObservation(
            **{
                "rgb": torch.Tensor(obs._rgb[cam]),
                "depth": torch.Tensor(obs._depth[cam]),
                "mask": torch.Tensor(obs._mask[cam]).to(torch.uint8),
                "extr": torch.Tensor(obs._extr[cam]),
                "intr": torch.Tensor(obs._intr[cam]),
            },
            batch_size=empty_batchsize,
        )

    multicam_obs = dict_to_tensordict(
        {"_order": CameraOrder._create(obs._camera_names)} | camera_obs
    )

    joint_pos = torch.Tensor(obs._joint_pos)
    joint_vel = torch.Tensor(obs._joint_vel)
    ee_pose = torch.Tensor(obs.ee_pose)
    ee_state = torch.Tensor([obs.ee_state])

    object_pose_len = 7
    object_poses_list = obs._low_dim_object_poses.reshape(-1, object_pose_len)

    object_poses = dict_to_tensordict(
        {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses_list)},
    )

    object_state_len = 1
    object_states_list = obs._low_dim_object_states.reshape(-1, object_state_len)

    object_states = dict_to_tensordict(
        {
            f"obj{i:03d}": torch.Tensor(state)
            for i, state in enumerate(object_states_list)
        },
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


def tapas_format(obs: CalvinObservation, task: Task) -> SceneObservation:  # type: ignore
    # This is a hack for changing the ee_pose to the origin for reversed models
    # It does nothing for standard models
    if task.value.reversed:
        obs.ee_pose = _origin_ee_tp_pose
    return _to_rlbench_format(obs)


class Observation:
    __slots__ = "_states"

    def __init__(
        self,
        obs: CalvinObservation,
    ):
        self._states: dict[State, np.ndarray] = {}
        self._states[State.EE_Transform] = obs.ee_pose[:3].astype(np.float32)
        self._states[State.EE_Quat] = obs.ee_pose[-4:].astype(np.float32)
        self._states[State.EE_State] = np.array([obs.ee_state], dtype=np.float32)
        for k, pose in obs.object_poses.items():
            self._states[State.from_string(f"{k}_euler")] = pose[:3].astype(np.float32)
            self._states[State.from_string(f"{k}_quat")] = pose[-4:].astype(np.float32)

        for k, val in obs.object_states.items():
            self._states[State.from_string(k)] = np.array([val], dtype=np.float32)

    @property
    def states(self) -> dict[State, np.ndarray]:
        """Returns the scalar states of the observation."""
        return self._states
