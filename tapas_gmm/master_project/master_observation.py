import numpy as np
import torch

from tapas_gmm.master_project.master_definitions import (
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


class Observation:
    def __init__(
        self,
        obs: CalvinObservation,
    ):
        self._obs = obs
        self._pose_states: dict[State, float | np.ndarray] = {
            State.EE_Pose: obs.ee_pose,
            **{State.from_string(f"{k}_pose"): v for k, v in obs.object_poses.items()},
        }

        self._euler_states: dict[State, float | np.ndarray] = {
            State.EE_Transform: obs.ee_pose[:3],
            **{
                State.from_string(f"{k}_euler"): v[:3]
                for k, v in obs.object_poses.items()
            },
        }

        self._quat_states: dict[State, float | np.ndarray] = {
            State.EE_Quat: obs.ee_pose[-4:],
            **{
                State.from_string(f"{k}_quat"): v[-4:]
                for k, v in obs.object_poses.items()
            },
        }

        self._scalar_states: dict[State, float] = {
            State.EE_State: obs.ee_state,
            **{State.from_string(k): v for k, v in obs.object_states.items()},
        }

        self._states: dict[State, float | np.ndarray] = {
            **self._pose_states,
            **self._euler_states,
            **self._quat_states,
            **self._scalar_states,
        }

    @property
    def states(self) -> dict[State, float | np.ndarray]:
        """Returns the scalar states of the observation."""
        return self._states

    def tapas_format(self, task: Task) -> SceneObservation:  # type: ignore
        # This is a hack for changing the ee_pose to the origin for reversed models
        # It does nothing for standard models
        if task.value.reversed:
            self._obs.ee_pose = _origin_ee_tp_pose
        return _to_rlbench_format(self._obs)

    def update_ee_pose(self, ee_pose: np.ndarray):
        self._obs.ee_pose = ee_pose
