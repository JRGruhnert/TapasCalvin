from dataclasses import dataclass
import pathlib
from typing import Dict, Set

from loguru import logger
import numpy as np
from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    Task,
    State,
    _origin_ee_tp_pose,
)

from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    Gaussian,
    ModelType,
    TPGMMConfig,
)


@dataclass(frozen=True)
class HRLHelper:
    """Utility class for returning meaningful model sets."""

    @classmethod
    def c_states(cls) -> Set[State]:
        return {
            m
            for m in State
            if (
                not (
                    m.value.identifier.endswith("euler")
                    or m.value.identifier.endswith("quat")
                )
            )
        }

    @classmethod
    def models(cls, dataset: ActionSpace) -> Set[Task]:
        return {m for m in Task if (m.value.action_space == dataset)}

    @classmethod
    def static(cls) -> Set[Task]:
        return {m for m in Task if (m.value.action_space == ActionSpace.STATIC)}

    @classmethod
    def dynamic(cls) -> Set[Task]:
        return {Task.DrawerDoClose, Task.DrawerDoOpen}

    @classmethod
    def button(cls) -> Set[Task]:
        return {Task.ButtonPress, Task.ButtonPressReversed}

    @classmethod
    def sliding_doors(cls) -> Set[Task]:
        return {Task.SliderLeftDoOpen, Task.SliderRightDoOpen}

    @classmethod
    def convert_observation(
        cls, task: Task, obs: HRLPolicyObservation
    ) -> HRLPolicyObservation:
        if task.value.reversed:
            obs.update_ee_pose(
                _origin_ee_tp_pose,
            )
        return obs

    @classmethod
    def get_tp_from_task(
        cls,
        task: Task,
        split_pose: bool,
        active_states: list[State],
    ) -> Dict[State, np.ndarray]:
        policy: GMMPolicy = HRLHelper.load_policy(task)
        tpgmm = policy.model
        result: Dict[State, np.ndarray] = {}
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                if split_pose:
                    transform_key, quaternion_key = State.get_tp_by_index(
                        frame_idx, True
                    )
                    if transform_key in active_states:
                        if frame_idx == 0:
                            # Zero means its the ee_pose
                            result[transform_key] = task.value.ee_hrl_start[:3]
                        else:
                            result[transform_key] = task.value.obj_start[:3]
                    if quaternion_key in active_states:
                        if frame_idx == 0:
                            # Zero means its the ee_pose
                            result[quaternion_key] = task.value.ee_hrl_start[-4:]
                        else:
                            result[quaternion_key] = task.value.obj_start[-4:]
                else:
                    pose_key = State.get_tp_by_index(frame_idx, False)
                    if frame_idx == 0:
                        # Zero means its the ee_pose
                        result[pose_key] = task.value.ee_hrl_start
                    else:
                        result[pose_key] = task.value.obj_start
        for key, value in task.value.precondition.items():
            result[key] = value
        return result

    @classmethod
    def _get_gaussians_from_model(cls, tpgmm: AutoTPGMM) -> Dict[int, Gaussian]:
        result: Dict[int, Gaussian] = {}
        # This gives me one HMM per frame; each HMM has K Gaussians and a K×K transition matrix.
        frame_hmms = tpgmm.get_frame_marginals(time_based=False)
        for i, segment in enumerate(tpgmm.segment_frames):
            for j, frame_idx in enumerate(segment):
                for gaussian in frame_hmms[i][j].gaussians:
                    mu3, sigma1 = gaussian.get_mu_sigma(
                        mu_on_tangent=False, as_np=False
                    )
                if result.get(frame_idx) is None:
                    # for gaussian in frame_hmms[i][j].gaussians:
                    #    mu1, sigma1 = gaussian.get_mu_sigma(mu_on_tangent=False, as_np=True)
                    #    mu2, sigma2 = gaussian.get_mu_sigma(mu_on_tangent=True, as_np=True)
                    #    mu3, sigma1 = gaussian.get_mu_sigma(
                    #        mu_on_tangent=False, as_np=False
                    #    )
                    #    mu4, sigma2 = gaussian.get_mu_sigma(mu_on_tangent=True, as_np=False)

                    result[frame_idx] = frame_hmms[i][j].gaussians[0]
                    mu, sigman = result[frame_idx].get_mu_sigma(
                        mu_on_tangent=True, as_np=False
                    )
        return result

    @classmethod
    def retrieve_task(cls, index: int) -> Task:
        return Task.get_enum_by_index(index)

    @classmethod
    def load_policy(cls, task: Task) -> GMMPolicy:
        config = cls._get_config(task.value.reversed)
        Policy = import_policy("gmm")
        policy: GMMPolicy = Policy(config).to(device)

        file_name = cls._policy_checkpoint_name(task.name)  # type: ignore
        logger.info("Loading policy checkpoint from {}", file_name)
        policy.from_disk(file_name)
        policy.eval()
        return policy

    @classmethod
    def _policy_checkpoint_name(cls, task_name: str) -> pathlib.Path:
        return (
            pathlib.Path("data")
            / task_name
            / ("demos" + "_" + "gmm" + "_policy" + "-release")
        ).with_suffix(".pt")

    @classmethod
    def _get_config(cls, reversed: bool) -> GMMPolicyConfig:
        """
        Get the configuration for the OpenDrawer policy.
        """
        return GMMPolicyConfig(
            suffix="release",
            model=AutoTPGMMConfig(
                tpgmm=TPGMMConfig(
                    n_components=20,
                    model_type=ModelType.HMM,
                    use_riemann=True,
                    add_time_component=True,
                    add_action_component=False,
                    position_only=False,
                    add_gripper_action=True,
                    reg_shrink=1e-2,
                    reg_diag=2e-4,
                    reg_diag_gripper=2e-2,
                    reg_em_finish_shrink=1e-2,
                    reg_em_finish_diag=2e-4,
                    reg_em_finish_diag_gripper=2e-2,
                    trans_cov_mask_t_pos_corr=False,
                    em_steps=1,
                    fix_first_component=True,
                    fix_last_component=True,
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=True,
            binary_gripper_closed_threshold=0.95,
            dbg_prediction=False,
            # the kinematics model in RLBench is just to unreliable -> leads to mistakes
            topp_in_t_models=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            # ---- Changing often ----
            postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
            return_full_batch=True,
            batch_predict_in_t_models=True,  # Change if visualization is needed
            invert_prediction_batch=reversed,
        )
