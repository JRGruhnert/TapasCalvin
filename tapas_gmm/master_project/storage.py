from dataclasses import dataclass
import pathlib
from loguru import logger
import numpy as np
from tapas_gmm.master_project.definitions import State, Task
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import TPGMM, AutoTPGMMConfig, ModelType, TPGMMConfig
from tapas_gmm.utils.select_gpu import device


def _load_policy(task: Task) -> GMMPolicy:
    config = _get_config(task.value.reversed)
    Policy = import_policy("gmm")
    policy: GMMPolicy = Policy(config).to(device)

    file_name = _policy_checkpoint_name(task.name)  # type: ignore
    logger.info("Loading policy checkpoint from {}", file_name)
    policy.from_disk(file_name)
    policy.eval()
    return policy


def _policy_checkpoint_name(task_name: str) -> pathlib.Path:
    return (
        pathlib.Path("data")
        / task_name
        / ("demos" + "_" + "gmm" + "_policy" + "-release")
    ).with_suffix(".pt")


def _get_config(reversed: bool) -> GMMPolicyConfig:
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


@dataclass
class StorageConfig:
    pass


class Storage:
    def __init__(self, config: StorageConfig, tasks: list[Task], states: list[State]):
        self.config = config
        self.tasks = tasks
        self.states = states
        self.policy_storage: dict[Task, GMMPolicy] = {
            task: _load_policy(task) for task in self.tasks
        }

    def task_parameter(self) -> dict[Task, dict[State, np.ndarray]]:
        task_param_dict: dict[Task, dict[State, np.ndarray]] = {}
        for task in self.tasks:
            task_param_dict[task] = self.get_tp_from_task(task)
        for key, value in task_param_dict.items():
            print(key.name)
            for key1 in value.keys():
                print(key1.name)
        return task_param_dict

    def get_tp_from_task(
        self,
        task: Task,
        split_pose: bool = True,
    ) -> dict[State, np.ndarray]:
        tpgmm: TPGMM = self.get_policy(task).model
        result: dict[State, np.ndarray] = {}
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                if split_pose:
                    transform_key, quaternion_key = State.get_tp_by_index(
                        frame_idx, True
                    )
                    if transform_key in self.states:
                        if frame_idx == 0:
                            # Zero means its the ee_pose
                            result[transform_key] = task.value.ee_hrl_start[:3]
                        else:
                            result[transform_key] = task.value.obj_start[:3]
                    if quaternion_key in self.states:
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

    def get_policy(self, task: Task):
        policy = self.policy_storage.get(task)
        if policy is None:
            raise ValueError(f"No policy found for task: {task}")
        return policy
