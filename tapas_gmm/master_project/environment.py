from dataclasses import dataclass

import numpy as np
from tapas_gmm.env.calvin import Calvin
from tapas_gmm.master_project.definitions import (
    State,
    StateSpace,
    Task,
    TaskSpace,
    convert_to_states,
    convert_to_tasks,
)
from tapas_gmm.master_project.observation import Observation, tapas_format
from tapas_gmm.master_project.evaluator import (
    Evaluator,
    EvaluatorConfig,
)
from tapas_gmm.master_project.sampler import (
    Sampler,
    SamplerConfig,
)
from tapas_gmm.master_project.storage import (
    Storage,
    StorageConfig,
)


@dataclass
class MasterEnvConfig:
    task_space: TaskSpace
    state_space: StateSpace
    eval_mode: bool
    pybullet_vis: bool
    debug_vis: bool
    evaluator: EvaluatorConfig
    storage: StorageConfig
    sampler: SamplerConfig


class MasterEnv:

    def __init__(
        self,
        config: MasterEnvConfig,
    ):
        self.config = config
        self.tasks = convert_to_tasks(config.task_space)
        self.states = convert_to_states(config.state_space)

        if self.config.task_space == TaskSpace.SMALL:
            self.config.evaluator.allowed_steps = 6  # Max 6 Steps needed
        else:
            self.config.evaluator.allowed_steps = 18  # Max 16 Steps needed

        self.env = Calvin(eval=config.eval_mode, vis=config.pybullet_vis)
        self.evaluator = Evaluator(config.evaluator, self.tasks, self.states)
        self.sampler = Sampler(config.sampler, self.states)
        self.policy_storage = Storage(config.storage, self.tasks, self.states)
        self.obs: Observation = None

    def publish(
        self,
    ) -> tuple[list[Task], list[State], dict[Task, dict[State, np.ndarray]]]:
        return self.tasks, self.states, self.policy_storage.task_parameter()

    def reset(self) -> tuple[Observation, Observation]:
        calvin_rnd, _, _, _ = self.env.reset(settle_time=0)
        ### Sampling goal and current state
        scene_obs = self.sampler.sample_pre_condition(calvin_rnd.scene_obs)
        scene_goal = self.sampler.sample_post_condition(scene_obs)
        # Reset environment twice to get CalvinObservation (maybe find a better way)
        # NOTE do not switch order of resets!!
        calvin_goal, _, _, _ = self.env.reset(scene_goal, static=False, settle_time=50)
        self.calvin_obs, _, _, _ = self.env.reset(
            scene_obs, static=False, settle_time=50
        )
        self.obs = Observation(self.calvin_obs)
        goal = Observation(calvin_goal)
        self.evaluator.reset(self.obs, goal)
        return self.obs, goal

    def step(
        self, task_id: int, verbose: bool = False
    ) -> tuple[float, bool, Observation]:
        task = Task.get_enum_by_index(task_id)
        viz_dict = {}  # TODO: Make available
        # Loads Tapas Policy for that Task (batch predict config)
        policy = self.policy_storage.get_policy(task)
        policy.reset_episode(self.env)
        # Batch prediction for the given observation
        try:
            prediction, _ = policy.predict(tapas_format(self.calvin_obs, task))
            for action in prediction:
                ee_action = np.concatenate((action.ee, action.gripper))
                self.calvin_obs, _, _, _ = self.env.step(
                    ee_action, self.config.debug_vis, viz_dict
                )
                if verbose:
                    print(self.calvin_obs.ee_pose)
                self.obs = Observation(self.calvin_obs)
        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")
        reward, done = self.evaluator.evaluate(self.obs)
        return reward, done, self.obs

    def close(self):
        self.env.close()
