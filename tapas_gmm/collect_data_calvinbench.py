import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm
from tapas_gmm.env.calvinbench import CalvinBenchEnvironmentConfig, CalvinBenchEnvironment
from conversion import calvin_to_tapas_representation
from tapas_gmm.policy.motion_planner import Action, CloseGripperAction, MotionPlannerPolicy, MotionTask, OpenGripperAction
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.env.environment import BaseEnvironmentConfig
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    get_dataset_name,
    get_full_task_name,
    loop_sleep,
)

# from tapas_gmm.utils.random import configure_seeds
@dataclass
class Task:
    task_name: str
    task_sequence: MotionTask

@dataclass
class TestTask(Task):
    task_name: str = "test"
    task_sequence: MotionTask = MotionTask([OpenGripperAction(), CloseGripperAction(), OpenGripperAction(), CloseGripperAction()])


@dataclass
class Config:
    n_episodes: int = 2
    horizon: int = 200

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    env_config: CalvinBenchEnvironmentConfig
    task: TestTask


def main(config: Config = Config()) -> None:
    env = CalvinBenchEnvironment(config.env_config)
    policy = MotionPlannerPolicy(env=env, sequence=config.task.task_sequence) # type: ignore

    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / config.task.task_name

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    replay_memory = SceneDataset(
        allow_creation=True,
        config=config.dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    env.reset()  # extra reset to correct set up of camera poses in first obs
    obs = env.reset()
    obs = calvin_to_tapas_representation(obs)

    time.sleep(5)

    logger.info("Go!")

    episodes_count = 0
    timesteps = 0

    # Max number of timesteps in an episode
    horizon = config.horizon or np.inf

    try:
        with tqdm(total=config.n_episodes) as ebar:
            with tqdm(total=horizon) as tbar:
                tbar.set_description("Time steps")
                while episodes_count < config.n_episodes:
                    ebar.set_description("Running episode")
                    start_time = time.time()

                    obs = calvin_to_tapas_representation(obs)

                    action, info = policy.predict(obs)
                    next_obs, _, done, _ = env.step(action)
                    obs.action = torch.Tensor(action)
                    obs.feedback = torch.Tensor([1])
                    replay_memory.add_observation(obs)
                    obs = next_obs

                    timesteps += 1
                    tbar.update(1)

                    if done:
                        # logger.info("Saving trajectory.")
                        ebar.set_description("Saving trajectory")
                        replay_memory.save_current_traj()

                        obs = env.reset()
                        policy.reset_episode(env)

                        episodes_count += 1
                        ebar.update(1)

                        timesteps = 0
                        tbar.reset()

                        done = False

                    elif timesteps >= horizon:
                        # logger.info("Resetting without saving traj.")
                        ebar.set_description("Resetting without saving traj")
                        replay_memory.reset_current_traj()

                        obs = env.reset()
                        policy.reset_episode(env)

                        timesteps = 0
                        tbar.reset()

                    else:
                        loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


if __name__ == "__main__":
    main()
