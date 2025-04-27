import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any
from mplib.pymp import Pose
import numpy as np
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
import torch
from loguru import logger
from tqdm.auto import tqdm

from tapas_gmm.collect_data import Config
from tapas_gmm.env import Environment
from tapas_gmm.env.calvin import CalvinConfig, Calvin

from tapas_gmm.env.environment import BaseEnvironmentConfig
from tapas_gmm.policy import PolicyEnum
from tapas_gmm.policy.manual_policy import ManualCalvinPolicy
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    get_dataset_name,
    loop_sleep,
)

from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.utils.observation import SceneObservation


@dataclass
class Config:

    task: str
    n_episodes: int
    sequence_len: int | None

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    env: Environment
    env_config: BaseEnvironmentConfig

    policy_type: PolicyEnum
    policy: Any

    horizon: int | None = 300  # None

    pretraining_data: bool = False


def main(config: Config) -> None:
    env = Calvin(config.env_config, eval=False)
    keyboard_obs = KeyboardObserver()
    policy = ManualCalvinPolicy(config, env, keyboard_obs)
    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / config.task

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

    env.reset()
    obs, _, _, _ = env.reset()

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

                    prediction, policy_done, policy_success = policy.predict(
                        obs
                    )  # Action is relative
                    try:
                        next_obs, step_reward, env_done, _ = env.step(prediction)
                    except RuntimeError as e:
                        logger.error(f"Raw action: {prediction}")
                        logger.error(f"Error: {e}")
                        raise e

                    # actual_action = env._get_action(obs, next_obs)
                    # logger.debug(f"Saved Demo Action: {actual_action.round(5)}")
                    ee_delta = env.compute_ee_delta(obs, next_obs)
                    obs.action = torch.Tensor(ee_delta)
                    obs.reward = torch.Tensor([step_reward])
                    replay_memory.add_observation(obs.to_rlbench_format())

                    obs = next_obs

                    timesteps += 1
                    tbar.update(1)

                    if env_done or policy_success:
                        # logger.info("Saving trajectory.")
                        ebar.set_description("Saving trajectory")
                        replay_memory.save_current_traj()

                        obs, _, policy_done, _ = env.reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        episodes_count += 1
                        ebar.update(1)

                        timesteps = 0
                        tbar.reset()

                    elif policy_done and not policy_success or timesteps >= horizon:
                        # logger.info("Resetting without saving traj.")
                        ebar.set_description("Resetting without saving traj")
                        replay_memory.reset_current_traj()

                        obs, _, _, _ = env.reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        timesteps = 0
                        tbar.reset()

                    else:
                        loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


def complete_config(config: DictConfig) -> DictConfig:
    config.env_config.task = config.data_naming.task
    config.task = config.data_naming.task
    config.dataset_config.data_root = config.data_naming.data_root

    config.data_naming.feedback_type = get_dataset_name(config)

    return config


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, extra_args={})
    dict_config = complete_config(dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
