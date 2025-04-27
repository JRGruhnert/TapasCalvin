from omegaconf import MISSING
from conf._machine import data_naming_config
from conf.dataset.scene.calvin import scene_dataset_config
from conf.env.calvin.env_collect_conf import calvin_env_config
from tapas_gmm.master_collect_data import Config
from tapas_gmm.env import Environment
from tapas_gmm.policy import PolicyEnum

config = Config(
    task=MISSING,
    n_episodes=5,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env=Environment.CALVIN,
    env_config=calvin_env_config,
    policy_type=PolicyEnum.MANUAL,
    policy=None,
)
