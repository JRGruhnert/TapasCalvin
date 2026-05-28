from conf._machine import data_naming_config
from conf.dataset.scene.franka import scene_dataset_config
from conf.env.panda.handguide_coffee import franka_env_config
from tapas_gmm_modified.collect_data import Config
from tapas_gmm_modified.env import Environment
from tapas_gmm_modified.policy import PolicyEnum

config = Config(
    n_episodes=20,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env=Environment.PANDA,
    env_config=franka_env_config,
    policy=PolicyEnum.MANUAL,
    policy=None,
    horizon=None,
)
