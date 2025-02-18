from conf._machine import data_naming_config
from conf.dataset.scene.calvinbench import scene_dataset_config
from conf.env.calvinbench.extract_demos import calvin_tapas_env_config
from tapas_gmm.collect_data import Config
from tapas_gmm.env import Environment
from tapas_gmm.policy import PolicyEnum

config = Config(
    n_episodes=20,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env=Environment.CALVINBENCH,
    env_config=calvin_tapas_env_config,
    policy=PolicyEnum.MANUAL,
    policy=None,
)
