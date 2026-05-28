from conf._machine import data_naming_config
from tapas_gmm_modified.env import Environment
from tapas_gmm_modified.tsdf_fusion import Config

config = Config(
    data_naming=data_naming_config,
    env=Environment.RLBENCH,
)
