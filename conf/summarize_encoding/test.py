from conf._machine import data_naming_config
from tapas_gmm_modified.summarize_encoding import Config
from tapas_gmm_modified.utils.observation import ObservationConfig

observation_config = ObservationConfig(
    cameras=("base",),
    image_crop=None,
)

config = Config(
    data_naming=data_naming_config,
    observation=observation_config,
)
