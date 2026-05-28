from tapas_gmm_modified.utils.misc import DataNamingConfig
from tapas_gmm_modified.visualize_demo import Config

data_naming_config = DataNamingConfig(
    task="Test",  # If None, values are taken from data_naming_config
    feedback_type="test",
    data_root="data",
)

config = Config(
    encoder_name="test_vit_keypoints_encoder",
    encoding_name="dbg",
    data_naming=data_naming_config,
)
