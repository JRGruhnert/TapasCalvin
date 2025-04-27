from omegaconf import MISSING

from tapas_gmm.env.calvin import CalvinConfig

calvin_env_config = CalvinConfig(
    task=MISSING,
    cameras=("wrist", "front"),
    camera_pose={},
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    gripper_plot=False,
    postprocess_actions=True,
)
