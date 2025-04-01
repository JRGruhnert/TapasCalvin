from omegaconf import MISSING

from tapas_gmm.env.calvinbench import CalvinEnvironmentConfig

calvin_tapas_env_config = CalvinEnvironmentConfig(
    task=MISSING,
    cameras=("front", "wrist"),
    camera_pose={},
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    postprocess_actions=False,
    invert_xy=False,
    gripper_plot=False,
    render_sapien=False,
    background=None,
    model_ids=None,
    real_depth=False,
    seed=None,
)
