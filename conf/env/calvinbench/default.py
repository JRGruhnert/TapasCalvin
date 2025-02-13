from omegaconf import MISSING

from tapas_gmm.env.calvinbench import CalvinTapasBridgeEnvironmentConfig

calvinbench_env_config = CalvinTapasBridgeEnvironmentConfig(
    task=MISSING,
    cameras=("static", "gripper"),
    camera_pose={
        "base": (0.2, 0.0, 0.2, 0, 0.194, 0.0, -0.981),
        # "overhead": (0.2, 0.0, 0.2, 7.7486e-07, -0.194001, 7.7486e-07, 0.981001),
    },
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    gripper_plot=False,
    postprocess_actions=True,
)
