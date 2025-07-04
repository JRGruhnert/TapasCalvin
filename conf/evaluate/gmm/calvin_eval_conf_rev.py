from conf._machine import data_naming_config
from conf.env.calvin.env_eval_conf import calvin_env_config
from conf.policy.models.tpgmm.master_project import auto_tpgmm_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.gmm import GMMPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

eval = EvalConfig(
    n_episodes=10,
    seed=0,
    obs_dropout=None,
    viz=False,
    kp_per_channel_viz=False,
    show_channels=None,
    horizon=100,
)

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="demos",
    data_root=None,
)

policy_config = GMMPolicyConfig(
    suffix="release",
    model=auto_tpgmm_config,
    time_based=True,
    predict_dx_in_xdx_models=False,
    binary_gripper_action=True,
    binary_gripper_closed_threshold=0.95,
    dbg_prediction=False,
    # the kinematics model in RLBench is just to unreliable -> leads to mistakes
    topp_in_t_models=False,
    force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
    time_scale=1.0,
    # ---- Changing often ----
    postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
    return_full_batch=True,
    batch_predict_in_t_models=True,  # Change if visualization is needed
    invert_prediction_batch=True,
)


config = Config(
    env=calvin_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
    policy_type="gmm",
)


def get_eval_config(task_name: str) -> Config:
    """
    Get the evaluation configuration for a specific task.
    """
    config.data_naming.task = task_name
    return config
