from enum import Enum


class Environment(Enum):
    PANDA = "panda"
    MANISKILL = "maniskill"
    RLBENCH = "rlbench"
    CALVIN = "calvin"


def get_env(env_str):
    return Environment[env_str.upper()]


def import_env(config):
    env_type = config.env_type
    if env_type is Environment.PANDA:
        from tapas_gmm_modified.env.franka import FrankaEnv as Env
    elif env_type is Environment.MANISKILL:
        from tapas_gmm_modified.env.mani_skill import ManiSkillEnv as Env
    elif env_type is Environment.RLBENCH:
        from tapas_gmm_modified.env.rlbench import RLBenchEnvironment as Env
    elif env_type is Environment.CALVIN:
        from tapas_gmm_modified.env.calvin import Calvin as Env
    else:
        raise ValueError("Invalid environment {}".format(config.env))

    return Env
