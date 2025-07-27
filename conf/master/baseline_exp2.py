from conf.master.env.master_env import env
from conf.master.agent.agents import baseline_v1
from tapas_gmm.master_project.definitions import RewardMode, StateSpace, TaskSpace
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="baselinev1_2",
    env=env,
    agent=baseline_v1,
    reward_mode=RewardMode.SPARSE,
    task_space=TaskSpace.SMALL,
    state_space=StateSpace.ALL,
)
