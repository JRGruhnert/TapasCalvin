from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.definitions import RewardMode, StateSpace, TaskSpace
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig


# USING RELU
agent = AgentConfig(
    name="baseline_v3",
    network=Network.BASELINEV1,
)


config = MasterConfig(
    tag="baselinev1_3",
    env=env_2,
    agent=agent,
)
