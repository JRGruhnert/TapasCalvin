from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.definitions import RewardMode, StateSpace, TaskSpace
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig

baseline_v1 = AgentConfig(
    name="baseline_v1",
    network=Network.BASELINEV1,
)


config = MasterConfig(
    tag="baselinev1_2",
    env=env_2,
    agent=baseline_v1,
)
