from conf.master.env.master_env import env
from conf.master.agent.agents import baseline_v1
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="baselinev1.1",
    env=env,
    agent=baseline_v1,
)
