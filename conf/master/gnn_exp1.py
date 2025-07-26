from conf.master.env.master_env import env
from conf.master.agent.agents import gnn_v1
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    env=env,
    agent=gnn_v1,
)
