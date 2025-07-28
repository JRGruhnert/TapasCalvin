from conf.master.env.master_env import env_2
from conf.master.agent.agents import gnn_v5
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="gnn5_1",
    env=env_2,
    agent=gnn_v5,
)
