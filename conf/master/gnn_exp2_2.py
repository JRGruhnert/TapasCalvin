from conf.master.env.master_env import env_2
from conf.master.agent.agents import gnn_v2_2
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="gnn2_2",
    env=env_2,
    agent=gnn_v2_2,
)
