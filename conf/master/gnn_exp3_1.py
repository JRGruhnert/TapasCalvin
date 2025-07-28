from conf.master.env.master_env import env_1
from conf.master.agent.agents import gnn_v3_1
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="gnn3_1",
    env=env_1,
    agent=gnn_v3_1,
)
