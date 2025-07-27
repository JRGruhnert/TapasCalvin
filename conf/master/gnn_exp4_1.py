from conf.master.env.master_env import env
from conf.master.agent.agents import gnn_v4
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="gnn4_1",
    env=env,
    agent=gnn_v4,
)
