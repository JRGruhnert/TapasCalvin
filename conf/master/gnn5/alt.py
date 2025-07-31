from conf.master.shared.env import env1
from conf.master.shared.agent import experiment1
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="alt",
    env=env1,
    agent=experiment1,
)
