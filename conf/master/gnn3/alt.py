from conf.master.shared.env import env1
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="alt",
    nt=NetworkType.GNN_V3,
    env=env1,
    agent=experiment1,
)
