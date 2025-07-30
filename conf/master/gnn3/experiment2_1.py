from conf.master.shared.env import env2
from conf.master.shared.agent import experiment1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="experiment2_1",
    nt=NetworkType.GNN_V3,
    env=env2,
    agent=experiment1,
)
