from conf.master.shared.env import env1
from conf.master.shared.agent import experiment1
from tapas_gmm_modified.master_project.networks import NetworkType
from tapas_gmm_modified.master_train import MasterConfig

config = MasterConfig(
    tag="experiment1_1",
    nt=NetworkType.GNN_V6,
    env=env1,
    agent=experiment1,
)
