from conf.master.shared.env import env1
from conf.master.shared.agent import test
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="test",
    nt=NetworkType.GNN_TEST,
    env=env1,
    agent=test,
)
