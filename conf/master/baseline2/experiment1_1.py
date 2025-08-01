from conf.master.shared.agent import experiment1
from conf.master.shared.env import env1
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="experiment1_1",
    nt=NetworkType.BASELINE_V2,
    env=env1,
    agent=experiment1,
)
