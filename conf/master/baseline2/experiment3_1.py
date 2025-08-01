from conf.master.shared.agent import experiment1
from conf.master.shared.env import env3
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="experiment3_1",
    nt=NetworkType.BASELINE_V2,
    env=env3,
    agent=experiment1,
)
