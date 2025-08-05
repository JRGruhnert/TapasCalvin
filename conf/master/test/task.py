from conf.master.shared.env import test_env
from conf.master.shared.agent import test
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


config = MasterConfig(
    tag="test",
    nt=NetworkType.BASELINE_TEST,
    env=test_env,
    agent=test,
)
