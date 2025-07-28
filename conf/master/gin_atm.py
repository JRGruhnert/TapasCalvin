from tapas_gmm.master_project.agent import AgentConfig

from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig
from conf.master.env.master_env import env_2

agent = AgentConfig(
    name="original",
    network=Network.GNNV3,
    lr_actor=0.0003,
    lr_annealing=False,
    early_stop_patience=20,
)

config = MasterConfig(
    tag="original",
    env=env_2,
    agent=agent,
)
