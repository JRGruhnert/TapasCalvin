from tapas_gmm.master_project.agent import AgentConfig

from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig
from conf.master.env.master_env import env_1

agent = AgentConfig(
    name="test",
    network=Network.GNNV2,
    batch_size=8,
    mini_batch_size=4,
)

config = MasterConfig(
    tag="test",
    env=env_1,
    agent=agent,
)
