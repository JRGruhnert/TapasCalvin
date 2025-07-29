from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig

gnn_v2_1 = AgentConfig(
    name="gnn_v2",
    network=Network.GNNV2,
)

config = MasterConfig(
    tag="gnn2_1",
    env=env_2,
    agent=gnn_v2_1,
)
