from conf.master.env.master_env import env_1
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig

gnn_v1 = AgentConfig(
    name="gnn_v1",
    network=Network.GNNV1,
)


config = MasterConfig(
    tag="gnn1_1",
    env=env_1,
    agent=gnn_v1,
)
