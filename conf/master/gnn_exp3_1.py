from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig


# USING RELU
gnn_v3_1 = AgentConfig(
    name="gnn_v3",
    network=Network.GNNV3,
)


config = MasterConfig(
    tag="gnn3_1",
    env=env_2,
    agent=gnn_v3_1,
)
