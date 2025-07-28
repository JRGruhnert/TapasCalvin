from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig

# USING RELU
gnn_v4_1 = AgentConfig(
    name="gnn_v4",
    network=Network.GNNV4,
    lr_actor=0.0003,
    early_stop_patience=10,
)

config = MasterConfig(
    tag="gnn4_1",
    env=env_2,
    agent=gnn_v4_1,
)
