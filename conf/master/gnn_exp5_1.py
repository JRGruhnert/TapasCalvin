from conf.master.env.master_env import env_2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig


gnn_v5_1 = AgentConfig(
    name="gnn_v5",
    network=Network.GNNV5,
    lr_actor=0.0005,
    early_stop_patience=10,
)


config = MasterConfig(
    tag="gnn5_1",
    env=env_2,
    agent=gnn_v5_1,
)
