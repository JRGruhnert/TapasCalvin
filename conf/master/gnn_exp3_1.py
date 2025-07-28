from conf.master.env.master_env import env_1
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network
from tapas_gmm.master_train import MasterConfig

gnn_v3_1 = AgentConfig(
    name="gnn_v3",
    network=Network.GNNV3,
    lr_actor=0.0005,
    early_stop_patience=10,
)


config = MasterConfig(
    tag="gnn3_1",
    env=env_1,
    agent=gnn_v3_1,
)
