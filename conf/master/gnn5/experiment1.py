from conf.master.shared.env import env2
from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import MasterConfig


gnn_v5_1 = AgentConfig(
    name="gnn_v5",
    network=NetworkType.GNN_V5,
)


config = MasterConfig(
    tag="gnn5_1",
    env=env2,
    agent=gnn_v5_1,
)
