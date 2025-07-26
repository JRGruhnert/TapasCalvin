from tapas_gmm.master_project.master_agent import AgentConfig
from tapas_gmm.master_project.networks import Network

baseline_v1 = AgentConfig(
    name="baseline_v1",
    network=Network.BASELINEV1,
)

gnn_v1 = AgentConfig(
    name="gnn_v1",
    network=Network.GNNV1,
)

gnn_v2 = AgentConfig(
    name="gnn_v2",
    network=Network.GNNV2,
)

gnn_v3 = AgentConfig(
    name="gnn_v3",
    network=Network.GNNV3,
)
