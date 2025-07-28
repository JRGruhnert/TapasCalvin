from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.networks import Network

baseline_v1 = AgentConfig(
    name="baseline_v1",
    network=Network.BASELINEV1,
)

gnn_v1 = AgentConfig(
    name="gnn_v1",
    network=Network.GNNV1,
)

gnn_v2_1 = AgentConfig(
    name="gnn_v2",
    network=Network.GNNV2,
    early_stop_patience=10,
)

gnn_v2_2 = AgentConfig(
    name="gnn_v2",
    network=Network.GNNV2,
    lr_actor=0.0002,
    early_stop_patience=10,
    lr_annealing=False,
)

gnn_v3_1 = AgentConfig(
    name="gnn_v3",
    network=Network.GNNV3,
    lr_actor=0.0005,
    early_stop_patience=10,
)

gnn_v3_2 = AgentConfig(
    name="gnn_v3",
    network=Network.GNNV3,
    lr_actor=0.0002,
    early_stop_patience=10,
    lr_annealing=False,
)


gnn_v4_1 = AgentConfig(
    name="gnn_v4",
    network=Network.GNNV4,
    lr_actor=0.0005,
    early_stop_patience=10,
)

gnn_v4_2 = AgentConfig(
    name="gnn_v4",
    network=Network.GNNV4,
    lr_actor=0.0002,
    early_stop_patience=10,
    lr_annealing=False,
)


gnn_v5_1 = AgentConfig(
    name="gnn_v5",
    network=Network.GNNV5,
    lr_actor=0.0005,
    early_stop_patience=10,
)

gnn_v5_2 = AgentConfig(
    name="gnn_v5",
    network=Network.GNNV5,
    lr_actor=0.0002,
    early_stop_patience=10,
    lr_annealing=False,
)
