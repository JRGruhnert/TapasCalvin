from enum import Enum


class NetworkType(Enum):
    BASELINE_V1 = "baseline1"
    BASELINE_V2 = "baseline2"
    GNN_V1 = "gnn1"
    GNN_V2 = "gnn2"
    GNN_V3 = "gnn3"
    GNN_V4 = "gnn4"
    GNN_V5 = "gnn5"
    GNN_V6 = "gnn6"
    BASELINE_TEST = "test_baseline"
    GNN_TEST = "test_gnn"


def to_nt(network_str):
    return NetworkType[network_str.upper()]


def import_network(network_type):
    if network_type is NetworkType.GNN_TEST:
        from tapas_gmm.master_project.networks.gnn.gnn6 import Gnn as Network
    elif network_type is NetworkType.GNN_V1:
        from tapas_gmm.master_project.networks.gnn.gnn1 import Gnn as Network
    elif network_type is NetworkType.GNN_V2:
        from tapas_gmm.master_project.networks.gnn.gnn2 import Gnn as Network
    elif network_type is NetworkType.GNN_V3:
        from tapas_gmm.master_project.networks.gnn.gnn3 import Gnn as Network
    elif network_type is NetworkType.GNN_V4:
        from tapas_gmm.master_project.networks.gnn.gnn4 import Gnn as Network
    elif network_type is NetworkType.GNN_V5:
        from tapas_gmm.master_project.networks.gnn.gnn5 import Gnn as Network
    elif network_type is NetworkType.GNN_V6:
        from tapas_gmm.master_project.networks.gnn.gnn6 import Gnn as Network
    elif network_type is NetworkType.BASELINE_TEST:
        from tapas_gmm.master_project.networks.baseline.baseline1 import (
            Baseline as Network,
        )
    elif network_type is NetworkType.BASELINE_V1:
        from tapas_gmm.master_project.networks.baseline.baseline1 import (
            Baseline as Network,
        )
    elif network_type is NetworkType.BASELINE_V2:
        from tapas_gmm.master_project.networks.baseline.baseline1 import (
            Baseline as Network,
        )
    else:
        raise ValueError(f"Invalid network {network_type.value}")
    return Network
