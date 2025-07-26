from enum import Enum


class Network(Enum):
    BASELINEV1 = "baselinev1"
    GNNV1 = "gnnv1"
    GNNV2 = "gnnv2"
    GNNV3 = "gnnv3"


def get_network(env_str):
    return Network[env_str.upper()]


def import_network(network_type):
    if network_type is Network.BASELINEV1:
        from tapas_gmm.master_project.networks.baseline import BaselineV1 as Net
    elif network_type is Network.GNNV1:
        from tapas_gmm.master_project.networks.gnn import GnnV1 as Net
    elif network_type is Network.GNNV2:
        from tapas_gmm.master_project.networks.gnn import GnnV2 as Net
    elif network_type is Network.GNNV3:
        from tapas_gmm.master_project.networks.gnn import GnnV3 as Net
    else:
        raise ValueError("Invalid network")

    return Net
