import numpy as np
from torch_geometric.data import HeteroData

from tapas_gmm.policy.gmm import GMMPolicy


def extract_keypoints(model: GMMPolicy) -> list[int]:
    # Extract keypoints from the model
    # Maps keypoints to integer values
    # TODO: Implement
    return model.get_keypoints()


class HRLModelData:
    def __init__(self, data: list[GMMPolicy]):
        self.data = data
        for i, model in enumerate(data):
            self.data[i] = extract_keypoints(model)

    def indeces_array(self) -> int:
        # Return the number of models
        return np.arange(len(self.data) + 1)  # +1 for the exit Node

    def task_params(self) -> int:
        # Return the task parameters
        return np.array([model.get_task_params() for model in self.data])


class HRLStateData:
    def __init__(self, data: list):
        self.data = data

    def prepare_data(self):
        # Prepare data for the state
        # This is a placeholder function and should be replaced with actual data preparation logic
        self.data = [encode_observation(obs) for obs in self.data]
        return self.data


class HRLBipartiteData(HeteroData):
    def __init__(self, md: HRLModelData, sd: HRLStateData):
        super().__init__()
        self.data["models"].index = md.indeces_array()
        self.data["models"].param = md.task_params()

        self.data["states"].goal_state = sd.prepare_data()
        self.data["states"].next_model = md.indeces_array()
        self.data["states"].y = sd.prepare_data()
        self.data["model", "to", "state"].edge_index = md.prepare_data()
        self.data["state", "to", "model"].edge_index = sd.prepare_data()
        self.data["models"].edge_index = md.prepare_data()
        self.data["states"].edge_index = sd.prepare_data()
        self.data["models"].edge_attr = md.prepare_data()
        self.data["states"].edge_attr = sd.prepare_data()
        self.data["models"].edge_attr = md.prepare_data()
        self.data["model"].x = md.data

    def reset(self):
        # Reset the data to its initial state
        self.data["model"].x = None
        self.data["model"].y = None
        self.data["state"].x = None
        self.data["state"].y = None
        self.data["model", "to", "state"].edge_index = None
        self.data["state", "to", "model"].edge_index = None
        self.data["model"].edge_index = None
        self.data["state"].edge_index = None
        self.data["model"].edge_attr = None
        self.data["state"].edge_attr = None

    def __getitem__(self, index):
        return self.data[index]


def encode_observation(models: GMMPolicy, states: list) -> HRLBipartiteData:
    # Encode the observation into a format suitable for the model
    # This is a placeholder function and should be replaced with actual encoding logic
    return HRLBipartiteData(
        HRLModelData(models),
        HRLStateData(states),
    )


def decode_observation(encoded_data: HRLBipartiteData) -> list:
    # Decode the observation from the model's output
    # This is a placeholder function and should be replaced with actual decoding logic
    return encoded_data.data
