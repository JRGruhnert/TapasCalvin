import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, Linear

from tapas_gmm.policy.models.master_project.hrl_data import HRLData
from tapas_gmm.policy.gmm import GMMPolicy


def get_task_parameters(child_policies: list[GMMPolicy]) -> list:
    # Extract task parameters from child policies
    task_params = []
    for policy in child_policies:
        params = policy.get_task_params()
        task_params.append(params)
    return task_params


def get_edge_index(model_task_params, state_set, object_count):
    # Create edge index based on model task parameters and state set
    # This is a placeholder function and should be replaced with actual logic
    edge_index = []
    for i, params in enumerate(model_task_params):
        for j, state in enumerate(state_set):
            if params == state:
                edge_index.append((i, j))
    return torch.tensor(edge_index).t().contiguous()  # Convert to tensor format


def get_hrl_reward(init_state, current_state, goal_state, object_count):
    # Calculate the reward based on the initial state, current state, and goal state
    # This is a placeholder function and should be replaced with actual reward calculation logic
    reward = 0.0
    for i in range(object_count):
        reward += abs(current_state[i] - goal_state[i])
    return reward


# ======================
# GATv2 Policy Network
# ======================
class HRLPolicy(nn.Module):
    def __init__(
        self, child_policies: list[GMMPolicy], state_set, object_count, horizon=8
    ):
        super().__init__()

        self.child_policies = child_policies
        self.state_set = state_set
        self.object_count = object_count
        self.horizon = horizon
        # Number of state nodes
        self.state_node_count = len(state_set) + object_count * 7
        # For exit model  # Number of model nodes
        self.model_node_count = len(child_policies) + 1
        # Model node encoder
        # self.model_encoder = Linear(state_node_count, hidden_dim)
        # State node encoder
        # self.state_encoder = Linear(model_node_count, hidden_dim)

        self.model_task_params = get_task_parameters(child_policies)
        self.edge_index = get_edge_index(
            self.model_task_params, state_set, object_count
        )

        self.state_feature_dim = (
            128  # [x1, x2, x3, ..., or1, or2]  # Hidden dimension for state encoder
        )
        self.model_feature_dim = 128  # Hidden dimension for model encoder

        # Hier vorher noch delta differences berechnen?

        # GATv2 layer for model->state attention
        # Gives dynamic Attention between state features and model features
        self.conv = GATv2Conv(
            in_channels=(self.state_feature_dim, self.model_feature_dim),
            out_channels=self.model_feature_dim,
            heads=1,
            add_self_loops=False,
        )

        # Layer for model -> prediction array
        # Prediction has Horizon length and is a sequence of models
        # Predictor: maps the LSTM hidden state to logits over model actions.

        inner_dim = 128
        self.predictor = nn.Sequential(
            nn.Linear(self.model_feature_dim, inner_dim),
            nn.ReLU(),  # ReLU(x)=max(0,x)
            nn.Linear(
                inner_dim, self.model_node_count
            ),  # Output logits over model indices
        )

        # LSTM cell to incorporate recurrence.
        self.lstm = nn.LSTMCell(self.model_feature_dim, self.model_feature_dim)

        # Embedding layer for the selected action (to feed back into the LSTM).
        self.action_embedding = nn.Embedding(
            self.model_node_count, self.model_feature_dim
        )

    def forward(
        self, state_features, model_features, initial_hidden=None, initial_cell=None
    ):
        """
        Forward pass that generates a sequence of actions and corresponding log probabilities.

        Args:
            state_features: Tensor of shape (num_state_nodes, state_feature_dim)
            model_features: Tensor of shape (num_model_nodes, model_feature_dim)
            initial_hidden, initial_cell: Optional LSTM states (shape (1, model_feature_dim))

        Returns:
            actions: list of sampled model indices (one per timestep)
            log_probs: list of log-probabilities corresponding to each sampled action
            (h_t, c_t): Final LSTM states (can be passed for further recurrence)
        """
        # Inputs the new state features (From Calvin Environment) and model features (From GMMPolicy)
        # into the GATv2 layer to get updated model features.
        # The model features are updated based on the state features and the edge index.
        # The edge index defines the connections between the model and state nodes.

        updated_model_features = self.conv(
            (state_features, model_features), self.edge_index
        )
        # Pool to get a global model feature (here by taking the mean).
        global_model_feature = updated_model_features.mean(
            dim=0
        )  # Shape: (model_feature_dim)

        device = global_model_feature.device

        # Initialize LSTM hidden and cell states if not provided.
        h_t = (
            initial_hidden
            if initial_hidden is not None
            else torch.zeros(1, self.model_feature_dim, device=device)
        )
        c_t = (
            initial_cell
            if initial_cell is not None
            else torch.zeros(1, self.model_feature_dim, device=device)
        )

        # Start LSTM input with the global feature.
        lstm_input = global_model_feature.unsqueeze(0)  # Shape: (1, model_feature_dim)

        actions = []  # To store sampled actions
        log_probs = []  # To store log probabilities for each action

        # Loop over the planning horizon.
        for t in range(self.horizon):
            h_t, c_t = self.lstm(lstm_input, (h_t, c_t))
            logits = self.predictor(h_t)  # Shape: (1, model_node_count)
            logits = logits.squeeze(0)  # Shape: (model_node_count,)

            # Create a categorical distribution from logits and sample an action.
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

            # Embed the action to use as input for next LSTM step.
            lstm_input = self.action_embedding(action).unsqueeze(0)

        return actions, log_probs, (h_t, c_t)

    def forward(self, data: dict):

        init_state = data["init_state"]
        current_state = data["current_state"]
        goal_state = data["goal_state"]

        # Prepare the data for the model
        # This is a placeholder function and should be replaced with actual data preparation logic
        reward = get_hrl_reward(
            init_state, current_state, goal_state, self.object_count
        )

        # Encode the model and state features
        x_model = self.model_encoder(data["model"].x)
        x_state = self.state_encoder(data["state"].x)

        # Perform message passing
        x_state = self.conv((x_model, x_state), self.edge_index)

        # Aggregate state information
        global_state = x_state.mean(dim=0)  # [hidden_dim]

        # Compute model logits
        model_logits = self.policy_head(x_model).squeeze(-1)  # [num_models]

        return model_logits
