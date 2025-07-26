import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from torch_geometric.data import Batch, HeteroData
from tapas_gmm.master_project.master_converter import Converter
from tapas_gmm.master_project.master_definitions import State, StateType, Task
from tapas_gmm.master_project.master_observation import Observation
from tapas_gmm.master_project.networks.master_modules import (
    QuaternionEncoder,
    ScalarEncoder,
    TransformEncoder,
)
from tapas_gmm.utils.select_gpu import device


class ActorCriticBase(nn.Module, ABC):
    def __init__(
        self,
        tasks: list[Task],
        states: list[State],
        tps: dict[Task, list[State]],
    ):
        super().__init__()
        self.converter = Converter(tasks, states, tps)
        self.tasks = tasks
        self.states = states
        self.dim_state = len(states)
        self.dim_tasks = len(tasks)
        self.dim_encoder = 32
        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(self.dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(self.dim_encoder),
                StateType.Scalar.name: ScalarEncoder(self.dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Transform.name: TransformEncoder(self.dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(self.dim_encoder),
                StateType.Scalar.name: ScalarEncoder(self.dim_encoder),
            }
        )

    @abstractmethod
    def forward(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def to_batch(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ):
        pass

    def act(
        self,
        obs: Observation,
        goal: Observation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward([obs], [goal])
        dist = Categorical(logits=logits)
        action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action, logprob, value

    def evaluate(
        self,
        obs: list[Observation],
        goal: list[Observation],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, goal)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy


class BaselineBase(ActorCriticBase):
    def to_batch(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ):
        obs_dicts = [self.converter.tensor_type_dict_values(o) for o in obs]
        goal_dicts = [self.converter.tensor_type_dict_values(g) for g in goal]

        keys = obs_dicts[0].keys()

        tensor_obs = {
            k: torch.stack([d[k] for d in obs_dicts], dim=0).detach().to(device)
            for k in keys
        }
        tensor_goal = {
            k: torch.stack([d[k] for d in goal_dicts], dim=0).detach().to(device)
            for k in keys
        }

        return tensor_obs, tensor_goal


class GnnBase(ActorCriticBase, ABC):
    @abstractmethod
    def to_data(self, obs: Observation, goal: Observation) -> HeteroData:
        pass

    def to_batch(
        self,
        obs: list[Observation],
        goal: list[Observation],
    ) -> Batch:
        data = []
        for o, g in zip(obs, goal):
            data.append(self.to_data(o, g))

        return Batch.from_data_list(data).to(device)
