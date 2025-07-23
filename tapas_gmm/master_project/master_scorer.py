from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from tapas_gmm.master_project.master_converter import NodeConverter
from tapas_gmm.master_project.master_definitions import (
    RewardMode,
    State,
    StateType,
    Task,
)
from tapas_gmm.master_project.master_observation import MasterObservation


@dataclass
class ScorerConfig:
    reward_mode: RewardMode
    max_reward: float
    min_reward: float
    allowed_steps: int
    success_threshold: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 0.05,
            StateType.Quat: 0.1,
            StateType.Scalar: 0.05,
        }
    )
    reward_scale: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 2.0,
            StateType.Quat: 1.0,
            StateType.Scalar: 2.0,
        }
    )


class Scorer:
    def __init__(
        self,
        config: ScorerConfig,
        active_states: list[State],
        active_tasks: list[Task],
    ):
        self.config = config
        self.last: MasterObservation = None
        self.goal: MasterObservation = None
        self.steps_left = config.allowed_steps
        self.active_states = active_states
        self.active_tasks = active_tasks
        total = sum(config.reward_scale.values())
        self.normalized_reward_scale = {
            key: value / total for key, value in config.reward_scale.items()
        }
        self.surfaces = {
            "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.52]],
            # "slider_left": [[-0.32, 0.05, 0.46], [-0.16, 0.12, 0.46]],
            "slider_right": [[-0.05, 0.05, 0.46], [0.13, 0.12, 0.52]],
            "drawer_open": [[0.0, -0.35, 0.38], [0.40, 0.12, 0.44]],
        }  # changed drawer box since its a movable surface
        # NOTE: Coords for original surfaces
        # table: [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]]
        # slider_left: [[-0.32, 0.05, 0.46], [-0.16, 0.12, 0.46]]
        # slider_right: [[-0.05, 0.05, 0.46], [0.13, 0.12, 0.46]]
        # drawer_open: [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]]
        self.converter = NodeConverter(
            state_list=self.active_states,
            task_list=self.active_tasks,
            normalized=True,
        )
        self.prev_dist: Dict[State, torch.Tensor] = {}

    def reset(self, obs: MasterObservation, goal: MasterObservation):
        self.prev_dist = self.converter.dict_distance(obs, goal)
        self.steps_left = self.config.allowed_steps
        self.last = obs
        self.goal = goal

    def reward(self, obs: MasterObservation) -> tuple[float, bool]:

        # Compute distances to goal
        current_dist = self.converter.dict_distance(obs, self.goal)

        if self.steps_left == 1:
            # Last Step -> terminal
            terminal = True
        else:
            # Either goal reached or not
            terminal = self._is_terminal(obs, self.active_states, current_dist)

        if self.config.reward_mode is RewardMode.SPARSE:
            if termi

    def _difference_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.state_type]
            # Multiply and accumulate
            total += scale * diff

        return total / len(prev)

    def _on_off_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.state_type]
            if diff < 0:
                total -= 1.0 * scale
            elif diff > 0:
                total += 1.0 * scale
        return total / len(prev)

    def _is_terminal(
        self,
        obs: MasterObservation,
        active_states: list[State],
        dist: Dict[State, torch.Tensor],
    ) -> bool:
        ##### Checking if goal is reached
        goal_reached = True
        for state in active_states:
            state_type = state.value.state_type
            if state_type == StateType.Transform and state is not State.EE_Transform:
                goal_value = self.goal.transform_states[state]
                goal_surface = self._check_surface(goal_value)
                next_value = obs.transform_states[state]
                next_surface = self._check_surface(next_value)
                if goal_surface != next_surface:
                    goal_reached = False
                    break
            else:  # Scalars and EE States
                if dist[state] > self.config.success_threshold[state_type]:
                    goal_reached = False
                    break
        return goal_reached

    def _check_surface(self, transform) -> str | None:
        for name, (min_corner, max_corner) in self.surfaces.items():
            box_min = np.array(min_corner)
            box_max = np.array(max_corner)
            if np.all(transform >= box_min) and np.all(transform <= box_max):
                return name
        return None
