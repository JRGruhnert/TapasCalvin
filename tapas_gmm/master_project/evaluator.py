from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from tapas_gmm.master_project.converter import Converter
from tapas_gmm.master_project.definitions import (
    RewardMode,
    State,
    StateType,
    SuccessMode,
    Task,
)
from tapas_gmm.master_project.observation import Observation


@dataclass
class EvaluatorConfig:
    allowed_steps: int = 18
    reward_mode: RewardMode = RewardMode.SPARSE
    success_mode: SuccessMode = SuccessMode.POSITION
    max_reward: float = 100.0
    min_reward: float = 0.0
    success_threshold: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 0.05,
            StateType.Quaternion: 0.1,
            StateType.Scalar: 0.05,
        }
    )
    reward_scale: Dict[StateType, float] = field(
        default_factory=lambda: {
            StateType.Transform: 2.0,
            StateType.Quaternion: 1.0,
            StateType.Scalar: 2.0,
        }
    )


class Evaluator:
    def __init__(
        self,
        config: EvaluatorConfig,
        tasks: list[Task],
        states: list[State],
    ):
        self.config = config
        self.steps_left = config.allowed_steps
        self.states = states
        self.tasks = tasks
        total = sum(config.reward_scale.values())
        self.normalized_reward_scale = {
            key: value / total for key, value in config.reward_scale.items()
        }
        self.converter = Converter(states=self.states, tasks=self.tasks)
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

        # Internal States
        self.prev_dist: Dict[State, torch.Tensor] = {}
        self.last: Observation = None
        self.goal: Observation = None
        self.terminal = False

    def reset(self, obs: Observation, goal: Observation):
        self.prev_dist = self.converter.dict_distance(obs, goal)
        self.steps_left = self.config.allowed_steps
        self.last = obs
        self.goal = goal
        self.terminal = False

    def evaluate(self, obs: Observation) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        self.steps_left -= 1
        # Compute distances to goal
        current_dist = self.converter.dict_distance(obs, self.goal)

        # NOTE: Replace prev_dist when other reward modes implemented
        # For sparse its not needed
        self.prev_dist = current_dist
        if self.config.reward_mode is RewardMode.SPARSE:
            terminal = self.is_terminal(obs, current_dist)
            if terminal:  # Success
                self.terminal = terminal
                return self.config.max_reward, terminal
            else:
                if self.steps_left == 0:  # Failure
                    self.terminal = terminal
                    return self.config.min_reward, True
                else:  # Normal Step
                    self.terminal = terminal
                    return self.config.min_reward, False
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def difference_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.type]
            # Multiply and accumulate
            total += scale * diff

        return total / len(prev)

    def on_off_reward(
        self,
        prev: dict[State, torch.Tensor],
        next: dict[State, torch.Tensor],
    ) -> float:
        total = 0.0
        for key in prev:
            diff = prev[key].item() - next[key].item()
            scale = self.normalized_reward_scale[key.value.type]
            if diff < 0:
                total -= 1.0 * scale
            elif diff > 0:
                total += 1.0 * scale
        return total / len(prev)

    def is_terminal(
        self,
        obs: Observation,
        dist: Dict[State, torch.Tensor],
    ) -> bool:
        ##### Checking if goal is reached
        goal_reached = True
        for state in self.states:
            state_type = state.value.type
            if self.config.success_mode is SuccessMode.AREA:
                if (
                    state_type is StateType.Transform
                    and state is not State.EE_Transform
                ):
                    goal_value = self.goal.states[state]
                    goal_surface = self.check_surface(goal_value)
                    next_value = obs.states[state]
                    next_surface = self.check_surface(next_value)
                    if goal_surface != next_surface:
                        goal_reached = False
                        break
                elif state_type is StateType.Quaternion:
                    pass  # Ignores Rotation
                else:  # Scalars and EE States
                    if dist[state] > self.config.success_threshold[state_type]:
                        goal_reached = False
                        break
            elif self.config.success_mode is SuccessMode.POSITION:
                if state_type is StateType.Quaternion:
                    pass  # Ignores Rotation
                else:  # Scalars and EE States
                    if dist[state] > self.config.success_threshold[state_type]:
                        goal_reached = False
                        break
            else:
                raise NotImplementedError(
                    f"Success Mode {self.config.success_mode.name} is not implemented yet."
                )
        return goal_reached

    def check_surface(self, transform) -> str | None:
        for name, (min_corner, max_corner) in self.surfaces.items():
            box_min = np.array(min_corner)
            box_max = np.array(max_corner)
            if np.all(transform >= box_min) and np.all(transform <= box_max):
                return name
        return None
