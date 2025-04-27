import numpy as np


class TapasTask:
    def __init__(
        self, name: str, horizon: int, state_mask: np.ndarray, start_state: np.ndarray, goal_state: np.ndarray
    ):
        self._horizon = horizon
        self._state_mask = state_mask
        self._start_state = start_state
        self._goal_state = goal_state

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def state_mask(self) -> np.ndarray:
        return self._state_mask

    @property
    def start_state(self) -> np.ndarray:
        return self._start_state

    @property
    def goal_state(self) -> np.ndarray:
        return self._goal_state
