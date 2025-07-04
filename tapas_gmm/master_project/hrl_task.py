import numpy as np
from calvin_env.envs.observation import CalvinObservation


class HRLTask:
    def __init__(
        self,
        name: str,
        horizon: int,
        state_mask: np.ndarray,
        start_state: np.ndarray,
        goal_state: np.ndarray,
    ):
        self._name = name
        self._horizon = horizon
        self._state_mask = state_mask
        self._start_state = start_state
        self._goal_state = goal_state

    def is_done(self, obs: CalvinObservation) -> bool:
        """
        Check if the task is done based on the observation.
        :param obs: The observation from the environment.
        :return: True if the task is done, False otherwise.
        """
        raise NotImplementedError

    def hrl_reward(self, obs: CalvinObservation) -> float:
        """
        Calculate the reward for the task based on the final outcome.
        :param obs: The observation from the environment.
        :return: The reward for the task.
        """
        raise NotImplementedError

    def tapas_reward(self, obs: CalvinObservation) -> float:
        """
        Calculate the reward for the task based on the final outcome.
        :param obs: The observation from the environment.
        :return: The reward for the task.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

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

    def get_hrl_task(self) -> dict:
        """
        Get the HRL task representation.
        :return: A dictionary representing the HRL task.
        """
        return {
            "name": self._name,
            "horizon": self._horizon,
            "state_mask": self._state_mask,
            "start_state": self._start_state,
            "goal_state": self._goal_state,
        }

    def get_tapas_task(self) -> TapasTask:
        return TapasTask()


class PressButton(CalvinTask):
    def __init__(self, horizon: int, start_state: np.ndarray, goal_state: np.ndarray):
        state_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        super().__init__("PressButton", horizon, state_mask, start_state, goal_state)

    def is_done(self, obs: CalvinObservation) -> bool:
        return False

    def tapas_reward(self, obs: CalvinObservation) -> float:
        return 0

    def hrl_reward(self, obs: CalvinObservation) -> float:
        return 0
