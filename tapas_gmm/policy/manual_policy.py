from loguru import logger
import numpy as np

from tapas_gmm.env.calvin import Calvin
from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.utils.human_feedback import correct_action
from calvin_env.envs.observation import (
    CalvinObservation,
)
from tapas_gmm.utils.keyboard_observer import KeyboardObserver


class ManualCalvinPolicy:
    def __init__(self, config, env: Calvin, keyboard_obs: KeyboardObserver, **kwargs):
        self.keyboard_obs = keyboard_obs

        self.env = env

        self.gripper_open = 0.9

    def from_disk(self, file_name):
        pass  # nothing to load

    def predict(
        self, obs: CalvinObservation  # type: ignore
    ) -> tuple[np.ndarray, bool, bool]:

        # Is default action pose
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_open])

        # Checks for updates and corrects the action
        if self.keyboard_obs.has_joints_cor() or self.keyboard_obs.has_gripper_update():
            action = correct_action(self.keyboard_obs, action)
            self.gripper_open = action[-1]

        done = False
        success = False
        if self.keyboard_obs.is_reset():
            success = self.keyboard_obs.success
            self.keyboard_obs.reset()
            done = True

        return action, done, success

    def reset_episode(self, env: BaseEnvironment | None = None):
        # TODO: add this to all other policies as well and use it to store
        # the LSTM state as well?
        self.gripper_open = 0.9
