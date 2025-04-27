from loguru import logger
import numpy as np
from overrides import override
from tapas_gmm.calvin_project.calvin.envs.observation import CalvinObservation
from tapas_gmm.calvin_project.calvin.envs.master_tasks.tapas_task import TapasTask

import numpy as np


class CalvinTask:
    def __init__(
        self,
        obs: CalvinObservation = None,
    ):
        """
        Base class for a task.
        """
        self._is_done = False
        self._dense_reward = 0.0
        self._sparse_reward = 0.0
        if obs is None:
            self._object_poses = {}
            self._object_states = {}
            self._tcp_pose = np.zeros(7)
            self._tcp_state = 0.0
        else:
            self._object_poses = obs.object_poses
            self._object_states = obs.object_states
            self._tcp_pose = obs.tcp_pose
            self._tcp_state = obs.tcp_state

    @property
    def tcp_pose(self) -> np.ndarray:
        return self._tcp_pose

    @property
    def tcp_state(self) -> float:
        return self._tcp_state

    @property
    def object_poses(self) -> dict:
        return self._object_poses

    @property
    def object_states(self) -> dict:
        return self._object_states

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def dense_reward(self) -> float:
        return self._dense_reward

    @property
    def sparse_reward(self) -> float:
        return self._sparse_reward

    def step(self, obs: CalvinObservation) -> tuple[float, bool]:
        self._dense_reward, self._is_done = self._step(obs)
        if self._is_done:
            self._sparse_reward = self._calc_sparse_reward(obs)
        return self.dense_reward, self.is_done

    def result(self) -> float:
        return self.sparse_reward

    def reset(self, obs: CalvinObservation) -> tuple[float, bool]:
        self._is_done = False
        self._dense_reward = 0.0
        self._sparse_reward = 0.0
        self._object_poses = obs.object_poses
        self._object_states = obs.object_states
        self._ee_pose = obs.ee_pose
        self._ee_state = obs.ee_state
        return self.dense_reward, self.is_done

    def _step(self, obs: CalvinObservation) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs: CalvinObservation) -> float:
        raise NotImplementedError

    def extract_partition_a_nodes(self) -> list[TapasTask]:
        """
        Convert the task to a Tapas task.
        :return: A dictionary representing the Tapas task.
        """
        raise NotImplementedError

    def create_partition_b(self) -> list[TapasTask]:
        """
        Convert the task to a Tapas task.
        :return: A dictionary representing the Tapas task.
        """
        raise NotImplementedError

    def to_multiply_task(self) -> list[TapasTask]:
        """
        Convert the task to a Tapas task.
        :return: A dictionary representing the Tapas task.
        """
        raise NotImplementedError


class PressButton(CalvinTask):

    @override
    def _step(self, obs: CalvinObservation) -> tuple[float, bool]:
        # 'base__slide', 'base__drawer', 'base__button', 'base__switch',
        # 'lightbulb', 'led', 'block_red', 'block_blue', 'block_pink'
        start_pose = self.object_poses.get("base__button")
        start_state = self.object_states.get("base__button")
        start_gripper_pose = self.tcp_pose
        current_pose = obs.object_poses.get("base__button")
        current_state = obs.object_states.get("base__button")
        current_gripper_pose = obs.ee_pose

        rel_state = start_state - current_state
        # Button can only be 0 or 1
        if abs(rel_state) > 0:
            # Button was pressed
            return 1.0, True
        else:
            # Button was not pressed or too much pressed
            start_distance = np.linalg.norm(start_pose[3:] - start_gripper_pose[3:])
            current_distance = np.linalg.norm(current_pose[3:] - current_gripper_pose[3:])

            delta = start_distance - current_distance
            if delta <= 0:
                # Situation is equal or worse than start
                return 0.0, False
            else:
                # Situation is better than start
                norm = (start_distance - current_distance) / start_distance
                return min(0.8, norm), False

    @override
    def _calc_sparse_reward(self, obs: CalvinObservation) -> float:
        return 1.0 if self._is_done else 0.0


class OpenDoor(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class CloseDoor(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PickCube1(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PlaceCube1(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PickCube2(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PlaceCube2(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PickCube3(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class PlaceCube3(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveUp(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveDown(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveLeft(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveRight(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveForward(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class MoveBackward(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class OpenGripper(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError


class CloseGripper(CalvinTask):
    def _step(self, obs) -> tuple[float, bool]:
        raise NotImplementedError

    def _calc_sparse_reward(self, obs) -> float:
        raise NotImplementedError
