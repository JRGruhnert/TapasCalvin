import numpy as np

from tapas_gmm.utils.observation import SceneObservation


class HRLPolicyObservation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(
        self,
        object_poses: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        tapas_observation: SceneObservation,  # type: ignore
    ):
        self._object_poses = object_poses
        self._object_states = object_states
        self._tapas_observation = tapas_observation

    @property
    def _tapas_observation(
        self,
    ) -> SceneObservation:  # type: ignore
        return self._tapas_observation

    @property
    def state_nodes(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "state_nodes is not implemented for HRLPolicyObservation"
        )

    @property
    def position_nodes(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "position_nodes is not implemented for HRLPolicyObservation"
        )

    @property
    def task_nodes(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "task_nodes is not implemented for HRLPolicyObservation"
        )
