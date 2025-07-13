import numpy as np
import pytest

from tapas_gmm.master_project.master_converter import (
    NodeConverter,
    QuaternionConverter,
    ScalarConverter,
    TransformConverter,
)
from tapas_gmm.master_project.master_data_def import State, StateSpace, StateType, Task
from tapas_gmm.master_project.master_observation import HRLPolicyObservation

state_list = State.list_by_state_space(StateSpace.STATIC)
task_list = Task.list_by_action_space(StateSpace.STATIC)
normalized = True


@pytest.mark.parametrize(
    "state, value, goal, exp_value, exp_dist",
    [
        (State.EE_State, 1.0, -1.0, 1.0, 1.0),
        (State.EE_State, -1.0, -1.0, 0.0, 0.0),
        (State.EE_State, 1.0, 1.0, 1.0, 0.0),
        (State.EE_State, -1.0, 1.0, 0.0, 1.0),
        (State.Slide_State, -1.0, 0.1, 0.0, 0.0),
    ],
)
def test_converter(
    state: State,
    value: float | np.ndarray,
    goal: float | np.ndarray,
    exp_value: float | np.ndarray,
    exp_dist: float | np.ndarray,
) -> None:
    if state.value.state_type == StateType.Scalar:
        converter = ScalarConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    elif state.value.state_type in [StateType.Transform, StateType.Quat]:
        converter = TransformConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    else:
        converter = QuaternionConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
