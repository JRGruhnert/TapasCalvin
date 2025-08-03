import numpy as np
import pytest

from tapas_gmm.master_project.converter import (
    Converter,
    QuaternionConverter,
    ScalarConverter,
    TransformConverter,
)
from tapas_gmm.master_project.definitions import (
    State,
    StateSpace,
    StateType,
    Task,
)
from tapas_gmm.master_project.observation import Observation


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
    if state.value.type == StateType.Scalar:
        converter = ScalarConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    elif state.value.type in [StateType.Transform, StateType.Quaternion]:
        converter = TransformConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
    else:
        converter = QuaternionConverter(state=state, normalized=normalized)
        assert converter.value(value) == exp_value
        assert converter.distance(value, goal) == exp_dist
