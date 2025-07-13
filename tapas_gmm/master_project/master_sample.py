import itertools
import random
from typing import Dict, List

from loguru import logger
import numpy as np

from tapas_gmm.master_project.master_data_def import (
    ActionSpace,
    State,
    StateSpace,
    StateType,
)


def sample_from_values(values):
    return random.choice(values)


def parse_scene_obs(scene_obs):
    # an object pose is composed of position (3) and orientation (4 for quaternion)  / (3 for euler)
    n_obj = 3
    n_doors = 2
    n_buttons = 1
    n_switches = 1
    n_lights = 2

    split_ids = np.cumsum([n_doors, n_buttons, n_switches, n_lights])
    door_info, button_info, switch_info, light_info, obj_info = np.split(
        scene_obs, split_ids
    )

    assert len(door_info) == n_doors
    assert len(button_info) == n_buttons
    assert len(switch_info) == n_switches
    assert len(light_info) == n_lights
    assert len(obj_info) // n_obj in [6, 7]  # depending on euler angles or quaternions

    obj_info = np.split(obj_info, n_obj)

    return door_info, button_info, switch_info, light_info, obj_info


def update_scene_obs(
    scene_dict: Dict[State, np.ndarray | float], scene_obs: np.ndarray
) -> np.ndarray:
    """Return state information of the doors, drawers and shelves."""
    door_states, button_states, switch_states, light_states, object_poses = (
        parse_scene_obs(scene_obs)
    )
    door_states = [
        scene_dict.get(State.Slide_State, door_states[0]),  # Update only if key exists
        scene_dict.get(State.Drawer_State, door_states[1]),  # Same for index 1
    ]
    button_states = [scene_dict.get(State.Button_State, button_states[0])]
    switch_states = [scene_dict.get(State.Switch_State, switch_states[0])]

    light_states = [
        scene_dict.get(State.Lightbulb_State, light_states[0]),
        scene_dict.get(State.Led_State, light_states[1]),
    ]

    object_poses = list(
        itertools.chain(
            *[
                scene_dict.get(State.Red_Pose, object_poses[0]),
                scene_dict.get(State.Blue_Pose, object_poses[1]),
                scene_dict.get(State.Pink_Pose, object_poses[2]),
            ]
        )
    )

    return np.concatenate(
        [door_states, button_states, switch_states, light_states, object_poses]
    )


def sample_pre_condition(scene_obs: np.ndarray, state_space: StateSpace) -> np.ndarray:
    scene_dict: Dict[State, np.ndarray | float] = {}

    for state in list(State):
        if state in State.list_by_state_space(state_space):
            if state.value.state_type == StateType.Scalar:
                # TODO: Change that by adding a model that opens the gripper or something
                if (
                    state != State.EE_State
                    and state != State.Switch_State
                    and state != State.Lightbulb_State
                    and state != State.Red_State
                    and state != State.Blue_State
                    and state != State.Pink_State
                ):
                    scene_dict[state] = sample_from_values(
                        [state.value.min, state.value.max]
                    )
            if (
                state.value.state_type == StateType.Transform
                or state.value.state_type == StateType.Quat
            ):
                pass
                # raise NotImplementedError("Not Supported.")
            if state.value.state_type == StateType.Pose:
                pass
                # raise NotImplementedError("Not Implemented so far.")

    # Hack to make light states depending on button and switch states
    # if State.Switch_State in scene_dict:
    #    if scene_dict[State.Switch_State] > 0.0:
    #        scene_dict[State.Lightbulb_State] = 1.0
    #    else:
    #        scene_dict[State.Lightbulb_State] = 0.0
    if State.Button_State in scene_dict:
        if scene_dict[State.Button_State] > 0.0:
            scene_dict[State.Led_State] = 1.0
        else:
            scene_dict[State.Led_State] = 0.0

    return update_scene_obs(scene_dict, scene_obs)


def sample_post_condition(scene_obs: np.ndarray, state_space: StateSpace) -> np.ndarray:
    """Samples an environment state that is different to the current one"""

    candidate = sample_pre_condition(scene_obs, state_space)

    while np.array_equal(candidate, scene_obs):
        candidate = sample_pre_condition(scene_obs, state_space)

    return candidate
