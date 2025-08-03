from enum import Enum
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from git import List
import numpy as np


_origin_ee_tp_pose: np.ndarray = np.array(
    [
        0.02586799,
        -0.23131344,
        0.57128022,
        0.73157951,
        0.68112164,
        0.02806045,
        0.00879429,
    ]
)

_origin_obj_tp_pose = np.array(
    [-0.00699564, 0.40082628, -0.03604347, 0.0, 0.0, 0.0, 1.0],
)


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


class TaskSpace(Enum):
    SMALL = 0
    ALL = 1


class StateSpace(Enum):
    SMALL = 0
    ALL = 1
    UNUSED = 2


class StateType(Enum):
    Transform = 0
    Quaternion = 1
    Scalar = 2


class StateSuccess(Enum):
    AREA = 0  # Has to be in area
    PRECISE = 1  # Has to be precise (with threshold)
    IGNORE = 2  # Is not used in Success calculation
    # (mostly quaternions cause i am not able to make models that precise)


@dataclass
class StateInfo:
    identifier: str
    type: StateType
    success: StateSuccess = StateSuccess.IGNORE
    space: StateSpace = StateSpace.UNUSED
    min: float | np.ndarray = np.array([-1.0, -1.0, -1.0])
    max: float | np.ndarray = np.array([1.0, 1.0, 1.0])

    def __eq__(self, other):
        # Always return False unless it's the same object
        return self is other

    def __hash__(self):
        # Use object's identity as hash
        return id(self)


class State(Enum):
    @classmethod
    def from_string(cls, name: str):
        for member in cls:
            if member.value.identifier in name:
                return member
        raise NotImplementedError(f"Enum for {name} does not exist.")

    @classmethod
    def get_tp_by_index(
        cls, index: int, split_pose: bool = False
    ) -> Union["State", tuple["State", "State"]]:
        """Get enum member by index"""
        if index < 0 or index >= len(cls):
            raise IndexError(f"Index {index} out of range for ObservationState.")
        if split_pose:
            # Return both transform and quat for pose states
            return (
                list(cls)[index + 10],
                list(cls)[index + 20],
            )  # transform and quat

        return list(cls)[index]

    EE_Transform = StateInfo(
        identifier="ee_euler",
        type=StateType.Transform,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Slide_Transform = StateInfo(
        identifier="base__slide_euler",
        type=StateType.Transform,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Drawer_Transform = StateInfo(
        identifier="base__drawer_euler",
        type=StateType.Transform,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Button_Transform = StateInfo(
        identifier="base__button_euler",
        type=StateType.Transform,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Switch_Transform = StateInfo(
        identifier="base__switch_euler",
        type=StateType.Transform,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Lightbulb_Transform = StateInfo(
        identifier="lightbulb_euler",
        type=StateType.Transform,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Led_Transform = StateInfo(
        identifier="led_euler",
        type=StateType.Transform,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Red_Transform = StateInfo(
        identifier="block_red_euler",
        type=StateType.Transform,
        space=StateSpace.ALL,
        success=StateSuccess.AREA,
    )
    Blue_Transform = StateInfo(
        identifier="block_blue_euler",
        type=StateType.Transform,
        space=StateSpace.ALL,
        success=StateSuccess.AREA,
    )
    Pink_Transform = StateInfo(
        identifier="block_pink_euler",
        type=StateType.Transform,
        space=StateSpace.ALL,
        success=StateSuccess.AREA,
    )
    EE_Quat = StateInfo(
        identifier="ee_quat",
        type=StateType.Quaternion,
        space=StateSpace.SMALL,
    )
    Slide_Quat = StateInfo(
        identifier="base__slide_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    Drawer_Quat = StateInfo(
        identifier="base__drawer_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    Button_Quat = StateInfo(
        identifier="base__button_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    Switch_Quat = StateInfo(
        identifier="base__switch_quat",
        type=StateType.Quaternion,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Lightbulb_Quat = StateInfo(
        identifier="lightbulb_quat",
        type=StateType.Quaternion,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Led_Quat = StateInfo(
        identifier="led_quat",
        type=StateType.Quaternion,
        space=StateSpace.SMALL,
    )
    Red_Quat = StateInfo(
        identifier="block_red_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    Blue_Quat = StateInfo(
        identifier="block_blue_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    Pink_Quat = StateInfo(
        identifier="block_pink_quat",
        type=StateType.Quaternion,
        space=StateSpace.ALL,
    )
    EE_State = StateInfo(
        identifier="ee_state",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )  # Gripper
    Slide_State = StateInfo(
        identifier="base__slide",
        type=StateType.Scalar,
        min=0.0,
        max=0.28,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Drawer_State = StateInfo(
        identifier="base__drawer",
        type=StateType.Scalar,
        min=0.0,
        max=0.22,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Button_State = StateInfo(
        identifier="base__button",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Switch_State = StateInfo(
        identifier="base__switch",
        type=StateType.Scalar,
        min=0.0,
        max=0.088,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Lightbulb_State = StateInfo(
        identifier="lightbulb",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        # UNUSED CAUSE NOT ABLE TO RECORD
        # Not relevant cause i already have enough tasks
    )
    Led_State = StateInfo(
        identifier="led",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        space=StateSpace.SMALL,
        success=StateSuccess.PRECISE,
    )
    Red_State = StateInfo(
        identifier="block_red",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        # UNUSED CAUSE MEANINGLESS STATE
    )
    Blue_State = StateInfo(
        identifier="block_blue",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        # UNUSED CAUSE MEANINGLESS STATE
    )
    Pink_State = StateInfo(
        identifier="block_pink",
        type=StateType.Scalar,
        min=0.0,
        max=1.0,
        # UNUSED CAUSE MEANINGLESS STATE
    )


@dataclass
class TaskInfo:
    precondition: Dict[State, float | np.ndarray]
    space: TaskSpace = TaskSpace.SMALL
    reversed: bool = False
    ee_tp_start: np.ndarray = _origin_ee_tp_pose
    obj_start: np.ndarray = _origin_obj_tp_pose
    ee_hrl_start: np.ndarray = _origin_ee_tp_pose

    def __eq__(self, other):
        # Always return False unless it's the same object
        return self is other

    def __hash__(self):
        # Use object's identity as hash
        return id(self)


class Task(Enum):

    @classmethod
    def get_enum_by_index(enum_cls, index: int):
        return list(enum_cls)[index]

    DrawerDoClose = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
    )
    SliderLeftMoveTo = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Slide_State: State.Slide_State.value.max,
        },
        reversed=True,
        ee_hrl_start=np.array(
            [
                -0.24200995,
                0.03103676,
                0.57855496,
                0.72666244,
                0.6863869,
                0.02884163,
                -0.00169593,
            ]
        ),
    )
    SliderRightMoveTo = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Slide_State: State.Slide_State.value.min,
        },
        reversed=True,
        ee_hrl_start=np.array(
            [
                0.0446754,
                0.03459689,
                0.56971713,
                0.73415732,
                0.67768943,
                0.04181756,
                -0.00116969,
            ]
        ),
    )
    DrawerMoveToClosed = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Drawer_State: State.Drawer_State.value.min,
        },
        reversed=True,
        ee_hrl_start=np.array(
            [
                0.1799327,
                -0.20690583,
                0.46871324,
                0.73168841,
                0.68082266,
                0.0325737,
                0.00717814,
            ]
        ),
    )
    DrawerMoveToOpen = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        reversed=True,
        ee_hrl_start=np.array(
            [
                0.18521255,
                -0.43961252,
                0.43864139,
                0.73337219,
                0.67915659,
                0.02904117,
                0.00825612,
            ]
        ),
    )
    DrawerDoOpen = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.min,
        },
    )
    ButtonPress = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            # State.Button_State: State.Button_State.value.min,
        },
    )
    ButtonPressReversed = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            # State.Button_State: State.Button_State.value.max,
        },
        reversed=True,
        ee_hrl_start=np.array(
            [
                -0.11145425,
                -0.12517733,
                0.47878784,
                0.73138852,
                0.68118917,
                0.03111799,
                0.00915563,
            ]
        ),
    )
    SliderLeftDoOpen = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
    )
    SliderRightDoOpen = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.max,
        },
    )
    # Cut
    BlockTableRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    BlockTablePinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    BlockTableBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockTableRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockTablePinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockTableBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockTableAllMoveTo = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    # Cut
    BlockDrawerRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    BlockDrawerPinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    BlockDrawerBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockDrawerRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockDrawerPinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockDrawerBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockDrawerAllMoveTo = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
        space=TaskSpace.ALL,
    )
    # Cut
    BlockCabinetRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    BlockCabinetPinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    BlockCabinetBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockCabinetRedGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockCabinetPinkGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockCabinetBlueGrab = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )
    ReversedBlockCabinetAllMoveTo = TaskInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
        space=TaskSpace.ALL,
    )


def convert_to_states(state_space: StateSpace) -> List[State]:
    states = []
    for member in State:
        if member.value.space.value <= state_space.value:
            states.append(member)
    return states


def convert_to_tasks(task_space: TaskSpace) -> List[Task]:
    states = []
    for member in Task:
        if member.value.space.value <= task_space.value:
            states.append(member)
    return states
