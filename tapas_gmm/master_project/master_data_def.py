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

class ActionSpace(Enum):
    SMALL = "small"
    STATIC = "static"
    DYNAMIC = "dynamic"
    UNUSED = "unused"


class StateSpace(Enum):
    STATIC = 0
    DYNAMIC = 1
    UNUSED = 2


class StateType(Enum):
    Pose = 0
    Transform = 1
    Quat = 2
    Scalar = 3


@dataclass
class StateInfo:
    identifier: str
    state_type: StateType
    state_space: StateSpace = StateSpace.UNUSED
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

    @classmethod
    def from_pose_string(cls, name: str) -> list:
        """Return both euler and quat enum members for a pose string"""
        if not name.endswith("_pose"):
            raise ValueError(f"Expected string ending with '_pose', got: {name}")

        # Generate both candidate names
        base_name = name.rsplit("_", 1)[0]  # Remove '_pose' suffix
        euler_name = f"{base_name}_euler"
        quat_name = f"{base_name}_quat"

        # Find matching enums for both
        results = []
        for candidate in [euler_name, quat_name]:
            for member in cls:
                if member.value.identifier == candidate:
                    results.append(member)
                    break
        return results

    @classmethod
    def count_by_state_space(cls, space_type: StateSpace) -> int:
        count = 0
        for member in cls:
            state_info = member.value
            if state_info.state_space.value <= space_type.value:
                count += 1
        return count

    @classmethod
    def list_by_state_space(cls, space_type: StateSpace) -> List["State"]:
        states = []
        for member in cls:
            state_info = member.value
            if state_info.state_space.value <= space_type.value:
                states.append(member)
        return states

    EE_Pose = StateInfo(
        identifier="ee_pose",
        state_type=StateType.Pose,
    )
    Slide_Pose = StateInfo(
        identifier="base__slide_pose",
        state_type=StateType.Pose,
    )
    Drawer_Pose = StateInfo(
        identifier="base__drawer_pose",
        state_type=StateType.Pose,
    )
    Button_Pose = StateInfo(
        identifier="base__button_pose",
        state_type=StateType.Pose,
    )
    Switch_Pose = StateInfo(
        identifier="base__switch_pose",
        state_type=StateType.Pose,
    )
    Lightbulb_Pose = StateInfo(
        identifier="lightbulb_pose",
        state_type=StateType.Pose,
    )
    Led_Pose = StateInfo(
        identifier="led_pose",
        state_type=StateType.Pose,
    )
    Red_Pose = StateInfo(
        identifier="block_red_pose",
        state_type=StateType.Pose,
    )
    Blue_Pose = StateInfo(
        identifier="block_blue_pose",
        state_type=StateType.Pose,
    )
    Pink_Pose = StateInfo(
        identifier="block_pink_pose",
        state_type=StateType.Pose,
    )
    EE_Transform = StateInfo(
        identifier="ee_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.STATIC,
    )
    Slide_Transform = StateInfo(
        identifier="base__slide_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Drawer_Transform = StateInfo(
        identifier="base__drawer_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Button_Transform = StateInfo(
        identifier="base__button_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Switch_Transform = StateInfo(
        identifier="base__switch_euler",
        state_type=StateType.Transform,
    )
    Lightbulb_Transform = StateInfo(
        identifier="lightbulb_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Led_Transform = StateInfo(
        identifier="led_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Red_Transform = StateInfo(
        identifier="block_red_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Blue_Transform = StateInfo(
        identifier="block_blue_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    Pink_Transform = StateInfo(
        identifier="block_pink_euler",
        state_type=StateType.Transform,
        state_space=StateSpace.DYNAMIC,
    )
    EE_Quat = StateInfo(
        identifier="ee_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.STATIC,
    )
    Slide_Quat = StateInfo(
        identifier="base__slide_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Drawer_Quat = StateInfo(
        identifier="base__drawer_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Button_Quat = StateInfo(
        identifier="base__button_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Switch_Quat = StateInfo(
        identifier="base__switch_quat",
        state_type=StateType.Quat,
    )
    Lightbulb_Quat = StateInfo(
        identifier="lightbulb_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Led_Quat = StateInfo(
        identifier="led_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Red_Quat = StateInfo(
        identifier="block_red_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Blue_Quat = StateInfo(
        identifier="block_blue_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    Pink_Quat = StateInfo(
        identifier="block_pink_quat",
        state_type=StateType.Quat,
        state_space=StateSpace.DYNAMIC,
    )
    EE_State = StateInfo(
        identifier="ee_state",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        state_space=StateSpace.STATIC,
    )  # Gripper
    Slide_State = StateInfo(
        identifier="base__slide",
        state_type=StateType.Scalar,
        min=0.0,
        max=0.28,
        state_space=StateSpace.STATIC,
    )
    Drawer_State = StateInfo(
        identifier="base__drawer",
        state_type=StateType.Scalar,
        min=0.0,
        max=0.22,
        state_space=StateSpace.STATIC,
    )
    Button_State = StateInfo(
        identifier="base__button",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        state_space=StateSpace.STATIC,
    )
    Switch_State = StateInfo(
        identifier="base__switch",
        state_type=StateType.Scalar,
        min=0.0,
        max=0.088,
    )
    Lightbulb_State = StateInfo(
        identifier="lightbulb",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
    )
    Led_State = StateInfo(
        identifier="led",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        state_space=StateSpace.STATIC,
    )
    Red_State = StateInfo(
        identifier="block_red",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        #state_space=StateSpace.DYNAMIC,
    )
    Blue_State = StateInfo(
        identifier="block_blue",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        #state_space=StateSpace.DYNAMIC,
    )
    Pink_State = StateInfo(
        identifier="block_pink",
        state_type=StateType.Scalar,
        min=0.0,
        max=1.0,
        #state_space=StateSpace.DYNAMIC,
    )


@dataclass
class ModelInfo:
    precondition: Dict[State, float | np.ndarray]
    action_space: ActionSpace = ActionSpace.STATIC
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

    @classmethod
    def count_by_action_space(cls, action_space: ActionSpace) -> int:
        count = 0
        for member in cls:
            model_info = member.value
            if model_info.action_space == action_space:
                count += 1
        return count

    @classmethod
    def list_by_action_space(cls, action_space: ActionSpace) -> List["Task"]:
        tasks = []
        for member in cls:
            model_info = member.value
            if model_info.action_space == action_space:
                tasks.append(member)
        return tasks

    CloseDrawer = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.max,
        },
    )
    MoveToDoorLeftReversed = ModelInfo(
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
    MoveToDoorRightReversed = ModelInfo(
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
    MoveToDrawerClosedReversed = ModelInfo(
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
    MoveToDrawerOpenReversed = ModelInfo(
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
    OpenDrawer = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Drawer_State: State.Drawer_State.value.min,
        },
    )
    PressButton = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            #State.Button_State: State.Button_State.value.min,
        },
    )
    PressButtonReversed = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.min,
            #State.Button_State: State.Button_State.value.max,
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
    SlideDoorLeft = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.min,
        },
    )
    SlideDoorRight = ModelInfo(
        precondition={
            State.EE_State: State.EE_State.value.max,
            State.Slide_State: State.Slide_State.value.max,
        },
    )


class TaskModel(Enum):
    # A.1 Model Set 1 - Static Elements
    PRESS_BUTTON = "Press Button"
    PRESS_BUTTON_REVERSED = "Press Button (Reversed)"
    OPEN_GRIPPER = "Open Gripper"
    CLOSE_GRIPPER = "Close Gripper"
    OPEN_DRAWER = "Open Drawer"
    CLOSE_DRAWER = "Close Drawer"
    MOVETO_DRAWER_REVERSED = "MoveTo Drawer (Reversed)"
    OPEN_CABINET = "Open Cabinet"
    CLOSE_CABINET = "Close Cabinet"
    MOVETO_CABINET_REVERSED = "MoveTo Cabinet (Reversed)"
    UP_LEVER = "Up Lever"
    DOWN_LEVER = "Down Lever"
    MOVETO_LEVER_REVERSED = "MoveTo Lever (Reversed)"

    # A.2 Model Set 2 - Dynamic Elements
    PICKUP_BLUEBLOCK_INSIDE_CABINET = "PickUp BlueBlock Inside Cabinet"
    PICKUP_BLUEBLOCK_INSIDE_DRAWER = "PickUp BlueBlock Inside Drawer"
    PICKUP_BLUEBLOCK_INFRONT_CABINET = "PickUp BlueBlock Infront Cabinet"
    PICKUP_BLUEBLOCK_INFRONT_LEVER = "PickUp BlueBlock Infront Lever"
    PICKUP_BLUEBLOCK_INSIDE_CABINET_REVERSED = (
        "PickUp BlueBlock Inside Cabinet (Reversed)"
    )
    PICKUP_BLUEBLOCK_INSIDE_DRAWER_REVERSED = (
        "PickUp BlueBlock Inside Drawer (Reversed)"
    )
    PICKUP_BLUEBLOCK_INFRONT_CABINET_REVERSED = (
        "PickUp BlueBlock Infront Cabinet (Reversed)"
    )
    PICKUP_BLUEBLOCK_INFRONT_LEVER_REVERSED = (
        "PickUp BlueBlock Infront Lever (Reversed)"
    )

    PICKUP_REDBLOCK_INSIDE_CABINET = "PickUp RedBlock Inside Cabinet"
    PICKUP_REDBLOCK_INSIDE_DRAWER = "PickUp RedBlock Inside Drawer"
    PICKUP_REDBLOCK_INFRONT_CABINET = "PickUp RedBlock Infront Cabinet"
    PICKUP_REDBLOCK_INFRONT_LEVER = "PickUp RedBlock Infront Lever"
    PICKUP_REDBLOCK_INSIDE_CABINET_REVERSED = (
        "PickUp RedBlock Inside Cabinet (Reversed)"
    )
    PICKUP_REDBLOCK_INSIDE_DRAWER_REVERSED = "PickUp RedBlock Inside Drawer (Reversed)"
    PICKUP_REDBLOCK_INFRONT_CABINET_REVERSED = (
        "PickUp RedBlock Infront Cabinet (Reversed)"
    )
    PICKUP_REDBLOCK_INFRONT_LEVER_REVERSED = "PickUp RedBlock Infront Lever (Reversed)"

    PICKUP_PURPLEBLOCK_INSIDE_CABINET = "PickUp PurpleBlock Inside Cabinet"
    PICKUP_PURPLEBLOCK_INSIDE_DRAWER = "PickUp PurpleBlock Inside Drawer"
    PICKUP_PURPLEBLOCK_INFRONT_CABINET = "PickUp PurpleBlock Infront Cabinet"
    PICKUP_PURPLEBLOCK_INFRONT_LEVER = "PickUp PurpleBlock Infront Lever"
    PICKUP_PURPLEBLOCK_INSIDE_CABINET_REVERSED = (
        "PickUp PurpleBlock Inside Cabinet (Reversed)"
    )
    PICKUP_PURPLEBLOCK_INSIDE_DRAWER_REVERSED = (
        "PickUp PurpleBlock Inside Drawer (Reversed)"
    )
    PICKUP_PURPLEBLOCK_INFRONT_CABINET_REVERSED = (
        "PickUp PurpleBlock Infront Cabinet (Reversed)"
    )
    PICKUP_PURPLEBLOCK_INFRONT_LEVER_REVERSED = (
        "PickUp PurpleBlock Infront Lever (Reversed)"
    )

    MOVETO_INSIDE_CABINET = "MoveTo Inside Cabinet"
    MOVETO_INSIDE_DRAWER = "MoveTo Inside Drawer"
    MOVETO_INFRONT_CABINET = "MoveTo Infront Cabinet"
    MOVETO_INFRONT_LEVER = "MoveTo Infront Lever"
    MOVETO_INSIDE_CABINET_REVERSED = "MoveTo Inside Cabinet (Reversed)"
    MOVETO_INSIDE_DRAWER_REVERSED = "MoveTo Inside Drawer (Reversed)"
    MOVETO_INFRONT_CABINET_REVERSED = "MoveTo Infront Cabinet (Reversed)"
    MOVETO_INFRONT_LEVER_REVERSED = "MoveTo Infront Lever (Reversed)"
