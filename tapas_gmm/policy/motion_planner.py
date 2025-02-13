from enum import Enum
import numpy as np
import pybullet as p
import torch
from loguru import logger
from typing import List
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from scipy.spatial.transform import Rotation as R



from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.env.calvinbench import CalvinTapasBridgeEnvironment
from tapas_gmm.policy.models.motion_planner import MotionPlanner
from tapas_gmm.utils.observation import SceneObservation
# from tapas_gmm.utils.select_gpu import device

zero_movement = torch.tensor([0.0] * 6)
close_gripper = torch.tensor([-0.9])
open_gripper = torch.tensor([0.9])

class ActionType(Enum):
    MOVE_TO = 0
    CLOSE_GRIPPER = 1
    OPEN_GRIPPER = 2
    NOOP = 3


class Action():
    def __init__(self, action_type, goal=None, with_screw=True) -> None:
        self.action_type = action_type
        self.goal = goal
        self.taskId: str = "block_red"
        self.with_screw = with_screw

class MoveToAction(Action):
    def __init__(self, goal, with_screw=True) -> None:
        super().__init__(ActionType.MOVE_TO, goal, with_screw)

class OpenGripperAction(Action):
    def __init__(self) -> None:
        super().__init__(ActionType.OPEN_GRIPPER)

class CloseGripperAction(Action):
    def __init__(self) -> None:
        super().__init__(ActionType.CLOSE_GRIPPER)

class NoopAction(Action):
    def __init__(self) -> None:
        super().__init__(ActionType.NOOP)

class ActionSequence:
    def __init__(self, solution_sequence: List[Action]) -> None:
        self.solution_sequence: List[Action] = solution_sequence

    def __len__(self):
        return len(self.solution_sequence)
    
    def __iter__(self):
        return iter(self.solution_sequence)
    
    def pop(self, index):
        return self.solution_sequence.pop(index)
    


class MotionPlannerPolicy:
    def __init__(self, env: CalvinTapasBridgeEnvironment, sequence: ActionSequence, **kwargs):
        self.time_step = 1 / 20

        self.motion_planner = MotionPlanner(env)
        self.env = env
        self.sequence = sequence

        self.reset_episode(env)

    def from_disk(self, file_name):
        pass

    @staticmethod
    def _normalize_gripper_state(
        gripper_state: torch.Tensor, finger_width: float = 0.04
    ) -> torch.Tensor:
        # average the two fingers - should be identical anyway
        gripper_state_1d = gripper_state.mean().unsqueeze(0)

        # map from [0, finger_width] to [-1, 1]
        return gripper_state_1d * 2 / finger_width - 1

    @staticmethod
    def _binary_gripper_state(
        gripper_state: torch.Tensor, closed_threshold: float = 0.85
    ) -> torch.Tensor:
        normalized_state = MotionPlannerPolicy._normalize_gripper_state(gripper_state)

        return torch.where(
            normalized_state < closed_threshold, close_gripper, open_gripper
        )

    @staticmethod
    def _get_current_joint_pos(obs: SceneObservation): # type: ignore
        # TODO: check if this is the correct way to get the current pose
        proprio_obs = obs.joint_pos
        print(f"Current Joint Pos: {len(proprio_obs)}")
        return proprio_obs

    def _make_plan(self, obs: SceneObservation, goal: Action): # type: ignore
        """
        Returns a plan for the given goal. The plan is given as a list of
        actions.

        THE ACTION SPACE DEPENDS ON THE GOAL TYPE:
        - MOVE_TO: joint positions that need to be translated to EE Delta Pose,
                   which only works online as the two envs/controllers used for
                   translation need to be synchronized
        - OPEN_GRIPPER/CLOSE_GRIPPER: EE Delta Pose and normalized gripper
                                      state, ready to use


        Parameters
        ----------
        obs : SceneObservation
            The current observation.
        goal : Action
            Action containing ActionType (MOVE_TO, OPEN_GRIPPER, CLOSE_GRIPPER)
            and goal in case of MOVE_TO.

        Returns
        -------
        list[torch.Tensor]
            The plan for the specified goal.

        Raises
        ------
        ValueError
            _description_
        """
        if goal.action_type is ActionType.MOVE_TO:
            plan = self.motion_planner.plan_to_goal(
                goal.goal,
                self._get_current_joint_pos(obs).numpy(),
                time_step=self.time_step,
                with_screw=goal.with_screw,
            )

            return self._convert_plan_to_euler(plan)

        elif goal.action_type is ActionType.OPEN_GRIPPER:
            return self._make_gripper_open_plan(obs)

        elif goal.action_type is ActionType.CLOSE_GRIPPER:
            return self._make_gripper_close_plan(obs)

        elif goal.action_type is ActionType.NOOP:
            return self._make_noop_plan(obs)

        else:
            raise ValueError("Unknown action type.")

    @staticmethod
    def _make_gripper_close_plan(obs: SceneObservation, repeat: int = 10): # type: ignore
        """
        Returns a plan for closing the gripper while keeping the arm still.

        Actions are given as EE Delta Pose and normalized gripper state.

        Parameters
        ----------
        obs : SceneObservation
            Current observation.
        repeat : int, optional
            Number of timesteps for which to repeat the action, by default 10

        Returns
        -------
        list[torch.Tensor]
            Plan for closing the gripper as EE Delta Pose and gripper action.
        """
        return [torch.cat((zero_movement, close_gripper))] * repeat
        # this line gives the identical plan but as joint positions
        # return [torch.cat((obs.joint_pos[:-1], close_gripper))] * repeat

    @staticmethod
    def _make_gripper_open_plan(obs: SceneObservation, repeat: int = 10): # type: ignore
        """
        Returns a plan for opening the gripper while keeping the arm still.

        Actions are given as EE Delta Pose and normalized gripper state.

        Parameters
        ----------
        obs : SceneObservation
            Current observation.
        repeat : int, optional
            Number of timesteps for which to repeat the action, by default 10

        Returns
        -------
        list[torch.Tensor]
            Plan for opening the gripper as EE Delta Pose and gripper action.
        """
        return [torch.cat((zero_movement, open_gripper))] * repeat
        # this line gives the identical plan but as joint positions
        # return [torch.cat((obs.joint_pos[:-1], close_gripper))] * repeat

    @staticmethod
    def _make_noop_plan(obs: SceneObservation, duration: int = 5): # type: ignore
        """
        Returns a plan for doing nothing for a given number of timesteps.

        Actions are given as EE Delta Pose and normalized gripper state.

        Parameters
        ----------
        obs : SceneObservation
            Current observation.
        duration : int, optional
            Number of timesteps for which to do nothing, by default 5

        Returns
        -------
        list[torch.Tensor]
            Plan for doing nothing as EE Delta Pose and gripper action.
        """
        current_gripper = MotionPlannerPolicy._binary_gripper_state(obs.gripper_state)

        return [torch.cat((zero_movement, current_gripper))] * duration

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        if self.goal_list is None:
            self.goal_list = self.sequence
        
        info = {}
        if not self.current_plan:
            if len(self.goal_list) == 0:
                logger.info("End of plan. Waiting for env to settle.")
                info["done"] = True
                return None, True
            else:
                self.current_goal = self.goal_list.pop(0)
                self.current_plan = self._make_plan(obs, self.current_goal)


        action = self.current_plan.pop(0)

        assert self.current_goal is not None

        # Need to translate the goal from joint pos to EE delta pose online
        if self.current_goal.action_type is ActionType.MOVE_TO:
            # action spaces are normalized to [-1, 1] in Maniskill, so map the
            # current gripper state to the corresponding gripper action
            gripper_action = self._binary_gripper_state(obs.gripper_state)
            #action = self.joint_to_ee_pose(action, self.robot_uid, ee_link_index)
            action = torch.cat((torch.tensor(action), gripper_action))


        else:
            assert self.current_goal.action_type in (
                ActionType.OPEN_GRIPPER,
                ActionType.CLOSE_GRIPPER,
                ActionType.NOOP,
            )

        action = action.numpy()
        return action, False

    def reset_episode(self, env: BaseEnvironment):
        self.goal_list = None
        self.current_goal = None
        self.current_plan = []

    def _convert_plan_to_euler(self, plan):
        euler_plan = []

        for joint_pos in plan["position"]:  # Plan contains joint positions, NOT EE poses
            joint_pos = np.array(joint_pos)  # Ensure it's a NumPy array

            # Perform forward kinematics
            pin.forwardKinematics(self.motion_planner.model, self.motion_planner.data, joint_pos)

            # Get the EE frame index (change "ee_link_name" to your actual EE link name)
            ee_id = self.motion_planner.model.getFrameId("ee_link_name")  

            # Extract EE transformation (SE(3) pose)
            ee_pose = self.motion_planner.data.oMf[ee_id]
            position = ee_pose.translation  # Extract position (x, y, z)
            rotation_matrix = ee_pose.rotation  # Extract rotation matrix (3x3)

            # Convert rotation matrix to Euler angles
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=False)  # Convert to roll, pitch, yaw

            # Combine position and Euler angles
            euler_pose = np.concatenate((position, euler_angles))
            euler_plan.append(euler_pose)

        return euler_plan

    