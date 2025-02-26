from typing import Tuple
import mplib
import numpy as np
from loguru import logger
import torch
from mplib.pymp import Pose

from calvin_env.robot.robot import Robot
from tapas_gmm.env.calvinbench import CalvinEnvironment

class MotionPlanner:
    def __init__(self, env: CalvinEnvironment):
        self.env = env
        self.robot_uid = self.env.calvin_env.robot.robot_uid
        self.p = self.env.calvin_env.p

        # Get link names
        self.link_names = [self.p.getBodyInfo(self.robot_uid)[0].decode()]  # Root/base link
        for i in range(self.p.getNumJoints(self.robot_uid)):
            self.link_names.append(self.p.getJointInfo(self.robot_uid, i)[12].decode())

        # Get active joint names (ignore fixed joints)
        self.joint_names = []
        self.active_joint_indices = []
        for i in range(self.p.getNumJoints(self.robot_uid)):
            joint_info = self.p.getJointInfo(self.robot_uid, i)
            joint_type = joint_info[2]  # Joint type
            joint_name = joint_info[1].decode()

            if joint_type in [self.p.JOINT_REVOLUTE, self.p.JOINT_PRISMATIC]:  # Movable joints
                self.joint_names.append(joint_name)
                if joint_type == self.p.JOINT_REVOLUTE:
                    self.active_joint_indices.append(i)

        #print(f"joint_names", self.joint_names)
        #print(f"active_joint_indices", self.active_joint_indices)

        self.move_group_idx = 13
        self.move_group_name = self.link_names[self.move_group_idx]
        self.move_group_name = "tcp"
        #print(f"move_group_name", self.move_group_name)
        self.move_group_idx = self.link_names.index(self.move_group_name)

        # Define joint velocity and acceleration limits
        joint_vel_limits = np.ones(len(self.active_joint_indices)) * 1.0
        joint_acc_limits = np.ones(len(self.active_joint_indices)) * 1.0
        
        args = ""
        args = "_mplib"
        urdf_path = f"calvin_env_motionplanner/data/franka_panda/panda_longer_finger{args}.urdf"
        srdf_path = f"calvin_env_motionplanner/data/franka_panda/panda_longer_finger{args}.srdf"
        # Initialize MPLib Planner
        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=self.link_names,
            user_joint_names=self.joint_names,
            move_group=self.move_group_name,
            joint_vel_limits=joint_vel_limits,
            joint_acc_limits=joint_acc_limits,
        )
        print(self.planner.pinocchio_model)
        self.planner.set_base_pose(Pose(tuple(self.env.calvin_env.robot.base_position), tuple(self.env.calvin_env.robot.base_orientation)))


    @staticmethod
    def _check_success(result):
        return result["status"] == "Success"

    def plan_to_goal(self, goal_pose, current_qpos, time_step=1 / 250, with_screw=True):
        success = False
        plan = {}

        if with_screw:
            plan = self.planner.plan_screw(goal_pose, current_qpos, time_step=time_step, verbose=True)
            success = self._check_success(plan)

            if not success:
                logger.warning("Planning with screw failed. Trying again without.")

        if not with_screw or not success:
            # Here 9 joints problem
            print(f"current_pose", current_qpos)
            #TODO: plan_qpos and make kinematics here
            #current_qpos = current_qpos[:7]
            self.robot: Robot = self.env.calvin_env.robot
            plan = self.planner.plan_pose(
                goal_pose, current_qpos,planning_time=10,  verbose=True, time_step=time_step,mask=[True,True,True,True,True,True,True,False,False]
            )
            print(f"succes state", plan["status"])
            success = self._check_success(plan)

        if not success:
            logger.error("Planning failed.")

            plan["position"] = []
            plan["velocity"] = []

        # Unpack ndarrays into lists to allow pop
        plan["position"] = [p for p in plan["position"]]
        plan["velocity"] = [v for v in plan["velocity"]]
        plan["acceleration"] = [a for a in plan["acceleration"]]
        plan["time"] = [t for t in plan["time"]]
        plan["ee_pose"] = [p for p in plan["position"]]

        return plan
    
    def joint_to_ee(self, ):
        ee_state = self.p.getLinkState(self.robot_uid, self.move_group_idx, computeLinkVelocity=True)
        ee_pose = ee_state[0] + ee_state[1]
        return ee_pose
