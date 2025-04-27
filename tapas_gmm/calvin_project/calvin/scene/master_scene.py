import itertools
import logging
import os
from pathlib import Path

import numpy as np

# A logger for this file
from omegaconf import OmegaConf
from tensordict import TensorDict
import torch

from tapas_gmm.calvin_project.calvin.camera.camera import Camera
from tapas_gmm.calvin_project.calvin.scene.objects.button import Button
from tapas_gmm.calvin_project.calvin.scene.objects.door import Door
from tapas_gmm.calvin_project.calvin.scene.objects.fixed_object import FixedObject
from tapas_gmm.calvin_project.calvin.scene.objects.light import Light
from tapas_gmm.calvin_project.calvin.scene.objects.movable_object import MovableObject
from tapas_gmm.calvin_project.calvin.scene.objects.switch import Switch

log = logging.getLogger(__name__)


REPO_BASE = Path(__file__).parents[2]


class Scene:
    def __init__(self, objects, data_path, euler_obs, p, cid, global_scaling, surfaces, np_random, **kwargs):
        self.p = p
        self.cid = cid
        self.global_scaling = global_scaling
        self.euler_obs = euler_obs
        self.surfaces = surfaces
        self.np_random = np_random
        if os.path.isabs(data_path):
            self.data_path = Path(data_path)
        else:
            self.data_path = REPO_BASE / data_path
        self.p.setAdditionalSearchPath(self.data_path.as_posix())
        self.object_cfg = OmegaConf.to_container(objects)
        self.fixed_objects, self.movable_objects = [], []
        self.doors, self.buttons, self.switches, self.lights = [], [], [], []

    def load(self):
        for name, obj_cfg in self.object_cfg.get("movable_objects", {}).items():
            self.movable_objects.append(
                MovableObject(
                    name,
                    obj_cfg,
                    self.p,
                    self.cid,
                    self.data_path,
                    self.global_scaling,
                    self.euler_obs,
                    self.surfaces,
                    self.np_random,
                )
            )

        for name, obj_cfg in self.object_cfg.get("fixed_objects", {}).items():
            fixed_obj = FixedObject(name, obj_cfg, self.p, self.cid, self.data_path, self.global_scaling)
            self.fixed_objects.append(fixed_obj)

            if "joints" in obj_cfg:
                for joint_name, cfg in obj_cfg["joints"].items():
                    door = Door(joint_name, cfg, fixed_obj.uid, self.p, self.cid)
                    self.doors.append(door)

            if "buttons" in obj_cfg:
                for button_name, cfg in obj_cfg["buttons"].items():
                    button = Button(button_name, cfg, fixed_obj.uid, self.p, self.cid)
                    self.buttons.append(button)

            if "switches" in obj_cfg:
                for switch_name, cfg in obj_cfg["switches"].items():
                    switch = Switch(switch_name, cfg, fixed_obj.uid, self.p, self.cid)
                    self.switches.append(switch)

            if "lights" in obj_cfg:
                for light_name, cfg in obj_cfg["lights"].items():
                    light = Light(light_name, cfg, fixed_obj.uid, self.p, self.cid)
                    self.lights.append(light)

        for light in self.lights:
            for button_switch in itertools.chain(self.buttons, self.switches):
                if button_switch.effect == light.name:
                    button_switch.add_effect(light)

        self.p.loadURDF(os.path.join(self.data_path, "plane/plane.urdf"), physicsClientId=self.cid)

    def reset(self, scene_obs=None):
        """Reset objects and doors to initial position."""
        if scene_obs is None:
            for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights):
                obj.reset()
            self.reset_movable_objects()
        else:
            door_info, button_info, switch_info, light_info, obj_info = self.parse_scene_obs(scene_obs)

            for door, state in zip(self.doors, door_info):
                door.reset(state)
            for light, state in zip(self.lights, light_info):
                light.reset(state)
            for button, state in zip(self.buttons, button_info):
                button.reset(state)
            for switch, state in zip(self.switches, switch_info):
                switch.reset(state)
            for obj, state in zip(self.movable_objects, obj_info):
                obj.reset(state)

    def parse_scene_obs(self, scene_obs):
        # an object pose is composed of position (3) and orientation (4 for quaternion)  / (3 for euler)
        n_obj = len(self.movable_objects)
        n_doors = len(self.doors)
        n_buttons = len(self.buttons)
        n_switches = len(self.switches)
        n_lights = len(self.lights)

        split_ids = np.cumsum([n_doors, n_buttons, n_switches, n_lights])
        door_info, button_info, switch_info, light_info, obj_info = np.split(scene_obs, split_ids)
        assert len(door_info) == n_doors
        assert len(button_info) == n_buttons
        assert len(switch_info) == n_switches
        assert len(light_info) == n_lights
        assert len(obj_info) // n_obj in [6, 7]  # depending on euler angles or quaternions

        obj_info = np.split(obj_info, n_obj)

        return door_info, button_info, switch_info, light_info, obj_info

    def reset_movable_objects(self):
        """reset movable objects such that there are no pairwise contacts"""
        num_sampling_iterations = 1000
        for i in range(num_sampling_iterations):
            for obj in self.movable_objects:
                obj.reset()
            self.p.stepSimulation()
            contact = False
            for obj_a, obj_b in itertools.combinations(self.movable_objects, 2):
                if np.any(len(self.p.getContactPoints(bodyA=obj_a.uid, bodyB=obj_b.uid, physicsClientId=self.cid))):
                    contact = True
                    break
            if not contact:
                return
        log.error(f"Could not place objects in {num_sampling_iterations} iterations without contacts")
        return

    def step(self):
        for button_switch in itertools.chain(self.buttons, self.switches):
            button_switch.step()

    def get_dictionary_object_poses(self) -> dict:
        """Return pose information of the doors, drawers and shelves."""
        object_poses = {}
        for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights, self.movable_objects):
            pose = obj.get_pose(euler_obs=self.euler_obs)
            if isinstance(pose, np.ndarray):
                object_poses[obj.name] = pose
            else:
                log.error(f"Object {obj.name} has invalid pose {pose}")
                object_poses[obj.name] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return object_poses

    def get_low_dim_object_poses(self):
        """Return pose information of the doors, drawers and shelves."""
        state = []
        for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights, self.movable_objects):
            temp_state = obj.get_pose()
            if state is not None:
                if isinstance(temp_state, np.ndarray):
                    state.extend(np.array(temp_state.tolist()))
                else:
                    log.error(f"Object {obj.name} has invalid state {temp_state}")
                    state.extend(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # return np.concatenate(list_of_states)
        # door_states = [door.get_state() for door in self.doors]  # also joints
        # button_states = [button.get_state() for button in self.buttons]
        # switch_states = [switch.get_state() for switch in self.switches]
        # light_states = [light.get_state() for light in self.lights]
        # object_poses = list(itertools.chain(*[obj.get_state() for obj in self.movable_objects]))
        return np.array(state).flatten()

    def get_dictionary_object_states(self) -> dict:
        """Return state information of the doors, drawers and shelves."""
        object_states = {}
        for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights, self.movable_objects):
            state = obj.get_state()
            if isinstance(state, float):
                object_states[obj.name] = state
            else:
                log.error(f"Object {obj.name} has invalid state {state}")
                object_states[obj.name] = 0.0
        return object_states

    def get_low_dim_object_states(self) -> np.ndarray:
        """Return state information of the doors, drawers and shelves."""
        state = []
        # (nominal) Door State: float, joint state meaning the extent of the door
        # Button State: float, 0:off, 1:on
        # Switch State: float, 0:off, 1:on # Object pose shows position and orientation
        # Light State: float, 0:off, 1:on
        # (Nominal) Movebale Object State: float, relative velocity to gripper
        for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights, self.movable_objects):
            temp_state = obj.get_state()
            if state is not None:
                if isinstance(temp_state, float):
                    state.append(temp_state)
                else:
                    log.error(f"Object {obj.name} has invalid state {temp_state}")
                    state.append(0.0)

        return np.array(state).flatten()

    def get_object_states(self, poses_ground_truth=False):
        """Return state information of the objects in the scene."""
        object_poses = []
        for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights):
            object_poses.append(obj.get_state())
        if poses_ground_truth:
            for obj in self.movable_objects:
                object_poses.append(obj.get_state())
        return {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses)}

    def _get_misc(self):
        def _get_cam_data(cam: Camera, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    "%s_extrinsics" % name: cam.get_matrix(),
                    "%s_intrinsics" % name: cam.get_intrinsic_matrix(),
                    "%s_near" % name: cam.get_near_clipping_plane(),
                    "%s_far" % name: cam.get_far_clipping_plane(),
                }
            return d

        misc = _get_cam_data(self._cam_over_shoulder_left, "left_shoulder_camera")
        misc.update(_get_cam_data(self._cam_over_shoulder_right, "right_shoulder_camera"))
        misc.update(_get_cam_data(self._cam_overhead, "overhead_camera"))
        misc.update(_get_cam_data(self._cam_front, "front_camera"))
        misc.update(_get_cam_data(self._cam_wrist, "wrist_camera"))
        misc.update({"variation_index": self._variation_index})
        if self._joint_position_action is not None:
            # Store the actual requested joint positions during demo collection
            misc.update({"joint_position_action": self._joint_position_action})
        joint_poses = [j.get_pose() for j in self.robot.arm.joints]
        misc.update({"joint_poses": joint_poses})
        return misc

    def get_info(self):
        """
        get dictionary of information about the objects in the scene
        self.objects:
            obj1:
                joints:
                    joint1:
                        joint_index: int
                        initial_state: float  # revolute
                        current_state: float
                    ...
                current_pos: [x, y, z]
                current_orn: [x, y, z, w]  # quaternion
                contacts: output of pybullet getContactPoints(...)
                links:  # key exists only if object has num_joints > 0
                    link1: link_id  #  name: id
            ...
        """
        info = {}
        info["fixed_objects"] = {}
        info["movable_objects"] = {}
        info["doors"] = {}
        info["buttons"] = {}
        info["switches"] = {}
        info["lights"] = {}

        for obj in self.fixed_objects:
            info["fixed_objects"][obj.name] = obj.get_info()
        for obj in self.movable_objects:
            info["movable_objects"][obj.name] = obj.get_info()
        for obj in self.doors:
            info["doors"][obj.name] = obj.get_info()
        for obj in self.buttons:
            info["buttons"][obj.name] = obj.get_info()
        for obj in self.switches:
            info["switches"][obj.name] = obj.get_info()
        for obj in self.lights:
            info["lights"][obj.name] = obj.get_info()
        return info

    def get_scene_obs_labels(self):
        raise NotImplementedError

    def get_objects(self):
        return itertools.chain(self.fixed_objects, self.movable_objects)

    def serialize(self):
        data = {
            "fixed_objects": [obj.serialize() for obj in self.fixed_objects],
            "movable_objects": [obj.serialize() for obj in self.movable_objects],
            "lights": [obj.serialize() for obj in self.lights],
        }
        return data

    def reset_from_storage(self, data):
        for fixed_obj in data["fixed_objects"]:
            for i, (value, velocity, *_) in enumerate(fixed_obj["joints"]):
                self.p.resetJointState(
                    bodyUniqueId=fixed_obj["uid"],
                    jointIndex=i,
                    targetValue=value,
                    targetVelocity=velocity,
                    physicsClientId=self.cid,
                )
        for movable_obj in data["movable_objects"]:
            self.p.resetBasePositionAndOrientation(
                bodyUniqueId=movable_obj["uid"],
                posObj=movable_obj["pose"][0],
                ornObj=movable_obj["pose"][1],
                physicsClientId=self.cid,
            )
        for light, state in zip(self.lights, data["lights"]):
            light.reset(state["logical_state"])
