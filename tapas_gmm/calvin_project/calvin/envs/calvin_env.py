import datetime
import logging
from math import pi
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
import time
from loguru import logger
import cv2
import gym
import gym.utils
import gym.utils.seeding
import hydra
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import torch

from tapas_gmm.calvin_project.calvin.camera.camera import Camera
from tapas_gmm.calvin_project.calvin.envs.observation import CalvinObservation
from tapas_gmm.calvin_project.calvin.envs.master_tasks.calvin_task import (
    CalvinTask,
    CloseDoor,
    OpenDoor,
    PickCube1,
    PickCube2,
    PickCube3,
    PlaceCube1,
    PlaceCube2,
    PlaceCube3,
    PressButton,
)
from tapas_gmm.calvin_project.calvin.robot.robot import Robot
from tapas_gmm.calvin_project.calvin.scene.master_scene import Scene
from tapas_gmm.calvin_project.calvin.utils.utils import FpsController, get_git_commit_hash
from tapas_gmm.env import calvin

# A logger for this file
log = logging.getLogger(__name__)

import numpy as np

from typing import Tuple

import numpy as np

registered_tasks: dict[str, CalvinTask] = {
    "Test1": PressButton,
    "Test2": PressButton,
    "PressButton": PressButton,
    "PressButtonNew": PressButton,
    "OpenDoor": OpenDoor,
    "CloseDoor": CloseDoor,
    "PickCube1": PickCube1,
    "PickCube2": PickCube2,
    "PickCube3": PickCube3,
    "PlaceCube1": PlaceCube1,
    "PlaceCube2": PlaceCube2,
    "PlaceCube3": PlaceCube3,
}


class CalvinEnvironment(gym.Env):

    def __init__(
        self,
        robot_cfg,
        seed,
        use_vr,
        bullet_time_step,
        cameras,
        show_gui,
        scene_cfg,
        use_scene_info,
        use_egl,
        task,
        control_freq,
        action_mode,
    ):
        self.physics_client = p
        # for calculation of FPS
        self.t = time.time()
        self.prev_time = time.time()
        self.fps_controller = FpsController(bullet_time_step)
        self.use_vr = use_vr
        self.show_gui = show_gui
        self.use_scene_info = use_scene_info
        self.cid = -1
        self.ownsPhysicsClient = False
        self.use_egl = use_egl
        self.control_freq = control_freq
        self.action_repeat = int(bullet_time_step // control_freq)
        render_width = max([cameras[cam].width for cam in cameras]) if cameras else None
        render_height = max([cameras[cam].height for cam in cameras]) if cameras else None
        self.initialize_bullet(bullet_time_step, render_width, render_height)
        self.np_random = None
        self.seed(seed)
        self.robot: Robot = hydra.utils.instantiate(robot_cfg, cid=self.cid)
        self.scene: Scene = hydra.utils.instantiate(
            scene_cfg, p=self.physics_client, cid=self.cid, np_random=self.np_random
        )

        self.task: CalvinTask = registered_tasks.get(task)()
        # Load Env
        self.load()

        # init cameras after scene is loaded to have robot id available
        self.cameras: list[Camera] = [
            hydra.utils.instantiate(
                cameras[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in cameras
        ]

        self.action_mode = action_mode
        self.camera_map: dict[str, Camera] = {}
        for cam in self.cameras:
            self.camera_map[cam.name] = cam

    def __del__(self):
        self.close()

    # From pybullet gym_manipulator_envs code
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_manipulator_envs.py
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        if self.cid < 0:
            self.ownsPhysicsClient = True
            if self.use_vr:
                self.physics_client = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.physics_client._client
                if cid < 0:
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.physics_client.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.physics_client = bc.BulletClient(connection_mode=p.GUI)
                cid = self.physics_client._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
                self.physics_client.resetDebugVisualizerCamera(
                    cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0]
                )
            elif self.use_egl:
                options = f"--width={render_width} --height={render_height}"
                self.physics_client = p
                cid = self.physics_client.connect(p.DIRECT, options=options)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)
                egl = pkgutil.get_loader("eglRenderer")
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    log.error("\nPlugin Failed to load!\n")
                    sys.exit()
                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                log.info("Successfully loaded egl plugin")
            else:
                self.physics_client = bc.BulletClient(connection_mode=p.DIRECT)
                cid = self.physics_client._client
                if cid < 0:
                    log.error("Failed to start DIRECT bullet mode.")
            log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.physics_client.resetSimulation(physicsClientId=self.cid)
            self.physics_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_GUI, 0)
            log.info(f"Connected to server with id: {self.cid}")
            self.physics_client.setTimeStep(1.0 / bullet_time_step, physicsClientId=self.cid)
            return cid

    def load(self):
        log.info("Resetting simulation")
        self.physics_client.resetSimulation(physicsClientId=self.cid)
        log.info("Setting gravity")
        self.physics_client.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        self.robot.load()
        self.scene.load()

    def close(self):
        if self.ownsPhysicsClient:
            if self.cid >= 0 and self.physics_client is not None:
                try:
                    self.physics_client.disconnect(physicsClientId=self.cid)
                except TypeError:
                    pass

        else:
            print("does not own physics client id")

    def update_prediction_marker(self, points: list):
        self.camera_map["front"].update_marker_points(points)

    def render(self, obs: CalvinObservation, info: dict = None, mode="human"):
        """render is gym compatibility function"""
        if mode == "human":
            # Resize images to the desired size
            rgb_static = cv2.resize(obs.rgb["front"], (500, 500))
            # depth_static = cv2.resize(obs.depth["front"], (500, 500))
            rgb_gripper = cv2.resize(obs.rgb["wrist"], (500, 500))
            # depth_gripper = cv2.resize(obs.depth["wrist"], (500, 500))

            # Convert BGR to RGB for correct color display
            rgb_static = cv2.cvtColor(rgb_static, cv2.COLOR_BGR2RGB)
            rgb_gripper = cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)

            # Normalize depth images and convert to BGR
            # depth_static_normalized = cv2.normalize(depth_static, None, 0, 255, cv2.NORM_MINMAX)
            # depth_static_normalized = np.uint8(depth_static_normalized)
            # depth_static_color = cv2.cvtColor(depth_static_normalized, cv2.COLOR_GRAY2BGR)

            # depth_gripper_normalized = cv2.normalize(depth_gripper, None, 0, 255, cv2.NORM_MINMAX)
            # depth_gripper_normalized = np.uint8(depth_gripper_normalized)
            # depth_gripper_color = cv2.cvtColor(depth_gripper_normalized, cv2.COLOR_GRAY2BGR)

            # Create a 2x2 grid of images
            # top_row = np.hstack((rgb_static, depth_static_color))
            # bottom_row = np.hstack((rgb_gripper, depth_gripper_color))
            # combined = np.vstack((top_row, bottom_row))

            combined = np.hstack((rgb_static, rgb_gripper))
            # Add info text overlay
            if info:
                combined = self.add_info_overlay(combined, info)

            # Display the combined image
            cv2.imshow("Combined View", combined)
            cv2.waitKey(1)
        else:
            raise NotImplementedError

    def add_info_overlay(self, img, info):
        """Add info dictionary as text overlay on right side with black background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White text
        thickness = 1
        line_type = cv2.LINE_AA
        margin = 20
        line_height = 40
        bg_padding = 10

        # Starting position (right side)
        x_start = img.shape[1] - 400  # 400px from right edge
        y_start = 50

        # Add timestamp with background
        # if self.recording_start_time is None:
        #    self.recording_start_time = datetime.now()
        # elapsed = datetime.now() - self.recording_start_time
        # time_text = f"Time: {elapsed.total_seconds():.2f}s"

        # Draw text with background
        img = self._draw_text_with_bg(
            img, None, (x_start, y_start), font, font_scale, font_color, thickness, line_type, bg_padding
        )
        y = y_start + line_height

        # Add info dictionary content
        for key, value in info.items():
            text = f"{key}: {value}"
            img = self._draw_text_with_bg(
                img, text, (x_start, y), font, font_scale, font_color, thickness, line_type, bg_padding
            )
            y += line_height

            # Prevent overflow
            if y > img.shape[0] - 50:
                break

        return img

    def _draw_text_with_bg(self, img, text, pos, font, font_scale, color, thickness, line_type, padding):
        """Helper function to draw text with background"""
        x, y = pos
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate background rectangle coordinates
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + padding

        # Draw background
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # Black background

        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, line_type)

        return img

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self, robot_obs=None, scene_obs=None) -> Tuple[CalvinObservation, float, bool, dict]:
        self.robot.reset(robot_obs)
        self.scene.reset(scene_obs)
        self.physics_client.stepSimulation(physicsClientId=self.cid)
        obs = self._get_observation()
        info = self._get_info()
        reward, done = self.task.reset(obs)

        # add values to observation for SceneObservation
        obs.reward = reward
        obs.done = done
        # obs, reward, done, info
        return obs, reward, done, info

    def step(self, action, action_mode) -> Tuple[CalvinObservation, float, bool, dict]:
        action = {"action": action, "type": action_mode}

        self.robot.apply_action(action)
        for i in range(self.action_repeat):
            self.physics_client.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self._get_observation()
        info = self._get_info()
        reward, done = self.task.step(obs)

        # add values to observation for SceneObservation
        obs.reward = reward
        obs.done = done
        # obs, reward, done, info
        return obs, reward, done, info

    def _get_observation(
        self,
        has_joint_forces=True,
        has_gripper_touch_forces=True,
    ) -> CalvinObservation:

        wrist_rgb, wrist_depth, wrist_pcd, wrist_mask = self.camera_map["wrist"].render()
        front_rgb, front_depth, front_pcd, front_mask = self.camera_map["front"].render()

        _, robot_obs = self.robot.get_observation()  # get state observation
        # scene_obs = self.scene.get_low_dim_state()
        arm_joint_forces = robot_obs["arm_joint_forces"]
        arm_joint_velocities = robot_obs["arm_joint_velocities"]
        arm_joint_positions = robot_obs["arm_joint_positions"]

        joint_forces = None
        if has_joint_forces:
            joint_forces = self.robot.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(arm_joint_forces, arm_joint_velocities)])
            )

        ee_forces_flat = None
        if has_gripper_touch_forces:
            ee_forces = robot_obs["gripper_finger_forces"]
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        rgb_dict: dict = {
            "wrist": wrist_rgb,
            "front": front_rgb,
        }
        depth_dict: dict = {
            "wrist": wrist_depth,
            "front": front_depth,
        }
        pcd_dict: dict = {
            "wrist": wrist_pcd,
            "front": front_pcd,
        }
        mask_dict: dict = {
            "wrist": wrist_mask,
            "front": front_mask,
        }
        camera_settings = self._get_misc()

        extr_dict: dict = {
            "wrist": camera_settings["wrist"]["extrinsics"],
            "front": camera_settings["front"]["extrinsics"],
        }
        intr_dict: dict = {
            "wrist": camera_settings["wrist"]["intrinsics"],
            "front": camera_settings["front"]["intrinsics"],
        }
        obs = CalvinObservation(
            camera_names=["wrist", "front"],
            rgb=rgb_dict,
            depth=depth_dict,
            pcd=pcd_dict,
            mask=mask_dict,
            extr=extr_dict,
            intr=intr_dict,
            object_poses=self.scene.get_dictionary_object_poses(),
            object_states=self.scene.get_dictionary_object_states(),
            low_dim_object_poses=self.scene.get_low_dim_object_poses(),
            low_dim_object_states=self.scene.get_low_dim_object_states(),
            joint_vel=self.robot.joint_velocities_noise.apply(np.array(arm_joint_velocities)),
            joint_pos=self.robot.joint_positions_noise.apply(np.array(arm_joint_positions)),
            joint_forces=joint_forces,
            gripper_matrix=robot_obs["gripper_view_matrix"],
            gripper_pose=robot_obs["gripper_pose"],
            gripper_state=robot_obs["gripper_opening_state"],
            tcp_pose=robot_obs["tcp_pose"],
            tcp_state=robot_obs["tcp_state"],
            gripper_touch_forces=ee_forces_flat,
            gripper_joint_positions=robot_obs["gripper_finger_positions"],
        )
        return obs

    def _get_info(self):
        _, robot_info = self.robot.get_observation()
        info = {"robot_info": robot_info}
        if self.use_scene_info:
            info["scene_info"] = self.scene.get_info()
        return info

    def _get_misc(self):
        def _get_cam_data(cam: Camera):
            d = {
                "extrinsics": np.array(cam.viewMatrix).reshape((4, 4)).T,
                "intrinsics": np.array(cam.projectionMatrix).reshape((3, 3)).T,
                "near": cam.nearval,
                "far": cam.farval,
            }
            return d

        camera_dict = {}
        for cam in self.cameras:
            camera_dict[cam.name] = cam

        misc = {}
        misc.update({"wrist": _get_cam_data(camera_dict["wrist"])})
        misc.update({"front": _get_cam_data(camera_dict["front"])})
        misc.update({"variation_index": self.seed()})
        return misc

    def reset_from_storage(self, filename):
        """
        Args:
            filename: file to load from.
        Returns:
            observation
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)

        self.robot.reset_from_storage(data["robot"])
        self.scene.reset_from_storage(data["scene"])

        self.physics_client.stepSimulation(physicsClientId=self.cid)

        return data["state_obs"], data["done"], data["info"]

    def serialize(self):
        data = {"time": time.time_ns() / (10**9), "robot": self.robot.serialize(), "scene": self.scene.serialize()}
        return data


def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")

    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    if "scene" in kwargs:
        scene_cfg = OmegaConf.load(Path(calvin.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")
        render_conf.scene = scene_cfg
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


def get_env_from_cfg(task: str, eval: bool = False) -> CalvinEnvironment:
    """Bypass Hydra's execution context and create the environment manually."""
    with hydra.initialize(config_path="../../conf", version_base="1.1"):
        config_name = "master_config_eval" if eval else "master_config"
        cfg = hydra.compose(config_name=config_name)
        # env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
        env = CalvinEnvironment(
            robot_cfg=cfg.robot,  # Robot basic cfg load here
            seed=cfg.seed,  # ignore for now
            use_vr=cfg.env.use_vr,  # correct
            bullet_time_step=cfg.env.bullet_time_step,  # ignore for now
            cameras=cfg.cameras,
            show_gui=cfg.env.show_gui,
            scene_cfg=cfg.scene,
            use_scene_info=cfg.env.use_scene_info,
            use_egl=cfg.env.use_egl,
            control_freq=cfg.env.control_freq,
            task=task,
            action_mode="action_mode",
        )
        assert env is not None, "Failed to create CustomSimEnv"
        return env
