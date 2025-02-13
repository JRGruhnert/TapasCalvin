import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any, List
from mplib.pymp import Pose
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from tapas_gmm.env.calvinbench import CalvinTapasBridgeEnvironmentConfig, CalvinTapasBridgeEnvironment

from conversion import calvin_to_tapas_representation
from tapas_gmm.policy.motion_planner import Action, CloseGripperAction, MotionPlannerPolicy, ActionSequence, MoveToAction, OpenGripperAction
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    get_dataset_name,
    get_full_task_name,
    loop_sleep,
)
from tapas_gmm.utils.observation import SceneObservation

# from tapas_gmm.utils.random import configure_seeds
@dataclass
class Task:
    task_name: str
    task_sequence: ActionSequence
    horizon: int
    feedback_type: str

@dataclass
class TestTask(Task):
    task_name: str = "Test"
    task_sequence: ActionSequence = ActionSequence([OpenGripperAction(), CloseGripperAction(), OpenGripperAction(), CloseGripperAction()])
    horizon: int = 50
    feedback_type: str = "test"

class Test2Task(Task):
    task_name: str = "Test2"
    task_sequence: ActionSequence = ActionSequence([OpenGripperAction(), MoveToAction(goal = Pose((0,0,0),(0,0,0,0))), CloseGripperAction(), OpenGripperAction(), CloseGripperAction()])
    horizon: int = 100
    feedback_type: str = "test2"

@dataclass
class Config:
    task: Task = Test2Task
    data_naming: DataNamingConfig = DataNamingConfig(feedback_type=task.feedback_type, task=task.task_name, data_root="data")
    dataset_config: SceneDatasetConfig = SceneDatasetConfig(data_root="data", camera_names=["gripper", "static"], image_size=(256, 256))

    env_config: CalvinTapasBridgeEnvironmentConfig = CalvinTapasBridgeEnvironmentConfig(
    camera_pose={
        "base": (0.2, 0.0, 0.2, 0, 0.194, 0.0, -0.981),
        # "overhead": (0.2, 0.0, 0.2, 7.7486e-07, -0.194001, 7.7486e-07, 0.981001),
    },
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    gripper_plot=False,
    postprocess_actions=True,
    task= task.task_name,
)

    horizon: int = task.horizon

    n_episodes: int = 2



def main(config: Config = Config()) -> None:
    env = CalvinTapasBridgeEnvironment(config.env_config)
    policy = MotionPlannerPolicy(env=env, sequence=config.task.task_sequence) # type: ignore
    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / config.task.task_name

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    replay_memory = SceneDataset(
        allow_creation=True,
        config=config.dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    env.reset()  # extra reset to correct set up of camera poses in first obs
    obs, info = env.reset()

    # Basically find the specific task in the task sequence and set the goal to the task pose
    for action in policy.sequence:
        if isinstance(action, MoveToAction):
            taskId = action.taskId
            scene_info = info["scene_info"]
            movable_objects = scene_info["movable_objects"]
            object_pos = movable_objects[taskId]["current_pos"]
            object_orn = movable_objects[taskId]["current_orn"]
            task_pose = Pose(object_pos, object_orn)
            print(f"Task Pose: {task_pose}")
            action.goal = task_pose

    time.sleep(5)
    logger.info("Go!")

    episodes_count = 0
    timesteps = 0

    # Max number of timesteps in an episode
    horizon = config.horizon or np.inf

    try:
        with tqdm(total=config.n_episodes) as ebar:
            with tqdm(total=horizon) as tbar:
                tbar.set_description("Time steps")
                while episodes_count < config.n_episodes:
                    ebar.set_description("Running episode")
                    start_time = time.time()

                    logger.info(f"Episode {episodes_count + 1}")
                    #logger.info(f"Observation: {obs}")
                    action, done = policy.predict(obs)
                    logger.info(f"Action: {action}")
                    logger.info(f"block_red: {task_pose}")
                    # Gettting the next observation
                    next_obs, _, _, _ = env.step(action)
                    if not done:
                        if next_obs is not None:
                            next_obs.action = torch.Tensor(env._get_action(obs, next_obs))
                            next_obs.feedback = torch.Tensor([1])
                            replay_memory.add_observation(obs)

                        timesteps += 1
                        tbar.update(1)
                        obs = next_obs

                    if done:
                        logger.info("Saving trajectory.")
                        ebar.set_description("Saving trajectory")
                        replay_memory.save_current_traj()

                        obs = env.reset()
                        policy.reset_episode(env)

                        episodes_count += 1
                        ebar.update(1)

                        timesteps = 0
                        tbar.reset()

                        done = False

                    elif timesteps >= horizon:
                        logger.info("Resetting without saving traj.")
                        ebar.set_description("Resetting without saving traj")
                        replay_memory.reset_current_traj()

                        obs = env.reset()
                        policy.reset_episode(env)

                        timesteps = 0
                        tbar.reset()

                    else:
                        loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


if __name__ == "__main__":
    main()
