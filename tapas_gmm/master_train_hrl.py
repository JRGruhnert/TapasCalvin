from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
import os
from typing import Dict
import numpy as np

from tapas_gmm.master_project.master_converter import P_C_NodeConverter
from tapas_gmm.master_project.master_data_def import State
from tapas_gmm.master_project.master_helper import HRLHelper
from tapas_gmm.master_project.master_sample import (
    sample_post_condition,
    sample_pre_condition,
)
from tapas_gmm.master_project.master_agent import GNNAgent, PPOAgent, Agent, RLConfig
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.env.calvin import Calvin


# Custom encoder that converts Enums to their names
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name  # or obj.value if you prefer the numeric value
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def serialize(obj):
    if isinstance(obj, Enum):
        return obj.name  # or obj.value
    elif is_dataclass(obj):
        return {k: serialize(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {serialize(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    else:
        return obj  # assume it's a primitive or already serializable


def train_agent(
    is_baseline: bool = True,
    n_episodes: int = 25,  # Number of episodes to train
    horizon: int = 6,  # Max steps the agent can take in one task
    saving_path: str = "results/",
    vis: bool = False,  # Visualize the environment
    use_gpu: bool = True,  # Use GPU for training
    # Agent Parameters
    parameters: RLConfig = RLConfig(),
):
    # Initialize the environment and agent
    env = Calvin(eval=True, vis=not use_gpu)
    agent: Agent = None
    if is_baseline:
        agent = PPOAgent(
            parameters=parameters,
        )
        agent_name = "baseline"
    else:
        agent = GNNAgent(
            parameters=parameters,
        )
        agent_name = "gnn"

    directory_path = saving_path + agent_name + "/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    #### get number of log files in log directory
    current_num_subdirectory = next(os.walk(directory_path))[1]
    run_num = len(current_num_subdirectory)

    #### run directory path
    run_directory_path = directory_path + f"run_{run_num}/"
    if not os.path.exists(run_directory_path):
        os.makedirs(run_directory_path)

    ### save parameters to json file
    config_path = run_directory_path + f"config{run_num}.json"
    with open(config_path, "w") as f:
        json.dump(serialize(parameters), f, indent=4, cls=DataclassJSONEncoder)

    checkpoint_path = run_directory_path + "model{}_ep-{}.pth".format(
        run_num,
        n_episodes,
    )

    #### log files for a run
    log_directory_path = run_directory_path + f"logs/"
    if not os.path.exists(log_directory_path):
        os.makedirs(log_directory_path)

    print("Checkpoint will be saved at: " + checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # Starting values
    running_reward = 0
    episode = 1
    while episode <= n_episodes:
        ep_reward = 0
        batch_step = 0
        while batch_step < parameters.batch_size:
            # print(f"Batch: {batch_step}/{n_steps}")
            random_obs, _, _, _ = env.reset(settle_time=0)
            # print("Sampling Precondition")
            scene_starting_obs = sample_pre_condition(
                random_obs.scene_obs, state_space=parameters.state_space
            )

            # print("Sampling Goalcondition")
            scene_goal_obs = sample_post_condition(
                scene_starting_obs, state_space=parameters.state_space
            )
            # Reset environment twice to get CalvinObservation (maybe find a better way)
            calvin_goal_obs, _, _, _ = env.reset(scene_goal_obs, static=False, settle_time=50)
            hrl_goal_obs = HRLPolicyObservation(calvin_goal_obs)
            calvin_obs, _, _, _ = env.reset(scene_starting_obs, static=False, settle_time=50)
            hrl_obs = HRLPolicyObservation(calvin_obs)
            viz_dict: Dict[str, bool] = {
                key.name: value == hrl_obs.scalar_states[key]
                for key, value in hrl_goal_obs.scalar_states.items()
            }
            for task_step in range(1, horizon + 1):  # Training loop
                last_step = task_step == horizon
                # Retrieve policy from agent
                task_id = agent.act(hrl_obs, hrl_goal_obs)
                selected_task = HRLHelper.retrieve_task(task_id)
                # print(f"{task_step}/{horizon}: {selected_task.name}")
                # Decides wether selected policy has a good starting position
                # based of preconditions and current observation
                skip = False
                pre_con_check = False
                selection_threshold = 0.5  # Threshold for precondition check
                if pre_con_check:
                    con_eval = P_C_NodeConverter(selected_task)
                    score = con_eval.node_features(hrl_obs, True)
                    if np.any(score > selection_threshold):
                        step_reward = -10.0
                        task_done = False
                        skip = True

                if not skip:
                    # Loads Tapas Policy for that Task (batch predict config)
                    policy = HRLHelper.load_policy(selected_task)
                    policy.reset_episode(env)
                    # This is a hack for changing the ee_pose to the origin for reversed models
                    # It does nothing for standard models
                    hrl_obs = HRLHelper.convert_observation(selected_task, hrl_obs)
                    # Batch prediction for the given observation
                    try:
                        prediction, _ = policy.predict(hrl_obs.tapas_format)
                        for action in prediction:
                            ee_action = np.concatenate((action.ee, action.gripper))
                            calvin_obs, _, _, _ = env.step(ee_action, vis, viz_dict)
                        new_hrl_obs = HRLPolicyObservation(calvin_obs)
                        step_reward, task_done = agent.step(
                            hrl_obs, new_hrl_obs, hrl_goal_obs, last_step
                        )
                    except FloatingPointError:
                        # At some point the model crashes.
                        # Have to debug if its because of bad input
                        print(f"Error happened!")
                        new_hrl_obs = hrl_obs
                        step_reward, task_done = agent.step(
                            hrl_obs, new_hrl_obs, hrl_goal_obs, True
                        )
                    hrl_obs = new_hrl_obs
                ep_reward += step_reward
                batch_step += 1

                # If episode is done break and start new episode
                if task_done or batch_step >= parameters.batch_size:
                    break
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Batch is ready - updating agent and saving log!")
        agent.save_buffer(log_directory_path, episode)
        agent.update()

        print("Saving updated agent!")
        agent.save(checkpoint_path)
        # print average reward till last episode
        print_avg_reward = running_reward / episode
        print_avg_reward = round(print_avg_reward, 2)

        print(
            "Elapsed Time: ",
            datetime.now().replace(microsecond=0) - start_time,
        )
        print(
            "Episode: {} \t Average Reward: {} \t Average Ep. Reward: {}".format(
                episode, print_avg_reward, ep_reward
            )
        )
        print(
            "--------------------------------------------------------------------------------------------"
        )

        running_reward += ep_reward
        episode += 1

    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )


def entry_point():
    train_agent(n_episodes=500)


if __name__ == "__main__":
    entry_point()
