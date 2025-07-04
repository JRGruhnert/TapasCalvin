from datetime import datetime
import os
from typing import Dict
from loguru import logger
import numpy as np

from tapas_gmm.master_project.master_converter import P_C_NodeConverter
from tapas_gmm.master_project.master_data_def import ActionSpace, StateSpace, StateType
from tapas_gmm.master_project.master_helper import HRLHelper
from tapas_gmm.master_project.master_sample import (
    sample_post_condition,
    sample_pre_condition,
)
from tapas_gmm.master_project.master_agent import GNNAgent, PPOAgent, Agent
from tapas_gmm.master_project.master_observation import HRLPolicyObservation
from tapas_gmm.env.calvin import Calvin


def train_agent(
    is_baseline: bool = True,
    pre_con_check: bool = False,
    num_episodes: int = 1000,
    batch_size: int = 50,
    horizon: int = 8,
    saving_frequence: int = 100,
    action_space: ActionSpace = ActionSpace.STATIC,
    state_space: StateSpace = StateSpace.DYNAMIC,
    selection_threshold: float = 0.25,
):
    # Initialize the environment and agent
    env = Calvin(eval=True)
    dummy_obs, _, _, _ = env.reset()

    agent: Agent = None
    if is_baseline:
        agent = PPOAgent(
            action_space=action_space,
            state_space=state_space,
            goal=HRLPolicyObservation(dummy_obs),
            threshold=selection_threshold,
        )
        agent_name = "baseline"
    else:
        agent = GNNAgent(
            action_space=action_space,
            state_space=state_space,
            goal=HRLPolicyObservation(dummy_obs),
            threshold=selection_threshold,
        )
        agent_name = "gnn"

    directory = "results/training/" + agent_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "Model_ep{}_as{}_ss{}_bs{}.pth".format(
        num_episodes, action_space.name, state_space.name, batch_size
    )
    print("Checkpoint will be saved at: " + checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # Starting values
    running_reward = 0
    running_episodes = 0
    time_step = 0
    episode = 0

    while time_step <= num_episodes:
        # Starting Conditions
        print("Sampling Precondition")
        scene_starting_obs = sample_pre_condition(
            dummy_obs.scene_obs, state_space=StateSpace.SMALL
        )

        print("Sampling Goalcondition")
        # Goal Conditions
        scene_goal_obs = sample_post_condition(
            scene_starting_obs, state_space=StateSpace.SMALL
        )

        # Reset environment twice to get CalvinObservation (maybe make it more elegant)
        calvin_goal_obs, _, _, _ = env.reset(scene_goal_obs)
        hrl_goal_obs = HRLPolicyObservation(calvin_goal_obs)
        calvin_obs, _, _, _ = env.reset(scene_starting_obs)
        hrl_obs = HRLPolicyObservation(calvin_obs)

        viz_dict: Dict[str, float] = {
            key.name: value  # Prefix all keys with "new_"
            for key, value in hrl_goal_obs.scalar_states.items()
        }
        agent.reset(hrl_goal_obs)
        ep_reward = 0

        for episode_step in range(horizon):  # Training loop
            # Retrieve policy from agent
            task_id = agent.act(hrl_obs)
            # Helper to convert index to Task enum (stores task specific informations)
            selected_task = HRLHelper.retrieve_task(task_id)

            # Decides wether selected policy has a good starting position
            # based of preconditions and current observation
            skip = False
            if pre_con_check:
                con_eval = P_C_NodeConverter(selected_task)
                score = con_eval.node_features(hrl_obs, True)
                if np.any(score > selection_threshold):
                    step_reward = -10.0
                    ep_done = False
                    skip = True

            if not skip:
                # Loads Tapas Policy for that Task (batch predict config)
                policy = HRLHelper.load_policy(selected_task)
                policy.reset_episode(env)

                # This is a hack for changing the ee_pose to the origin for reversed models
                # It does nothing for standard models
                hrl_obs = HRLHelper.convert_observation(selected_task, hrl_obs)

                # Batch prediction for the given observation
                prediction, _ = policy.predict(hrl_obs.tapas_format)

                for action in prediction:
                    ee_action = np.concatenate((action.ee, action.gripper))
                    calvin_obs, _, _, _ = env.step(ee_action, viz_dict)

                new_hrl_obs = HRLPolicyObservation(calvin_obs)
                step_reward, ep_done = agent.get_reward_step(hrl_obs, new_hrl_obs)

                hrl_obs = new_hrl_obs
            time_step += 1
            ep_reward += step_reward

            if time_step % batch_size == 0:
                print("Batch ready!")
                agent.update()

            # save model weights
            if time_step % saving_frequence == 0:
                agent.save(checkpoint_path)
                # print average reward till last episode
                print_avg_reward = running_reward / running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print("model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        episode, time_step, print_avg_reward
                    )
                )

                print(
                    "--------------------------------------------------------------------------------------------"
                )
                running_reward = 0
                running_episodes = 0

            # If episode is done break and start new episode
            if ep_done:
                break

        running_reward += ep_reward
        running_episodes += 1
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
    train_agent(num_episodes=500)


if __name__ == "__main__":
    entry_point()
