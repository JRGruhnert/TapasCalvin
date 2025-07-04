def calculate_policy_reward(
    start_obs,
    goal_obs,
    number_of_steps,
) -> float:
    """
    Calculate the reward for the policy based on the start and goal observations and the number of steps taken.
    This function is used to evaluate the performance of the policy in a reinforcement learning setting.

    Args:
        policy: The policy object.
        start_obs: The initial observation.
        goal_obs: The target observation.
        number_of_steps: The number of steps taken to reach the goal.

    Returns:
        float: The calculated reward for the policy.
    """
    reward = 0.0
    # Calculate the percentual difference of every property
    # between the start and goal observation
    # TODO: implement this

    # Maybe include the total steps taken to reach the goal
    # TODO: implement this

    return reward


def calculate_model_reward(
    policy,
    obs,
    action,
    next_obs,
    reward,
    done,
    info,
) -> float:
    """
    Calculate the reward for the model based on the observation, action, next observation, and reward.
    This function is used to evaluate the performance of the model in a reinforcement learning setting.

    Args:
        policy: The policy object.
        obs: The current observation.
        action: The action taken by the agent.
        next_obs: The next observation after taking the action.
        reward: The reward received after taking the action.
        done: A boolean indicating if the episode is done.
        info: Additional information about the environment.

    Returns:
        float: The calculated reward for the model.
    """
    return reward
