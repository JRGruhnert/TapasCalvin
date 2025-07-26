from tqdm import tqdm
from typing import Optional, Tuple

from tapas_gmm.env.calvin import Calvin
from tapas_gmm.policy.models.master_project.hrl_enums import TaskModel, State
from tapas_gmm.policy.policy import Policy
from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.viz.live_keypoint import LiveKeypoints


# edge list construction
# - states manual (cause not recorded)
# - tapas model with index list
# - need to load each model and then take index list
def run_episode_hrl(
    env: Calvin,
    keyboard_obs: Optional[KeyboardObserver],
    parent_policy,  # Your GNN-based HRL policy
    child_policies: list[Policy],  # TAPAS sub-policies
    goal_state,
    horizon: Optional[int],
    keypoint_viz: Optional[LiveKeypoints],
    obs_dropout: Optional[float],
    fragment_len: int,
    disturbe_at_step: Optional[int],
    hold_until_step: Optional[int],
) -> Tuple[float, int]:
    episode_reward = 0.0
    obs, _, done, _ = env.reset()
    obs = obs.to_rlbench_format()

    tapas_model_dict = {}
    task_parameter_dict = {}
    step_no = 0
    pbar = tqdm(total=horizon or 1000)

    while not done and step_no < (horizon or 1000):
        # --- GNN parent policy selects child TAPAS model based on current and goal states
        child_index = parent_policy.predict(
            obs, goal_state
        )  # Returns index or key to child policy
        selected_policy = child_policies[child_index]

        selected_policy.reset_episode(env=env)
        child_done = False

        while not child_done and not done:
            action, _ = selected_policy.predict(obs)

            for _ in range(selected_policy.repeat_steps):
                obs, reward, done, _ = env.step(action)
                obs = obs.to_rlbench_format()
                episode_reward += reward
                step_no += 1
                pbar.update(1)

                if keypoint_viz:
                    keypoint_viz.update_from_info({}, obs)
                    keypoint_viz.propose_update_visualization({})

                if horizon and step_no >= horizon:
                    done = True
                    break

            child_done = selected_policy.has_finished(obs)

        if keyboard_obs:
            if keyboard_obs.success:
                episode_reward += 1.0
                done = True
            elif keyboard_obs.reset_button:
                done = True

    return episode_reward, step_no + 1
