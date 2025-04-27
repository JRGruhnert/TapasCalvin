# ======================
# RL Training Setup
# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from tapas_gmm.env.calvin import Calvin
from tapas_gmm.policy.models.master_project.hrl_data import HRLBipartiteData
from tapas_gmm.policy.models.master_project.hrl_policy import HRLPolicy


goal_states = {
    "turn_on_light": [0.5, 0.5, 0.5],
    "turn_off_light": [0.5, 0.5, 0.5],
    "pick_up_object": [0.5, 0.5, 0.5],
    "put_down_object": [0.5, 0.5, 0.5],
    "move_to_location": [0.5, 0.5, 0.5],
    "open_door": [0.5, 0.5, 0.5],
}


class HRLAgent:
    def __init__(
        self,
        env: Calvin,
        data: HRLBipartiteData,
        policy: HRLPolicy,
        lr=1e-3,
        gamma=0.99,
    ):
        self.env = env
        self.data = data
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma

        self.initial_state = env.reset()
        self.current_state = self.initial_state

    def train_episode(self):
        # Initialize episode
        self.initial_state = self.env.reset()
        self.current_state = self.initial_state
        goal_state = goal_states[self.env.task_name]
        log_probs = []
        rewards = []

        # Run episode
        done = False
        while not done:
            # Get action probabilities
            model_logits = self.policy(obs)
            probs = torch.softmax(model_logits, dim=0)

            # Take step in environment
            new_obs, reward, done = self.env.step(action.item())

            # Store results
            log_probs.append(log_prob)
            rewards.append(reward)

            obs = new_obs

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return sum(rewards)


# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    # Initialize components
    env: Calvin = Calvin()
    data = HRLBipartiteData()
    policy = HRLPolicy()
    agent = HRLAgent(env, data, policy)

    # Training loop
    for episode in range(100):
        reward = agent.train_episode()
        print(f"Episode {episode}, Total Reward: {reward:.2f}")

    # Inference example
    obs = env.reset()
    with torch.no_grad():
        model_logits = policy(obs)
        selected_model = torch.argmax(model_logits).item()
    print(f"Optimal initial model: {selected_model}")
