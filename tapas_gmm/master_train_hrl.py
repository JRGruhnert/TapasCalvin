from tapas_gmm.env import calvin
from env.environment import BaseEnvironment
import gym
import numpy as np
from policy.gmm import GMMPolicy
import torch
import torch.nn as nn
import torch.optim as optim


class BasePolicy:
    def generate_sequence(self, state):
        raise NotImplementedError

    def update(self, reward, state, done):
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space

    def generate_sequence(self, state):
        return self.action_space.sample()


class DQNPolicy(BasePolicy, nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.action_size = action_size
        self.replay_buffer = []
        self.batch_size = 32
        self.gamma = 0.99

    def forward(self, x):
        return self.net(x)

    def generate_sequence(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self(state)
            return torch.argmax(q_values).item()

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = np.random.choice(
            len(self.replay_buffer), self.batch_size, replace=False
        )
        batch = [self.replay_buffer[i] for i in batch]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RLAgent:
    def __init__(self, policy: BasePolicy):
        self.policy: BasePolicy = policy

    def act(self, state) -> list[GMMPolicy]:
        return self.policy.generate_sequence(state)

    def update(self, reward: float, state, done: bool):
        self.policy.update(reward, state, done)


def sample_goal_state(env: BaseEnvironment):
    # Sample a goal state from the environment
    # This is a placeholder function. You should implement your own logic.
    return env.sample_goal_state()


class BaseRewardMode:
    def __init__(self, env: BaseEnvironment):
        self.env = env

    def get_reward(self, state, action, next_state):
        # Placeholder for reward calculation
        return 0.0


def train_agent(
    agent: RLAgent,
    env: BaseEnvironment,
    reward_strategy: BaseRewardMode,
    num_episodes=1000,
):
    rewards = []
    avg_rewards = []

    for episode in range(num_episodes):  # Training loop
        goal_state = sample_goal_state(env)
        state = env.reset()
        ep_done = False
        total_reward = 0

        while not ep_done:  # Episode loop
            # Generate a sequence of models
            sq_prediction = agent.act(state)

            # Takes first model in the sequence (for simplicity)
            # In a real scenario, you might want to use all models in the sequence
            # But we want to repredict the sequence after the first model of the sequence is used
            # to have a tighter feedback loop
            model = sq_prediction[0]

            done = False

            while not done:  # Model loop
                # Predict the next action using the model
                prediction = model.predict(state)
                # Execute the action in the environment
                next_state, reward, done, info = env.step(prediction)
                # Update the total reward
                total_reward += reward
                # Update the model with the new state
                state = next_state

            ep_reward = reward_strategy.get_reward(state, sq_prediction, next_state)
            agent.update(ep_reward, state, done)

        rewards.append(total_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | Reward: {total_reward:4.1f} | Avg: {avg_rewards[-1]:4.1f}"
            )

    # Save the model
    # agent.policy.save_model("model.pth")

    return rewards, avg_rewards


if __name__ == "__main__":
    # Example usage with agent concept preserved
    env = None

    # Train random agent
    print("Training Random Agent:")
    random_agent = RLAgent(RandomPolicy(env.action_space))
    train_agent(random_agent, num_episodes=500)
