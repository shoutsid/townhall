import matplotlib.pyplot as plt
from tinygrad.tensor import Tensor
import numpy as np


class ImprovedEnvironment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = [ImprovedAgent(i) for i in range(num_agents)]
        self.performance = {i: []
                            for i in range(num_agents)}  # Store performance
        self.agent_positions = {i: []
                                for i in range(num_agents)}  # Store agent positions
        self.rewards = {i: [] for i in range(num_agents)}  # Store rewards

    def step(self):
        for agent in self.agents:
            loss, reward = agent.observe()
            self.performance[agent.id].append(loss)
            self.agent_positions[agent.id].append(
                np.random.rand(2))  # Random 2D positions
            self.rewards[agent.id].append(reward)  # Store rewards


class ImprovedAgent:
    def __init__(self, id):
        self.id = id
        self.model = ImprovedLLM()

    def observe(self):
        observations = np.random.rand(10)
        target = np.sum(observations)
        loss = self.model.train(observations, target)
        reward = self.calculate_reward(loss)
        return loss, reward

    def calculate_reward(self, loss):
        # Define a simple reward function (can be more complex)
        return -loss  # Negative loss as reward


class ImprovedLLM:
    def __init__(self):
        self.w1 = Tensor(np.random.randn(5, 10), requires_grad=True)
        self.w2 = Tensor(np.random.randn(5), requires_grad=True)

    def forward(self, x):
        x = Tensor(x).reshape(1, -1)
        x = self.w1.dot(x.transpose()).relu()
        return self.w2.dot(x)

    def train(self, x, y_true):
        y_pred = self.forward(x)
        loss = ((y_pred - Tensor(y_true)) ** 2).sum()
        loss.backward()
        learning_rate = 0.01
        self.w1 = self.w1 - self.w1.grad * learning_rate
        self.w2 = self.w2 - self.w2.grad * learning_rate
        return loss.numpy()


# Create environment with 3 agents
env = ImprovedEnvironment(3)

# Assume 100 steps for training
for step in range(100):
    env.step()

# Plot agent positions
plt.figure(figsize=(8, 8))
for agent_id, positions in env.agent_positions.items():
    positions = np.array(positions)
    plt.scatter(positions[:, 0], positions[:, 1],
                label=f'Agent {agent_id}', alpha=0.5)

plt.legend()
plt.title('Agent Movements')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

# Plot performance
plt.figure(figsize=(10, 6))
for agent_id, losses in env.performance.items():
    plt.plot(losses, label=f'Agent {agent_id}')

plt.legend()
plt.title('Agent Performance')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Plot rewards
plt.figure(figsize=(10, 6))
for agent_id, rewards in env.rewards.items():
    plt.plot(rewards, label=f'Agent {agent_id}')

plt.legend()
plt.title('Agent Rewards')
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
