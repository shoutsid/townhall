"""
A module containing the ImprovedAgent class, which represents an improved agent
that uses a Deep Q-Network (DQN) to learn to perform multiple tasks.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from app.models.transition import Transition
from app.models.dqn import DQN
from app.models.replay_memory import ReplayMemory
from app.models.life_long_model import LifeLongModel
from app.models.constants import BATCH_SIZE, GAMMA


class ImprovedAgent:
    """
    A class representing an improved agent that uses a Deep Q-Network (DQN) to learn to perform multiple tasks.

    Attributes:
    - id (int): The ID of the agent.
    - num_tasks (int): The number of tasks the agent can perform.
    - num_features (int): The number of features in the state space.
    - models (list): A list of ImprovedLLM models, one for each task.
    - device (torch.device): The device (CPU or GPU) on which to run the DQN.
    - target_net (DQN): The target DQN used for computing the target Q-values.
    - policy_net (DQN): The policy DQN used for computing the current Q-values.
    - optimizer (torch.optim.RMSprop): The optimizer used for updating the policy DQN.
    - memory (ReplayMemory): The replay memory used for storing transitions.
    - previous_loss (list): A list of the previous loss for each task.
    """

    def __init__(self, id, num_tasks, num_features):
        """
        Initializes a new instance of the ImprovedAgent class.

        Args:
        - id (int): The ID of the agent.
        - num_tasks (int): The number of tasks the agent can perform.
        - num_features (int): The number of features in the state space.
        """
        self.id = id
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.models = [LifeLongModel(num_features) for _ in range(num_tasks)]

        # DQN-related attributes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.target_net = DQN(num_features).to(self.device)
        self.policy_net = DQN(num_features).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        # Initialize with zeros for each task
        self.previous_loss = [0] * num_tasks

    def observe(self, task):
        """
        Observe the current state of the environment and update the agent's models and DQN.

        Args:
            task (int): The task ID for the current environment.

        Returns:
            tuple: A tuple containing the loss and reward values for the current observation.
        """
        observations = np.random.rand(
            self.num_features)  # Expanded state space
        target = np.sum(observations)
        loss = self.models[task].train(observations, target)

        # Calculate reward based on the difference between current and previous loss
        reward = self.calculate_reward(loss, task)

        # Update DQN
        self.update_dqn(observations, reward, task)

        return loss, reward

    def calculate_reward(self, loss, task):
        """
        Calculates the reward for the agent based on the difference between the current and previous loss.

        Args:
            loss (float): The current loss value for the task.
            task (str): The name of the task for which the loss is being calculated.

        Returns:
            float: The reward value for the agent.
        """
        reward = self.previous_loss[task] - loss
        self.previous_loss[task] = loss  # Store current loss for the next step
        return reward

    def update_dqn(self, state, reward, task):
        """
        Updates the Deep Q-Network (DQN) by training it on a batch of transitions from the replay memory.

        Args:
            state (torch.Tensor): The current state of the agent.
            reward (float): The reward received by the agent for taking the last action.
            task (int): The task ID of the current task.

        Returns:
            None
        """
        self.memory.push(state, task, None, reward)  # Pass None for next_state
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        if any(non_final_mask):  # Check if at least one non-final state is present
            # Filter out empty tensors from non_final_next_states
            non_final_next_states = torch.stack(
                [s for s, mask in zip(batch.next_state, non_final_mask) if mask])

            state_batch = torch.stack(
                [s for s, mask in zip(batch.state, non_final_mask) if mask])
            action_batch = torch.tensor(
                [a for a, mask in zip(batch.action, non_final_mask) if mask], device=self.device)
            reward_batch = torch.tensor(
                [r for r, mask in zip(batch.reward, non_final_mask) if mask], device=self.device)

            next_state_values = torch.zeros(
                BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (
                next_state_values * GAMMA) + reward_batch
            q_values = self.policy_net(state_batch).gather(
                1, action_batch.unsqueeze(1))
            loss = F.smooth_l1_loss(
                q_values, expected_state_action_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
