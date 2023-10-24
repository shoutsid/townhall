from typing import List, Tuple
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from townhall.models.transition import Transition
from townhall.models.dqn_torch import DQN
from townhall.models.replay_memory import ReplayMemory
from townhall.models.lifelong_learning_model import LifelongLearningModel
from townhall.models.constants import BATCH_SIZE, GAMMA


class ImprovedAgent:
    """
    A module containing the ImprovedAgent class, which represents an improved agent
    that uses a Deep Q-Network (DQN) to learn to perform multiple tasks.
    """

    def __init__(self, agent_id: int, num_tasks: int, num_features: int):
        """
        Initializes a new instance of the ImprovedAgent class.

        Args:
        - agent_id (int): The ID of the agent.
        - num_tasks (int): The number of tasks the agent can perform.
        - num_features (int): The number of features in the state space.
        """
        self.agent_id = agent_id
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.models: List[LifelongLearningModel] = [
            LifelongLearningModel(num_features) for _ in range(num_tasks)]

        # DQN-related attributes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.target_net: DQN = DQN(num_features).to(self.device)
        self.policy_net: DQN = DQN(num_features).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer: optim.RMSprop = optim.RMSprop(self.policy_net.parameters())
        self.memory: ReplayMemory = ReplayMemory(10000)
        self.previous_loss: List[float] = [0] * num_tasks

    def observe(self, task: int) -> Tuple[float, float]:
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
        reward = self.calculate_reward(loss, task)
        self.update_dqn(observations, reward, task)
        return loss, reward

    def calculate_reward(self, loss: float, task: int) -> float:
        """
        Calculates the reward for the agent based on the difference between the current and previous loss.

        Args:
            loss (float): The current loss value for the task.
            task (int): The task ID of the current task.

        Returns:
            float: The reward value for the agent.
        """
        reward = self.previous_loss[task] - loss
        self.previous_loss[task] = loss
        return reward

    def update_dqn(self, state: torch.Tensor, reward: float, task: int) -> None:
        """
        Updates the Deep Q-Network (DQN) by training it on a batch of transitions from the replay memory.

        Args:
            state (torch.Tensor): The current state of the agent.
            reward (float): The reward received by the agent for taking the last action.
            task (int): The task ID of the current task.

        Returns:
            None
        """
        self.memory.push(state, task, None, reward)
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        if any(non_final_mask):
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
