"""
This module contains classes for multi-agent environments.
"""


import random
from app.models import CooperativeAgent2


class BaseMultiAgentEnvironment:
    """
    Base class for multi-agent environments.

    Args:
        num_agents (int): Number of agents in the environment.
        num_targets (int): Number of targets in the environment.
        grid_size (int): Size of the grid.
        num_features (int): Number of features for each agent.

    Attributes:
        num_agents (int): Number of agents in the environment.
        num_targets (int): Number of targets in the environment.
        grid_size (int): Size of the grid.
        agents (list): List of CooperativeAgent objects.
        targets (list): List of target positions.
    """

    def __init__(self, num_agents, num_targets, grid_size, num_features):
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.grid_size = grid_size
        self.agents = [CooperativeAgent2(
            i, grid_size, num_features) for i in range(num_agents)]
        self.targets = [self.random_position() for _ in range(num_targets)]

    def random_position(self):
        """
        Returns a random position within the grid.

        Returns:
            tuple: A tuple containing the x and y coordinates of the position.
        """
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def step(self):
        """
        This method should be overridden by subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclass")


class MultiAgentEnvironment1(BaseMultiAgentEnvironment):
    """
    A multi-agent environment where each agent takes an action and receives a reward based on the state of the environment.

    Attributes:
        agents (list): A list of Agent objects representing the agents in the environment.
        targets (list): A list of Target objects representing the targets in the environment.
    """

    def step(self):
        """
        Executes one step of the environment, where each agent takes an action and receives a reward based on the state of the environment.

        Returns:
            list: A list of rewards, where each reward corresponds to an agent in the environment.
        """
        for agent in self.agents:
            agent.take_action(self.targets)
        rewards = [agent.calculate_reward(self.targets)
                   for agent in self.agents]
        return rewards


class MultiAgentEnvironment2(BaseMultiAgentEnvironment):
    """
    A multi-agent environment that simulates interactions between agents and targets.

    This environment contains a list of agents and a list of targets. On each step, each agent sends a message to all targets,
    receives messages from all other agents, takes an action based on the received messages, and calculates a reward based on
    its action and the current state of the targets. The rewards for all agents are returned as a list.

    Attributes:
        agents (list): A list of Agent objects representing the agents in the environment.
        targets (list): A list of Target objects representing the targets in the environment.
    """

    def step(self):
        messages = [agent.send_message(self.targets) for agent in self.agents]
        for agent in self.agents:
            agent.receive_messages(messages)
            agent.take_action(self.targets)
        rewards = [agent.calculate_reward(self.targets)
                   for agent in self.agents]
        return rewards


class MultiAgentEnvironment3(BaseMultiAgentEnvironment):
    """
    A multi-agent environment that extends the BaseMultiAgentEnvironment class.

    This environment contains a list of agents that can take actions and receive rewards based on their current position
    and the positions of other agents and targets in the environment.

    Attributes:
        agents (list): A list of Agent objects representing the agents in the environment.
        targets (list): A list of Target objects representing the targets in the environment.
    """

    def step(self, policy_net, memory):
        """
        Takes a step in the environment by having each agent take an action and updating the memory with the results.

        Args:
            policy_net (nn.Module): A neural network that takes in an observation and outputs an action.
            memory (ReplayMemory): A replay memory object that stores experiences for the agents.

        Returns:
            None
        """
        for agent in self.agents:
            current_position = agent.position
            action = agent.take_action(policy_net)
            next_position = agent.position
            reward = agent.calculate_reward(self.targets)
            memory.push(current_position, action, next_position, reward)

    def reset(self):
        """
        Resets the environment by randomly placing each agent in a new position.

        Returns:
            None
        """
        for agent in self.agents:
            agent.position = self.random_position()
            agent.positions = [agent.position]
