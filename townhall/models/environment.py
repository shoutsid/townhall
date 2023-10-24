"""
A module containing the Environment class, which represents the environment in which agents operate.
"""

import numpy as np
from townhall.models.improved_agent import ImprovedAgent


class Environment:
    """
    A class representing the environment in which agents operate.

    Attributes:
        num_agents (int): The number of agents in the environment.
        num_tasks (int): The number of tasks in the environment.
        num_features (int): The number of features in the environment.
        agents (List[ImprovedAgent]): A list of ImprovedAgent objects representing the agents in the environment.
        performance (Dict[int, Dict[int, List[float]]]): A dictionary containing the performance of each agent on each task.
        agent_positions (Dict[int, List[np.ndarray]]): A dictionary containing the positions of each agent.
        rewards (Dict[int, Dict[int, List[float]]]): A dictionary containing the rewards earned by each agent on each task.
        log_dir (str): The directory for TensorBoard logs.
    """

    def __init__(self, num_agents: int, num_tasks: int, num_features: int, log_dir: str):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.agents = [ImprovedAgent(i, num_tasks, num_features)
                       for i in range(num_agents)]
        self.performance = {i: {task: []
                                for task in range(num_tasks)} for i in range(num_agents)}
        self.agent_positions = {i: [] for i in range(num_agents)}
        self.rewards = {i: {task: []
                            for task in range(num_tasks)} for i in range(num_agents)}
        self.log_dir = log_dir  # Directory for TensorBoard logs

    def step(self) -> None:
        """
        Runs a single step of the environment simulation, where each agent observes each task and updates its performance
        and rewards accordingly. The agent's position is also updated randomly in 2D space.
        """
        for agent in self.agents:
            for task in range(self.num_tasks):
                loss, reward = agent.observe(task)
                self.performance[agent.agent_id][task].append(loss)
                self.rewards[agent.agent_id][task].append(reward)
            self.agent_positions[agent.agent_id].append(
                np.random.rand(2))  # Random 2D positions
