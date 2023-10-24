"""
This module contains the base class for cooperative agents in the Townhall simulation.
"""

import secrets
import numpy as np
from typing import List, Optional, Tuple
from tinygrad.tensor import Tensor
from townhall.models.message import Message


class BaseCooperativeAgent:
    """
    Base class for cooperative agents in the Townhall simulation.

    Attributes:
        id (int): The unique identifier of the agent.
        grid_size (int): The size of the grid in which the agent operates.
        position (tuple): The current position of the agent on the grid.
        positions (list): A list of all positions the agent has visited.
    """

    def __init__(self, agent_id: int, grid_size: int, position: Optional[Tuple[int, int]] = None) -> None:
        self.id = agent_id
        self.grid_size = grid_size
        if position is None:
            self.position = self.random_position()
        else:
            self.position = position
        self.positions = [self.position]

    def random_position(self) -> Tuple[int, int]:
        """
        Returns a random position on the grid.

        Returns:
            tuple: A tuple representing the x and y coordinates of the position.
        """

        return (secrets.randbelow(self.grid_size), secrets.randbelow(self.grid_size))

    def take_action(self, targets: List[Tuple[int, int]]) -> None:
        """
        Takes an action based on the current state of the simulation.

        Args:
            targets (list): A list of targets in the simulation.

        Raises:
            NotImplementedError: This method should be overridden by subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclass")

    def calculate_reward(self, targets: List[Tuple[int, int]]) -> None:
        """
        Calculates the reward for the agent based on the current state of the simulation.

        Args:
            targets (list): A list of targets in the simulation.

        Raises:
            NotImplementedError: This method should be overridden by subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclass")


class CooperativeAgent1(BaseCooperativeAgent):
    """
    A cooperative agent that takes actions based on a neural network model.

    Args:
        id (int): The ID of the agent.
        grid_size (int): The size of the grid.
        num_features (int): The number of features in the neural network model.

    Attributes:
        num_features (int): The number of features in the neural network model.
        positions (list): A list of positions the agent has visited.
        position (tuple): The current position of the agent on the grid.

    Methods:
        take_action(targets): Takes an action based on the neural network model.
        calculate_reward(targets): Calculates the reward for the agent based on its current position and the targets.
    """

    def __init__(self, agent_id: int, grid_size: int, num_features: int, position: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(agent_id, grid_size, position=position)
        self.num_features = num_features

    def take_action(self, targets: List[Tuple[int, int]]) -> int:
        epsilon = 0.1
        if secrets.randbelow(100) < epsilon * 100:
            action = secrets.randbelow(4)
        else:
            state = Tensor(
                np.array(self.position, dtype=np.float32)).reshape(1, 2)
            action = targets(state).argmax().item()

        if action == 0 and self.position[1] < self.grid_size - 1:
            self.position = (self.position[0], self.position[1] + 1)
        elif action == 1 and self.position[0] < self.grid_size - 1:
            self.position = (self.position[0] + 1, self.position[1])
        elif action == 2 and self.position[1] > 0:
            self.position = (self.position[0], self.position[1] - 1)
        elif action == 3 and self.position[0] > 0:
            self.position = (self.position[0] - 1, self.position[1])

        self.positions.append(self.position)
        return action

    def calculate_reward(self, targets: List[Tuple[int, int]]) -> float:
        """
        Calculates the reward for the agent based on its current position and the targets.

        Args:
            targets (list): A list of target positions.

        Returns:
            reward (float): The reward for the agent.
        """
        reward = 0
        for target in targets:
            # Calculate the old and new distances to the target
            old_distance = ((self.positions[-2][0] - target[0]) **
                            2 + (self.positions[-2][1] - target[1]) ** 2) ** 0.5
            new_distance = (
                (self.position[0] - target[0]) ** 2 + (self.position[1] - target[1]) ** 2) ** 0.5

            # Provide a positive reward for moving closer, and a negative reward for moving farther away
            reward += old_distance - new_distance

        return reward


class CooperativeAgent2(BaseCooperativeAgent):
    """
    A cooperative agent that sends and receives messages to/from other agents, takes actions based on the received messages,
    and calculates reward based on the distance traveled towards the targets.

    Attributes:
    - id (int): The unique identifier of the agent.
    - grid_size (int): The size of the grid.
    - messages (list): The list of messages received by the agent.
    - positions (list): The list of positions visited by the agent.

    Methods:
    - send_message(targets: List[Tuple[int, int]]) -> Message: Sends a message to the specified targets.
    - receive_messages(messages: List[Message]) -> None: Receives messages from other agents.
    - take_action(targets: List[Tuple[int, int]]) -> None: Takes an action based on the received messages and the current position of the agent.
    - calculate_reward(targets: List[Tuple[int, int]]) -> float: Calculates the reward based on the distance traveled towards the targets.
    """

    def __init__(self, agent_id: int, grid_size: int, position: Tuple[int, int] = None):
        super().__init__(agent_id, grid_size, position=position)
        self.messages: List[Message] = []

    def send_message(self, targets: List[Tuple[int, int]]) -> Message:
        """
        Sends a message to the specified targets.

        Args:
            targets (list): A list of target positions.

        Returns:
            Message: A message object containing the sender's ID, position, and perceived distances to the targets.
        """
        perceived_distances = [((self.position[0] - target[0]) ** 2 +
                                (self.position[1] - target[1]) ** 2) ** 0.5 for target in targets]
        return Message(self.id, self.position, perceived_distances)

    def receive_messages(self, messages: List[Message]) -> None:
        """
        Receives a list of messages and sets the agent's messages attribute to the given list.

        Args:
            messages (list): A list of messages to be received by the agent.

        Returns:
            None
        """
        self.messages = messages

    def take_action(self, targets: List[Tuple[int, int]]) -> None:
        """
        Takes an action for the cooperative agent based on the given targets.

        Args:
            targets (list): A list of tuples representing the coordinates of the targets.

        Returns:
            None
        """
        closest_target_distances = [float('inf')] * len(targets)
        for message in self.messages:
            for i, distance in enumerate(message.perceived_distances):
                closest_target_distances[i] = min(
                    closest_target_distances[i], distance)

        my_distances = [((self.position[0] - target[0]) ** 2 +
                         (self.position[1] - target[1]) ** 2) ** 0.5 for target in targets]
        target_priorities = [my_distance if my_distance <= closest_target_distance else float(
            'inf') for my_distance, closest_target_distance in zip(my_distances, closest_target_distances)]

        # Find the index of the target with the highest priority
        chosen_target_index = target_priorities.index(min(target_priorities))
        chosen_target = targets[chosen_target_index]

        # Calculate the new position based on the chosen target
        new_position = (self.position[0] + np.sign(chosen_target[0] - self.position[0]),
                        self.position[1] + np.sign(chosen_target[1] - self.position[1]))

        # Update the agent's position and positions history
        self.position = new_position
        self.positions.append(new_position)

    def calculate_reward(self, targets: List[Tuple[int, int]]) -> float:
        """
        Calculates the reward for the agent based on the distance to the given targets.

        Args:
            targets (list): A list of target positions in the form [(x1, y1), (x2, y2), ...]

        Returns:
            float: The reward for the agent based on the change in distance to the targets.
        """
        reward = 0
        for target in targets:
            old_distance = ((self.positions[-2][0] - target[0]) **
                            2 + (self.positions[-2][1] - target[1]) ** 2) ** 0.5
            new_distance = (
                (self.position[0] - target[0]) ** 2 + (self.position[1] - target[1]) ** 2) ** 0.5
            reward += old_distance - new_distance
        return reward
