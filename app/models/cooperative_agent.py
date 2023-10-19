"""
This module contains the base class for cooperative agents in the Townhall simulation.
"""

import torch
import random
import numpy as np
from app.models.message import Message


class BaseCooperativeAgent:
    """
    Base class for cooperative agents in the Townhall simulation.

    Attributes:
        id (int): The unique identifier of the agent.
        grid_size (int): The size of the grid in which the agent operates.
        position (tuple): The current position of the agent on the grid.
        positions (list): A list of all positions the agent has visited.
    """

    def __init__(self, id, grid_size, position=None):
        self.id = id
        self.grid_size = grid_size
        if position is None:
            self.position = self.random_position()
        else:
            self.position = position
        self.positions = [self.position]

    def random_position(self):
        """
        Returns a random position on the grid.

        Returns:
            tuple: A tuple representing the x and y coordinates of the position.
        """
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def take_action(self, targets):
        """
        Takes an action based on the current state of the simulation.

        Args:
            targets (list): A list of targets in the simulation.

        Raises:
            NotImplementedError: This method should be overridden by subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclass")

    def calculate_reward(self, targets):
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

    def __init__(self, id, grid_size, num_features, position=None):
        super().__init__(id, grid_size, position=position)
        self.num_features = num_features

    def take_action(self, targets):
        epsilon = 0.1
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            state = torch.tensor(
                self.position, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = targets(state).max(1)[1].item()

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

    def calculate_reward(self, targets):
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
    - send_message(targets): Sends a message to the specified targets.
    - receive_messages(messages): Receives messages from other agents.
    - take_action(targets): Takes an action based on the received messages and the current position of the agent.
    - calculate_reward(targets): Calculates the reward based on the distance traveled towards the targets.
    """

    def __init__(self, id, grid_size, position=None):
        super().__init__(id, grid_size, position=position)
        self.messages = []

    def send_message(self, targets):
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

    def receive_messages(self, messages):
        """
        Receives a list of messages and sets the agent's messages attribute to the given list.

        Args:
            messages (list): A list of messages to be received by the agent.

        Returns:
            None
        """
        self.messages = messages

    def take_action(self, targets):
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

        chosen_target_index = target_priorities.index(min(target_priorities))
        chosen_target = targets[chosen_target_index]

        new_position = (self.position[0] + np.sign(chosen_target[0] - self.position[0]),
                        self.position[1] + np.sign(chosen_target[1] - self.position[1]))
        self.position = new_position
        self.positions.append(new_position)

    def calculate_reward(self, targets):
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
