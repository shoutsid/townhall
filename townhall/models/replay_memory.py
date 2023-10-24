"""
Replay memory module, which contains the ReplayMemory class for storing transitions.
"""

from typing import List, Tuple
import random
from townhall.models.transition import Transition


class ReplayMemory:
    """
    A cyclic buffer of fixed size that stores transitions observed from the environment.
    The transitions can be sampled randomly to train a reinforcement learning agent.

    Args:
        capacity (int): The maximum number of transitions that can be stored in the memory.

    Attributes:
        capacity (int): The maximum number of transitions that can be stored in the memory.
        memory (list): A list of transitions stored in the memory.
        position (int): The index of the next available slot in the memory.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.memory: List[Transition] = []
        self.position: int = 0

    def push(self, *args: Tuple) -> None:
        """
        Saves a transition to the replay memory.

        Args:
            *args: The transition to be saved.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Randomly samples a batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the length of the replay memory.

        Returns:
            int: The length of the replay memory.
        """
        return len(self.memory)
