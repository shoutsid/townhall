"""
    This module contains the Message class.
    A Message object represents a message sent by a life long model.
"""

from typing import List, Tuple


class Message:
    """
    A class representing a message sent by a life long model

    Attributes:
        sender_id (int): The ID of the user who sent the message.
        position (tuple): The (x, y) position of the user who sent the message.
        perceived_distances (list): A list of perceived distances between the user who sent the message and other users.
    """

    def __init__(self, sender_id: int, position: Tuple[int, int], perceived_distances: List[float]):
        self.sender_id = sender_id
        self.position = position
        self.perceived_distances = perceived_distances
