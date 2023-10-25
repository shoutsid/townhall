"""
This module defines the UserAgent class, which represents a user agent that can send messages
 to the system.
"""

import threading
from autogen import UserProxyAgent


class UserAgent(UserProxyAgent):
    """A class representing a user agent that can send messages to the system.

    Attributes:
      current_message (str): The current message from the user.
      message_event (threading.Event): An event that signals when a message is available.

    Methods:
      set_current_message(message: str) -> None: Sets the current message from the user and
                                                  signals that a message is available.
      get_human_input(prompt: str) -> str: Blocks until a message is available, then returns it.
    """

    def __init__(self, name: str = None, **kwargs):
        if name is None:
            name: str = "User"
        super().__init__(name=name, **kwargs)
        self.current_message = None
        self.message_event = threading.Event()

    def set_current_message(self, message: str) -> None:
        """
        Sets the current message from the user and signals that a message is available.

        Args:
          message (str): The message to set as the current message.

        Returns:
          None
        """
        self.current_message = message


    def get_human_input(self, prompt: str) -> str:
        """
        Blocks until a message is available, then returns it.

        Args:
          prompt (str): The prompt to display to the user.

        Returns:
          str: The message entered by the user.
        """
        self.set_current_message(input(prompt))
        message = self.current_message
        return message
