"""
This module defines the UserAgent class, which represents a user agent that can send messages
 to the system.
"""

import threading
from typing import Optional, Dict, Callable
from autogen import UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class RAGUserAgent(RetrieveUserProxyAgent):
    """
    A user agent class that inherits from the RetrieveUserProxyAgent class.
    This class is responsible for setting and getting the current message from the user.

    Args:
            name (str): The name of the agent.
            human_input_mode (Optional[str]): The mode of human input. Defaults to "ALWAYS".
            is_termination_msg (Optional[Callable[[Dict], bool]]):
                A callable function that returns a boolean value indicating whether a message
                is a termination message. Defaults to None.
            retrieve_config (Optional[Dict]):
                A dictionary containing the configuration for the retrieval model. Defaults to None.
            **kwargs: Additional keyword arguments.

    Attributes:
            current_message (str): The current message from the user.

    Methods:
            set_current_message(message: str) -> None:
                Sets the current message from the user and signals that a message is available.
            get_human_input(prompt: str) -> str:
                Blocks until a message is available, then returns it.
    """

    def __init__(
            self,
            name="RAGUserAgent",
            human_input_mode: Optional[str] = "ALWAYS",
            is_termination_msg: Optional[Callable[[Dict], bool]] = None,
            retrieve_config: Optional[Dict] = None,
            **kwargs,
    ):
        self.current_message: str = None
        super().__init__(
                name=name,
                human_input_mode=human_input_mode,
                is_termination_msg=is_termination_msg,
                retrieve_config=retrieve_config,
                **kwargs,
        )

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
