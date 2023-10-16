"""
This module contains code for creating an instance of the AssistantAgent and UserProxyAgent classes,
and initiating a chat between them.
"""

from autogen import AssistantAgent, UserProxyAgent
from settings import (
    GPT3_5_TURBO_0613,
    CONFIG_LIST,
)


class Buddy:
    """
    A class representing a chatbot that can initiate a chat between a user and an assistant agent.

    Attributes:
    None

    Methods:
    start(): Initializes an instance of the AssistantAgent and UserProxyAgent classes,
              prompts the user to enter a message to send to the assistant,
              and initiates a chat between the user_proxy and assistant.
    """

    def __init__(self):
        pass

    def start(self, message: str):
        """
        Creates an instance of the AssistantAgent class with the specified configuration settings.
        Creates an instance of the UserProxyAgent class with the specified configuration settings.
        Prompts the user to enter a message to send to the assistant,
        then initiates a chat between the user_proxy and assistant.
        """
        assistant = AssistantAgent(
            name="assistant",
            llm_config={
                "temperature": 0,
                "request_timeout": 600,
                "seed": 42,
                "model": GPT3_5_TURBO_0613,
                "config_list": CONFIG_LIST,
            },
        )

        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "workspace"},
        )

        user_proxy.initiate_chat(assistant, message=message)
