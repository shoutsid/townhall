"""
This module contains code for creating an instance of the AssistantAgent and UserProxyAgent classes,
and initiating a chat between them.
"""


from autogen import AssistantAgent
from settings import (
    GPT3_5_TURBO_0613,
    CONFIG_LIST,
)
from app.agents.user_agent import UserAgent


class Buddy:
    """
    A class representing a chatbot for chatting with the assistant.

    Attributes:
        conversation_context (list): A list of dictionaries representing the conversation context.
        assistant (AssistantAgent): An instance of the AssistantAgent class.
        user_proxy (UserAgent): An instance of the UserAgent class.
    """

    def __init__(self):
        self.conversation_context = []
        self.assistant = AssistantAgent(
            name="assistant",
            llm_config={
                "temperature": 0,
                "request_timeout": 600,
                "seed": 42,
                "model": GPT3_5_TURBO_0613,
                "config_list": CONFIG_LIST,
            },
        )
        self.user_proxy = UserAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "workspace"},
        )

    def start(self, message: str):
        """
        Initiates a chat between the user and the assistant.

        Args:
            message (str): The message sent by the user.

        Returns:
            str: The reply sent by the assistant.
        """
        # Store the user's message to the conversation context
        self.conversation_context.append({"role": "user", "content": message})

        # Set the current_message for UserAgent
        self.user_proxy.current_message = message

        # If there's an existing conversation context, continue the conversation
        if len(self.conversation_context) > 1:  # Adjusted this line
            self.user_proxy.send(
                message=message, recipient=self.assistant, request_reply=True
            )
        else:
            self.user_proxy.initiate_chat(
                recipient=self.assistant, message=message)

        # Store the assistant's reply to the conversation context
        reply = self.user_proxy.last_message()["content"]
        self.conversation_context.append(
            {"role": "assistant", "content": reply})

        return reply
