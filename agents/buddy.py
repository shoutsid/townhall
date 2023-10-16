"""
This module contains code for creating an instance of the AssistantAgent and UserProxyAgent classes,
and initiating a chat between them.
"""


import threading
from autogen import AssistantAgent, UserProxyAgent
from settings import (
    GPT3_5_TURBO_0613,
    CONFIG_LIST,
)


class UserAgent(UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_message = None
        self.message_event = threading.Event()

    def set_current_message(self, message: str):
        """Sets the current message from the user and signals that a message is available."""
        self.current_message = message
        self.message_event.set()

    def get_human_input(self, prompt: str) -> str:
        """Blocks until a message is available, then returns it."""
        self.message_event.wait()  # Block until a message is available
        message = self.current_message
        self.message_event.clear()  # Reset the event for the next message
        return message


class Buddy:
    """A class representing a chatbot for chatting with the assistant."""

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
        """Initiates a chat between the user and the assistant."""
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
            self.user_proxy.initiate_chat(recipient=self.assistant, message=message)

        # Store the assistant's reply to the conversation context
        reply = self.user_proxy.last_message()["content"]
        self.conversation_context.append({"role": "assistant", "content": reply})

        return reply
