"""
This module contains the GroupChatService class,
which is responsible for managing group chat sessions between users and assistants.
"""

from typing import List, Dict, Callable
# import autogen
from autogen import GroupChat, GroupChatManager
from townhall.agents.user_agent import RAGUserAgent
from townhall.agents.utils import is_termination_msg
from settings import CONFIG_LIST

class GroupChatService:
    """
    A service class for managing group chat sessions between users and assistants.
    """

    def __init__(
        self,
        config_list: Dict = None,
        function_map: List[Callable] = None,
        agents: List[str] = None
    ):
        self.config_list = config_list if config_list is not None else {}
        self.function_map = function_map if function_map is not None else []
        self.agents = agents if agents is not None else []
        self.messages = []
        self.user_agent = RAGUserAgent(
            name="User",
            is_termination_msg=is_termination_msg,
            human_input_mode="TERMINATE",
            system_message=
                "The user who ask questions and give tasks, and is the ultimate authority.",
        )
        self.group_chat = GroupChat(
            agents=agents, messages=self.messages, max_round=12
        )
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=CONFIG_LIST
        )

    def initiate_chat(self, message: str):
        """
            Initiates a chat session between the user and the assistant.

            Args:
                message (str): The initial message from the user.

            Returns:
                None
        """
        self.user_agent.initiate_chat(self.group_chat_manager, problem=message)