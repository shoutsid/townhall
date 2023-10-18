from app.agents.user_agent import UserAgent
from autogen import AssistantAgent


class ChatService:
    """
    A class that represents a chat service between a user and an assistant agent.

    Attributes:
      assistant (AssistantAgent): The assistant agent used in the chat service.
      user_proxy (UserAgent): The user agent used in the chat service.
    """

    def __init__(self, config_list, function_map, assistant=None, user_proxy=None):
        if assistant is None:
            self.assistant = AssistantAgent(
                name="assistant",
                llm_config={"config_list": config_list,
                            "functions": function_map},
            )
        else:
            self.assistant = assistant

        if user_proxy is None:
            self.user_proxy = UserAgent(
                name="user_proxy",
                human_input_mode="TERMINATE",
                max_consecutive_auto_reply=0,
                function_map=function_map,
            )
        else:
            self.user_proxy = user_proxy

    def initiate_chat(self, message):
        """
        Initiates a chat session between the user and the assistant.

        Args:
          message (str): The initial message from the user.

        Returns:
          None
        """
        self.user_proxy.initiate_chat(self.assistant, message=message)
