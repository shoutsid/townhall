"""
A module that contains the ChatService class,
which represents a chat service between a user and an assistant agent.
"""

from typing import List
from IPython.core.interactiveshell import InteractiveShell
from autogen import AssistantAgent, GroupChat, GroupChatManager
from townhall.agents.user_agent import UserAgent
from townhall.agents.utils import is_termination_msg
from townhall.services.function_service import FunctionService
from settings import CONFIG_LIST, LLM_CONFIG

class ChatService:
    """
    A class that represents a chat service between a user and an assistant agent.

    Attributes:
      assistant (AssistantAgent): The assistant agent used in the chat service.
      user_proxy (UserAgent): The user agent used in the chat service.
    """

    def __init__(
            self,
            config_list: dict = None,
            assistants: List = None,
            user_proxy: UserAgent = None
    ):
        if config_list is None:
            config_list = CONFIG_LIST

        self.function_service = FunctionService()

        if assistants is None:
            self.assistants = [
                AssistantAgent(
                    name="assistant",
                    system_message=("""
                        For coding tasks, only use the functions you have been provided with.
                        You argument should follow json format. Reply TERMINATE when the task is done.
                    """),
                    llm_config={
                        "config_list": config_list,
                        "functions": self.function_service.openai_functions_list
                    }
                )
            ]

        if user_proxy is None:
            self.user_proxy = UserAgent(
                name="user_proxy",
                is_termination_msg=is_termination_msg,
                max_consecutive_auto_reply=10
            )
        else:
            self.user_proxy = user_proxy

        self.setup_functions()

        self.messages: List = []

        # Setup group chat and manager
        self.chat = GroupChat(agents=[user_proxy, *assistants], messages=self.messages, max_round=50)
        self.manager = GroupChatManager(groupchat=self.chat, llm_config=LLM_CONFIG)

    def setup_functions(self):
        """
        Registers the available functions for the user proxy.

        Available functions:
        - python: Executes a Python code snippet.
        - sh: Executes a shell command.
        """
        self.user_proxy.register_function(
                function_map={
                    "python": self.exec_python,
                    "sh": self.exec_sh,
                }
            )

    def exec_python(self, cell):
        """
        Executes a Python code cell using IPython and returns the log of the execution.

        Args:
            cell (str): The Python code cell to execute.

        Returns:
            str: The log of the execution, including any output or errors.
        """
        shell = InteractiveShell.instance()
        result = shell.run_cell(cell)
        log = str(result.result)
        if result.error_before_exec is not None:
            log += f"\n{result.error_before_exec}"
        if result.error_in_exec is not None:
            log += f"\n{result.error_in_exec}"
        return log

    def exec_sh(self, script):
        """
        Executes a shell script using the user proxy.

        Args:
            script (str): The shell script to execute.

        Returns:
            The result of the shell script execution.
        """
        return self.user_proxy.execute_code_blocks([("sh", script)])

    def initiate_chat(self, message, clear_history: bool | None = True):
        """
        Initiates a chat session between the user and the assistant.

        Args:
          message (str): The initial message from the user.

        Returns:
          None
        """
        self.user_proxy.initiate_chat(self.manager, message=message, clear_history=clear_history)
