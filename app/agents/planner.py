"""
This module contains the main function to initiate a chat with the assistant.
"""

# add project root to path to allow imports
import sys
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
sys.path.append(str(Path(__file__).resolve().parents[2]))

from settings import (
    CONFIG_LIST,
)
from app.services.planner_service import PlannerService
from app.services.function_service import FunctionService
from app.services.chat_service import ChatService


PLANNER_SERVICE = PlannerService(CONFIG_LIST)
FUNCTION_SERVICE = FunctionService()
CHAT_SERVICE = ChatService(
    CONFIG_LIST, FUNCTION_SERVICE.openai_functions_list)

def exec_python(cell):
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


def exec_sh(script):
    """
    Executes a shell script using the user proxy.

    Args:
        script (str): The shell script to execute.

    Returns:
        The result of the shell script execution.
    """
    return CHAT_SERVICE.user_proxy.execute_code_blocks([("sh", script)])



if __name__ == "__main__":

    CHAT_SERVICE.user_proxy.register_function(
        function_map={
            "python": exec_python,
            "sh": exec_sh,
        }
    )

    message = input("Enter a message to send to the assistant: ")
    CHAT_SERVICE.initiate_chat(message)
