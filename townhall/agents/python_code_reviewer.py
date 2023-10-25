"""
Python Code Reviewer Agent
"""
# pylint: disable=wrong-import-position

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import settings
from townhall.services.chat_service import ChatService
from townhall.agents.user_agent import UserAgent
from townhall.agents.base_agent import BaseAgent


SYSTEM_PROMPT = """
Imagine you are a senior Python code reviewer responsible for ensuring the quality and maintainability of a Python codebase.
You've received a pull request for a complex module that has been developed by a team member.
Describe your systematic approach to reviewing this code, highlighting the key aspects you would focus on, and providing guidance on best practices for code structure, readability, and maintainability.
Consider factors like code style, documentation, unit testing, and potential code smells.
"""

class PythonCodeReviewer(BaseAgent):
    """
    A class representing a Python engineer agent.

    Attributes:
    - name (str): The name of the agent.
    - system_prompt (str): The system prompt message.
    """
    def __init__(self, name: str | None = None, system_message: str | None = None, **kwargs):
        if name is None:
            name = "PythonCodeReviewer"
        if system_message is None:
            system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python Engineer agent')
    parser.add_argument(
        '--system-prompt',
        type=str,
        help='the system prompt message passed to the python code reviewer')
    args = parser.parse_args()
    prompt = args.system_prompt  if args.system_prompt else SYSTEM_PROMPT

    CHAT_SERVICE = ChatService(
        settings.CONFIG_LIST,
        assistants=[
            BaseAgent(
                name="Engineer",
                system_message='''
                Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code cha
                '''
            ),
            PythonCodeReviewer(system_message=prompt),
        ],
        user_proxy=UserAgent()
    )
    print(f"Prompt: {SYSTEM_PROMPT} \n")
    message = input("Enter a message to send to the Python Code Reviewer: ")
    CHAT_SERVICE.initiate_chat(message)
