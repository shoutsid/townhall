"""
Product Manager Agent
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
You are an experienced product manager at a tech company.
You take instructions from your peers, most notably the User. You are responsible for
manging existing and new products. You are responsible for the success of the product.
Outline the key steps you would take to ensure the successful development, launch, and
maintenance of a new product. You may assume that you have access to a team of engineers,
scientists, and peers to interact with.
Consider aspects like market research, defining the MVP (Minimum Viable Product),
feature prioritization, user feedback, and project management methodologies you might employ.
"""

class ProductManager(BaseAgent):
    """
    A class representing a Python engineer agent.

    Attributes:
    - name (str): The name of the agent.
    - system_prompt (str): The system prompt message.
    """
    def __init__(self, name: str | None = None, system_message: None | str = None, **kwargs):
        if name is None:
            name = "ProductManager"
        if system_message is None:
            system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Product Manager agent')
    parser.add_argument(
        '--system-prompt', type=str, help='the system prompt message passed to the product manager')
    args = parser.parse_args()
    prompt = args.system_prompt  if args.system_prompt else SYSTEM_PROMPT

    CHAT_SERVICE = ChatService(
        settings.CONFIG_LIST,
        assistants=[
            ProductManager(system_message=prompt),
            BaseAgent(
                name="Engineer",
                system_message='''
                Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
                Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
                If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
                ''',
            ),
            UserAgent(
                name="Executor",
                system_message =
                    "Executor. Execute the code written by the engineer and report the result.",
                human_input_mode="NEVER",
                code_execution_config={"last_n_messages": 3, "work_dir": "agent_workplace"}
            )
        ],
        user_proxy=UserAgent()
    )
    print(f"Product Manager System Prompt: {SYSTEM_PROMPT} \n")
    message = input("Enter a message to send to the Product Manager: ")
    CHAT_SERVICE.initiate_chat(message)
