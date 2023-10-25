"""
Scientist Agent
"""
# pylint: disable=wrong-import-position

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import settings
from townhall.services.chat_service import ChatService
from townhall.agents.base_agent import BaseAgent
from townhall.agents.user_agent import UserAgent

SYSTEM_PROMPT = """
Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.
Use the function tools at your disposal, including the ability to search for papers, to categorize the papers.
"""

class Scientist(BaseAgent):
    """
    A class representing a Scientist agent.

    Attributes:
    - name (str): The name of the agent.
    - system_prompt (str): The system prompt message.
    """
    def __init__(self, name: str = None, system_message: str = None, **kwargs):
        if name is None:
            name = "Scientist"
        if system_message is None:
            system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Planner agent')
    parser.add_argument(
        '--system-prompt',
        type=str,
        help='the system prompt message passed to the Planner which the Scientist will follow'
    )
    args = parser.parse_args()
    prompt = args.system_prompt  if args.system_prompt else SYSTEM_PROMPT

    from townhall.agents.planner import Planner

    CHAT_SERVICE = ChatService(
        settings.CONFIG_LIST, assistants=[
            Planner(system_message=prompt),
            Scientist(system_message=prompt)
        ], user_proxy=UserAgent())
    print(f"Planner System Prompt: {SYSTEM_PROMPT} \n")
    message = input("Enter a message to send to the Scientist: ")
    CHAT_SERVICE.initiate_chat(message)


