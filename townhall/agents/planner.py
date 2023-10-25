"""
This module contains the main function to initiate a chat with the assistant.
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
Planner. Suggest a plan. Revise the plan based on feedback from User and your peers, until User approval.
The plan may involve an Engineer who can write code and a Scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by your peers, and which step is performed by which peer.
"""

class Planner(BaseAgent):
    """
    A class representing a Planner agent.

    Attributes:
    - name (str): The name of the agent.
    - system_prompt (str): The system prompt message.
    """

    def __init__(self, name: str = None, system_message: str = None, **kwargs):
        if name is None:
            name = "Planner"
        if system_message is None:
            system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Planner agent')
    parser.add_argument(
        '--system-prompt', type=str, help='the system prompt message passed to the planner')
    args = parser.parse_args()
    prompt = args.system_prompt  if args.system_prompt else SYSTEM_PROMPT

    CHAT_SERVICE = ChatService(
        settings.CONFIG_LIST, assistants=[
            Planner(system_message=prompt)
        ], user_proxy=UserAgent())
    print(f"Planner System Prompt: {SYSTEM_PROMPT} \n")
    message = input("Enter a message to send to the Planner: ")
    CHAT_SERVICE.initiate_chat(message)
