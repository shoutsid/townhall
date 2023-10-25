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
Critic. Double check plan, claims, code from other agents and provide feedback.
Check whether the plan includes adding verifiable info such as source URL.
"""

class Critic(BaseAgent):
    """
    A class representing a critic agent in the Townhall system.

    Attributes:
    - name (str): the name of the agent
    - system_message (str): the system message to display when the agent sends a message
    - kwargs: additional keyword arguments to pass to the BaseAgent constructor
    """
    def __init__(self, name: str = None, system_message: str = None, **kwargs):
        if name is None:
            name = "Critic"
        if system_message is None:
            system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)

