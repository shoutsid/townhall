"""
Scientist Agent
"""

from townhall.agents.base_agent import BaseAgent

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
    def __init__(self, name: str | None = None, system_message: str | None = None, **kwargs):
        if name is None:
            name = "Scientist"
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        super().__init__(name=name, system_message=system_message, **kwargs)
