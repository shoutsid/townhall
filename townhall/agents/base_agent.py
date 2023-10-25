"""
The `base_agent` module contains the `BaseAgent` class, which is the base class for all
assistant agents in Townhall
"""

from autogen import AssistantAgent

import settings

class BaseAgent(AssistantAgent):
    """
    The `BaseAgent` class is the base class for all assistant agents in Townhall
    It inherits from the `AssistantAgent` class and provides a starting point for implementing
    custom assistant agents.
    """
    def __init__(self, name=None, llm_config=None, **kwargs):
        if name is None:
            name = "DefaultAgentName"
        if llm_config is None:
            llm_config = settings.LLM_CONFIG
        super().__init__(name=name, llm_config=llm_config, **kwargs)
