"""
Python Engineer Agent
"""

from townhall.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """
You are a seasoned EXPERT PYTHON DEVELOPER tasked with optimizing a performance-critical Python code.
The script processes a large dataset, and the goal is to make it run as efficiently as possible.
Describe your step-by-step approach to identifying and addressing performance bottlenecks in the code.
Consider factors like algorithm complexity, data structures, profiling, and optimization techniques.
"""

class PythonEngineer(BaseAgent):
    """
    A class representing a Python engineer agent.

    Attributes:
    - name (str): The name of the agent.
    - system_prompt (str): The system prompt message.
    """
    def __init__(self, name="PythonEngineer", **kwargs):
        self.system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_prompt=SYSTEM_PROMPT, **kwargs)