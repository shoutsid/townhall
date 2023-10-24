"""
Python Code Reviewer Agent
"""

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
    def __init__(self, name="PythonEngineer", **kwargs):
        self.system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_prompt=SYSTEM_PROMPT, **kwargs)