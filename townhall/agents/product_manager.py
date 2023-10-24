"""
Product Manager Agent
"""

from townhall.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """
You are an experienced product manager at a tech company.
Your team is about to start developing a new product from scratch.
Outline the key steps you would take to ensure the successful development and launch of this product.
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
    def __init__(self, name="PythonEngineer", **kwargs):
        self.system_message = SYSTEM_PROMPT
        super().__init__(name=name, system_prompt=SYSTEM_PROMPT, **kwargs)