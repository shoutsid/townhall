"""
A module to manage the registration and querying of agents.
"""


class AgentRegistry:
    """
    A class to manage the registration and querying of agents.

    Attributes:
      registry (dict): A dictionary to store agent details.
    """

    def __init__(self):
        self.registry = {}

    async def register_agent(self, agent_id, details):
        """
        Register an agent with a given ID and details.

        Args:
          agent_id (str): The ID of the agent to register.
          details (dict): A dictionary containing details about the agent.

        Returns:
          str: A message indicating whether the agent was registered successfully.
        """
        self.registry[agent_id] = details
        return f"Agent {agent_id} registered successfully."

    async def unregister_agent(self, agent_id):
        """
        Unregister an agent with a given ID.

        Args:
          agent_id (str): The ID of the agent to unregister.

        Returns:
          str: A message indicating whether the agent was unregistered successfully.
        """
        if agent_id in self.registry:
            del self.registry[agent_id]
            return f"Agent {agent_id} unregistered successfully."
        else:
            return f"Agent {agent_id} not found."

    async def query_agent(self, agent_id):
        """
        Query details about a specific agent.

        Args:
          agent_id (str): The ID of the agent to query.

        Returns:
          dict or str: A dictionary containing details about the agent, or a message indicating that the agent was not found.
        """
        return self.registry.get(agent_id, f"Agent {agent_id} not found.")

    async def list_agents(self):
        """
        List all registered agents.

        Returns:
          dict: A dictionary containing details about all registered agents.
        """
        return self.registry
