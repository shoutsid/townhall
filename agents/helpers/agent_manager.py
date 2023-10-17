class AgentManager:
    """
    A class to manage agents in a registry.

    Attributes:
      registry (dict): A dictionary to store agent details.

    Methods:
      register_agent: Registers an agent with the given ID and details.
      unregister_agent: Unregisters an agent with the given ID.
      query_agent: Returns the details of an agent with the given ID.
      list_agents: Returns a dictionary of all registered agents and their details.
    """

    def __init__(self, inter_agent_comm):
        self.registry = {}
        self.inter_agent_comm = inter_agent_comm

    async def register_agent(self, agent_id, details):
        """
        Registers a new agent with the given ID and details.

        Args:
          agent_id (str): The ID of the agent to register.
          details (dict): A dictionary containing details about the agent.

        Returns:
          str: A message indicating that the agent was registered successfully.
        """
        self.registry[agent_id] = {"details": details, "messages": []}
        return f"Agent {agent_id} registered successfully."

    async def unregister_agent(self, agent_id):
        """
        Unregister an agent from the agent manager.

        Args:
            agent_id (str): The ID of the agent to unregister.

        Returns:
            str: A message indicating whether the agent was successfully unregistered or not.
        """
        if agent_id in self.registry:
            del self.registry[agent_id]
            return f"Agent {agent_id} unregistered successfully."
        else:
            return f"Agent {agent_id} not found."

    async def query_agent(self, agent_id):
        """
        Returns the details of the agent with the given ID.

        Args:
            agent_id (str): The ID of the agent to query.

        Returns:
            dict: A dictionary containing the details of the agent, or a string
                  indicating that the agent was not found.
        """
        return self.registry.get(agent_id, {}).get(
            "details", f"Agent {agent_id} not found."
        )

    async def list_agents(self):
        """
        Returns a dictionary containing the details of all registered agents.
        The keys of the dictionary are the agent IDs, and the values are the agent details.
        """
        return {k: v["details"] for k, v in self.registry.items()}
