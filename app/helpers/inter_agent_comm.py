"""
This class will be responsible for handling the sharing of context between different agents.
"""


class InterAgentComm:
    """
    A class for facilitating communication between different agents in a multi-agent system.
    """

    def __init__(self):
        self.shared_contexts = (
            {}
        )  # Initialize an empty dictionary for storing shared contexts

    async def share_context(self, from_agent_id, to_agent_id, context):
        """
        Share context from one agent to another.

        Args:
          from_agent_id (str): The ID of the agent sharing the context.
          to_agent_id (str): The ID of the agent receiving the context.
          context (Any): The context being shared.

        Returns:
          str: A message indicating that the context was successfully shared.
        """
        if from_agent_id not in self.shared_contexts:
            self.shared_contexts[from_agent_id] = {}
        self.shared_contexts[from_agent_id][to_agent_id] = context
        return f"Context shared from agent {from_agent_id} to agent {to_agent_id}."

    async def fetch_shared_context(self, from_agent_id, to_agent_id):
        """
        Fetch the shared context for a specific agent from another agent.

        Args:
          from_agent_id (str): The ID of the agent that the shared context is being fetched from.
          to_agent_id (str): The ID of the agent that the shared context is being fetched for.

        Returns:
          The shared context for the specified agent, or a message indicating that there is no shared context
          between the specified agents.
        """
        return self.shared_contexts.get(from_agent_id, {}).get(
            to_agent_id, f"No shared context from {from_agent_id} to {to_agent_id}."
        )
