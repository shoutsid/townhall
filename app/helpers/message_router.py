from collections import deque


class MessageRouter:
    """
    A class to handle sending and fetching messages for agents.

    Attributes:
    -----------
    agent_manager : AgentManager
      An instance of AgentManager class to manage agents.

    Methods:
    --------
    send_message(agent_id: str, message: str) -> str:
      Sends a message to the specified agent.

    fetch_messages(agent_id: str) -> Union[List[str], str]:
      Fetches all the messages for the specified agent.
    """

    def __init__(self, agent_manager, inter_agent_comm):
        self.agent_manager = agent_manager
        self.inter_agent_comm = inter_agent_comm

    async def send_message(self, agent_id, message):
        """
        Sends a message to the specified agent.

        Parameters:
        -----------
        agent_id : str
          The ID of the agent to send the message to.
        message : str
          The message to be sent.

        Returns:
        --------
        str
          A message indicating whether the message was sent successfully or not.
        """
        if agent_id in self.agent_manager.registry:
            self.agent_manager.registry[agent_id]["messages"].append(message)
            return f"Message sent to agent {agent_id}."
        else:
            return f"Agent {agent_id} not found."

    async def fetch_messages(self, agent_id):
        """
        Fetches all the messages for the specified agent.

        Parameters:
        -----------
        agent_id : str
          The ID of the agent to fetch messages for.

        Returns:
        --------
        Union[List[str], str]
          A list of messages if there are any, else a message indicating that the agent was not found.
        """
        if agent_id in self.agent_manager.registry:
            messages = list(self.agent_manager.registry[agent_id]["messages"])
            self.agent_manager.registry[agent_id]["messages"].clear()
            return messages
        else:
            return f"Agent {agent_id} not found."
