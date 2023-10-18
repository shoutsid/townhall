from autogen import AssistantAgent, UserProxyAgent


class PlannerService:
    """
    A service class that provides an interface for interacting with the planner agent.

    Attributes:
      planner (AssistantAgent): The planner agent used for generating plans.
      planner_user (UserProxyAgent): The user proxy agent used for communicating with the planner agent.
    """

    def __init__(self, config_list):
        self.config_list = config_list
        self.planner = AssistantAgent(name="planner")
        self.planner_user = UserProxyAgent(
            name="planner_user", max_consecutive_auto_reply=0, human_input_mode="NEVER"
        )

    def get_service(self, name):
        """
        Returns the service associated with the given name, or None if no such service exists.

        Args:
          name (str): The name of the service to retrieve.

        Returns:
          The service associated with the given name, or None if no such service exists.
        """
        for config in self.config_list:
            if config["name"] == name:
                return config["service"]
        return None

    def get_parameters(self, name):
        """
        Returns the parameters for a given configuration name.

        Args:
          name (str): The name of the configuration to retrieve parameters for.

        Returns:
          dict: A dictionary containing the parameters for the configuration, or None if the configuration
              with the given name does not exist.
        """
        for config in self.config_list:
            if config["name"] == name:
                return config["parameters"]
            return None

    def ask_planner(self, msg):
        """
        Sends a message to the planner and returns the response.

        Args:
          msg (str): The message to send to the planner.

        Returns:
          str: The response from the planner.
        """
        self.planner_user.initiate_chat(self.planner, message=msg)
        return self.planner_user.last_message()["content"]
