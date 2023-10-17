import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

from autogen import AssistantAgent, UserProxyAgent


class PlannerService:
    def __init__(self, config_list):
        self.planner = AssistantAgent(
            name="planner", llm_config={"config_list": config_list}
        )
        self.planner_user = UserProxyAgent(
            name="planner_user", max_consecutive_auto_reply=0, human_input_mode="NEVER"
        )

    def ask_planner(self, msg):
        self.planner_user.initiate_chat(self.planner, message=msg)
        return self.planner_user.last_message()["content"]
