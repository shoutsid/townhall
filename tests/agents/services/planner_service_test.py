import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from agents.services.planner_service import PlannerService


@pytest.mark.parametrize(
    "msg, expected", [("Hello", "Hello, Planner!"), ("Test", "Test message!")]
)
def test_ask_planner(msg, expected):
    with patch("agents.services.planner_service.AssistantAgent") as MockedAgent, patch(
        "agents.services.planner_service.UserProxyAgent"
    ) as MockedProxy:
        mocked_last_message = MagicMock()
        mocked_last_message.return_value = {"content": expected}
        MockedProxy.return_value.last_message = mocked_last_message
        service = PlannerService("some_config")
        assert service.ask_planner(msg) == expected
