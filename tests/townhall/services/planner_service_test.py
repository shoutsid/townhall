import pytest
from unittest.mock import patch
from townhall.services.planner_service import PlannerService


@pytest.fixture(scope="function")
def planner_service_with_mocked_chat():
    with patch(
        "townhall.services.planner_service.UserProxyAgent.initiate_chat"
    ) as MockedInitiateChat:
        MockedInitiateChat.return_value = (
            None  # No return value as we're mocking the method
        )
        yield PlannerService(
            [
                {
                    "name": "test_service",
                    "service": "TestService",
                    "parameters": {"param1": "value1"},
                }
            ]
        )


def test_get_service(planner_service_with_mocked_chat):
    assert planner_service_with_mocked_chat.get_service(
        "test_service") == "TestService"


def test_get_parameters(planner_service_with_mocked_chat):
    assert planner_service_with_mocked_chat.get_parameters("test_service") == {
        "param1": "value1"
    }


def test_ask_planner(planner_service_with_mocked_chat):
    with patch(
        "townhall.services.planner_service.UserProxyAgent.last_message"
    ) as MockedLastMessage:
        MockedLastMessage.return_value = {
            "content": "Test response from planner"}
        response = planner_service_with_mocked_chat.ask_planner(
            "buy groceries")
        assert response == "Test response from planner"
