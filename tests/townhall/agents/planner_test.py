import pytest
from unittest.mock import patch
from townhall.services.planner_service import PlannerService


@pytest.fixture
def planner_service():
    config_list = [
        {
            "name": "some_service",
            "service": "some_service_impl",
            "parameters": {"key": "value"},
        }
    ]
    service = PlannerService(config_list)
    return service


@patch("townhall.services.planner_service.UserProxyAgent.initiate_chat")
@patch("townhall.services.planner_service.UserProxyAgent.last_message")
def test_ask_planner(mocked_last_message, mocked_initiate_chat, planner_service):
    # Mock the last_message() method to return a dict with 'content' key
    mocked_last_message.return_value = {
        "content": "Go to the store and buy groceries"}

    response = planner_service.ask_planner("buy groceries")

    # Verify that the initiate_chat method was called once
    mocked_initiate_chat.assert_called_once()

    # Verify the response
    assert response == "Go to the store and buy groceries"


@patch("townhall.services.planner_service.UserProxyAgent.initiate_chat")
@patch("townhall.services.planner_service.UserProxyAgent.last_message")
def test_ask_planner_invalid_input(
    mocked_last_message, mocked_initiate_chat, planner_service
):
    # Mock the last_message() method to return a dict with 'content' key
    mocked_last_message.return_value = {"content": "Invalid input"}

    response = planner_service.ask_planner("")

    # Verify that the initiate_chat method was called once
    mocked_initiate_chat.assert_called_once()

    # Verify the response
    assert response == "Invalid input"
