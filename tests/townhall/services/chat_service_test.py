from unittest.mock import patch
import pytest
from townhall.services.chat_service import ChatService


@patch("autogen.oai.ChatCompletion.create")
@patch("threading.Event.wait")
def test_initiate_chat(mocked_wait, mocked_create):
    mocked_create.return_value = {"choices": [{"text": "Mocked text"}]}
    mocked_wait.return_value = None

    with patch("autogen.AssistantAgent") as MockedAssistant, patch(
        "townhall.agents.user_agent.UserAgent"
    ) as MockedProxy:
        service = ChatService(
            "some_config", {}, MockedAssistant.return_value, MockedProxy.return_value
        )

        assert (
            service.user_proxy == MockedProxy.return_value
        ), "UserAgent is not mocked properly"

        service.initiate_chat("Hello, Assistant!")

        MockedProxy.return_value.initiate_chat.assert_called_once()
