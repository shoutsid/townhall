"""
This module contains unit tests for the Buddy class.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from unittest.mock import Mock, patch
from agents.buddy import Buddy, UserAgent, AssistantAgent


class TestBuddy(unittest.TestCase):
    """
    This class contains unit tests for the Buddy class.
    """

    def setUp(self):
        self.buddy = Buddy()

    @patch.object(UserAgent, "get_human_input", return_value="Test message")
    @patch.object(UserAgent, "send")
    @patch.object(UserAgent, "initiate_chat")
    @patch.object(
        UserAgent, "last_message", return_value={"content": "Reply from assistant"}
    )
    @patch.object(AssistantAgent, "__init__", return_value=None)
    def test_start_without_existing_conversation(
        self,
        mock_assistant_init,
        mock_last_message,
        mock_initiate_chat,
        mock_send,
        mock_get_human_input,
    ):
        """
        Test the start method of the Buddy class without an existing conversation.
        """
        # Resetting the conversation context to simulate no existing conversation
        self.buddy.conversation_context = []

        message = "Hello, can you help me?"
        self.buddy.start(message)
        mock_initiate_chat.assert_called_once_with(
            recipient=self.buddy.assistant, message=message
        )
        mock_send.assert_not_called()

    @patch.object(UserAgent, "get_human_input", return_value="Test message")
    @patch.object(UserAgent, "send")
    @patch.object(UserAgent, "initiate_chat")
    @patch.object(
        UserAgent, "last_message", return_value={"content": "Reply from assistant"}
    )
    @patch.object(AssistantAgent, "__init__", return_value=None)
    def test_start_with_existing_conversation(
        self,
        mock_assistant_init,
        mock_last_message,
        mock_initiate_chat,
        mock_send,
        mock_get_human_input,
    ):
        """
        Test the start method of the Buddy class with an existing conversation.
        """
        # Simulate an existing conversation
        self.buddy.conversation_context = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        message = "Can you assist further?"
        self.buddy.start(message)

        mock_send.assert_called_once_with(
            message=message, recipient=self.buddy.assistant, request_reply=True
        )
        mock_initiate_chat.assert_not_called()
