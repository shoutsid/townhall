"""
This module contains unit tests for the Buddy class.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from unittest.mock import Mock, patch
from agents.buddy import Buddy


class TestBuddy(unittest.TestCase):
    """
    This class contains unit tests for the Buddy class.
    """

    def setUp(self):
        self.buddy = Buddy()

    @patch("agents.buddy.AssistantAgent")
    @patch("agents.buddy.UserProxyAgent")
    def test_start(self, mock_user_proxy_agent, mock_assistant_agent):
        """
        Test the start method of the Buddy class.
        """
        mock_user_proxy = Mock()
        mock_user_proxy_agent.return_value = mock_user_proxy

        mock_assistant = Mock()
        mock_assistant_agent.return_value = mock_assistant

        message = "Hello, can you help me?"

        self.buddy.start(message)

        mock_user_proxy.initiate_chat.assert_called_once_with(
            mock_assistant, message=message
        )
