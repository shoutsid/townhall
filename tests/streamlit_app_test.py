import sys
import os
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import main
from agents.buddy import Buddy


class DynamicMock(Mock):
    """
    Mock class to emulate dynamic attribute addition.
    """

    def __getattr__(self, item):
        if item not in self.__dict__:
            setattr(self, item, None)
        return super().__getattr__(item)

    def __contains__(self, item):
        return item in self.__dict__


class TestStreamlitApp(unittest.TestCase):
    @patch("streamlit_app.st")
    @patch.object(Buddy, "start", return_value="Test Response from Buddy")
    def test_initialization_of_session_state(self, mock_buddy_start, mock_st):
        # Mocking initial session state with dynamic attribute addition
        mock_st.session_state = DynamicMock()

        main()

        self.assertIsInstance(mock_st.session_state.buddy, Buddy)
        self.assertIsNone(mock_st.session_state.response)

    @patch("streamlit_app.st")
    def test_home_page_selection(self, mock_st):
        mock_st.sidebar.radio.return_value = "Home"

        main()

        mock_st.write.assert_called_with("Welcome to the Townhall Streamlit interface!")

    @patch("streamlit_app.st")
    @patch.object(Buddy, "start", return_value="Test Response from Buddy")
    def test_agent_interaction_with_buddy(self, mock_buddy_start, mock_st):
        mock_st.sidebar.radio.return_value = "Agents"
        mock_st.selectbox.return_value = "Buddy"
        mock_st.button.return_value = True
        user_input = "Hello Buddy!"
        mock_st.text_input.return_value = user_input

        main()

        # This ensures the buddy's start method was called with the user's input
        mock_buddy_start.assert_called_with(user_input)
        # This ensures a response was written to the streamlit interface
        mock_st.write.assert_called()

    @patch("streamlit_app.st")
    def test_agent_interaction_with_planner(self, mock_st):
        mock_st.sidebar.radio.return_value = "Agents"
        mock_st.selectbox.return_value = "Planner"
        mock_st.button.return_value = True

        main()

        mock_st.write.assert_called_with(
            "Planner Response: Planner agent interaction not yet implemented."
        )


if __name__ == "__main__":
    unittest.main()
