from app.agents.user_agent import UserAgent

import pytest


@pytest.fixture
def user_agent():
    """Fixture to create an instance of UserAgent for testing."""
    return UserAgent(name="test")


def test_set_current_message(user_agent):
    """Test the set_current_message method."""
    message = "Test message"

    # Set a message and check if it's correctly stored
    user_agent.set_current_message(message)
    assert user_agent.current_message == message

    # Set another message and check if it's updated
    new_message = "New test message"
    user_agent.set_current_message(new_message)
    assert user_agent.current_message == new_message


def test_get_human_input(user_agent):
    """Test the get_human_input method."""
    prompt = "Enter a message: "

    # Start a thread to simulate user input
    import threading

    def simulate_user_input():
        user_agent.set_current_message("User input")

    input_thread = threading.Thread(target=simulate_user_input)
    input_thread.start()

    # Check if the get_human_input method returns the user input
    user_input = user_agent.get_human_input(prompt)
    assert user_input == "User input"

    # Wait for the input thread to finish
    input_thread.join()


if __name__ == "__main__":
    pytest.main()
