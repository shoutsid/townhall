import pytest
import numpy as np
from app.models import CooperativeAgent2, Message


@pytest.fixture
def agent():
    return CooperativeAgent2(agent_id=1, grid_size=10, position=(0, 0))


def test_send_message(agent: CooperativeAgent2):
    """
    Test sending a message to multiple targets and checking if the message is sent correctly.

    Args:
        agent: An instance of the CooperativeAgent class.
    """
    agent.position = (0, 0)
    message = agent.send_message(targets=[(3, 4), (5, 5)])
    assert message.sender_id == agent.id
    assert message.perceived_distances == [5.0, 7.0710678118654755]


def test_receive_messages(agent: CooperativeAgent2):
    """
    Test that the agent can receive messages correctly.

    Args:
        agent (CooperativeAgent2): The agent to test.
    """
    # Create mock messages
    messages = [
        Message(sender_id=2, position=(1, 1),
                perceived_distances=[5.0, 6.0]),
        Message(sender_id=3, position=(2, 2),
                perceived_distances=[1.0, 2.0])
    ]

    # Agent receives the messages
    agent.receive_messages(messages)

    assert agent.messages == messages


def test_take_action(agent: CooperativeAgent2):
    """
    Test the take_action method of the CooperativeAgent2 class.

    The function tests whether the agent's position changes after taking an action based on the current state and received messages.

    Args:
        agent (CooperativeAgent2): An instance of the CooperativeAgent2 class.
    """
    initial_position = agent.position

    # Mock received messages and state
    agent.received_messages = [
        Message(sender_id=2, position=(1, 1),
                perceived_distances=[5.0, 6.0])  # Added position
    ]

    mock_targets = [(3, 4), (5, 5)]

    # Take action based on the current state and received messages
    agent.take_action(targets=mock_targets)

    assert agent.position != initial_position


def test_calculate_reward(agent: CooperativeAgent2):
    """
    Test the calculate_reward method of the CooperativeAgent2 class.

    The method should calculate the reward obtained by the agent after moving to a new position.

    The test sets up a scenario where the agent is at position (1, 1) and has to move to one of two targets (0, 0) or (4, 4).
    The expected reward is manually calculated by comparing the distances between the agent's previous and new positions to each target.
    The method is then called and the obtained reward is compared to the expected reward.
    """
    targets = [(0, 0), (4, 4)]
    agent.position = (1, 1)
    agent.positions = [(0, 0), (1, 0), (1, 1)]

    # Manually calculate the expected reward
    old_distances = [np.linalg.norm(np.array(pos) - np.array(target))
                     for pos, target in zip([agent.positions[-2]]*2, targets)]
    new_distances = [np.linalg.norm(np.array(pos) - np.array(target))
                     for pos, target in zip([agent.positions[-1]]*2, targets)]
    expected_reward = np.sum(np.array(old_distances) - np.array(new_distances))

    reward = agent.calculate_reward(targets)
    assert reward == expected_reward
