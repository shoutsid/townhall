from townhall.models.transition import Transition


def test_transition():
    """
    Test the Transition class to ensure that it correctly initializes with the expected attributes.
    """
    transition = Transition(state=1, action=2, next_state=3, reward=4)
    assert transition.state == 1
    assert transition.action == 2
    assert transition.next_state == 3
    assert transition.reward == 4
