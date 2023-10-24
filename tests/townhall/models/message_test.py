from townhall.models.message import Message


def test_message_attributes():
    """
    Test that the Message object has the correct attributes after initialization.
    """
    sender_id = 1
    position = (0, 0)
    perceived_distances = [1, 2, 3]
    message = Message(sender_id, position, perceived_distances)

    assert message.sender_id == sender_id
    assert message.position == position
    assert message.perceived_distances == perceived_distances
