"""
This module contains utility functions for agents.
"""

def is_termination_msg(x):
    """
    Check if the given message is a termination message.

    Args:
        x (dict): The message to check.

    Returns:
        bool: True if the message is a termination message, False otherwise.
    """
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()
