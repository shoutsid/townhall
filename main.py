"""
This module contains the main function for the townhall application.
"""

from agents.buddy import Buddy


def main():
    """
    Main function for the townhall application.
    """
    message = input("Enter a message to send to the assistant: ")
    Buddy().start(message)


if __name__ == "__main__":
    main()
