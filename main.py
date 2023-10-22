"""
This module contains the main function for the townhall application.
"""

from app.agents.buddy import Buddy


def main():
    """
    Main function for the townhall application.
    """
    message = input("OpenAI Buddy, no functions yet. Planner deals with functions. \n " \
                    "Enter a message to send to the assistant: ")
    bud: Buddy = Buddy()
    while True:
        bud.start(message)


if __name__ == "__main__":
    main()
