"""
This module contains the main function for the townhall application.
"""

from app.agents.buddy import Buddy


def main():
    """
    Main function for the townhall application.
    """
    message = input("Basic OpenAI Buddy, no functions.\n " \
                    "To use functions, run the models.planner.py in the app folder.\n" \
                    "All PR's are welcome and features will be tracked & planned. \n" \
                    "Enter a message to send to the assistant: ")
    bud: Buddy = Buddy()
    while True:
        bud.start(message)


if __name__ == "__main__":
    main()
