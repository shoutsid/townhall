# Townhall

Townhall is a Python-based application designed to facilitate interactions using the AssistantAgent and UserProxyAgent classes.

## Features

- **Planner**: The [planner](https://github.com/shoutsid/townhall/blob/main/agents/planner.py) module provides functionalities to truncate conversations, initiate chats with the planner, and manage a registry of functions.

- **Function Registry**: The [function registry](https://github.com/shoutsid/townhall/blob/main/agents/function_registry.py) module manages a registry of functions, allowing users to add, execute, and list functions.

- **Buddy**: The [buddy](https://github.com/shoutsid/townhall/blob/main/agents/buddy.py) module represents a chatbot that can initiate a chat between a user and an assistant agent.

- **Settings**: The [settings](https://github.com/shoutsid/townhall/blob/main/settings.py) module loads environment variables and defines configurations for the application.

- **Setup**: The [setup script](https://github.com/shoutsid/townhall/blob/main/setup.sh) helps in setting up a virtual environment and installing necessary dependencies.

## Getting Started

1. Clone the repository.
2. Run the [setup script](https://github.com/shoutsid/townhall/blob/main/setup.sh) to set up the virtual environment and install dependencies.
3. Update the `.env` file with your `OPENAI_API_KEY` using the provided [.env.example](https://github.com/shoutsid/townhall/blob/main/.env.example) as a reference.
4. Execute the [main.py](https://github.com/shoutsid/townhall/blob/main/main.py) script to start the application.

## Dependencies

The application has several dependencies which can be found in the [requirements.txt](https://github.com/shoutsid/townhall/blob/main/requirements.txt) file.

## Contribution

Feel free to fork the repository, make changes, and create pull requests. Your contributions are always welcome!
