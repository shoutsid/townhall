![Townhall Banner](docs/banner.png)

# Townhall

A Python-based chatbot framework built on the autogen foundation, utilizing advanced agents for dynamic conversations and function orchestration, enhancing and expanding traditional chatbot capabilities.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [Testing](#testing)
7. [Roadmap](#roadmap)
8. [Credits](#credits)
9. [License](#license)



## Features

- **Planner**: The [planner](https://github.com/shoutsid/townhall/blob/main/agents/planner.py) module provides functionalities to truncate conversations, initiate chats with the planner, and manage a registry of functions.
- **Function Registry**: The [function registry](https://github.com/shoutsid/townhall/blob/main/agents/function_registry.py) module manages a registry of functions, allowing users to add, execute, and list functions.
- **Buddy**: The [buddy](https://github.com/shoutsid/townhall/blob/main/agents/buddy.py) module represents a chatbot that can initiate a chat between a user and an assistant agent.
- **Settings**: The [settings](https://github.com/shoutsid/townhall/blob/main/settings.py) module loads environment variables and defines configurations for the application.
- **Setup**: The [setup script](https://github.com/shoutsid/townhall/blob/main/setup.sh) helps in setting up a virtual environment and installing necessary dependencies.
- **User Agent**: The [user agent](https://github.com/shoutsid/townhall/blob/main/agents/user_agent.py) represents a user in the system and can send messages to other agents.
- **Chat Service**: The [chat service](https://github.com/shoutsid/townhall/blob/main/agents/services/chat_service.py) module handles the logic for initiating and managing chats between agents.
- **Function Service**: The [function service](https://github.com/shoutsid/townhall/blob/main/agents/services/function_service.py) module provides functionalities for executing specific tasks.
- **Planner Service**: The [planner service](https://github.com/shoutsid/townhall/blob/main/agents/services/planner_service.py) module aids in planning and sequencing tasks for the chatbot.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- pip package manager

## Installation

The easiest way to start playing is click below to use the Github Codespace.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shoutsid/townhall?quickstart=1)

### For Linux/Mac

```bash
git clone https://github.com/shoutsid/townhall.git
cd townhall
pip install -r requirements.txt
```

### For Windows

```bash
git clone https://github.com/shoutsid/townhall.git
cd townhall
pip install -r requirements.txt
```

## Usage

To start the chatbot, run the following command:

```bash
python main.py
```

For more advanced usage, refer to the [documentation (todo)](#).

## Contributing

If you would like to contribute to Townhall, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Testing

To run the tests:

```bash
pytest
```

## Roadmap

- Token prediction and compression techniques
- Implement multi-agent conversations
- ORM for opensource vector space models
- Prompt templating for forced choice questions/answers
- Add support for voice commands
- Improve UI/UX
- Interactive chat interface
  - interactive chatroom with multiple agents
  - Add support for document embedding
  - Add support for image embedding
  - Add support for video embedding
  - Add support for audio embedding

## Credits

Developed by @shoutsid.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE . See the [LICENSE.md](LICENSE.md) file for details.
