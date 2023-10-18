![Townhall Banner](docs/banner.png)

# Townhall

Under heavy development and in transition from a previous personal project. Not ready for production use.

[![GitHub contributors](https://img.shields.io/github/contributors/shoutsid/townhall.svg)]()
[![GitHub issues](https://img.shields.io/github/issues/shoutsid/townhall.svg)]()
[![GitHub pull requests](https://img.shields.io/github/issues-pr/shoutsid/townhall.svg)]()
[![GitHub forks](https://img.shields.io/github/forks/shoutsid/townhall.svg?style=social&label=Fork)]()
[![GitHub stars](https://img.shields.io/github/stars/shoutsid/townhall.svg?style=social&label=Stars)]()
[![GitHub watchers](https://img.shields.io/github/watchers/shoutsid/townhall.svg?style=social&label=Watch)]()
[![GitHub followers](https://img.shields.io/github/followers/shoutsid.svg?style=social&label=Follow)]()
[![GitHub license](https://img.shields.io/github/license/shoutsid/townhall.svg)]()
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

Townhall is a cutting-edge chatbot framework crafted in Python and grounded on the robust Autogen foundation. This isn't just another chatbot; Townhall leverages the power of advanced agents to breathe life into conversations and elevate them to a whole new level. 

### The Autogen Foundation: The Bedrock of Innovation
At its core, Townhall is built upon the Autogen framework, a pioneering platform for LLMs. Autogen enables the creation of agents that are not only customizable but also conversational. These agents can interact with each other, and seamlessly incorporate human inputs, setting the stage for more dynamic and intelligent dialogues.

### Advanced Agents: The Symphony Conductors with Function Orchestration
Our advanced agents go beyond merely responding to user queries; they orchestrate multiple functions to provide a cohesive and engaging user experience. Think of them as the conductors of a grand symphony, where each instrument is a unique function or feature. They coordinate these functions to create a harmonious and effective dialogue, far outclassing traditional chatbots which often feel like disjointed sets of scripted responses.
The advanced agents adapt and learn, making each conversation better than the last. They can switch between various modes, employing a blend of LLMs, human inputs, and specialized tools to deliver a personalized conversational experience.

In traditional chatbots, functions are often isolated, activated only by specific commands. Townhall changes this by introducing function orchestration. Functions can now work in tandem, sharing data and context to provide a more comprehensive solution to the user’s needs. This not only enhances the chatbot's utility but also makes the interaction more natural and intuitive.

Join us on this exciting journey as we redefine what chatbots can achieve, making them more responsive, intelligent, and, most importantly, more human.


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

- **Planner**: The [planner](https://github.com/shoutsid/townhall/blob/main/app/agents/planner.py) initiate chats with the planner, and manage a registry of functions.
- **Function Registry**: The [function registry](https://github.com/shoutsid/townhall/blob/main/app/helpers/function_registry.py) module manages a registry of functions, allowing users to add, execute, and list functions.
- **Buddy**: The [buddy](https://github.com/shoutsid/townhall/blob/main/app/agents/buddy.py) module represents a chatbot that can initiate a chat between a user and an assistant agent.
- **Settings**: The [settings](https://github.com/shoutsid/townhall/blob/main/app/settings.py) module loads environment variables and defines configurations for the application.
- **User Agent**: The [user agent](https://github.com/shoutsid/townhall/blob/main/app/agents/user_agent.py) represents a user in the system and can send messages to other agents.
- **Chat Service**: The [chat service](https://github.com/shoutsid/townhall/blob/main/app/services/chat_service.py) module handles the logic for initiating and managing chats between agents.
- **Function Service**: The [function service](https://github.com/shoutsid/townhall/blob/main/app/services/function_service.py) module provides functionalities for executing specific tasks.
- **Planner Service**: The [planner service](https://github.com/shoutsid/townhall/blob/main/app/services/planner_service.py) module aids in planning and sequencing tasks for the chatbot.


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
