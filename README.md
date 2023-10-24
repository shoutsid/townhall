![Townhall Banner](docs/banner.png)

# Townhall

🚧 Under heavy development and in transition from a previous personal project. Not ready for production use. 🚧

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

### 🧱 The Autogen Foundation: The Bedrock of Innovation
At its core, Townhall is built upon the Autogen framework, a pioneering platform for LLMs. Autogen enables the creation of agents that are not only customizable but also conversational. These agents can interact with each other, and seamlessly incorporate human inputs, setting the stage for more dynamic and intelligent dialogues.

Our advanced agents go beyond merely responding to user queries; they orchestrate multiple functions to provide a cohesive and engaging user experience. Think of them as the conductors of a grand symphony, where each instrument is a unique function or feature. They coordinate these functions to create a harmonious and effective dialogue, far outclassing traditional chatbots which often feel like disjointed sets of scripted responses. The advanced agents adapt and learn, making each conversation better than the last. They can switch between various modes, employing a blend of LLMs, human inputs, and specialized tools to deliver a personalized conversational experience.

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

## 📝 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10 or higher
- pip package manager

## 🛠️ Installation

### Docker Compose

```bash
git clone --recurse-submodules https://github.com/shoutsid/townhall.git
cd townhall
docker compose up -d
docker compose exec townhall bash
./setup.sh
```

### For Linux

```bash
git clone --recurse-submodules https://github.com/shoutsid/townhall.git
cd townhall
./setup.sh
```

### For Mac/Windows
The easiest way to get setup on windows is to start playing is click below to use the Github Codespace. Otherwise this was developed on WSL Ubuntu.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shoutsid/townhall?quickstart=1)


## 🌐 Usage

### Planner

To start the planner that can execute shell scripts and run python,
run the following command:

```bash
export OPENAI_API_KEY=<your-api-key>
python3 townhall/models/planner.py
```

### LLaMa Integration
To start the Llama module, run the following commands:

```bash
pip install -r requirements.txt
cd townhall/models/llama/weights/
bash pull_llama.sh
cd ../../../..
python3 townhall/models/llama/llama.py
```

### GPT-2 Integration

To start the GPT-2 module, run the following commands:

```bash
python3 townhall/models/gpt2/gpt2.py
```

## 🤝 Contributing

If you would like to contribute to Townhall, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## 🧪 Testing

To run the tests:

```bash
pytest
```

## 🗺️ Roadmap

For the detailed roadmap of upcoming features, please visit our [Project Board](https://github.com/users/shoutsid/projects/1).

## 👏 Credits

- **Prompt Contributions**: A big thank you to [Josh-XT](https://github.com/Josh-XT) for the various prompts. Check out his repository [AGiXT](https://github.com/Josh-XT/AGiXT) for more details.
- **Autogen Foundation**: Townhall is built upon the robust [Autogen Framework](https://github.com/microsoft/autogen/), a pioneering platform for LLMs by Microsoft.
- **OpenAI Assistant**: Special mention to [OpenAI Assistant](https://chat.openai.com/) for aiding in the development process.
- **tinygrad**: Special thanks to [tinygrad](https://github.com/tinygrad/tinygrad) for providing the foundational machine learning framework that enhances our project's capabilities.


Developed by @shoutsid.


## 📜 License

This project is licensed under the GNU GENERAL PUBLIC LICENSE . See the [LICENSE.md](LICENSE.md) file for details.
