#!/bin/bash

# Incase we ever decide to use lfs for the models
# git lfs install
# git lfs pull

# Not in a virtual environment, do we need to create one?
# if [ ! -d ".venv" ]; then
#     echo "Creating virtual environment"
#     python3 -m venv .venv
# fi

# # check if we are in active venv, if not activate it
# if [ -z "$VIRTUAL_ENV" ]; then
#   echo "Not in a virtual environment, activating venv"
#   source .venv/bin/activate
# fi

pip install -r ./requirements.txt

cd ext/tinygrad
python3 -m pip install -e .

# For llama + tinygrad on Mac Metal GPU
## check if root user, if not install pocl-opencl-icd as sudo
# if [ "$EUID" -ne 0 ]; then
#     echo "Not root user, installing pocl-opencl-icd as sudo"
#     sudo apt install -y pocl-opencl-icd
# else
#     echo "Root user, installing pocl-opencl-icd"
#     apt install -y pocl-opencl-icd
# fi


# python -m fastchat.serve.controller
# python -m fastchat.serve.model_worker --model-path chatglm2-6b
# python -m fastchat.serve.openai_api_server --host localhost --port 8000