import os
from dotenv import load_dotenv

load_dotenv()

# Validate essential environment variables
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Missing essential environment variable: OPENAI_API_KEY")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GPT3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
GPT4_0613 = "gpt-4-0613"
VICUNA_7B_V1_5 = "vicuna-7b-v1.5"

if "GPT4" in os.environ:
    CONFIG_LIST = [
        {
            "model": GPT4_0613,
            "api_key": OPENAI_API_KEY,
        }
    ]
elif "LOCAL" in os.environ:
    CONFIG_LIST = [
        {
            "model": GPT3_5_TURBO_0613,
            "api_key": "None",
            "api_base": "http://localhost:8000/v1",
            "api_type": "open_ai",
        }
    ]
else:
    CONFIG_LIST = [
        {
            "model": GPT3_5_TURBO_0613,
            "api_key": OPENAI_API_KEY,
        }
    ]

LLM_CONFIG = {
    "request_timeout": 60,
    "seed": 42,
    "config_list": CONFIG_LIST,
    "temperature": 0,
}
