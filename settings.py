import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GPT3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
GPT4_0613 = "gpt-4-0613"
VICUNA_7B_V1_5 = "vicuna-7b-v1.5"

CONFIG_LIST = [
    {
        "model": GPT3_5_TURBO_0613,
        "api_key": OPENAI_API_KEY,
    }
]

LOCAL_CONFIG_LIST = [
    {
        "model": GPT3_5_TURBO_0613,
        "api_key": "None",
        "api_base": "http://localhost:8000/v1",
        "api_type": "open_ai",
    }
]
