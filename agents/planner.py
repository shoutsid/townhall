from autogen import AssistantAgent, UserProxyAgent

from settings import (
    CONFIG_LIST,
)
from agents.function_registry import FunctionRegistry


def truncate_conversation(conversation, max_tokens=4090):
    """
    Given a conversation (list of messages), truncate it to fit within max_tokens.
    Prioritize the most recent messages.
    """
    total_tokens = sum([len(msg["content"]) for msg in conversation])

    while total_tokens > max_tokens:
        # Remove the oldest message
        removed_message = conversation.pop(0)
        total_tokens -= len(removed_message["content"])

    return conversation


registry = FunctionRegistry()

planner = AssistantAgent(
    name="planner",
    llm_config={"config_list": CONFIG_LIST},
)
planner_user = UserProxyAgent(
    name="planner_user", max_consecutive_auto_reply=0, human_input_mode="NEVER"
)


def ask_planner(msg: str):
    """
    Initiates a chat with the planner and returns the last message content.

    Args:
        msg (str): The message to send to the planner.

    Returns:
        str: The content of the last message received from the planner.
    """
    planner_user.initiate_chat(planner, message=msg)
    return planner_user.last_message()["content"]


registry.add_function("ask_planner", ask_planner)


def add_function(name, function_body):
    """
    Adds a new function to the global registry.

    Args:
        name (str): The name of the function.
        function_body (str): The body of the function as a string.

    Returns:
        str: A message indicating that the function was added successfully.
    """
    exec(function_body, None, globals())
    new_func = globals()[name]
    registry.add_function(name, new_func)
    return f"Function '{name}' added!"


def execute_function(name, *args, **kwargs):
    """
    Executes a function with the given name and arguments.

    Args:
        name (str): The name of the function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the executed function.
    """
    return registry.execute_function(name, *args, **kwargs)


def add_multiple_functions(functions):
    """
    Adds multiple functions to the planner.

    Args:
        functions (dict): A dictionary containing function names as keys and function bodies as values.

    Returns:
        str: A string containing the responses from adding each function, separated by newlines.
    """
    responses = []
    for name, function_body in functions.items():
        response = add_function(name, function_body)
        responses.append(response)
    return "\n".join(responses)


def execute_multiple_functions(function_calls):
    """
    Executes multiple functions and returns their responses as a string.

    Args:
        function_calls (list): A list of dictionaries representing function calls.
            Each dictionary should have a "name" key with the function name, and optional
            "args" and "kwargs" keys with the function arguments and keyword arguments.

    Returns:
        str: A string containing the responses of all the executed functions, separated by newlines.
    """
    responses = []
    for call in function_calls:
        name = call["name"]
        args = call.get("args", [])
        kwargs = call.get("kwargs", {})
        response = execute_function(name, *args, **kwargs)
        responses.append(response)
    return "\n".join(responses)


registry.add_function("add_function", add_function)
registry.add_function("execute_function", execute_function)
registry.add_function("add_multiple_functions", add_multiple_functions)
registry.add_function("execute_multiple_functions", execute_multiple_functions)


assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "request_timeout": 600,
        "seed": 42,
        "model": "gpt-3.5-turbo-0613",
        # "model": "vicuna-7b-v1.5",
        "config_list": CONFIG_LIST,
        "functions": [
            {
                "name": "ask_planner",
                "description": (
                    "ask planner to: 1. get a plan for finishing a task, "
                    "2. verify the execution result of the plan and potentially suggest new plan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "question to ask planner. Make sure the question includes enough context, "
                                "such as the code and the execution result. The planner does not know the "
                                "conversation between you and the user, unless you share the conversation with the planner."
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    },
    function_map=registry.functions,  # This will make sure the assistant always knows about the functions.
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=0,
    code_execution_config={"work_dir": "planning"},
    function_map=registry.functions,
)

message = input("Enter a message to send to the assistant: ")
user_proxy.initiate_chat(
    assistant,
    message=message,
)
