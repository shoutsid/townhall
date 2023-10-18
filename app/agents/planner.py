"""
This module contains the main function to initiate a chat with the assistant.
"""

from settings import (
    CONFIG_LIST,
)

from app.services.planner_service import PlannerService
from app.services.function_service import FunctionService
from app.services.chat_service import ChatService

if __name__ == "__main__":
    planner_service = PlannerService(CONFIG_LIST)
    function_service = FunctionService()
    chat_service = ChatService(
        CONFIG_LIST, function_service.registry.functions)

    message = input("Enter a message to send to the assistant: ")
    chat_service.initiate_chat(message)
