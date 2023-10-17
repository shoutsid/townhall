import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from agents.helpers.agent_manager import AgentManager
from agents.helpers.message_router import MessageRouter


@pytest.mark.asyncio
async def test_message_routing():
    # Initialize AgentManager and MessageRouter
    agent_manager = AgentManager()
    message_router = MessageRouter(agent_manager)

    # Register an agent for testing
    await agent_manager.register_agent(
        "001", {"name": "Agent Smith", "status": "active"}
    )

    # Test message sending
    assert (
        await message_router.send_message("001", "Hello, Agent Smith.")
        == "Message sent to agent 001."
    )

    # Test message fetching
    assert await message_router.fetch_messages("001") == ["Hello, Agent Smith."]

    # Test message sending after unregistration
    await agent_manager.unregister_agent("001")
    assert (
        await message_router.send_message("001", "Hello, Agent Smith.")
        == "Agent 001 not found."
    )
