from app.helpers.message_router import MessageRouter
from app.helpers.agent_manager import AgentManager
from app.helpers.inter_agent_comm import InterAgentComm
import pytest


@pytest.mark.asyncio
async def test_message_routing():
    # Initialize InterAgentComm, AgentManager and MessageRouter
    inter_agent_comm = InterAgentComm()
    agent_manager = AgentManager(inter_agent_comm)
    message_router = MessageRouter(agent_manager, inter_agent_comm)

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
