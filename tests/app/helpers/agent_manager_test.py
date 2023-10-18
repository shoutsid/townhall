import pytest

from app.helpers.inter_agent_comm import InterAgentComm
from app.helpers.agent_manager import AgentManager


@pytest.mark.asyncio
async def test_agent_management():
    # Initialize AgentManager
    inter_agent_comm = InterAgentComm()
    agent_manager = AgentManager(inter_agent_comm=inter_agent_comm)

    # Test agent registration
    assert (
        await agent_manager.register_agent(
            "001", {"name": "Agent Smith", "status": "active"}
        )
        == "Agent 001 registered successfully."
    )

    # Test agent query
    assert await agent_manager.query_agent("001") == {
        "name": "Agent Smith",
        "status": "active",
    }

    # Test agent unregistration
    assert (
        await agent_manager.unregister_agent("001")
        == "Agent 001 unregistered successfully."
    )

    # Test agent query after unregistration
    assert await agent_manager.query_agent("001") == "Agent 001 not found."
