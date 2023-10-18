from app.helpers.agent_registry import AgentRegistry
import pytest


@pytest.mark.asyncio
async def test_register_agent():
    registry = AgentRegistry()
    response = await registry.register_agent(
        "001", {"name": "Agent Smith", "status": "active"}
    )
    assert response == "Agent 001 registered successfully."
    assert "001" in registry.registry


@pytest.mark.asyncio
async def test_unregister_agent():
    registry = AgentRegistry()
    await registry.register_agent("001", {"name": "Agent Smith", "status": "active"})
    response = await registry.unregister_agent("001")
    assert response == "Agent 001 unregistered successfully."
    assert "001" not in registry.registry


@pytest.mark.asyncio
async def test_query_agent():
    registry = AgentRegistry()
    await registry.register_agent("001", {"name": "Agent Smith", "status": "active"})
    response = await registry.query_agent("001")
    assert response == {"name": "Agent Smith", "status": "active"}


@pytest.mark.asyncio
async def test_list_agents():
    registry = AgentRegistry()
    await registry.register_agent("001", {"name": "Agent Smith", "status": "active"})
    await registry.register_agent(
        "002", {"name": "Agent Johnson", "status": "inactive"}
    )
    response = await registry.list_agents()
    assert response == {
        "001": {"name": "Agent Smith", "status": "active"},
        "002": {"name": "Agent Johnson", "status": "inactive"},
    }
