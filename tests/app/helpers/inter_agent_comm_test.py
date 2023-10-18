import pytest
from app.helpers.inter_agent_comm import InterAgentComm


@pytest.mark.asyncio
async def test_inter_agent_communication():
    # Initialize InterAgentComm
    inter_agent_comm = InterAgentComm()

    # Test context sharing
    assert (
        await inter_agent_comm.share_context(
            "001", "002", {"shared_data": "important_info"}
        )
        == "Context shared from agent 001 to agent 002."
    )

    # Test fetching shared context
    assert await inter_agent_comm.fetch_shared_context("001", "002") == {
        "shared_data": "important_info"
    }
