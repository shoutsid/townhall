import pytest
from townhall.models.environment import Environment


@pytest.fixture
def environment():
    return Environment(num_agents=2, num_tasks=3, num_features=4, log_dir="logs/")


def test_environment_initialization(environment):
    assert environment.num_agents == 2
    assert environment.num_tasks == 3
    assert environment.num_features == 4
    assert len(environment.agents) == 2
    assert len(environment.performance) == 2
    assert len(environment.agent_positions) == 2
    assert len(environment.rewards) == 2


def test_environment_step(environment):
    environment.step()
    assert len(environment.performance[0][0]) == 1
    assert len(environment.rewards[0][0]) == 1
    assert len(environment.agent_positions[0]) == 1
    assert len(environment.performance[1][0]) == 1
    assert len(environment.rewards[1][0]) == 1
    assert len(environment.agent_positions[1]) == 1