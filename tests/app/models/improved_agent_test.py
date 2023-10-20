import pytest
import torch
import numpy as np
from app.models.improved_agent import ImprovedAgent
from app.models.replay_memory import ReplayMemory


@pytest.fixture
def improved_agent():
    return ImprovedAgent(agent_id=1, num_tasks=3, num_features=5)


def test_improved_agent_init(improved_agent):
    assert improved_agent.agent_id == 1
    assert improved_agent.num_tasks == 3
    assert improved_agent.num_features == 5
    assert len(improved_agent.models) == 3
    assert isinstance(improved_agent.target_net, torch.nn.Module)
    assert isinstance(improved_agent.policy_net, torch.nn.Module)
    assert isinstance(improved_agent.optimizer, torch.optim.RMSprop)
    assert isinstance(improved_agent.memory, ReplayMemory)
    assert isinstance(improved_agent.previous_loss, list)
    assert len(improved_agent.previous_loss) == 3


def test_improved_agent_observe(improved_agent):
    loss, reward = improved_agent.observe(task=0)
    assert isinstance(loss, np.ndarray)
    assert loss.dtype == np.float32
    assert isinstance(reward, float)


def test_improved_agent_calculate_reward(improved_agent):
    loss = 0.5
    reward = improved_agent.calculate_reward(loss, task=0)
    assert isinstance(reward, float)


def test_improved_agent_update_dqn(improved_agent):
    state = torch.randn(5)
    reward = 1.0
    task = 0
    improved_agent.update_dqn(state, reward, task)
    assert len(improved_agent.memory) == 1
