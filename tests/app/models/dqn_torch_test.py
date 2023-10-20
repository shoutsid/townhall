import pytest
import torch
from app.models.dqn_torch import DQN


@pytest.fixture
def dqn():
    return DQN(input_size=4, hidden_sizes=[32, 16], output_size=2)


def test_dqn_forward(dqn):
    x = torch.randn(3, 4)
    output = dqn.forward(x)
    assert output.shape == (3, 2)


def test_dqn_no_hidden_sizes():
    dqn = DQN(input_size=4, output_size=2)
    x = torch.randn(3, 4)
    output = dqn.forward(x)
    assert output.shape == (3, 2)
