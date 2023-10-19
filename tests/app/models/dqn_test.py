import torch
from app.models.dqn import DQN


def test_dqn_forward():
    """
    Test the forward method of the DQN model.

    Creates a DQN model with 2 input features, 2 hidden layers with 3 and 4 units, and 1 output unit.
    Creates a batch of 2 input tensors of shape (2,).
    Passes the input through the model.
    Checks that the output has shape (2, 1).
    """
    model = DQN(input_size=2, hidden_sizes=[3, 4], output_size=1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = model.forward(x)
    assert output.shape == (2, 1)


def test_dqn_output():
    """
    Test the output of a DQN model with 2 input features, 2 hidden layers with 3 and 4 units, and 1 output unit.

    Creates a DQN model with the specified architecture, generates a batch of 2 input tensors of shape (2,),
    passes the input through the model, and checks that the output values are between -1 and 1.
    """
    model = DQN(input_size=2, hidden_sizes=[3, 4], output_size=1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = model.forward(x)
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
