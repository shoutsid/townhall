from tinygrad.tensor import Tensor
from app.models.dqn import DQN


def test_dqn_forward_pass():
    # Create a DQN with 2 input features, 2 hidden layers with 3 and 4 units, and 2 output units
    dqn = DQN(input_size=2, hidden_sizes=[3, 4], output_size=2)

    # Create an input tensor of shape (1, 2)
    x = Tensor([[1.0, 2.0]])

    # Compute the output of the DQN for the input tensor
    y = dqn.forward(x)

    # Check that the output tensor has shape (1, 2)
    assert y.shape == (1, 2)


def test_dqn_default_hidden_sizes():
    # Create a DQN with 2 input features and default hidden layer sizes
    dqn = DQN(input_size=2)

    # Check that the DQN has 2 hidden layers with sizes 32 and 16
    assert len(dqn.layers) == 2
    assert dqn.layers[0].shape == (2, 32)
    assert dqn.layers[1].shape == (32, 16)


def test_dqn_custom_hidden_sizes():
    # Create a DQN with 2 input features and custom hidden layer sizes
    dqn = DQN(input_size=2, hidden_sizes=[4, 5, 6])

    # Check that the DQN has 3 hidden layers with sizes 4, 5, and 6
    assert len(dqn.layers) == 3
    assert dqn.layers[0].shape == (2, 4)
    assert dqn.layers[1].shape == (4, 5)
    assert dqn.layers[2].shape == (5, 6)
