"""
DQN model class, based on the DQN model from the PyTorch tutorial:
"""
from typing import List
from tinygrad.tensor import Tensor


class DQN:
    """
    Deep Q-Network (DQN) class.

    Args:
        input_size (int): The number of input features.
        hidden_sizes (List[int]): The number of units in each hidden layer.
        output_size (int): The number of output units.

    Attributes:
        layers (List[Tensor]): The list of hidden layers.
        output_layer (Tensor): The output layer.

    Methods:
        forward(x): Forward pass through the network.

    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = None, output_size: int = 1):
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        self.layers = []
        in_size = input_size

        # Create hidden layers
        for size in hidden_sizes:
            self.layers.append(Tensor.uniform(in_size, size))
            in_size = size

        # Create output layer
        self.output_layer = Tensor.uniform(in_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the DQN network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_actions).
        """
        for layer in self.layers:
            x = x.dot(layer).relu()
        return x.dot(self.output_layer)
