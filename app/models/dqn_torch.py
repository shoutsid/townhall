"""
DQN model class, based on the DQN model from the PyTorch tutorial:
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class.

    Args:
        input_size (int): The number of input features.
        hidden_sizes (List[int]): The number of units in each hidden layer.
        output_size (int): The number of output units.

    Attributes:
        layers (List[nn.Linear]): The list of hidden layers.
        output_layer (nn.Linear): The output layer.

    Methods:
        forward(x): Forward pass through the network.

    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = None, output_size: int = 1):
        super(DQN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        self.layers = nn.ModuleList()
        in_size = input_size

        # Create hidden layers
        for size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, size))
            in_size = size

        # Create output layer
        self.output_layer = nn.Linear(in_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DQN network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_actions).
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
