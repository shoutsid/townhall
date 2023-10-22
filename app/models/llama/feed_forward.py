"""
Feedforward neural network module.
"""

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear


class FeedForward:
    """
    A feedforward neural network with three linear layers.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension.
        multiple_of (int): The output dimension will be a multiple of this value.
        linear (nn.Module): The linear layer module to use. Default is tinygrad.nn.Linear.
        ffn_dim_multiplier (float): A multiplier for the hidden dimension. Default is None.

    Attributes:
        w1 (nn.Module): The first linear layer.
        w2 (nn.Module): The second linear layer.
        w3 (nn.Module): The third linear layer.
    """

    def __init__(self, dim, hidden_dim, multiple_of, linear=Linear, ffn_dim_multiplier=None):
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = linear(dim, hidden_dim, bias=False)
        self.w2 = linear(hidden_dim, dim, bias=False)
        self.w3 = linear(dim, hidden_dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Applies the feedforward neural network to the input tensor x.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the feedforward neural network.
        """
        return self.w2(self.w1(x).silu() * self.w3(x))
