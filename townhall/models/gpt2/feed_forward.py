"""
Feedforward layer with GELU activation function.
"""

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class FeedForward: # pylint: disable=too-few-public-methods
    """
    A feedforward neural network layer with a GELU activation function.

    Args:
        dim (int): The input dimension of the layer.
        hidden_dim (int): The hidden dimension of the layer.
        linear (nn.Module): The linear layer module to use. Default is `tingrad.nn.Linear`.

    Attributes:
        c_fc (nn.Module): The linear layer module for the feedforward computation.
        c_proj (nn.Module): The linear layer module for the projection computation.

    Methods:
        __call__(x: Tensor) -> Tensor:
            Computes the feedforward layer output for the given input tensor.

    Example:
        >>> ff = FeedForward(dim=512, hidden_dim=2048)
        >>> x = randn(1, 512)
        >>> y = ff(x)
        >>> y.shape
        Size([1, 512])
    """
    def __init__(self, dim, hidden_dim, linear=Linear):
        self.c_fc = linear(dim, hidden_dim, bias=True)
        self.c_proj = linear(hidden_dim, dim, bias=True)

    def __call__(self, x:Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())
