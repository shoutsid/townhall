"""
Layer Normalization (https://arxiv.org/abs/1607.06450)
"""

from tinygrad.tensor import Tensor

class LayerNorm:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Args:
        dim (int): The size of the last dimension of the input tensor.
        eps (float, optional):
          A value added to the denominator for numerical stability. Default: 1e-5.
    """
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to the input tensor `x`.

        Args:
            x (Tensor): The input tensor to normalize.

        Returns:
            Tensor: The normalized tensor, with the same shape as `x`.
        """
        return (x.layernorm(eps=self.eps)) * self.weight + self.bias
