"""
Root Mean Square (RMS) normalization layer.
"""

from tinygrad.tensor import Tensor


class RMSNorm:
    """
    Root Mean Square (RMS) normalization layer.

    Args:
        dim (int): The number of dimensions in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
    """

    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Applies the RSM normalization to the input tensor `x`.

        Args:
            x (Tensor): The input tensor to normalize.

        Returns:
            Tensor: The normalized tensor.
        """
        return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight
