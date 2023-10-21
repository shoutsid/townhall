"""
This module contains the AbsmaxQuantizedLinear class,
which represents a quantized linear layer with 8-bit integer representation.
"""

from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes


class AbsmaxQuantizedLinear:
    """
    A class representing a quantized linear layer with 8-bit integer representation.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): Whether to include bias in the layer. Default is False.
    """

    def __init__(self, in_features, out_features, bias=False):
        assert not bias
        self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
        self.scale = Tensor.ones(out_features, dtype=dtypes.half)

    def __call__(self, x):
        return x.dot(self.weight.cast(dtype=dtypes.half).T*self.scale)

    @staticmethod
    def quantize(tensors):
        """
        Quantizes the given tensors using 8-bit integer representation.

        Args:
            tensors (dict): A dictionary of tensors to be quantized.

        Returns:
            dict: A dictionary of quantized tensors.
        """
        new_tensors = {}
        for name, v in tensors.items():
            if "feed_forward" in name or ("attention.w") in name or name == "output.weight":
                scale = v.abs().max(axis=1) / 127.0
                int8_weight = (v.T/scale).T.cast(dtype=dtypes.int8)
                new_tensors[name] = int8_weight
                new_tensors[name.replace('weight', 'scale')] = scale
            else:
                new_tensors[name] = v
        return new_tensors
