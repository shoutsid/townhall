"""
Residual Attention Block
"""

import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from multi_head_attention import MultiHeadAttention

class ResidualAttentionBlock:
    """
    A residual attention block that applies multi-head attention and
        a multi-layer perceptron (MLP) to the input.

    Args:
        n_state (int): The number of features in the input.
        n_head (int): The number of attention heads.
        cross_attention (bool): Whether to apply cross-attention to the input.

    Attributes:
        attn (MultiHeadAttention): The multi-head attention layer.
        attn_ln (nn.LayerNorm): The layer normalization layer for the multi-head attention layer.
        cross_attn (MultiHeadAttention):
            The cross-attention layer if `cross_attention` is True, otherwise None.
        cross_attn_ln (nn.LayerNorm):
            The layer normalization layer for the cross-attention
                layer if `cross_attention` is True, otherwise None.
        mlp (list): The list of layers in the multi-layer perceptron.
        mlp_ln (nn.LayerNorm): The layer normalization layer for the multi-layer perceptron.
    """
    def __init__(self, n_state, n_head, cross_attention=False):
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None):
        """
        Apply the residual attention block to the input tensor `x`.

        Args:
            x (Tensor): The input tensor of shape `(batch_size, seq_len, hidden_size)`.
            xa (Tensor, optional):
                The auxiliary input tensor of shape `(batch_size, seq_len, hidden_size)`.
            mask (Tensor, optional):
                The attention mask tensor of shape `(batch_size, seq_len, seq_len)`.

        Returns:
            Tensor: The output tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn is not None:
            # pylint: disable=not-callable
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
            # pylint: enable=not-callable
        x = x + self.mlp_ln(x).sequential(self.mlp)
        return x
