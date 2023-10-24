"""
Transformer block module for GPT-2.
"""

from typing import Optional, Tuple
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from layer_norm import LayerNorm
from attention import Attention
from feed_forward import FeedForward

class TransformerBlock:
    """
    A single transformer block that applies multi-head attention and
    feedforward layers to its input.

    Args:
        dim (int): The input and output dimension of the transformer block.
        n_heads (int): The number of attention heads to use.
        norm_eps (float): The epsilon value used in layer normalization.
        linear (type): The type of linear layer to use. Default is `Linear`.

    Attributes:
        attn (Attention): The multi-head attention layer.
        mlp (FeedForward): The feedforward layer.
        ln_1 (LayerNorm): The layer normalization layer before the attention layer.
        ln_2 (LayerNorm): The layer normalization layer before the feedforward layer.

    Methods:
        __call__(
            self,
            x: Tensor,
            cache_k: Optional[Tensor],
            cache_v: Optional[Tensor],
            start_pos: int,
            mask: Optional[Tensor]
        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            Applies the transformer block to the input tensor `x`.

            Args:
                x (Tensor): The input tensor of shape `(batch_size, seq_len, dim)`.
                cache_k (Optional[Tensor]):
                  The cached key tensor for fast decoding. Default is `None`.
                cache_v (Optional[Tensor]):
                  The cached value tensor for fast decoding. Default is `None`.
                start_pos (int): The starting position for decoding. Default is `0`.
                mask (Optional[Tensor]):
                  The attention mask tensor of shape `(batch_size, n_heads, seq_len, seq_len)`.
                  Default is `None`.

            Returns:
                A tuple containing:
                - The output tensor of shape `(batch_size, seq_len, dim)`.
                - The updated cached key tensor for fast decoding.
                    - If `cache_k` is `None`, returns `None`.
                - The updated cached value tensor for fast decoding.
                    - If `cache_v` is `None`, returns `None`.
    """

    def __init__(self, dim, n_heads, norm_eps, linear=Linear):
        self.attn = Attention(dim, n_heads, linear)
        self.mlp = FeedForward(dim, 4*dim, linear)
        self.ln_1 = LayerNorm(dim, norm_eps)
        self.ln_2 = LayerNorm(dim, norm_eps)

    def __call__(
        self,
        x: Tensor,
        cache_k: Optional[Tensor],
        cache_v: Optional[Tensor],
        start_pos: int,
        mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Apply the transformer block to the input tensor `x`.

        Args:
            x (Tensor): The input tensor of shape `(batch_size, seq_len, hidden_size)`.
            cache_k (Optional[Tensor]):
                The cached key tensor of shape `(batch_size, num_heads, seq_len, head_size)`.
                If `None`, the key tensor is computed from scratch.
            cache_v (Optional[Tensor]):
                The cached value tensor of shape `(batch_size, num_heads, seq_len, head_size)`.
                If `None`, the value tensor is computed from scratch.
            start_pos (int): The starting position for decoding. Used only during inference.
            mask (Optional[Tensor]):
                The attention mask tensor of shape `(batch_size, seq_len, seq_len)`.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: A tuple containing:
                - The output tensor of shape `(batch_size, seq_len, hidden_size)`.
                - The updated cached key tensor of shape
                    - `(batch_size, num_heads, seq_len + start_pos, head_size)`.
                - The updated cached value tensor of shape
                    - `(batch_size, num_heads, seq_len + start_pos, head_size)`.
        """
        output, cache_k, cache_v = self.attn(self.ln_1(x), cache_k, cache_v, start_pos, mask)
        h = x + output
        h = (h + self.mlp(self.ln_2(h)))
        return h, cache_k, cache_v
