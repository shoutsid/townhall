"""
Attention layer for GPT-2
"""

from typing import Optional
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear


class Attention: # pylint: disable=too-few-public-methods
    """
    Multi-head scaled dot-product attention layer.

    Args:
        dim (int): dimension of the input and output tensors.
        n_heads (int): number of attention heads.
        linear (callable): function used to create the linear layers. Default is `Linear`.

    Attributes:
        c_attn (Linear):
            linear layer used to project the input tensor into query, key and value tensors.
        c_proj (Linear):
            linear layer used to project the concatenated output of the attention heads.
        n_heads (int): number of attention heads.
        dim (int): dimension of the input and output tensors.
        head_dim (int): dimension of each attention head.

    Methods:
        __call__(
            self,
            x: Tensor,
            cache_k: Optional[Tensor],
            cache_v: Optional[Tensor],
            start_pos: int,
            mask: Optional[Tensor]
        ) -> Tensor:
            Applies the multi-head attention layer to the input tensor.

    """
    def __init__(self, dim, n_heads, linear=Linear):
        self.c_attn = linear(dim, 3*dim, bias=True)
        self.c_proj = linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def __call__(
            self,
            x: Tensor,
            cache_k: Optional[Tensor],
            cache_v: Optional[Tensor],
            start_pos: int,
            mask: Optional[Tensor]
        ) -> Tensor:
        """
        Applies the multi-head attention layer to the input tensor.

        Args:
            x (Tensor):
                Input tensor of shape `(batch_size, sequence_length, dim)`.
            cache_k (Optional[Tensor]):
                Cached key tensor of shape `(batch_size, sequence_length - 1, dim)`.
                Default is `None`.
            cache_v (Optional[Tensor]):
                Cached value tensor of shape
                `(batch_size, sequence_length - 1, dim)`. Default is `None`.
            start_pos (int):
                Starting position of the input tensor. Used for caching.
                Default is `0`.
            mask (Optional[Tensor]):
                Mask tensor of shape `(batch_size, sequence_length)`.
                Default is `None`.

        Returns:
            output (Tensor): Output tensor of shape `(batch_size, sequence_length, dim)`.

        """
        xqkv = self.c_attn(x)
        xq, xk, xv = [xqkv.slice([None, None, (i*self.dim, (i+1)*self.dim)]) for i in range(3)]
        xq, xk, xv = [
            x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)
        ]

        bsz, seqlen, _, _ = xq.shape
        # kv caching!
        if start_pos == 0:
            keys, values = xk, xv
        else:
            assert cache_k, "no cache"
            assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
            keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

        # save the cache
        cache_k, cache_v = keys, values
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        output = xq.scaled_dot_product_attention(
            keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.c_proj(output), cache_k, cache_v
