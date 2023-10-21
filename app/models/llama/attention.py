"""
Attention layer for LLAMA.
"""

from typing import Optional, Tuple, Dict
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.shape.symbolic import Variable
from utils import repeat_kv, apply_rotary_emb

np.set_printoptions(linewidth=200)

class Attention:
    """
    Multi-head attention layer.

    Args:
        dim (int): Hidden size of the input sequence.
        n_heads (int): Number of attention heads.
        n_kv_heads (Optional[int]): Number of attention heads for key and value projections. If None, defaults to `n_heads`.
        linear (Callable): Linear layer constructor. Defaults to `Linear`.

    Attributes:
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of attention heads for key and value projections.
        head_dim (int): Dimension of each attention head.
        n_rep (int): Number of times to repeat each key and value vector.
        wq (Linear): Linear layer for query projection.
        wk (Linear): Linear layer for key projection.
        wv (Linear): Linear layer for value projection.
        wo (Linear): Linear layer for output projection.

    Methods:
        __call__(
            self,
            x: Tensor,
            cache_k: Optional[Tensor],
            cache_v: Optional[Tensor],
            start_pos: int,
            freqs_cis: Tensor,
            mask: Optional[Tensor],
            jit_ctx: Optional[Dict[Variable, int]] = None
        ) -> Tuple[Tensor, Tensor, Tensor]:
            Apply multi-head attention to the input sequence `x`.
    """

    def __init__(self, dim, n_heads, n_kv_heads, linear=Linear):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

    def __call__(
        self,
        x: Tensor,
        cache_k: Optional[Tensor],
        cache_v: Optional[Tensor],
        start_pos: int,
        freqs_cis: Tensor,
        mask: Optional[Tensor],
        jit_ctx: Optional[Dict[Variable, int]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply multi-head attention to the input sequence `x`.

        Args:
            x (Tensor): Input sequence of shape `(batch_size, seq_len, hidden_size)`.
            cache_k (Optional[Tensor]): Cached keys tensor of shape `(batch_size, seq_len, n_kv_heads, head_dim)`.
            cache_v (Optional[Tensor]): Cached values tensor of shape `(batch_size, seq_len, n_kv_heads, head_dim)`.
            start_pos (int): Starting position for the current sequence.
            freqs_cis (Tensor): Sine and cosine frequencies for rotary embeddings of shape `(seq_len, hidden_size)`.
            mask (Optional[Tensor]): Mask tensor of shape `(batch_size, seq_len)` or `(batch_size, seq_len, seq_len)`.
            jit_ctx (Optional[Dict[Variable,int]]): JIT context.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the output tensor of shape `(batch_size, seq_len, hidden_size)`,
            the cached keys tensor of shape `(batch_size, seq_len, n_kv_heads, head_dim)`, and the cached values tensor of shape
            `(batch_size, seq_len, n_kv_heads, head_dim)`.
        """

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # kv caching!
        if start_pos == 0:
            keys, values = xk, xv
        else:
            assert cache_k is not None and cache_v is not None, "no cache"
            assert start_pos == (cache_k.shape[1].val if isinstance(cache_k.shape[1], Variable) else cache_k.shape[1]) == (cache_v.shape[1].val if isinstance(cache_v.shape[1], Variable) else cache_v.shape[1]), f"cache has wrong shape, {start_pos=}, {cache_k.shape[1]=}, {cache_v.shape[1]=}"
            assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
            keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

        cache_k, cache_v = keys, values
        keys, values = repeat_kv(keys, self.n_rep).realize(), repeat_kv(values, self.n_rep).realize()
        attn = (Tensor.scaled_dot_product_attention(xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), mask)
                .transpose(1, 2)
                .reshape(bsz, seqlen, -1))

        return self.wo(attn).realize(), cache_k.realize(), cache_v.realize()
