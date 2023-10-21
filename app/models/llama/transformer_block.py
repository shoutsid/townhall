"""
Transformer block module.
"""

from typing import Optional, Dict, Tuple
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.shape.symbolic import Variable
from utils import JIT
from attention import Attention
from feed_forward import FeedForward
from rms_norm import RMSNorm


class TransformerBlock:
    """
    A transformer block that applies multi-head attention and feed-forward layers to an input tensor.

    Args:
        dim (int): The hidden size of the input tensor.
        multiple_of (int): The multiple of the hidden size for the feed-forward layer.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key-value attention heads.
        norm_eps (float): The epsilon value for normalization layers.
        linear (type): The type of linear layer to use.
        ffn_dim_multiplier (Optional[int]): The multiplier for the feed-forward layer hidden size.

    Attributes:
        attention (Attention): The multi-head attention layer.
        feed_forward (FeedForward): The feed-forward layer.
        attention_norm (RMSNorm): The normalization layer for the attention output.
        ffn_norm (RMSNorm): The normalization layer for the feed-forward output.

    Methods:
        __call__(self, x, cache_k, cache_v, start_pos, freqs_cis, mask=None, jit_ctx=None):
            Apply the transformer block to the input tensor `x`.
    """

    def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, linear=Linear, ffn_dim_multiplier=None):
        self.attention = Attention(dim, n_heads, n_kv_heads, linear)
        self.feed_forward = FeedForward(
            dim, 4*dim, multiple_of, linear, ffn_dim_multiplier)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def __call__(self, x: Tensor, cache_k: Optional[Tensor], cache_v: Optional[Tensor], start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor], jit_ctx: Optional[Dict[Variable, int]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply the transformer block to the input tensor `x`.

        Args:
            x (Tensor): The input tensor of shape `(batch_size, seq_len, hidden_size)`.
            cache_k (Optional[Tensor]): The cached key tensor of shape `(batch_size, seq_len, num_heads, head_size)`.
            cache_v (Optional[Tensor]): The cached value tensor of shape `(batch_size, seq_len, num_heads, head_size)`.
            start_pos (int): The starting position for positional encoding.
            freqs_cis (Tensor): The cosine frequencies tensor of shape `(seq_len, hidden_size)`.
            mask (Optional[Tensor]): The attention mask tensor of shape `(batch_size, num_heads, seq_len, seq_len)`.
            jit_ctx (Optional[Dict[Variable, int]]): The JIT context dictionary.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the output tensor of shape `(batch_size, seq_len, hidden_size)`,
            the updated cached key tensor of shape `(batch_size, seq_len, num_heads, head_size)`, and the updated cached value tensor
            of shape `(batch_size, seq_len, num_heads, head_size)`.
        """

        # bsz, seqlen, _ = x.shape
        if JIT and mask is None:
            assert cache_k is not None and cache_v is not None, "no cache"
            # BUG: Buffer expansion causes postion to exceed bounds, consider strategies.
            #
            #        /app/models/llama/llama.py", line 260, in <module>
            #          probs = llama.model(
            #        /app/models/llama/transformer.py", line 83, in __call__
            #          pos = Variable("pos", 1, 1024).bind(start_pos)
            #        /ext/tinygrad/tinygrad/shape/symbolic.py", line 155, in bind
            #          assert self.val is None and self.min<=val<=self.max, f"cannot bind {val} to {self}"
            pos = Variable("pos", 1, 1024).bind(start_pos)
            cache_k = cache_k.reshape(
                cache_k.shape[0], pos, cache_k.shape[2], cache_k.shape[3])
            cache_v = cache_v.reshape(
                cache_v.shape[0], pos, cache_v.shape[2], cache_v.shape[3])

        output, cache_k, cache_v = self.attention(self.attention_norm(
            x), cache_k, cache_v, start_pos, freqs_cis, mask, jit_ctx=jit_ctx)
        h = x + output
        return (h + self.feed_forward(self.ffn_norm(h))).realize(), cache_k.realize(), cache_v.realize()
