"""
Multi-Head Attention layer that takes in a sequence of vectors and returns
a weighted sum of the sequence.
"""

from typing import Optional
import tinygrad.nn as nn
from tinygrad.tensor import Tensor


class MultiHeadAttention:
    """
    Multi-Head Attention layer that takes in a sequence of
        vectors and returns a weighted sum of the sequence.
    Args:
        n_state (int): The number of features in the input vectors.
        n_head (int): The number of attention heads to use.
    """

    def __init__(self, n_state, n_head):
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None
        ) -> Tensor:
        """
        Apply multi-head attention to the input tensor.

        Args:
            x (Tensor): The input tensor.
            xa (Optional[Tensor]): The tensor to attend to. If None, use x.
            mask (Optional[Tensor]): An optional mask tensor.

        Returns:
            Tensor: The output tensor after applying multi-head attention.
        """
        q = self.query(x)
        k = self.key(xa or x)
        v = self.value(xa or x)
        wv, _qk = self.qkv_attention(q, k, v, mask)
        # NOTE: we aren't returning qk
        return self.out(wv)

    def qkv_attention(self, q, k, v, mask=None):
        """
        Computes the scaled dot-product attention between query (q), key (k) and value (v) tensors.
        Args:
            q (Tensor): Query tensor of shape (batch_size, sequence_length, hidden_size).
            k (Tensor): Key tensor of shape (batch_size, sequence_length, hidden_size).
            v (Tensor): Value tensor of shape (batch_size, sequence_length, hidden_size).
            mask (Tensor, optional): Mask tensor of shape (sequence_length, sequence_length).
                Defaults to None.

        Returns:
            Tensor: Tensor of shape (batch_size, sequence_length, hidden_size) representing the
                output of the attention mechanism.
            Tensor: Tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
                representing the attention weights.
        """
        n_ctx, n_state = q.shape[:2]
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = qk.softmax(-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
