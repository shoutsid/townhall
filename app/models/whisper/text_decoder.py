import numpy as np
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from residual_attention_block import ResidualAttentionBlock

class TextDecoder:
    """
    Transformer-based decoder for text generation.

    Args:
        n_vocab (int): size of the vocabulary.
        n_text_ctx (int): size of the context window.
        n_text_state (int): size of the hidden state.
        n_text_head (int): number of attention heads.
        n_text_layer (int): number of layers.
    """

    def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
        self.token_embedding = nn.Embedding(n_vocab, n_text_state)
        self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
        self.blocks = [ResidualAttentionBlock(
            n_text_state, n_text_head, cross_attention=True) for _ in range(n_text_layer)]
        self.ln = nn.LayerNorm(n_text_state)

    def __call__(self, x, xa):
        """
        Apply the transformer to the input sequence x.

        Args:
            x (Tensor):
                Input sequence of shape (batch_size, seq_len).
            xa (Tensor):
                Additional input sequence of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, vocab_size).
        """
        offset = 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]

        seqlen, start_pos = x.shape[1], 0

        mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=np.float32)
        mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
        mask = Tensor(mask)

        for block in self.blocks:
            x = block(x, xa, mask)
        x = self.ln(x)
        return x @ self.token_embedding.weight.T
