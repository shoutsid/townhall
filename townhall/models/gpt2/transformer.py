"""
The Transformer class represents a GPT-2 transformer model.
"""

from typing import Optional
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn import Linear, Embedding
from tinygrad.jit import TinyJit
from tinygrad.shape.symbolic import Variable
from layer_norm import LayerNorm
from transformer_block import TransformerBlock

MAX_CONTEXT = 128

class Transformer:
    """
    A Transformer class that represents a GPT-2 transformer model.

    Args:
        dim (int): The dimension of the model.
        n_heads (int): The number of heads in the multi-head attention layer.
        n_layers (int): The number of layers in the model.
        norm_eps (float): The epsilon value for normalization.
        vocab_size (int): The size of the vocabulary.
        linear (class): The linear layer class to use.
        max_seq_len (int): The maximum sequence length.

    Attributes:
        wte (Embedding): The word token embedding layer.
        wpe (Embedding): The positional embedding layer.
        h (list): A list of TransformerBlock objects representing the transformer blocks.
        kv_caches (None or dict): A dictionary of key-value caches for each transformer block.
        n_layers (int): The number of layers in the model.
        ln_f (LayerNorm): The final layer normalization layer.
        lm_head (Linear): The linear layer for the language model head.

    Methods:
        embed(tokens: Tensor, pos: Tensor) -> Tensor:
            Embeds the input tokens and positional encodings.

        postprocess(x: Tensor, temperature: Optional[float]) -> Tensor:
            Computes the logits for the output sequence.

        run_all_layers(
            tokens: Tensor, pos: Tensor, start_pos: int, temperature: float, **kv_cache
        ) -> Tuple[Tensor, dict]:
            Runs all the transformer blocks and returns the logits and key-value caches.

        __call__(tokens: Tensor, start_pos: int, temperature: Optional[float] = None) -> Tensor:
            Calls the model with the input tokens and returns the logits for the output sequence.
    """

    def __init__(
            self, dim, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_seq_len=1024
        ):
        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(max_seq_len, dim)
        self.h = [TransformerBlock(dim, n_heads, norm_eps, linear) for _ in range(n_layers)]
        self.kv_caches = None
        self.n_layers = n_layers
        self.ln_f = LayerNorm(dim, norm_eps)
        self.lm_head = linear(dim, vocab_size, bias=False)

    def embed(self, tokens: Tensor, pos: Tensor):
        """
        Embeds the input tokens and positions into a high-dimensional space.

        Args:
            tokens (Tensor): The input tokens to be embedded.
            pos (Tensor): The positions of the input tokens.

        Returns:
            Tensor: The embedded representation of the input tokens and positions.
        """
        tok_emb = self.wte(tokens)
        pos_emb = self.wpe(pos)
        h = tok_emb + pos_emb
        if getenv("FP16"):
            h = h.half()
        return h

    def postprocess(self, x, temperature: Optional[float]):
        """
        Postprocesses the output of the GPT-2 model.

        Args:
            x (Tensor): The output tensor from the GPT-2 model.
            temperature (float, optional): The temperature to apply to the logits before softmax.
                If None, no temperature is applied.

        Returns:
            Tensor: The postprocessed output tensor.
        """
        logits = self.lm_head(self.ln_f(x))
        if temperature is not None:
            return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten()
        return logits

    @TinyJit
    def run_all_layers(
        self,
        tokens: Tensor,
        pos: Tensor,
        start_pos: int,
        temperature: float,
        **kv_cache
    ):
        """
        Runs the input tokens through all the transformer layers and returns the output.

        Args:
            tokens (Tensor): The input tokens to be processed.
            pos (Tensor): The positional encoding for the input tokens.
            start_pos (int): The starting position for the positional encoding.
            temperature (float):
                The temperature value to use for sampling from the output distribution.
            **kv_cache: Optional keyword arguments for caching intermediate results.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                A tuple containing the output tensor and a
                dictionary of cached intermediate results.
        """
        h = self.embed(tokens, pos)

        for i, hi in enumerate(self.h):
            h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'] = hi(
                h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'], start_pos=start_pos, mask=None
            )

        # don't realize until here
        for v in kv_cache.values():
            v.realize()
        return self.postprocess(h, temperature).realize(), kv_cache

    def __call__(
            self,
            tokens: Tensor,
            start_pos: int,
            temperature: Optional[float] = None
        ) -> Tensor:
        """
        Apply the transformer to the given input tokens, starting from the specified position.

        Args:
            tokens (Tensor): The input tokens to process, with shape (batch_size, sequence_length).
            start_pos (int): The position in the sequence to start processing from.
            temperature (Optional[float]):
              The temperature to use when sampling from the model's output distribution.

        Returns:
            Tensor:
              The output of the transformer, with shape (batch_size, sequence_length, vocab_size).
        """
        _bsz, seqlen = tokens.shape
        if seqlen == 1 and start_pos > 0 and getenv("JIT"):
            start_pos_var = Variable("start_pos", 1, MAX_CONTEXT).bind(start_pos)
            #pylint: disable=access-member-before-definition
            pos = self.allpos.shrink((
                (0, self.allpos.shape[0]), (start_pos_var, start_pos_var + seqlen)
            )) # noqa
            #pylint: enable=access-member-before-definition
            for k,v in self.kv_caches.items():
                self.kv_caches[k] = v.reshape(v.shape[0], start_pos_var, v.shape[2], v.shape[3])
            logit_or_softmax, self.kv_caches = self.run_all_layers(
                tokens, pos, start_pos=start_pos, temperature=temperature, **self.kv_caches)
            return logit_or_softmax
        else:
            if start_pos == 0:
                if not hasattr(self, 'allpos'):
                    # pylint: disable=attribute-defined-outside-init
                    self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize() # noqa
                    # pylint: enable=attribute-defined-outside-init
                self.kv_caches = {
                    **{f'cache_k{i}':None for i in range(self.n_layers)},
                    **{f'cache_v{i}':None for i in range(self.n_layers)}
                }

            pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos, start_pos+seqlen)))
            mask = Tensor.full(
                (1, 1, seqlen, start_pos + seqlen),
                float("-inf"),
                dtype=dtypes.float16 if getenv('FP16') else dtypes.float32
            ).triu(start_pos+1).realize()
            mask = Tensor.full(
                (1, 1, seqlen, start_pos + seqlen),
                float("-inf"),
                dtype = dtypes.float16 if getenv('FP16') else dtypes.float32
            ).triu(start_pos+1).realize()
            h = self.embed(tokens, pos)

            for i, hi in enumerate(self.h):
                h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'] = hi(
                    h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'],
                    start_pos=start_pos, mask=mask
                )

            for v in self.kv_caches.values():
                v.realize()
            return self.postprocess(h, temperature).realize()
