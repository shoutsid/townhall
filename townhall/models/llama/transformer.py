"""
This module contains the Transformer class, which is a Transformer model that
can be used for natural language processing tasks such as language modeling and text generation.
"""

from typing import Optional
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding
from tinygrad.shape.symbolic import Variable
from tinygrad.helpers import dtypes
from transformer_block import TransformerBlock
from rms_norm import RMSNorm
from tinygrad.jit import TinyJit
from utils import precompute_freqs_cis, JIT


class Transformer:
    """
    A Transformer model that can be used for natural language processing tasks such as language modeling and text generation.

    Args:
        dim (int): The dimension of the model.
        multiple_of (int): The multiple of the model dimension that the sequence length should be divisible by.
        n_heads (int): The number of attention heads to use in the model.
        n_layers (int): The number of layers in the model.
        norm_eps (float): The epsilon value to use for normalization.
        vocab_size (int): The size of the vocabulary.
        linear (class): The linear layer class to use in the model.
        max_batch_size (int): The maximum batch size to use in the model.
        max_seq_len (int): The maximum sequence length to use in the model.
        ffn_dim_multiplier (Optional[int]): The multiplier to use for the feedforward network dimension.
        n_kv_heads (Optional[int]): The number of key-value attention heads to use in the model.
        rope_theta (int): The theta value to use for the random orthogonal parameterization.
    """

    def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_batch_size=32, max_seq_len=1024, ffn_dim_multiplier=None, n_kv_heads=None, rope_theta=10000):
        self.layers = [TransformerBlock(
            dim, multiple_of, n_heads, n_kv_heads, norm_eps, linear, ffn_dim_multiplier) for _ in range(n_layers)]
        self.kv_caches = [(None, None) for _ in range(n_layers)]
        self.norm = RMSNorm(dim, norm_eps)
        self.tok_embeddings = Embedding(vocab_size, dim)
        self.output = linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            dim // n_heads, max_seq_len * 2, rope_theta)
        self.norm_output = lambda x: self.output(self.norm(x))

        self.tok_embeddings_jitted = TinyJit(
            lambda x: self.tok_embeddings(x).realize())
        self.postprocess_jitted = TinyJit(self.postprocess)
        self.layers_jitted = [TinyJit(layer.__call__) for layer in self.layers]

    def postprocess(self, x, temperature: Optional[float]):
        """
        Postprocesses the output of the transformer model.

        Args:
            x (Tensor): The input tensor to the transformer model.
            temperature (float, optional): The temperature to use for softmax. If None, no temperature scaling is applied.

        Returns:
            Tensor: The postprocessed output tensor.
        """
        logits = self.output(self.norm(x))
        if temperature is not None:
            return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten().realize()
        return logits.realize()

    def __call__(self, tokens: Tensor, start_pos: int, temperature: Optional[float] = None) -> Tensor:
        """
        Apply the transformer to the given input tokens.

        Args:
            tokens (Tensor): The input tokens to apply the transformer to.
            start_pos (int): The starting position of the input tokens.
            temperature (Optional[float]): The temperature to use for sampling. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the transformer.
        """

        _bsz, seqlen = tokens.shape
        if seqlen == 1 and start_pos > 0 and JIT:
            pos = Variable("pos", 1, 1024).bind(start_pos)
            # get only the part of freqs_cis that we are using.
            freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (pos, pos+seqlen), (
                0, self.freqs_cis.shape[2]), (0, self.freqs_cis.shape[3]), (0, self.freqs_cis.shape[4])))
            h = self.tok_embeddings_jitted(tokens)
            for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers_jitted, self.kv_caches)):
                h, cache_k, cache_v = layer(h, cache_k, cache_v, start_pos=start_pos,
                                            freqs_cis=freqs_cis, mask=None, jit_ctx={pos.unbind()[0]: start_pos})
                self.kv_caches[i] = (cache_k, cache_v)
            return self.postprocess_jitted(h, temperature)
        else:
            freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (start_pos, start_pos+seqlen), (
                0, self.freqs_cis.shape[2]), (0, self.freqs_cis.shape[3]), (0, self.freqs_cis.shape[4])))
            mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"),
                               dtype=dtypes.float32).triu(start_pos+1).realize()
            h = self.tok_embeddings(tokens)
            for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers, self.kv_caches)):
                # need this reshape back to int shape in conversational mode because jitted and unjitted calls share the same cache
                if cache_k is not None and start_pos > 0:
                    cache_k = cache_k.reshape(
                        cache_k.shape[0], start_pos, cache_k.shape[2], cache_k.shape[3])
                    cache_v = cache_v.reshape(
                        cache_v.shape[0], start_pos, cache_v.shape[2], cache_v.shape[3])
                h, cache_k, cache_v = layer(
                    h, cache_k, cache_v, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
                self.kv_caches[i] = (cache_k, cache_v)
            return self.postprocess(h, temperature)
