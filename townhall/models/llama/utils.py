"""
Utilities for the llama model.
"""

from pathlib import Path
import json
from typing import List, Dict
from typing import Tuple
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.jit import JIT_SUPPORTED_DEVICE
from tinygrad.helpers import getenv, CI
from tinygrad.nn.state import safe_load, torch_load


JIT = getenv("JIT", 0 if CI else int(Device.DEFAULT in JIT_SUPPORTED_DEVICE))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequencies for the complex exponential used in the
    `precompute_cis` function.

    Args:
        dim (int): The dimension of the tensor to be transformed.
        end (int): The length of the tensor to be transformed.
        theta (float, optional): The scaling factor for the frequencies. Defaults to 10000.0.

    Returns:
        Tensor: A tensor of precomputed frequencies for the complex exponential.
            The tensor has shape (1, end, 1, dim//2, 2).
            The last dimension contains the cosine and sine values of the frequencies.
    """
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1)*freqs.unsqueeze(dim=0)
    return Tensor.stack([Tensor.cos(freqs), Tensor.sin(freqs)], dim=-1).reshape(1, end, 1, dim//2, 2)


def complex_mult(A, c, d):
    """
    Performs complex multiplication between a tensor A of shape (..., N, 2) and a complex number c + di.
    (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)

    Args:
    - A: tensor of shape (..., N, 2) representing N complex numbers
    - c: real part of the complex number to multiply with
    - d: imaginary part of the complex number to multiply with

    Returns:
    - tensor of shape (..., N, 2) representing the result of the complex multiplication
    """
    a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
    ro = a*c - b*d
    co = a*d + b*c
    return ro.cat(co, dim=-1)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Applies rotary embeddings to the input tensors xq and xk using the given frequency tensor freqs_cis.

    Args:
        xq (Tensor): The query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        xk (Tensor): The key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        freqs_cis (Tensor): The frequency tensor of shape (batch_size, num_heads, seq_len, head_dim, 2).

    Returns:
        A tuple of two tensors, each of shape (batch_size, num_heads, seq_len * head_dim * 2).

    This function applies rotary embeddings to the input tensors xq and xk using the given frequency tensor freqs_cis.
    The rotary embeddings are applied by multiplying the input tensors with complex-valued sinusoids, which are
    generated from the frequency tensor. The resulting tensors are flattened along the last dimension and returned
    as a tuple of two tensors.

    Note that the input tensors xq and xk should have the same shape, and the frequency tensor freqs_cis should have
    the same shape as the input tensors with an additional last dimension of size 2.
    """
    assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[
        1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == 5 and len(
        xk.shape) == 5 and len(freqs_cis.shape) == 5
    c, d = freqs_cis[:, :xq.shape[1], :, :,
                     0:1], freqs_cis[:, :xq.shape[1], :, :, 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Repeat the key or value tensor n_rep times along the last dimension.

    Args:
        x (Tensor): The input tensor of shape (batch_size, seq_len, n_kv_heads, head_dim).
        n_rep (int): The number of times to repeat the tensor along the last dimension.

    Returns:
        Tensor: The output tensor of shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim).

    Example:
        >>> x = randn(2, 3, 4, 5)
        >>> y = repeat_kv(x, 2)
        >>> y.shape
        Size([2, 3, 8, 5])
    """
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


def concat_weights(models: List) -> Dict:
    """
    Concatenates the weights of multiple models.

    Args:
        models (List): A list of models.

    Returns:
        dict: A dictionary containing the concatenated weights of the input models.
    """
    def convert(name) -> Tensor:
        disk_tensors = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=Device.DEFAULT)
        axis = 1 if name.startswith("tok_embeddings.") or name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
        lazy_tensors = [data.to(device=Device.DEFAULT) for data in disk_tensors]
        return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

    return {name: convert(name) for name in {name: None for model in models for name in model}}


def load(fn: str):
    """
    Load a model from a file.

    Args:
        fn (str): The path to the file containing the model.

    Returns:
        The loaded model.
    """
    if fn.endswith('.index.json'):
        with open(fn) as fp:
            weight_map = json.load(fp)['weight_map']
        parts = {n: load(Path(fn).parent / Path(n).name)
                 for n in set(weight_map.values())}
        return {k: parts[n][k] for k, n in weight_map.items()}
    elif fn.endswith(".safetensors"):
        return safe_load(fn)
    else:
        return torch_load(fn)


def convert_from_huggingface(weights, model):
    """
    Converts Hugging Face model weights to the format used by the LLAMA model.

    Args:
        weights (dict): A dictionary of Hugging Face model weights.
        model (LLAMA): An instance of the LLAMA model.

    Returns:
        dict: A dictionary of LLAMA model weights.
    """
    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        **{f"model.layers.{layer_index}.input_layernorm.weight": f"layers.{layer_index}.attention_norm.weight" for layer_index in range(len(model.layers))},
        **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(len(model.layers))},
        **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(len(model.layers))},
        **{f"model.layers.{layer_index}.mlp.{x}_proj.weight": f"layers.{layer_index}.feed_forward.w{y}.weight"
           for x, y in {"gate": "1", "down": "2", "up": "3"}.items()
           for layer_index in range(len(model.layers))},
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    return {keymap[k]: v for k, v in weights.items() if ".rotary_emb." not in k}
