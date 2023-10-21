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
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_seq_len=1024):
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps, linear) for _ in range(n_layers)]
    self.kv_caches = None
    self.n_layers = n_layers
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = linear(dim, vocab_size, bias=False)

  def embed(self, tokens:Tensor, pos:Tensor):
    tok_emb = self.wte(tokens)
    pos_emb = self.wpe(pos)
    h = tok_emb + pos_emb
    if getenv("FP16"): h = h.half()
    return h

  def postprocess(self, x, temperature:Optional[float]):
    logits = self.lm_head(self.ln_f(x))
    if temperature is not None: return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten()
    return logits

  @TinyJit
  def run_all_layers(self, tokens:Tensor, pos:Tensor, start_pos:int, temperature:float, **kv_cache):
    h = self.embed(tokens, pos)

    for i, hi in enumerate(self.h):
      h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'] = hi(h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'], start_pos=start_pos, mask=None)

    # don't realize until here
    for v in kv_cache.values(): v.realize()
    return self.postprocess(h, temperature).realize(), kv_cache

  def __call__(self, tokens:Tensor, start_pos:int, temperature:Optional[float]=None):
    _bsz, seqlen = tokens.shape
    if seqlen == 1 and start_pos > 0 and getenv("JIT"):
      start_pos_var = Variable("start_pos", 1, MAX_CONTEXT).bind(start_pos)
      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos_var, start_pos_var + seqlen)))
      for k,v in self.kv_caches.items():
        self.kv_caches[k] = v.reshape(v.shape[0], start_pos_var, v.shape[2], v.shape[3])
      logit_or_softmax, self.kv_caches = self.run_all_layers(tokens, pos, start_pos=start_pos, temperature=temperature, **self.kv_caches)
      return logit_or_softmax
    else:
      if start_pos == 0:
        if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
        self.kv_caches = {**{f'cache_k{i}':None  for i in range(self.n_layers)}, **{f'cache_v{i}':None for i in range(self.n_layers)}}

      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos, start_pos+seqlen)))
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float16 if getenv('FP16') else dtypes.float32).triu(start_pos+1).realize()
      h = self.embed(tokens, pos)
      for i, hi in enumerate(self.h):
        h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'] = hi(h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'], start_pos=start_pos, mask=mask)
      for v in self.kv_caches.values(): v.realize()
      return self.postprocess(h, temperature).realize()