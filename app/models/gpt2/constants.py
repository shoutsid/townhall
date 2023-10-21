"""
This module contains constants for GPT-2 model parameters.
"""

VOCAB_SIZE = 50257

MODEL_PARAMS = {
    'gpt2': {
        "n_layers": 12,
        "n_heads": 12,
        "dim": 768,
        "norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE
    },   # 124M params
    'gpt2-medium': {
        "n_layers": 24,
        "n_heads": 16,
        "dim": 1024,
        "norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE
    },  # 350M params
    'gpt2-large': {
        "n_layers": 36,
        "n_heads": 20,
        "dim": 1280,
        "norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE
    },  # 774M params
    'gpt2-xl': {
        "n_layers": 48,
        "n_heads": 25,
        "dim": 1600,
        "norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE
    },  # 1558M params
}
