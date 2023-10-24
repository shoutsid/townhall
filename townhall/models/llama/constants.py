"""
This file contains the model parameters for the LLAMA models.
"""

MODEL_PARAMS = {
    "1": {
        "7B": {
            "args": {
                "dim": 4096,
                "multiple_of": 256,
                "n_heads": 32,
                "n_layers": 32,
                "norm_eps": 1e-06,
                "vocab_size": 32000
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "multiple_of": 256,
                "n_heads": 40,
                "n_layers": 40,
                "norm_eps": 1e-06,
                "vocab_size": 32000
            },
            "files": 2,
        },
        "30B": {
            "args": {
                "dim": 6656,
                "multiple_of": 256,
                "n_heads": 52,
                "n_layers": 60,
                "norm_eps": 1e-06,
                "vocab_size": 32000
            },
            "files": 4,
        },
        "65B": {
            "args": {
                "dim": 8192,
                "multiple_of": 256,
                "n_heads": 64,
                "n_layers": 80,
                "norm_eps": 1e-05,
                "vocab_size": 32000
            },
            "files": 8,
        },
    },
    "2": {
        "7B": {
            "args": {
                "dim": 4096,
                "multiple_of": 256,
                "n_heads": 32,
                "n_layers": 32,
                "norm_eps": 1e-05,
                "vocab_size": 32000
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "multiple_of": 256,
                "n_heads": 40,
                "n_layers": 40,
                "norm_eps": 1e-05,
                "vocab_size": 32000
            },
            "files": 2,
        },
        "70B": {
            "args": {
                "dim": 8192,
                "multiple_of": 4096,
                "ffn_dim_multiplier": 1.3,
                "n_heads": 64,
                "n_kv_heads": 8,
                "n_layers": 80,
                "norm_eps": 1e-05,
                "vocab_size": 32000
            },
            "files": 8,
        },
    },
    "code": {
        "7B": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016
            },
            "files": 1,
        },
        "7B-Python": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000
            },
            "files": 1,
        },
        "7B-Instruct": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_heads": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016
            },
            "files": 2,
        },
        "13B-Python": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_heads": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000
            },
            "files": 2,
        },
        "13B-Instruct": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_headvocab_sizes": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000
            },
            "files": 2,
        },
        "34B": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016
            },
            "files": 4,
        },
        "34B-Python": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000
            },
            "files": 4,
        },
        "34B-Instruct": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000
            },
            "files": 4,
        },
    }
}
