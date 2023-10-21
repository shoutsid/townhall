"""
GPT2 model
"""

import argparse
import tiktoken # pylint: disable=import-error
import numpy as np
from tinygrad.helpers import Timing, getenv, DEBUG
from tinygrad.ops import GlobalCounters, Device
from tinygrad.tensor import Tensor
from tinygrad.nn.state import torch_load, load_state_dict
from extra.utils import fetch_as_file
from constants import MODEL_PARAMS
from tqdm import trange
from transformer import Transformer
from utils import get_url # pylint: disable=import-error

class GPT2:
    """
    A class representing the GPT-2 model.

    Attributes:
    model (Transformer): The GPT-2 transformer model.
    tokenizer (tiktoken.Encoding): The tokenizer used to encode and decode text for the model.
    """

    @staticmethod
    def build(model_size="gpt2"):
        """
        Builds a GPT-2 model with the specified size.

        Args:
        model_size (str): The size of the GPT-2 model to build. Defaults to "gpt2".

        Returns:
        GPT2: The built GPT-2 model.
        """
        tokenizer = tiktoken.get_encoding("gpt2")

        params = MODEL_PARAMS[model_size]
        model = Transformer(**params)
        weights = torch_load(fetch_as_file(get_url(model_size)))
        # special treatment for the Conv1D weights we need to transpose
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'
        ]
        for k in weights.keys():
            if any(k.endswith(w) for w in transposed):
                weights[k] = Tensor(weights[k].numpy().T)
        # lm head and wte are tied
        weights['lm_head.weight'] = Tensor(weights['wte.weight'].numpy())

        if getenv("FP16"):
            for k, v in weights.items():
                weights[k] = v.cpu().half().realize()
        load_state_dict(model, weights)
        return GPT2(model, tokenizer)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def greedy_until(
            self,
            prompt: str,
            max_length: int,
            temperature: float,
            timing: bool = False
    ) -> str:
        """
        Generates text starting from the given prompt using greedy decoding.

        Args:
            prompt (str): The prompt to start the generation from.
            max_length (int): The maximum length of the generated text.
            temperature (float): The temperature to use during generation.
            timing (bool, optional): Whether to print timing information. Defaults to False.

        Returns:
            str: The generated text.
        """
        toks = self.tokenizer.encode(prompt)
        start_pos = 0
        for _ in trange(max_length, disable=timing is True):
            GlobalCounters.reset()
            if timing:
                print("")
            st = GlobalCounters.time_sum_s
            with Timing("total ", enabled=timing):
                # pylint: disable=line-too-long
                with Timing("ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU"+
                            f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                            f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s") if DEBUG else None, enabled=timing):
                    probs = self.model(Tensor([toks[start_pos:]]), start_pos, temperature)
                # pylint: enable=line-too-long
                probs_np = probs.numpy()
                tok = int(np.random.choice(len(probs_np), p=probs_np))
            start_pos = len(toks)
            toks.append(tok)
            output = self.tokenizer.decode(toks)
        return output


if __name__ == "__main__":
    Tensor.no_grad = True
    print(f"using {Device.DEFAULT} backend")

    parser = argparse.ArgumentParser(
        description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--prompt',
        type=str,
        default="What is the answer to life, the universe, and everything?",
        help="Phrase to start with"
    )
    parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
    parser.add_argument(
        '--model_size',
        type=str,
        default="gpt2-medium",
        help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]"
    )
    parser.add_argument('--timing', action='store_true', help="Print timing per token")
    parser.add_argument('--seed', type=int, help="Set the random seed")
    args = parser.parse_args()

    if args.seed is not None:
        Tensor._seed = args.seed # pylint: disable=protected-access
        np.random.seed(args.seed)

    print(f"using {args.model_size}")
    gpt2 = GPT2.build(args.model_size)
    print('Generating text...')
    y = gpt2.greedy_until(args.prompt, args.count, args.temperature, timing=args.timing)
    print(y)
