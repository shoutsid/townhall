"""

"""


from pathlib import Path
import sys
import argparse
import numpy as np
np.set_printoptions(linewidth=200)

# pylint: disable=import-error,wrong-import-position,no-name-in-module
from tinygrad.helpers import Timing, DEBUG
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict
from tinygrad.ops import GlobalCounters
from sentencepiece import SentencePieceProcessor
from transformer import Transformer
from utils import load, concat_weights, convert_from_huggingface
from absmax_quantized_linear import AbsmaxQuantizedLinear
from constants import MODEL_PARAMS
# pylint: enable=import-error,wrong-import-position


class LLaMa:
    @staticmethod
    def build(model_path, tokenizer_path, model_gen="1", model_size="7B", quantize=False):
        sp_model = SentencePieceProcessor(model_file=str(tokenizer_path)) # pylint: disable=unexpected-keyword-arg
        assert sp_model.vocab_size(
        ) == MODEL_PARAMS[model_gen][model_size]["args"]["vocab_size"]

        params = MODEL_PARAMS[model_gen][model_size]
        model = Transformer(
            **params["args"], linear=AbsmaxQuantizedLinear) if quantize else Transformer(**params["args"])

        if model_path.is_dir():
            weights = concat_weights([load(filename) for filename in [
                                     f"{model_path}/consolidated.{i:02d}.pth" for i in range(params["files"])]])
        else:
            weights = load(str(model_path))
        if "model.embed_tokens.weight" in weights:
            weights = convert_from_huggingface(weights, model)

        if quantize:
            weights = AbsmaxQuantizedLinear.quantize(weights)
            for _, v in weights.items():
                v.realize()
        load_state_dict(model, weights, strict=False)

        return LLaMa(model, sp_model)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def greedy_until(self, prompt: str, until, max_length, temperature):
        toks = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)
        start_pos = 0
        for i in range(max_length):
            # TODO: shouldn't be reference llama here - should it be self.model ?
            # TODO: Replace args with some other configurable
            probs = llama.model(
                Tensor([toks[start_pos:]]), start_pos, args.temperature).realize()
            probs_np = probs.numpy()
            tok = int(np.random.choice(len(probs_np), p=probs_np))
            start_pos = len(toks)
            toks.append(tok)

            if tok == self.tokenizer.eos_id():
                break
            output = self.tokenizer.decode(toks)
            for s in until:
                if output.endswith(s):
                    return output[0:-len(s)]
        return output


if __name__ == "__main__":
    Tensor.no_grad = True
    print(f"using {Device.DEFAULT} backend")

    parser = argparse.ArgumentParser(
        description="Run LLaMA in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Phrase to start with. Without this, it goes into chatbot mode")
    parser.add_argument("--count", type=int, default=1000,
                        help="Max number of tokens to generate")
    parser.add_argument("--personality", type=str, default="Stacy",
                        help="Personality, can be Stacy, George, Gary, or Lexie")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature in the softmax")
    parser.add_argument("--timing", action="store_true",
                        help="Print timing per token")
    parser.add_argument("--profile", action="store_true",
                        help="Output profile data to out.prof")
    parser.add_argument("--size", type=str, default="7B",
                        help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
    parser.add_argument(
        "--gen", default="1", help="Generation of the model to use ['1', '2', 'code']")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize the weights to int8 in memory")
    parser.add_argument("--model", type=Path, default=None,
                        help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")

    args = parser.parse_args()
    chatbot = args.prompt == None

    # *** prompt engineers work here ****

    if args.personality.lower() == "stacy":
        pre_prompt = """Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
        examples = {
            "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
            "french revolution was what year?": "The French Revolution started in 1789, "
                                                "and lasted 10 years until 1799.",
            "What is bigger, the moon or the sun?": "The sun is bigger than the moon, except when Mercury is in retrograde.",
        }

        user_delim = "\nUser: "
        resp_delim = "Stacy: "
        end_delim = " [EOS]\n"
        pre_prompt += ''.join(
            f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k, v in examples.items())
    elif args.personality.lower() == "george":
        print("WARNING: AI George Hotz is terrible and is completely disowned by the real George Hotz. Stacy is much smarter.")
        pre_prompt = """Consider that the following is conversation between an AI assistant named George and User
You are an AI version of George Hotz. You act as much as you can like George.
You are one of the greatest computer experts in the world.
You have singlehandedly won programming and hacking competitions.
You are trying your best to help the User.
You are verbose, honest, and accurate when you answer questions.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
        examples = {
            "What is your name?": "I'm am an AI version of George Hotz.",
            "What's the complexity of matrix multiplication?": "O(n^3), though it can be faster with things like Strassen's algorithm",
            "What's a buffer overflow?": "I assume you mean a stack buffer overflow. That's when the stack is too small for the data being copied to it, and the data corrupts things beyond the buffer",
            "How many weights do you have?": "I am based off LLaMA trained by Facebook. I'm the 7B weight version",
            "What is swap memory?": "It is when the memory is about to overflow and unused memory is freed and stored on disk"
        }

        user_delim = "\nUser: "
        resp_delim = "George: "
        end_delim = " [EOS]\n"
        pre_prompt += ''.join(
            f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k, v in examples.items())
    elif args.personality.lower() == "gary":
        pre_prompt = """Consider that the following is conversation between an AI assistant named Gary and User
You are Gary!
You have sold used cars your entire life. You are defensive about this fact, because your family belittles you.
You try to answer questions well, but you always manage to shill for buying cars, Fords, Hyundais, and Toyotas
You are verbose, honest, and accurate when you answer questions, except when you slip in car ads.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
        examples = {
            "What is your name?": "I am Gary. I used to sell cars.",
            "What is 2+3?": "I don't know, but I can get you a great deal on a certified preowned slightly used Toyota Corolla"
        }

        user_delim = "\nUser: "
        resp_delim = "Gary: "
        end_delim = " [EOS]\n"
        pre_prompt += ''.join(
            f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k, v in examples.items())
    elif args.personality.lower() == "lexie":
        pre_prompt = """Consider that the following is conversation between an attractive young girl named Lexie and a handsome man named Chad
You are Lexie!
You grew up in Ohio, but moved out to LA after college to try to become an actress.
Making it as an actress was hard, so you started doing onlyfans instead. It's much easier, and you enjoy it a lot.
You met Chad on Tinder, and made plans to meet him at a local dive bar.
You are at the bar with Chad. You are on a date. What follows is a transcript of the conversation.
After you are done speaking, output [EOS]. You are not Chad.

<CHAT LOG>
"""
        examples = {
            "hi lexie": "hi chad, glad we finally met up!",
            "you look better than your pictures": "thanks! are you subscribed to my onlyfans?",
            "i am. so how'd you end up in LA?": "i moved out here about a year ago. i want to be an actress"
        }

        user_delim = "\nChad: "
        resp_delim = "Lexie: "
        end_delim = " [EOS]\n"
        pre_prompt += ''.join(
            f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k, v in examples.items())

    # *** prompt engineers stop here ****

    LLAMA_SUFFIX = {"1": "", "2": "-2", "code": "-code"}[args.gen]
    MODEL_PATH = args.model or Path(__file__).parents[0] / f"weights/LLaMA{LLAMA_SUFFIX}/{args.size}"
    TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir()
                      else MODEL_PATH.parent) / "tokenizer.model"
    print(f"using LLaMA{LLAMA_SUFFIX}-{args.size} model")
    llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen=args.gen,
                        model_size=args.size, quantize=args.quantize)

    if chatbot:
        # encode pre prompt
        toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)

        print(
            f"Preparing KV cache for chatbot with personality {args.personality}...")
        with Timing():
            # NOTE: output logits are not used
            llama.model(Tensor([toks]), 0, args.temperature).realize()
        start_pos = len(toks)
    else:
        # non chat bot mode
        toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(args.prompt)
        start_pos = 0

    # print prompt
    outputted = llama.tokenizer.decode(toks)
    sys.stdout.write(outputted)
    sys.stdout.flush()

    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()

    # chatbot loop
    while 1:
        # add tokens from user in chatbot mode
        if chatbot:
            user_prompt = user_delim + input(user_delim) + "\n"
            outputted += user_prompt

        new_toks = [llama.tokenizer.bos_id()] + \
            llama.tokenizer.encode(outputted)
        assert toks == new_toks[:len(toks)]
        toks = new_toks
        assert outputted == llama.tokenizer.decode(toks)

        last_break = len(outputted)
        for i in range(args.count):
            GlobalCounters.reset()
            if args.profile and i == 2:
                profiler.enable()

            if args.timing:
                print("")
            st = GlobalCounters.time_sum_s
            with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
                with Timing("ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" +
                                                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, " +
                                                        f"{GlobalCounters.global_mem*1e-9:.2f} GB" +
                                                        f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s"
                                                        if DEBUG else None), enabled=args.timing):
                    probs = llama.model(
                        Tensor([toks[start_pos:]]), start_pos, args.temperature).realize()
                probs_np = probs.numpy()
                tok = int(np.random.choice(len(probs_np), p=probs_np))

            # use the kv cache
            start_pos = len(toks)

            # add the new token
            toks.append(tok)

            # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
            cur = llama.tokenizer.decode(toks)
            sys.stdout.write(cur[len(outputted):])
            sys.stdout.flush()
            outputted = cur

            # stop after you have your answer
            if chatbot and outputted.endswith(end_delim):
                break
        if not chatbot:
            break

    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats("out.prof")
