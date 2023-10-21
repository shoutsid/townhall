# GPT-2 Integration in Townhall

## Overview

This module integrates the GPT-2 language model into the Townhall project using the Tinygrad framework. The GPT-2 model comes in different sizes including "gpt2-medium", and is useful for a variety of natural language processing tasks.

## Installation

The installation steps are the same as those of the Townhall project. Make sure to update your `requirements.txt` to include dependencies for GPT-2.

## File Structure


## Overview

This module integrates the GPT-2 language model into the Townhall project using the Tinygrad framework. The GPT-2 model comes in different sizes including "gpt2-medium", and is useful for a variety of natural language processing tasks.

## Installation

The installation steps are the same as those of the Townhall project. Make sure to update your `requirements.txt` to include dependencies for GPT-2.

## File Structure

-   `attention.py`: Implements the Attention mechanism used in the Transformer architecture.
-   `constants.py`: Holds constant values and configurations for the GPT-2 model.
-   `feed_forward.py`: Implements the Feed Forward neural network used within the Transformer block.
-   `gpt2.py`: The main file that builds and runs the GPT-2 model.
-   `layer_norm.py`: Implements Layer Normalization.
-   `transformer.py`: Implements the Transformer architecture.
-   `transformer_block.py`: Implements a block of the Transformer architecture, consisting of one attention layer and one feed-forward layer.
-   `utils.py`: Various utility functions.

## Usage

Run the `gpt2.py` script to use the GPT-2 model:

```bash
(.venv) python3 app/models/gpt2/gpt2.py
```

### Parameters

-   `--prompt`: The prompt text to start generating from.
-   `--count`: The maximum number of tokens to generate.
-   `--temperature`: Controls the randomness of the output.
-   `--model_size`: The size of the GPT-2 model to use (e.g., "gpt2-medium").
-   `--timing`: Enable timing logs.

## Performance Metrics

The GPT-2 model can be run using different configurations, and its performance can be observed as follows:

-   Loaded weights in 1599.03 ms, 1.63 GB loaded at 1.02 GB/s
-   Text generation speed varies based on the model size and hardware.

## Contributions

This integration is part of issue #50 in the Townhall project. Feel free to contribute by submitting pull requests or creating issues.

```bash
(.venv) james@Sid-Desktop:/mnt/e/workspace/townhall$ /mnt/e/workspace/townhall/.venv/bin/python /mnt/e/workspace/townhall/app/models/gpt2/gpt2.py
using GPU backend
using gpt2-medium
https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1.52G/1.52G [01:02<00:00, 24.4MB/s]
ram used:  1.63 GB, lm_head.weight                                    : 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 293/293 [00:01<00:00, 179.72it/s]
loaded weights in 1630.84 ms, 1.63 GB loaded at 1.00 GB/s
Generating text...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:55<00:00,  4.76s/it]
What is the answer to life, the universe, and everything? What is the goal of life? What is the meaning of my life? What am I entitled to for my life? These are the questions that are imperative to the human psyche. Understanding these questions is the foundation for these answers.

I define this concept of existence as living and functioning as separate individuals, not as an aggregate of things. I am not interested in the resoi e of those today who have lived a life of comfort, comfort, and comfort, and have been content to live
(.venv) james@Sid-Desktop:/mnt/e/workspace/townhall$ /mnt/e/workspace/townhall/.venv/bin/python /mnt/e/workspace/townhall/app/models/gpt2/gpt2.py
using GPU backend
using gpt2-medium
ram used:  1.63 GB, lm_head.weight                                    : 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 293/293 [00:01<00:00, 184.06it/s]
loaded weights in 1599.03 ms, 1.63 GB loaded at 1.02 GB/s
Generating text...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:31<00:00,  3.20it/s]
What is the answer to life, the universe, and everything? What is the meaning of life, the universe, and everything?

We are moving through an age of exploration, discovery, and possibility, and we are learning from ourselves.

With that in mind, the question is, What are we building that will allow us to live out our values in an era of uncertainty?

Here are four principles that help me visualize our journey through this new era:

1: Design for a future world

It is not enough to design
```
