"""
This module contains utility functions for the GPT-2 model.
"""

def get_url(model_size):
    """
    Returns the URL to download the pre-trained model weights for a given model size.

    Args:
        model_size (str): The size of the GPT-2 model to download.

    Returns:
        str: The URL to download the pre-trained model weights.
    """
    return f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'
