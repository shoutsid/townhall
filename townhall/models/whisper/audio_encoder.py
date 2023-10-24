"""
Audio encoder that converts mel-spectrogram inputs into a sequence of hidden states.
"""

import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from residual_attention_block import ResidualAttentionBlock


class AudioEncoder:
    """
    Audio encoder that converts mel-spectrogram inputs into a sequence of hidden states.

    Args:
        n_mels (int): Number of mel-spectrogram channels in the input.
        n_audio_ctx (int): Length of the input sequence.
        n_audio_state (int): Number of hidden units in the encoder.
        n_audio_head (int): Number of attention heads in each residual block.
        n_audio_layer (int): Number of residual blocks in the encoder.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer.
        conv2 (nn.Conv1d): Second convolutional layer.
        blocks (List[ResidualAttentionBlock]): List of residual attention blocks.
        ln_post (nn.LayerNorm): Layer normalization layer.
        positional_embedding (Tensor): Positional embedding tensor.

    Methods:
        __call__(x): Encodes the input mel-spectrogram into a sequence of hidden states.

    """

    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.blocks = [
            ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)
        ]
        self.ln_post = nn.LayerNorm(n_audio_state)
        self.positional_embedding = Tensor.empty(n_audio_ctx, n_audio_state)

    def __call__(self, x):
        """
        Applies the audio encoder to the input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_channels, num_samples).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_features, num_frames),
                where `num_features` is the number of output features and `num_frames` is the
                number of output frames.
        """
        x = self.conv1(x).gelu()
        x = self.conv2(x).gelu()
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding[:x.shape[1]]
        x = x.sequential(self.blocks)
        x = self.ln_post(x)
        return x
