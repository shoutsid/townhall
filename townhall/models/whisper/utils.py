"""
Utility functions for the Whisper model.
"""

import itertools
import base64
import librosa # pylint: disable=import-error
import matplotlib.pyplot as plt
import numpy as np
from tiktoken import Encoding
from tinygrad.tensor import Tensor
from tinygrad.nn.state import torch_load, load_state_dict
from pyaudio import PyAudio, paInt16
from extra.utils import download_file
from constants import (
    RATE, N_FFT, HOP_LENGTH, N_MELS, LANGUAGES,
    BASE, CHUNK, RECORD_SECONDS, MODEL_URLS
)
from whisper import Whisper

def prep_audio(waveform) -> Tensor:
    """
    Preprocesses the given waveform for use in the Whisper model.

    Args:
        waveform (np.ndarray): The audio waveform to preprocess.

    Returns:
        np.ndarray: The preprocessed audio data in the form of a log-mel spectrogram.
    """
    assert waveform is not None
    waveform = waveform.reshape(1, -1)

    stft = librosa.stft(
        waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.float32)
    magnitudes = stft[..., :-1] ** 2
    mel_spec = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, mel_spec.max() + 1e8))
    log_spec = (log_spec + 4.0) / 4.0

    # https://github.com/openai/whisper/blob/b38a1f20f4b23f3f3099af2c3e0ca95627276ddf/whisper/audio.py#L19
    n_frames = log_spec.shape[2]
    if n_frames < 3000:
        log_spec = np.pad(log_spec, ((0, 0), (0, 0), (0, 3000 - n_frames)))

    return log_spec

def get_encoding(n_vocab_in):
    """
    Downloads a file containing token ranks, calculates the number of special tokens,
        and returns an Encoding object.

    Args:
        n_vocab_in (int): The number of vocabulary items.

    Returns:
        Encoding:
            An object containing the name, explicit_n_vocab, pat_str,
            mergeable_ranks, and special_tokens.
    """

    download_file(
        "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
        BASE / "gpt2.tiktoken")
    with open(BASE / "gpt2.tiktoken", "r", encoding="utf-8") as f:
        ranks = {base64.b64decode(token): int(rank) for token, rank in (
            line.split() for line in f if line)}
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = dict(zip(specials, itertools.count(n_vocab)))
    n_vocab += len(specials)
    assert n_vocab == n_vocab_in
    return Encoding(
        name="bob",
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens)

def img(x):
    """
    Displays a given image using matplotlib.

    Args:
        x: A tensor representing the image to be displayed.
    """
    plt.imshow(x.numpy())
    plt.show()

def listener(q):
    """
    Records audio from the default input device and puts the audio data into a queue.

    Args:
        q (Queue): A queue to put the audio data into.

    Returns:
        None
    """
    p = PyAudio()
    stream = p.open(format=paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("listening")
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        waveform = (np.frombuffer(data, np.int16)/32768).astype(np.float32) * 3
        q.put(waveform)
    print("done listening")

def init_whisper(model_name="tiny.en"):
    """
    Initializes a Whisper model with the specified model name.

    Args:
        model_name (str): The name of the model to initialize. Defaults to "tiny.en".

    Returns:
        tuple: A tuple containing the initialized Whisper model and its encoding.
    """
    assert MODEL_URLS[model_name] is not None

    filename = BASE / f"whisper-{model_name}.pt"
    download_file(MODEL_URLS[model_name], filename)
    state = torch_load(filename)
    model = Whisper(state['dims'])
    load_state_dict(model, state['model_state_dict'])

    enc = get_encoding(state['dims']['n_vocab'])
    return model, enc

def transcribe_file(model, enc, filename):
    """
    Transcribes the audio file at the given filename using the provided model and encoder.

    Args:
        model: The model to use for transcription.
        enc: The encoder to use for transcription.
        filename: The path to the audio file to transcribe.

    Returns:
        The transcription of the audio file.
    """
    waveform, _sample_rate = librosa.load(filename, sr=RATE)
    log_spec = prep_audio(waveform)

    # pylint: disable=protected-access
    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    dat = model.encoder(Tensor(log_spec)).realize()

    for _ in range(50):
        out = model.decoder(Tensor([lst]), dat).realize()
        idx = int(out[0,-1].argmax().numpy().item())
        lst.append(idx)
        transcription = enc.decode(lst)
        print(transcription)
        if lst[-1] == enc._special_tokens["<|endoftext|>"]:
            return  transcription
    # pylint: enable=protected-access
