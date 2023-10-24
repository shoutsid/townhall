"""
Whisper model (https://github.com/openai/whisper)
"""

from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from audio_encoder import AudioEncoder
from text_decoder import TextDecoder


class Whisper:
    """
    A class representing the Whisper model, which converts audio to text.

    Attributes:
        encoder (AudioEncoder): An instance of the AudioEncoder class.
        decoder (TextDecoder): An instance of the TextDecoder class.
    """
    def __init__(self, dims):
        self.encoder = AudioEncoder(**dims)
        self.decoder = TextDecoder(**dims)

    def __call__(self, mel: Tensor, tokens: Tensor):
        """
        Generate audio waveform from mel-spectrogram and tokens.

        Args:
            mel (Tensor): Mel-spectrogram tensor.
            tokens (Tensor): Token tensor.

        Returns:
            Tensor: Audio waveform tensor.
        """
        return self.decoder(tokens, self.encoder(mel))


if __name__ == "__main__":
    import sys
    import multiprocessing
    import numpy as np
    from utils import prep_audio, listener, init_whisper, transcribe_file
    from constants import RATE, CHUNK, RECORD_SECONDS

    model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")

    if len(sys.argv) > 1:
        transcribe_file(model, enc, sys.argv[1])
    else:
        # online
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=listener, args=(q,))
        p.daemon = True
        p.start()

        # pylint: disable=protected-access
        lst = [enc._special_tokens["<|startoftranscript|>"],
                enc._special_tokens["<|notimestamps|>"]]
        TOTAL = None
        DID_READ = False
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            while not q.empty() or TOTAL is None:
                waveform = q.get()
                if TOTAL is None:
                    TOTAL = waveform
                else:
                    TOTAL = np.concatenate([TOTAL, waveform])
                DID_READ = True
            if DID_READ:
                log_spec = prep_audio(TOTAL)
                encoded_audio = model.encoder(Tensor(log_spec)).realize()
            out = model.decoder(Tensor([lst]), encoded_audio).realize()
            idx = int(out[0,-1].argmax().numpy().item())
            lst.append(idx)
            dec = enc.decode(lst)
            print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
            if dec.endswith("<|endoftext|>"):
                #total = total[:, 320*(len(lst)-1):]
                lst.pop()
        # pylint: enable=protected-access
