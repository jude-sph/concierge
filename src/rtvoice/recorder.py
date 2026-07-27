"""Dual-channel session recorder.

Channels are kept separate as well as mixed so user and model audio can be
analysed independently - which is how acoustic echo gets diagnosed rather
than guessed at. Recording is always on; sessions are curated afterwards.

The user (microphone, into SoulX-Duplug) and model (synthesized speech)
channels are not necessarily the same rate: the input path is fixed at
16 kHz, but TTS output is now played back at its native 24 kHz (see
tts.py's OUTPUT_SAMPLE_RATE) rather than needlessly -- and, as it turns out,
harmfully -- downsampled. Each file is tagged with the rate it actually
contains, so a replay is not pitch-shifted. Where the two differ, `close()`
resamples (anti-aliased, via resample.py -- never a naive stretch) up to a
common rate to build the stereo mix, since a WAV file cannot carry two
different sample rates in its two channels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .resample import resample_audio

SAMPLE_RATE = 16000


class SessionRecorder:
    def __init__(
        self,
        session_dir: str | Path,
        *,
        user_sample_rate: int = SAMPLE_RATE,
        model_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.user_sample_rate = user_sample_rate
        self.model_sample_rate = model_sample_rate
        self._user_file = sf.SoundFile(
            self.dir / "user.wav", "w", samplerate=user_sample_rate, channels=1, subtype="FLOAT"
        )
        self._model_file = sf.SoundFile(
            self.dir / "model.wav", "w", samplerate=model_sample_rate, channels=1, subtype="FLOAT"
        )
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        """Defensive cleanup: close file handles if close() was not called.

        Runs during garbage collection and is defensive against:
        - Partially initialized objects
        - Interpreter shutdown
        - Exceptions during initialization
        """
        try:
            # Guard against partially-initialized instances
            if not hasattr(self, "_closed") or not hasattr(self, "_user_file"):
                return
            if not self._closed and self._user_file is not None:
                try:
                    self._user_file.close()
                except Exception:
                    pass
            if not hasattr(self, "_model_file"):
                return
            if not self._closed and self._model_file is not None:
                try:
                    self._model_file.close()
                except Exception:
                    pass
        except Exception:
            # Swallow all exceptions during cleanup
            pass

    def write_user(self, chunk: np.ndarray) -> None:
        if not self._closed:
            self._user_file.write(np.asarray(chunk, dtype=np.float32))

    def write_model(self, chunk: np.ndarray) -> None:
        if not self._closed:
            self._model_file.write(np.asarray(chunk, dtype=np.float32))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._user_file.close()
        self._model_file.close()

        # Read back the written files to construct the mix
        user, _ = sf.read(self.dir / "user.wav", dtype="float32")
        model, _ = sf.read(self.dir / "model.wav", dtype="float32")

        # Handle scalar case (single sample becomes 0-d array)
        if user.ndim == 0:
            user = np.array([user], dtype=np.float32)
        if model.ndim == 0:
            model = np.array([model], dtype=np.float32)

        # A WAV file has one sample rate for both its channels, so if the two
        # inputs were recorded at different rates they must be brought to a
        # common one before they can share a stereo mix. Resample (anti-
        # aliased) up to the higher of the two rates rather than down to the
        # lower, so the mix never loses bandwidth either channel actually
        # had. When the rates already match (still the common case, and the
        # only case in every pre-existing recording), this is a no-op and
        # the mix is built exactly as before.
        mix_rate = max(self.user_sample_rate, self.model_sample_rate)
        if self.user_sample_rate != mix_rate:
            user = resample_audio(user, self.user_sample_rate, mix_rate)
        if self.model_sample_rate != mix_rate:
            model = resample_audio(model, self.model_sample_rate, mix_rate)

        # Create stereo mix with zero-padding for unequal lengths
        n = max(len(user), len(model))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[: len(user), 0] = user
        mix[: len(model), 1] = model
        sf.write(self.dir / "mix.wav", mix, mix_rate)
