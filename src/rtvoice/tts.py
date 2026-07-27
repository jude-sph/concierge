"""Kokoro TTS, streamed and interruptible.

Kokoro emits 24 kHz natively, and that is now also the output/playback rate:
downsampling synthesized speech to 16 kHz was never required by anything
downstream of TTS (16 kHz is a constraint of the *microphone* path into
SoulX-Duplug, not of what goes to the user's speakers), and doing it with
plain linear interpolation and no anti-aliasing filter was actively harmful
-- everything above the resulting 8 kHz Nyquist folded back into the audible
band as metallic aliasing. See `resample.py`, which is where any rate
conversion this module still needs (e.g. a caller resampling for its own
purposes) now goes through.

stop() bumps a generation counter checked between chunks, so barge-in halts
speech within one chunk rather than at the end of the utterance.
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from .resample import resample_audio

KOKORO_RATE = 24000
# The rate synthesized speech is produced -- and now played back -- at. Named
# separately from KOKORO_RATE so callers that care about "the TTS output
# rate" aren't implicitly coupled to it happening to equal Kokoro's native
# rate today.
OUTPUT_SAMPLE_RATE = KOKORO_RATE


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Anti-aliased rate conversion (see `resample.resample_audio`).

    Not used on the hot path anymore -- Kokoro's output is played back at
    its native rate -- but kept as this module's public resampling entry
    point for callers (and tests) that do need to convert rates.
    """
    return resample_audio(audio, src, dst)


class KokoroTTS:
    sample_rate = OUTPUT_SAMPLE_RATE

    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        from kokoro import KPipeline  # imported lazily; needs GPU deps

        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self._generation = 0

    def stop(self) -> None:
        self._generation += 1

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        generation = self._generation
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if generation != self._generation:
                return
            # Native rate, unmodified: no resampling, no aliasing.
            yield np.asarray(audio, dtype=np.float32)
