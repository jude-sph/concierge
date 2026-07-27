"""Kokoro TTS, streamed and interruptible.

Kokoro emits 24 kHz; everything else in this system is 16 kHz, so we resample
on the way out. stop() sets a flag checked between chunks, so barge-in halts
speech within one chunk rather than at the end of the utterance.
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np

SAMPLE_RATE = 16000
KOKORO_RATE = 24000


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst / src))
    idx = np.linspace(0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        from kokoro import KPipeline  # imported lazily; needs GPU deps

        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        self._stopped = False
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if self._stopped:
                return
            yield _resample(np.asarray(audio, dtype=np.float32), KOKORO_RATE, SAMPLE_RATE)
