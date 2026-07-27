"""Drives the loop from audio.

Everything else in this package is reactive: the orchestrator responds to turn
events, the voice service responds to audio chunks. Something has to actually
push audio in and pump the clock, and nothing did -- `feed_audio()` had no
caller, and `on_tick()` had none outside the tests, which made the silence
timeout dead code in production. This is that missing driver, kept separate
from both the voice service and the orchestrator so it can be exercised with
fakes and reused by any audio source (a WAV file, a microphone, a WebSocket).

Two things it is responsible for getting right:

  1. Turn events from the voice service reach the orchestrator, in order.
  2. on_tick is driven on the AUDIO clock, not the wall clock. Turn events are
     stamped with a position in the audio stream, and on_tick compares now_ms
     against those stamps. Feeding a WAV file as fast as the network allows
     would otherwise force-dispatch every partial utterance instantly, and a
     stalled feed would never time out at all.
"""
from __future__ import annotations

from typing import AsyncIterable, Callable, Iterable

import numpy as np

from .soulx_client import CHUNK_SAMPLES
from .states import TurnEvent

CHUNK_MS = 160


class AudioDriver:
    def __init__(
        self,
        voice,
        orch,
        *,
        tick_interval_ms: int = CHUNK_MS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.voice = voice
        self.orch = orch
        self.tick_interval_ms = tick_interval_ms
        self._clock = clock if clock is not None else (lambda: voice.stream_ms)
        self._last_tick_ms = 0
        self.chunks_fed = 0

    async def feed(self, chunk: np.ndarray) -> list[TurnEvent]:
        events = await self.voice.feed_audio(chunk)
        self.chunks_fed += 1
        for ev in events:
            await self.orch.on_turn_event(ev)
        await self.tick()
        return events

    async def tick(self) -> bool:
        """Pump the orchestrator's timer if enough audio has gone by."""
        now_ms = int(self._clock())
        if now_ms - self._last_tick_ms < self.tick_interval_ms:
            return False
        self._last_tick_ms = now_ms
        await self.orch.on_tick(now_ms)
        return True

    async def run(self, chunks: Iterable[np.ndarray] | AsyncIterable[np.ndarray]) -> None:
        if hasattr(chunks, "__aiter__"):
            async for chunk in chunks:
                await self.feed(chunk)
        else:
            for chunk in chunks:
                await self.feed(chunk)

    async def drain(self, silence_ms: int) -> None:
        """Feed trailing silence.

        A recording that ends the instant the speaker stops leaves the last
        utterance held forever: the turn-taking model needs silence to decide
        the turn is over, and the silence timer needs clock to advance. A live
        microphone supplies both for free; a file does not.
        """
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        for _ in range(max(0, silence_ms) // CHUNK_MS):
            await self.feed(silence.copy())
