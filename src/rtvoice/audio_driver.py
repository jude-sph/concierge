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

import asyncio
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
        # Live-audio stage two (see start/submit). None until started, so
        # every existing caller of feed() is completely unaffected.
        self._turns: "asyncio.Queue | None" = None
        self._worker: "asyncio.Task | None" = None

    async def feed(self, chunk: np.ndarray) -> list[TurnEvent]:
        """Feed one chunk and handle whatever it produced, inline.

        Everything happens before this returns, which is what every test and
        every file-replay wants. A LIVE microphone must not use this -- see
        `start()` and `submit()` below for why.
        """
        events = await self.voice.feed_audio(chunk)
        self.chunks_fed += 1
        for ev in events:
            await self.orch.on_turn_event(ev)
        await self.tick()
        return events

    # --- live audio: two stages, so the microphone is never blocked ----------
    #
    # `feed` above does everything in one call, and for a live microphone that
    # is wrong. Handling a turn means running the reasoner (~2s) and then
    # speaking the reply, and speaking BLOCKS FOR THE DURATION OF THE AUDIO --
    # deliberately, because that is what makes barge-in possible (see
    # VoiceService.speak). Do that inline and the next chunk is not read until
    # the system has finished talking.
    #
    # It did exactly that. Measured across live sessions by comparing the
    # audio clock against the wall clock, the pipeline ran 4-22 SECONDS behind
    # real time and never recovered. That is what "sometimes my speech takes
    # 5+ seconds to appear" was: not slow recognition -- Whisper is 0.01-0.24s
    # and SoulX averages 112ms against a 160ms budget -- but a backlog of
    # audio nobody had read yet.
    #
    # So the two jobs are separated. Stage one feeds the turn-taking model and
    # nothing else, which comfortably beats real time. Stage two consumes the
    # turn events it produces and may take as long as it likes. A queue joins
    # them, so a slow reply delays only replies.
    #
    # Both stages stay strictly ordered (one consumer each, FIFO): turn events
    # must reach the orchestrator in the order they happened, or a `speak`
    # could be processed before the `nonidle` that preceded it.

    async def start(self) -> None:
        """Begin the turn-handling stage. Call once before submit()."""
        if self._turns is None:
            self._turns = asyncio.Queue()
            self._worker = asyncio.create_task(self._handle_turns())

    async def submit(self, chunk: np.ndarray) -> None:
        """Feed one chunk of LIVE audio and return as soon as the turn-taking
        model has seen it. Turn handling happens elsewhere."""
        if self._turns is None:
            await self.start()
        events = await self.voice.feed_audio(chunk)
        self.chunks_fed += 1
        self._turns.put_nowait((events, int(self._clock())))

    async def _handle_turns(self) -> None:
        while True:
            events, now_ms = await self._turns.get()
            try:
                for ev in events:
                    await self.orch.on_turn_event(ev)
                if now_ms - self._last_tick_ms >= self.tick_interval_ms:
                    self._last_tick_ms = now_ms
                    await self.orch.on_tick(now_ms)
            except Exception as exc:
                # One bad turn must not end the session: this task is the only
                # consumer, and if it dies every later turn is silently lost.
                self.orch.log.append("turn_handler_error", error=repr(exc),
                                     error_type=type(exc).__name__)
            finally:
                self._turns.task_done()

    async def aclose(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
            self._turns = None

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
