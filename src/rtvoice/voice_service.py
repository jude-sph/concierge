"""Owns all audio: SoulX-Duplug in, Kokoro out, both channels recorded.

This is the relocatable boundary. If the tunnel measures badly (Task 1), this
service moves to the Mac and only text crosses the wire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .recorder import SessionRecorder
from .soulx_client import CHUNK_SAMPLES, SAMPLE_RATE, SoulXClient
from .states import StateAdapter, TurnEvent

CHUNK_MS = int(CHUNK_SAMPLES / SAMPLE_RATE * 1000)


class VoiceService:
    def __init__(
        self,
        session_dir: str | Path,
        soulx_url: str = "ws://localhost:8000/turn",
        *,
        tts_factory: Callable[[], object] | None = None,
    ):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.recorder = SessionRecorder(session_dir)
        self._tts = None
        self._tts_factory = tts_factory
        self._t_ms = 0

    # -- text to speech -------------------------------------------------------
    #
    # `tts` used to be a plain attribute set to None and never assigned by
    # anything, so the first spoken reply raised AttributeError -- swallowed
    # by the orchestrator's return_exceptions=True gather and logged as
    # turn_task_error, i.e. the system failed to speak invisibly. It is now
    # built on FIRST USE. Lazily, because Kokoro pulls in GPU dependencies
    # that must not be imported at module scope (and are not installed in the
    # test environment), and because constructing a VoiceService must stay
    # cheap and side-effect-free.

    @property
    def tts(self):
        if self._tts is None:
            self._tts = self._make_tts()
        return self._tts

    @tts.setter
    def tts(self, value) -> None:
        self._tts = value

    def _make_tts(self):
        if self._tts_factory is not None:
            return self._tts_factory()
        from . import tts as tts_module

        try:
            return tts_module.KokoroTTS()
        except Exception as exc:  # ImportError, missing model, no GPU, ...
            raise RuntimeError(
                "text-to-speech is unavailable: KokoroTTS could not be "
                "constructed. The `kokoro` package and its GPU dependencies "
                "are not installed in every environment this runs in. "
                "Install kokoro, or supply your own: "
                "VoiceService(session_dir, tts_factory=...) or voice.tts = <obj>."
            ) from exc

    @property
    def stream_ms(self) -> int:
        """Milliseconds of audio fed so far.

        This is the clock TurnEvents are stamped with, so it is also the clock
        Orchestrator.on_tick must be driven on -- wall-clock time would make
        the silence timeout meaningless whenever audio is not being played at
        exactly real time.
        """
        return self._t_ms

    async def feed_audio(self, chunk: np.ndarray) -> list[TurnEvent]:
        self.recorder.write_user(chunk)
        wire = await self.client.feed(chunk)
        events = self.adapter.feed(wire, self._t_ms)
        self._t_ms += CHUNK_MS
        return events

    async def speak(self, text: str, utterance_id: str) -> None:
        async for chunk in self.tts.stream(text):
            self.recorder.write_model(chunk)

    async def stop(self) -> None:
        # Deliberately reads the private slot: stopping speech must never be
        # the thing that constructs a (possibly unavailable) TTS engine.
        if self._tts is not None:
            self._tts.stop()

    def close(self) -> None:
        self.recorder.close()
