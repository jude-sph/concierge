"""Owns all audio: SoulX-Duplug in, Kokoro out, both channels recorded.

This is the relocatable boundary. If the tunnel measures badly (Task 1), this
service moves to the Mac and only text crosses the wire.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .recorder import SessionRecorder
from .soulx_client import CHUNK_SAMPLES, SoulXClient
from .states import StateAdapter, TurnEvent


class VoiceService:
    def __init__(self, session_dir: str | Path, soulx_url: str = "ws://localhost:8000/turn"):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.recorder = SessionRecorder(session_dir)
        self.tts = None  # set by the caller; lazily imported to avoid GPU deps in tests
        self._t_ms = 0

    async def feed_audio(self, chunk: np.ndarray) -> list[TurnEvent]:
        self.recorder.write_user(chunk)
        wire = await self.client.feed(chunk)
        events = self.adapter.feed(wire, self._t_ms)
        self._t_ms += int(CHUNK_SAMPLES / 16000 * 1000)
        return events

    async def speak(self, text: str, utterance_id: str) -> None:
        async for chunk in self.tts.stream(text):
            self.recorder.write_model(chunk)

    async def stop(self) -> None:
        if self.tts is not None:
            self.tts.stop()

    def close(self) -> None:
        self.recorder.close()
