"""WebSocket client for the SoulX-Duplug turn-taking server.

Wire protocol (from Soul-AILab/SoulX-Duplug server.py):
  send: {"type":"audio","session_id":str,"audio":base64(float32 PCM)}
  recv: {"type":"turn_state","session_id":str,"state":{...},"ts":float}

Inner state dict:
  {"state": "idle"|"nonidle"|"speak"|"blank",
   "asr_buffer": str,   # last ~3.2s, present when nonidle
   "asr_segment": str,  # current chunk, present when nonidle
   "text": str}         # full utterance, present when speak
"""
from __future__ import annotations

import base64
import json
import uuid

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2560  # 160 ms, matches config.yaml chunk_size


def encode_chunk(session_id: str, chunk: np.ndarray) -> dict:
    audio = np.asarray(chunk, dtype=np.float32)
    return {
        "type": "audio",
        "session_id": session_id,
        "audio": base64.b64encode(audio.tobytes()).decode(),
    }


def decode_response(wire: dict) -> dict:
    """Pull the inner state dict out of the envelope."""
    return wire.get("state", {})


class SoulXClient:
    def __init__(self, url: str = "ws://localhost:8000/turn", session_id: str | None = None):
        self.url = url
        self.session_id = session_id or uuid.uuid4().hex
        self._ws = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url, max_size=None)

    async def feed(self, chunk: np.ndarray) -> dict:
        """Send one chunk, return the inner state dict."""
        if self._ws is None:
            await self.connect()
        await self._ws.send(json.dumps(encode_chunk(self.session_id, chunk)))
        raw = await self._ws.recv()
        return decode_response(json.loads(raw))

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
