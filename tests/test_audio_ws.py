"""The browser audio path: `/audio` framing and wiring.

None of this needs a browser, a GPU, `kokoro`, or a real SoulX-Duplug server --
`chunk_pcm_bytes` is pure, and the websocket route is exercised with a fake
voice service standing in for VoiceService (same pattern as
tests/test_audio_path.py's ScriptedVoice).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtvoice.audio_ws import chunk_pcm_bytes
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator, create_app
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.soulx_client import CHUNK_SAMPLES
from rtvoice.states import TurnEvent, UserState

from fakes import FakeConcierge

# --- pure framing -------------------------------------------------------------


def test_chunk_pcm_bytes_emits_nothing_for_a_partial_chunk():
    incoming = np.zeros(100, dtype=np.float32).tobytes()
    chunks, leftover = chunk_pcm_bytes(b"", incoming)
    assert chunks == []
    assert leftover == incoming


def test_chunk_pcm_bytes_emits_an_exact_chunk_and_keeps_the_remainder():
    full = np.arange(CHUNK_SAMPLES, dtype=np.float32)
    extra = np.full(37, 9.0, dtype=np.float32)
    chunks, leftover = chunk_pcm_bytes(b"", full.tobytes() + extra.tobytes())
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], full)
    assert leftover == extra.tobytes()


def test_chunk_pcm_bytes_accumulates_a_chunk_split_across_two_calls():
    full = np.arange(CHUNK_SAMPLES, dtype=np.float32)
    midpoint = (CHUNK_SAMPLES // 2) * 4  # byte offset, sample-aligned
    first, second = full.tobytes()[:midpoint], full.tobytes()[midpoint:]

    chunks, buffer = chunk_pcm_bytes(b"", first)
    assert chunks == []

    chunks, buffer = chunk_pcm_bytes(buffer, second)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], full)
    assert buffer == b""


def test_chunk_pcm_bytes_emits_multiple_chunks_from_one_call():
    a = np.full(CHUNK_SAMPLES, 1.0, dtype=np.float32)
    b = np.full(CHUNK_SAMPLES, 2.0, dtype=np.float32)
    chunks, leftover = chunk_pcm_bytes(b"", a.tobytes() + b.tobytes())
    assert len(chunks) == 2
    assert np.array_equal(chunks[0], a)
    assert np.array_equal(chunks[1], b)
    assert leftover == b""


def test_chunk_pcm_bytes_handles_uneven_byte_boundaries_not_aligned_to_samples():
    """A browser's binary WebSocket frames can split anywhere, including
    mid-sample (not just mid-chunk). Framing must still land on the right
    boundaries once enough bytes accumulate."""
    full = np.arange(CHUNK_SAMPLES, dtype=np.float32)
    raw = full.tobytes()
    chunks, buf = chunk_pcm_bytes(b"", raw[:3])
    assert chunks == []
    chunks, buf = chunk_pcm_bytes(buf, raw[3:])
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], full)


# --- the websocket route --------------------------------------------------------


class FakeAudioVoice:
    """Stands in for VoiceService: matches the surface AudioDriver and the
    /audio route both need (feed_audio, stream_ms, on_audio_chunk, speak,
    stop), without a GPU, kokoro, or a turn-taking server."""

    def __init__(self, script=(), echo_reply: np.ndarray | None = None):
        self.script = [list(s) for s in script]
        self.fed: list[np.ndarray] = []
        self.spoken: list[str] = []
        self.stops = 0
        self.on_audio_chunk = None
        self._t_ms = 0
        # If set, every feed_audio call pushes this back out through the
        # on_audio_chunk hook -- standing in for "the assistant is speaking",
        # without needing a real TTS engine or a second call into speak().
        self.echo_reply = echo_reply

    @property
    def stream_ms(self) -> int:
        return self._t_ms

    async def feed_audio(self, chunk):
        self.fed.append(np.asarray(chunk, dtype=np.float32).copy())
        t_ms = self._t_ms
        pairs = self.script.pop(0) if self.script else []
        self._t_ms += 160
        if self.echo_reply is not None and self.on_audio_chunk is not None:
            await self.on_audio_chunk(self.echo_reply)
        return [TurnEvent(state, text, t_ms) for state, text in pairs]

    async def speak(self, text, utterance_id):
        self.spoken.append(text)

    async def stop(self):
        self.stops += 1


def make_app(tmp_path, voice, **orch_kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": [{"id": 1, "first_name": "Sarah"}]}))
    device = DeviceState(state, tmp_path / "j.jsonl")
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=voice,
        log=EventLog(tmp_path / "e.jsonl"), device=device,
        use_concierge=False, **orch_kw,
    )
    return create_app(orch), orch


def test_audio_ws_frames_incoming_bytes_into_fixed_chunks(tmp_path):
    voice = FakeAudioVoice()
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    a = np.full(CHUNK_SAMPLES, 0.1, dtype=np.float32)
    b = np.full(CHUNK_SAMPLES, 0.2, dtype=np.float32)
    with client.websocket_connect("/audio") as ws:
        ws.send_bytes(a.tobytes())
        # b arrives split across two frames -- must still land as one chunk.
        raw = b.tobytes()
        ws.send_bytes(raw[:1000])
        ws.send_bytes(raw[1000:])

    assert len(voice.fed) == 2
    assert np.allclose(voice.fed[0], a)
    assert np.allclose(voice.fed[1], b)


def test_audio_ws_forwards_turn_events_to_the_orchestrator(tmp_path):
    """The whole point of reusing AudioDriver: a turn event produced by
    feed_audio must reach the orchestrator and actually register a task,
    exactly as it would from a WAV file or a real microphone."""
    voice = FakeAudioVoice(script=[[(UserState.COMPLETE, "rename my contacts to Hans")]])
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    with client.websocket_connect("/audio") as ws:
        ws.send_bytes(np.zeros(CHUNK_SAMPLES, dtype=np.float32).tobytes())

    assert len(orch.registry.all()) == 1


def test_audio_ws_relays_synthesized_speech_back_over_the_socket(tmp_path):
    """Playback direction: when VoiceService (here, the fake) emits audio via
    its on_audio_chunk hook, the browser must receive it as a binary frame."""
    reply = np.full(8, 0.5, dtype=np.float32)
    voice = FakeAudioVoice(echo_reply=reply)
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    with client.websocket_connect("/audio") as ws:
        ws.send_bytes(np.zeros(CHUNK_SAMPLES, dtype=np.float32).tobytes())
        received = ws.receive_bytes()

    got = np.frombuffer(received, dtype=np.float32)
    assert np.allclose(got, reply)


def test_audio_ws_installs_and_clears_the_playback_hook(tmp_path):
    voice = FakeAudioVoice()
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    assert voice.on_audio_chunk is None
    with client.websocket_connect("/audio") as ws:
        assert voice.on_audio_chunk is not None
        ws.send_bytes(np.zeros(CHUNK_SAMPLES, dtype=np.float32).tobytes())
    assert voice.on_audio_chunk is None


def test_audio_ws_logs_connect_and_disconnect(tmp_path):
    voice = FakeAudioVoice()
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    with client.websocket_connect("/audio"):
        pass

    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "audio_ws_connected" in kinds
    assert "audio_ws_disconnected" in kinds


def test_index_page_is_served_at_root(tmp_path):
    voice = FakeAudioVoice()
    app, orch = make_app(tmp_path, voice)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "rtvoice" in res.text.lower()
