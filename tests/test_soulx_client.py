"""Pure encode/decode tests for the SoulX-Duplug wire protocol.

No network, no websocket connection: only encode_chunk / decode_response /
CHUNK_SAMPLES, which do not touch SoulXClient's connect/feed/close methods.
"""
import base64

import numpy as np

from rtvoice.soulx_client import CHUNK_SAMPLES, decode_response, encode_chunk


def test_chunk_samples_is_160ms_at_16khz():
    assert CHUNK_SAMPLES == 2560


def test_encode_chunk_produces_expected_envelope():
    chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    wire = encode_chunk("session-1", chunk)

    assert wire["type"] == "audio"
    assert wire["session_id"] == "session-1"
    assert isinstance(wire["audio"], str)
    assert base64.b64decode(wire["audio"]) == chunk.tobytes()


def test_encode_chunk_round_trips_samples_exactly():
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-1.0, 1.0, size=CHUNK_SAMPLES).astype(np.float32)

    wire = encode_chunk("session-2", chunk)
    decoded = np.frombuffer(base64.b64decode(wire["audio"]), dtype=np.float32)

    assert np.array_equal(decoded, chunk)


def test_encode_chunk_casts_non_float32_input():
    chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float64)
    wire = encode_chunk("session-3", chunk)
    decoded = np.frombuffer(base64.b64decode(wire["audio"]), dtype=np.float32)
    assert decoded.dtype == np.float32
    assert len(decoded) == CHUNK_SAMPLES


def test_decode_response_extracts_inner_state():
    wire = {
        "type": "turn_state",
        "session_id": "session-1",
        "state": {"state": "nonidle", "asr_buffer": "hel", "asr_segment": "hel"},
        "ts": 123.456,
    }
    assert decode_response(wire) == {
        "state": "nonidle", "asr_buffer": "hel", "asr_segment": "hel",
    }


def test_decode_response_missing_state_returns_empty_dict():
    assert decode_response({"type": "turn_state", "session_id": "x", "ts": 1.0}) == {}
