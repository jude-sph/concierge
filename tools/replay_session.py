"""Testing Layer 2: replay a recorded session's audio and assert the turn
states reproduce. Since recording is always on, the corpus accumulates for
free as the system gets used.

    uv run python tools/replay_session.py sessions/2026-07-27T14-32-05
"""
from __future__ import annotations

import asyncio, pathlib, sys
import numpy as np, soundfile as sf

from rtvoice.events import EventLog
from rtvoice.soulx_client import CHUNK_SAMPLES, SoulXClient
from rtvoice.states import StateAdapter


async def main(session_dir: str) -> int:
    root = pathlib.Path(session_dir)
    audio, sr = sf.read(root / "user.wav", dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    expected = [
        e.data["state"] for e in EventLog.read(root / "events.jsonl")
        if e.kind == "turn_event"
    ]

    client = SoulXClient()
    await client.connect()
    adapter = StateAdapter()
    actual: list[str] = []
    t_ms = 0
    for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        wire = await client.feed(audio[i:i + CHUNK_SAMPLES])
        for ev in adapter.feed(wire, t_ms):
            actual.append(ev.state.value)
        t_ms += 160
    await client.close()

    if actual == expected:
        print(f"MATCH  {len(actual)} states reproduced")
        return 0

    print(f"MISMATCH  expected {len(expected)} states, got {len(actual)}")
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            print(f"  first divergence at {i}: expected {a}, got {b}")
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1])))
