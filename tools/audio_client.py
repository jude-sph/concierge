#!/usr/bin/env python
"""Stream a 16 kHz mono WAV file through the whole loop.

The promised CLI audio client: audio in at one end, `device_state.json`
mutated at the other, with every state token, protocol message and task
transition in the session's `events.jsonl`. This is how the loop gets
exercised without a microphone or a browser.

    uv run python tools/audio_client.py fixtures/rename_contacts.wav

Needs the SoulX-Duplug turn-taking server reachable (--soulx-url). Kokoro TTS
is only built if something actually speaks; pass --mute to run the whole loop
with speech discarded, and --no-concierge to skip the vLLM concierge as well,
which leaves only the turn-taking server as an external dependency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from rtvoice.audio_driver import CHUNK_MS, AudioDriver
from rtvoice.orchestrator import build_default_orchestrator
from rtvoice.soulx_client import CHUNK_SAMPLES, SAMPLE_RATE


def read_wav(path: str | Path) -> np.ndarray:
    """Read a WAV as float32 mono at the system sample rate."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz. Resample first, "
            f"e.g. `sox in.wav -r {SAMPLE_RATE} -c 1 out.wav`."
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def chunk_audio(audio: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> Iterator[np.ndarray]:
    """Fixed-size chunks, with the final partial chunk zero-padded.

    The turn-taking server expects exactly one chunk_size frame per message;
    a short final chunk would be silently mis-framed.
    """
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < chunk_samples:
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        yield chunk


class MuteTTS:
    """Consumes speech without producing audio, so the loop can be driven end
    to end on a machine with no GPU and no `kokoro` installed."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def stop(self) -> None:
        pass

    async def stream(self, text: str):
        self.spoken.append(text)
        return
        yield  # pragma: no cover - makes this an async generator


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", help="16 kHz mono WAV to stream")
    p.add_argument("--session-dir", default=None,
                   help="where to write events.jsonl and the recordings")
    p.add_argument("--soulx-url", default=None, help="turn-taking server websocket URL")
    p.add_argument("--realtime", action="store_true",
                   help="pace the stream at 1x instead of as fast as possible")
    p.add_argument("--tail-silence-ms", type=int, default=4000,
                   help="silence appended after the file so the last utterance "
                        "finalises and the silence timeout can fire")
    p.add_argument("--mute", action="store_true", help="discard TTS instead of speaking")
    p.add_argument("--no-concierge", action="store_true",
                   help="bypass the concierge (no vLLM needed)")
    args = p.parse_args(argv)

    if args.soulx_url:
        import os
        os.environ["SOULX_URL"] = args.soulx_url

    audio = read_wav(args.wav)
    orch = build_default_orchestrator(args.session_dir)
    if args.no_concierge:
        orch.use_concierge = False
    if args.mute:
        orch.voice.tts = MuteTTS()

    driver = AudioDriver(orch.voice, orch)
    chunk_seconds = CHUNK_MS / 1000

    try:
        for chunk in chunk_audio(audio):
            await driver.feed(chunk)
            if args.realtime:
                await asyncio.sleep(chunk_seconds)
        await driver.drain(args.tail_silence_ms)
    finally:
        await orch.voice.client.close()
        orch.voice.close()

    print(f"streamed {driver.chunks_fed} chunks "
          f"({driver.chunks_fed * CHUNK_MS / 1000:.1f}s of audio)")
    for t in orch.registry.all():
        print(f"  [{t.task_id}] {t.understood_as} -> {t.status.value} {t.detail}")
    print(json.dumps(orch.device.snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
