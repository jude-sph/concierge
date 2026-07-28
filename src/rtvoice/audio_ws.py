"""The browser microphone/speaker path.

The orchestrator, VoiceService and AudioDriver already do all the real work --
this module's only job is to sit between them and a raw binary WebSocket:

  * frame the browser's binary Float32 PCM stream into fixed 2560-sample
    chunks (the framing logic, `chunk_pcm_bytes`, is a pure function so it is
    testable with no browser, no GPU and no turn-taking server involved), and
  * feed each chunk through the existing `AudioDriver` (which itself calls
    `VoiceService.feed_audio`, forwards TurnEvents to
    `Orchestrator.on_turn_event`, and drives `on_tick` -- see audio_driver.py;
    this module deliberately does not reimplement any of that), and
  * relay any audio VoiceService.speak() produces back out over the same
    socket, via the `on_audio_chunk` hook on VoiceService.
"""
from __future__ import annotations

import asyncio

import numpy as np
# Module level, not lazily inside install_audio_route: with `from __future__
# import annotations` active, FastAPI resolves a route function's `ws:
# WebSocket` annotation via typing.get_type_hints() against the function's
# *module* globals. A name imported only inside the enclosing function is
# invisible to that lookup, so the parameter silently falls back to "missing
# query parameter" and every connection is closed with code 1008 before the
# handler runs -- see the longer note in orchestrator.py, which had the same
# bug on /events. FastAPI has no import-time side effects, so this is safe to
# hoist.
from fastapi import WebSocket, WebSocketDisconnect

from .audio_driver import AudioDriver
from .soulx_client import CHUNK_SAMPLES

BYTES_PER_SAMPLE = 4  # float32
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


def chunk_pcm_bytes(
    buffer: bytes, incoming: bytes, chunk_samples: int = CHUNK_SAMPLES
) -> tuple[list[np.ndarray], bytes]:
    """Accumulate raw little-endian float32 bytes into fixed-size chunks.

    `buffer` is whatever was left over from the previous call (0 to
    chunk_bytes-1 bytes -- never a whole chunk, or it would already have been
    emitted). Returns (complete chunks found, new leftover). Pure: no I/O, no
    dependency on FastAPI, a browser, or a GPU -- the whole point is that this
    is testable on its own.
    """
    data = buffer + incoming
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
    chunks: list[np.ndarray] = []
    offset = 0
    while len(data) - offset >= chunk_bytes:
        raw = data[offset:offset + chunk_bytes]
        chunks.append(np.frombuffer(raw, dtype=np.float32).copy())
        offset += chunk_bytes
    return chunks, data[offset:]


def install_audio_route(app, orch) -> None:
    """Register `GET/WS /audio` on `app`.

    One AudioDriver is created here and reused for the lifetime of the app --
    it wraps the orchestrator's single VoiceService, matching how /inject and
    /state already treat the orchestrator as one shared session. This is a
    live-demo tool for one speaker at a time, not a multi-tenant server.
    """
    driver = AudioDriver(orch.voice, orch)

    @app.websocket("/audio")
    async def audio_ws(ws: WebSocket) -> None:
        await ws.accept()
        buffer = b""
        send_lock = asyncio.Lock()

        async def on_audio_chunk(chunk: np.ndarray) -> None:
            payload = np.asarray(chunk, dtype=np.float32).tobytes()
            async with send_lock:
                try:
                    await ws.send_bytes(payload)
                except Exception:
                    # The browser side may have gone away mid-utterance;
                    # losing the rest of this reply's audio is fine, the
                    # transcript already has the text. Never let a failed
                    # send here take down feed_audio's caller.
                    pass

        async def on_playback_cancel() -> None:
            # A control message, not audio -- binary frames on this socket
            # are always raw PCM (see on_audio_chunk above), so the browser
            # needs an unambiguous, differently-typed frame to know "this is
            # an instruction, not a sample buffer". A text/JSON frame is the
            # simplest scheme that can never be confused with a binary one.
            # Sent through the same send_lock as audio so it can never be
            # reordered around the very chunks it's telling the browser to
            # drop.
            async with send_lock:
                try:
                    await ws.send_text('{"type": "cancel"}')
                except Exception:
                    # Same reasoning as on_audio_chunk: the browser may
                    # already be gone, and that's fine -- there's nothing
                    # left to cancel for it anyway.
                    pass

        previous_hook = orch.voice.on_audio_chunk
        previous_cancel_hook = orch.voice.on_playback_cancel
        orch.voice.on_audio_chunk = on_audio_chunk
        orch.voice.on_playback_cancel = on_playback_cancel
        orch.log.append("audio_ws_connected")
        try:
            while True:
                # receive(), not receive_bytes(): a single text frame on this
                # socket raises KeyError('bytes') out of receive_bytes, which
                # ends the loop and takes the microphone down for the rest of
                # the session. Binary frames are the only ones that mean
                # anything here, so anything else is skipped rather than fatal.
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None:
                    continue
                chunks, buffer = chunk_pcm_bytes(buffer, data)
                for chunk in chunks:
                    # submit(), not feed(): this loop must return to
                    # receive_bytes at the speed of the turn-taking model, not
                    # at the speed of answering. feed() handles the turn
                    # inline, and handling a turn ends in speaking it -- which
                    # blocks for the whole duration of the reply. Live, that
                    # left the microphone unread for seconds at a time and the
                    # pipeline 4-22s behind real time. See AudioDriver.
                    await driver.submit(chunk)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            orch.log.append("audio_ws_error", error=repr(exc), error_type=type(exc).__name__)
        finally:
            await driver.aclose()
            orch.voice.on_audio_chunk = previous_hook
            orch.voice.on_playback_cancel = previous_cancel_hook
            orch.log.append("audio_ws_disconnected")
