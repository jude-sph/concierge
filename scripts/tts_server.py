"""Chatterbox Turbo, served over HTTP, on whichever machine can host it.

WHY THIS IS A SEPARATE PROCESS ON A SEPARATE MACHINE
----------------------------------------------------
Kokoro-82M is 82 million parameters. It runs at 20-50x realtime and sounds
like it: flat, and unmistakably synthetic. Chatterbox sounds markedly more
human, and it cannot live beside the rest of the system --

  * `chatterbox-tts` requires numpy<2, and downgrading numpy in the demo
    machine's environment would break the turn-taking model that shares it;
  * that machine is at 97% disk with ~3.4GB of VRAM left, both LLMs resident.

So it runs here, wherever "here" is (a laptop is fine -- it needs ~1-2GB and
is CPU/MPS-capable), and `rtvoice.tts.RemoteTTS` calls it. The orchestrator is
unchanged: it still asks for audio and streams whatever comes back.

ONE CLAUSE PER REQUEST
----------------------
The caller splits the reply into clauses and runs one clause ahead of
playback, so every round trip after the first is overlapped with speaking the
previous clause. Streaming a whole utterance from here would move that
buffering to the wrong side of the wire and make barge-in the server's
problem, which it is not: the client already halts within one 120ms frame.

VOICE
-----
Chatterbox has no voice list -- a voice IS a reference recording, cloned
zero-shot. Turbo's own default speaker is not the same one as the standard
model's, so `TTS_VOICE_REF` points at a clip of the voice actually wanted and
every request conditions on it.

Run:  uvicorn scripts.tts_server:app --host 127.0.0.1 --port 8020
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

# Chatterbox stamps an inaudible Resemble watermark on its output. That
# implementation imports pkg_resources, removed in setuptools 81+, so it lands
# as None and the constructor dies with a confusing TypeError. Stubbed to the
# no-op watermarker so a setuptools version can't take the voice out.
import perth
if getattr(perth, "PerthImplicitWatermarker", None) is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

VOICE_REF = os.environ.get("TTS_VOICE_REF") or None
DEVICE = os.environ.get("TTS_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI()

# Synthesis is not thread-safe and the GPU is one resource; serialise here
# rather than hoping requests never overlap. The client sends one clause at a
# time per utterance, so this is almost never contended.
_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        started = time.perf_counter()
        _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

        # Embed the voice ONCE, here, and never again.
        #
        # `generate(audio_prompt_path=...)` re-runs prepare_conditionals every
        # single call, which re-encodes the whole reference clip before a word
        # of the request is looked at. That is a fixed cost paid per clause,
        # and it dominated everything: "Right," -- six characters, half a
        # second of speech -- took 1.75s, essentially all of it re-learning a
        # voice the model had already been told about. prepare_conditionals
        # caches into `model.conds`, and generate() uses that when no prompt
        # is passed.
        if VOICE_REF:
            _model.prepare_conditionals(VOICE_REF)

        # The first generation is several times slower than the rest (kernel
        # autotuning, lazy weight materialisation). Spend that here, at
        # startup, rather than on someone's first sentence.
        _model.generate("Warming up.")
        print(f"[tts] chatterbox-turbo on {DEVICE} in "
              f"{time.perf_counter()-started:.1f}s, voice={VOICE_REF or 'default'}",
              flush=True)
    return _model


class Say(BaseModel):
    text: str


@app.on_event("startup")
def _warm() -> None:
    _get_model()


@app.get("/health")
def health() -> dict:
    return {"engine": "chatterbox-turbo", "device": DEVICE,
            "voice_ref": VOICE_REF, "sample_rate": 24000}


@app.post("/tts")
def tts(say: Say) -> Response:
    """One clause in, raw float32 PCM at 24 kHz out.

    Raw bytes rather than JSON or a WAV: the caller reads it straight into a
    numpy array and forwards it to the browser, and a container would only add
    a header to strip. 24 kHz matches what the rest of the pipeline plays at
    (see rtvoice.tts.OUTPUT_SAMPLE_RATE), so nothing is resampled anywhere.
    """
    text = (say.text or "").strip()
    if not text:
        return Response(content=b"", media_type="application/octet-stream")

    model = _get_model()
    with _lock, torch.no_grad():
        # No audio_prompt_path: the voice is already embedded (see
        # _get_model). Passing it here would re-encode the reference clip on
        # every clause, which is where nearly all the latency used to go.
        wav = model.generate(text)
    audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
    # int16, not float32: half the bytes for audio that is about to be played
    # through a speaker, where 16 bits is already beyond what anyone can hear.
    # This crosses a tunnel twice on the way to the browser, and the first
    # clause's transfer is part of the silence before speech starts.
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return Response(content=pcm.tobytes(),
                    media_type="application/octet-stream")
