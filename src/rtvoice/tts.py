"""Kokoro TTS: streamed clause by clause, and interruptible mid-word.

Kokoro emits 24 kHz natively, and that is also the playback rate. Downsampling
synthesized speech to 16 kHz was never required by anything downstream (16 kHz
is a constraint of the *microphone* path into SoulX-Duplug, not of what goes to
the user's speakers), and doing it with plain linear interpolation and no
anti-aliasing filter was actively harmful -- everything above the resulting
8 kHz Nyquist folded back into the audible band as metallic aliasing. See
`resample.py`, which is where any rate conversion this module still needs now
goes.

WHY THE TEXT IS SPLIT HERE
--------------------------
`KPipeline` is a generator, so this module always looked like it streamed. It
did not. The pipeline's default `split_pattern` is `\\n+`, and a spoken reply
is one line -- so the whole utterance was a single segment, and the generator
yielded exactly once, after synthesizing all of it. A three-second reply meant
three seconds of silence followed by three seconds of speech.

Splitting the text into clauses ourselves turns that into a first sound after
the first clause, which is typically a few hundred milliseconds. The split is
done here rather than by handing `split_pattern` a sentence regex because the
FIRST clause wants to be shorter than the rest (it sets the time-to-first-
audio, which is the number a listener actually perceives as latency) and the
later ones want to be long enough that prosody does not break up.

Synthesis runs one clause AHEAD of playback, in a worker thread. Both halves
matter: the thread keeps a multi-hundred-millisecond GPU call from stalling
the event loop that is simultaneously reading the microphone, and the
lookahead means clause n+1 is ready by the time clause n finishes playing, so
the seams are not audible.

BARGE-IN
--------
stop() bumps a generation counter that is checked before every emitted frame,
not merely between clauses. Audio is emitted in ~120ms frames, so speech halts
within about a frame of the interruption rather than at the end of the current
sentence. Cancelling generation is only half of barge-in -- frames already
sent are already in the browser's queue -- see VoiceService.on_playback_cancel
for the other half.
"""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Iterator

import httpx
import numpy as np

from .resample import resample_audio

KOKORO_RATE = 24000
# The rate synthesized speech is produced -- and now played back -- at. Named
# separately from KOKORO_RATE so callers that care about "the TTS output
# rate" aren't implicitly coupled to it happening to equal Kokoro's native
# rate today.
OUTPUT_SAMPLE_RATE = KOKORO_RATE

# Frames small enough that an interruption is inaudible-to-immediate, large
# enough that a websocket send per frame is not the bottleneck.
FRAME_MS = 120
FRAME_SAMPLES = OUTPUT_SAMPLE_RATE * FRAME_MS // 1000

# The first clause is capped hard: it is the only one whose synthesis time the
# listener experiences as latency. Later clauses are allowed to be longer,
# because by then audio is already playing and the only thing that matters is
# staying ahead of playback -- and because a longer span gives Kokoro more
# context to get the prosody right.
FIRST_CLAUSE_CHARS = 60
CLAUSE_CHARS = 160

# Sentence enders first, then internal punctuation. Splitting only on sentence
# enders leaves a single long clause-heavy sentence (the common shape for a
# spoken reply) unsplit, which is exactly the case this is here to fix.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def _split_once(text: str, limit: int) -> tuple[str, str]:
    """Take a prefix no longer than `limit`, broken at the best place available.

    Preference order is sentence end, then clause punctuation, then a word
    boundary. A hard character cut is the last resort and only happens for a
    single word longer than the limit, where there is no better answer.
    """
    text = text.strip()
    if len(text) <= limit:
        return text, ""

    for pattern in (_SENTENCE_END, _CLAUSE_BREAK):
        cuts = [m.end() for m in pattern.finditer(text) if m.end() <= limit]
        if cuts:
            return text[:cuts[-1]].strip(), text[cuts[-1]:].strip()

    space = text.rfind(" ", 0, limit)
    if space > 0:
        return text[:space].strip(), text[space:].strip()
    return text[:limit].strip(), text[limit:].strip()


def split_for_streaming(
    text: str,
    first_limit: int = FIRST_CLAUSE_CHARS,
    limit: int = CLAUSE_CHARS,
) -> list[str]:
    """Break a reply into speakable pieces, shortest first.

    Pure and GPU-free, so the thing that decides time-to-first-audio can be
    tested without a model.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    pieces: list[str] = []
    head, rest = _split_once(text, first_limit)
    if head:
        pieces.append(head)
    while rest:
        head, rest = _split_once(rest, limit)
        if not head:
            break
        pieces.append(head)
    return pieces


def _frames(audio: np.ndarray, size: int = FRAME_SAMPLES) -> Iterator[np.ndarray]:
    for start in range(0, len(audio), size):
        yield audio[start:start + size]


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Anti-aliased rate conversion (see `resample.resample_audio`).

    Not used on the hot path anymore -- Kokoro's output is played back at
    its native rate -- but kept as this module's public resampling entry
    point for callers (and tests) that do need to convert rates.
    """
    return resample_audio(audio, src, dst)


class StreamingTTS:
    """Clause splitting, lookahead synthesis and interruptible framing.

    Everything here is engine-independent, so a subclass only has to turn one
    clause of text into one array of samples. That division is what lets the
    engine be swapped -- a small local model, or a bigger one on another
    machine -- without touching the part that makes speech start early and
    stop on a barge-in, which is the part with the subtle bugs in it.
    """

    sample_rate = OUTPUT_SAMPLE_RATE

    def __init__(self) -> None:
        self._generation = 0

    def stop(self) -> None:
        self._generation += 1

    def synthesize(self, clause: str) -> np.ndarray:
        """One clause to one array. Synchronous; always called in a thread."""
        raise NotImplementedError

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        generation = self._generation
        clauses = split_for_streaming(text)
        if not clauses:
            return

        # Depth 1: synthesis stays exactly one clause ahead of playback.
        # Deeper would buy nothing (playback is real time, so it can never
        # fall further behind than one clause) and would cost more work to
        # throw away on a barge-in.
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        async def produce() -> None:
            try:
                for clause in clauses:
                    if generation != self._generation:
                        break
                    audio = await asyncio.to_thread(self.synthesize, clause)
                    if generation != self._generation:
                        break
                    await queue.put(audio)
            except Exception as exc:  # pragma: no cover - surfaced to consumer
                await queue.put(exc)
            else:
                await queue.put(None)  # end of stream

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                for frame in _frames(item):
                    # Checked per FRAME, not per clause: a barge-in during a
                    # long sentence must stop within a frame, not run to the
                    # end of the sentence.
                    if generation != self._generation:
                        return
                    if frame.size:
                        yield frame
        finally:
            # Covers every exit: end of stream, barge-in, and the consumer
            # abandoning the generator. Without this an interrupted utterance
            # leaves a thread synthesizing audio nobody will ever hear, and
            # (worse) a task blocked forever on a full queue.
            producer.cancel()


class KokoroTTS(StreamingTTS):
    """Kokoro-82M, in-process. Fast (20-50x realtime on a 3090) and flat --
    82 million parameters buys speed, not expressiveness."""

    def __init__(self, voice: str = "am_puck", lang_code: str = "a") -> None:
        from kokoro import KPipeline  # imported lazily; needs GPU deps

        super().__init__()
        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice

    def synthesize(self, clause: str) -> np.ndarray:
        parts = [np.asarray(audio, dtype=np.float32)
                 for _, _, audio in self._pipeline(clause, voice=self.voice)]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)


class RemoteTTS(StreamingTTS):
    """Synthesis on another machine, over HTTP, one clause per request.

    Chatterbox sounds markedly more human than Kokoro and cannot run beside
    the rest of the system: `chatterbox-tts` requires numpy<2, which would
    downgrade the numpy the turn-taking model depends on, and the demo box is
    at 97% disk with 3.4GB of VRAM left. So it runs on the laptop and this
    calls it.

    One clause per request, deliberately, rather than streaming a whole
    utterance from the server: the base class already runs a clause ahead of
    playback, so each round trip is overlapped with speaking the previous
    clause and only the FIRST one is ever waited on.

    A failure here is silence for one utterance, never an exception into the
    audio path -- the caller is Orchestrator._speak_facts, which is speaking a
    reasoner result the person is waiting on.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8020",
                 timeout: float = 30.0) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.failures = 0

    def synthesize(self, clause: str) -> np.ndarray:
        try:
            resp = self._client.post(f"{self.base_url}/tts", json={"text": clause})
            resp.raise_for_status()
            return np.frombuffer(resp.content, dtype=np.float32).copy()
        except Exception:
            self.failures += 1
            return np.zeros(0, dtype=np.float32)
