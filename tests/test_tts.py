"""Synthesis streams clause by clause and stops within a frame.

The contract changed. `stream()` used to hand back whatever Kokoro's pipeline
yielded, one array per pipeline chunk -- and because the pipeline's default
`split_pattern` is `\\n+`, a one-line spoken reply was a single segment, so it
yielded exactly once, after synthesizing the whole utterance. It looked like a
streaming API and behaved like a blocking one.

Now the text is split into clauses here, synthesized one clause ahead in a
worker thread, and emitted as fixed-size frames. So the tests assert what the
new contract actually promises:

  * the SAMPLES are preserved exactly (nothing is dropped or resampled),
  * they arrive in bounded frames, so playback can start early and an
    interruption lands within a frame rather than at the end of a sentence,
  * a stopped stream stays stopped, even against a concurrent one.
"""
import asyncio

import numpy as np
import pytest

from rtvoice.tts import (
    FRAME_SAMPLES,
    KokoroTTS,
    _resample,
    split_for_streaming,
)


class FakePipeline:
    """Stands in for Kokoro. Yields fixed audio per call, tracks call order."""

    def __init__(self, num_chunks: int = 3, base_value: float = 0.1,
                 samples: int = 1000):
        self.num_chunks = num_chunks
        self.base_value = base_value
        self.samples = samples
        self.texts: list[str] = []

    def __call__(self, text: str, voice: str = "af_heart"):
        self.texts.append(text)
        for _ in range(self.num_chunks):
            yield (0, "a", np.ones(self.samples, dtype=np.float32) * self.base_value)


def _tts(pipeline) -> KokoroTTS:
    """A KokoroTTS without its GPU-only constructor."""
    tts = object.__new__(KokoroTTS)
    tts._pipeline = pipeline
    tts.voice = "af_heart"
    tts._generation = 0
    return tts


async def _drain(stream) -> list[np.ndarray]:
    return [chunk async for chunk in stream]


# --- splitting: what decides time-to-first-audio --------------------------

def test_a_single_sentence_reply_is_still_split():
    """The whole bug in one test.

    A spoken reply is one line, so Kokoro's default `\\n+` split treated it as
    one segment and synthesized all of it before the first sample came out.
    """
    pieces = split_for_streaming(
        "Sure, I'm pulling up your calendar now, and I'll read it out in a moment."
    )
    assert len(pieces) > 1


def test_the_first_piece_is_the_shortest():
    """It is the only one whose synthesis time is heard as latency."""
    pieces = split_for_streaming(
        "Right, give me one second. I am looking through your contacts for "
        "anyone in the work group, and there are quite a few of them to check."
    )
    assert len(pieces[0]) <= len(pieces[1])


def test_splitting_preserves_every_word():
    text = ("Okay, hold on. I'm checking your messages from Marcus, "
            "then I'll look at the calendar; it won't take long.")
    assert " ".join(split_for_streaming(text)).split() == text.split()


def test_a_short_reply_is_a_single_piece():
    assert split_for_streaming("On it.") == ["On it."]


def test_empty_text_produces_nothing_to_say():
    assert split_for_streaming("") == []
    assert split_for_streaming("   ") == []


def test_a_word_longer_than_the_limit_still_terminates():
    """No natural break exists; a hard cut is correct, an infinite loop is not."""
    pieces = split_for_streaming("x" * 500)
    assert pieces and "".join(pieces) == "x" * 500


# --- framing ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_resample_24k_to_16k():
    audio_24k = np.ones(240, dtype=np.float32)  # 10 ms at 24 kHz
    audio_16k = _resample(audio_24k, 24000, 16000)
    assert len(audio_16k) == 160  # 10 ms at 16 kHz
    assert audio_16k.dtype == np.float32


@pytest.mark.asyncio
async def test_every_sample_survives_framing():
    """Framing is a repackaging, not a transformation.

    Kokoro's 24 kHz output is also the playback rate, so nothing is resampled
    on this path and the sample count must come through exactly.
    """
    pipeline = FakePipeline(num_chunks=3, samples=1000)
    frames = await _drain(_tts(pipeline).stream("Test text"))

    assert all(f.dtype == np.float32 for f in frames)
    assert sum(len(f) for f in frames) == 3000
    assert np.allclose(np.concatenate(frames), 0.1)


@pytest.mark.asyncio
async def test_frames_are_bounded_so_playback_can_start_early():
    pipeline = FakePipeline(num_chunks=1, samples=FRAME_SAMPLES * 4 + 7)
    frames = await _drain(_tts(pipeline).stream("Test text"))

    assert len(frames) > 1
    assert all(len(f) <= FRAME_SAMPLES for f in frames)
    assert all(len(f) > 0 for f in frames)


@pytest.mark.asyncio
async def test_each_clause_is_synthesized_separately():
    """This is what makes the first sound arrive before the last word exists."""
    pipeline = FakePipeline(num_chunks=1)
    text = "Okay, hold on a moment. I am checking that for you right now."
    await _drain(_tts(pipeline).stream(text))

    assert len(pipeline.texts) == len(split_for_streaming(text)) > 1


# --- barge-in --------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_halts_the_stream_within_a_frame():
    """Not at the end of the clause, and not at the end of the utterance.

    The generation counter is checked before every emitted frame, so an
    interruption costs at most one frame of audio the person did not want.
    """
    pipeline = FakePipeline(num_chunks=1, samples=FRAME_SAMPLES * 10)
    tts = _tts(pipeline)

    frames = []
    async for frame in tts.stream("A long spoken sentence that keeps going."):
        frames.append(frame)
        if len(frames) == 2:
            tts.stop()

    assert len(frames) == 2


@pytest.mark.asyncio
async def test_a_stopped_stream_never_resumes_behind_a_newer_one():
    """Two utterances must not interleave.

    Stream A is stopped mid-flight and B started; A's generator must be
    finished for good, not merely paused -- with a shared boolean flag,
    starting B would clear the flag and let A resume into B's audio.
    """
    tts = _tts(FakePipeline(num_chunks=10, base_value=0.5))

    generator_a = tts.stream("utterance A")
    first = await generator_a.__anext__()
    assert np.allclose(first, 0.5)

    tts.stop()

    tts._pipeline = FakePipeline(num_chunks=5, base_value=0.7)
    frames_b = await _drain(tts.stream("utterance B"))

    assert frames_b, "the new utterance must still be spoken"
    assert np.allclose(np.concatenate(frames_b), 0.7)

    with pytest.raises(StopAsyncIteration):
        await generator_a.__anext__()


@pytest.mark.asyncio
async def test_abandoning_the_stream_does_not_leave_synthesis_running():
    """A barge-in exits the `async for` without draining the generator.

    Synthesis runs one clause ahead, in a task feeding a depth-1 queue. If
    that task is not cancelled when the consumer walks away, it blocks forever
    on a queue nobody will read -- one leaked task and one busy worker thread
    per interruption, for the life of the process.
    """
    pipeline = FakePipeline(num_chunks=1, samples=FRAME_SAMPLES)
    tts = _tts(pipeline)

    stream = tts.stream("One. Two. Three. Four. Five. Six. Seven.")
    await stream.__anext__()
    await stream.aclose()

    await asyncio.sleep(0)
    assert len(pipeline.texts) < 7


@pytest.mark.asyncio
async def test_consecutive_streams_are_independent():
    tts = _tts(FakePipeline(num_chunks=3))

    first = await _drain(tts.stream("First"))
    second = await _drain(tts.stream("Second"))

    assert sum(len(f) for f in first) == 3000
    assert sum(len(f) for f in second) == 3000


@pytest.mark.asyncio
async def test_nothing_to_say_yields_nothing():
    tts = _tts(FakePipeline(num_chunks=3))
    assert await _drain(tts.stream("   ")) == []


# --- a remote engine, behind the same interface -----------------------------
#
# Chatterbox sounds far more human than Kokoro and cannot run beside the rest
# of the system: it requires numpy<2, which would downgrade the numpy the
# turn-taking model depends on. So it runs elsewhere and is called over HTTP --
# and everything that makes speech start early and stop on a barge-in has to
# keep working across that boundary, because it is the same code.

class _FakeHTTP:
    """Stands in for httpx.Client; returns raw float32 PCM per clause."""

    def __init__(self, samples=1200, exc=None):
        self.samples = samples
        self.exc = exc
        self.sent = []

    def post(self, url, json=None, **kw):
        self.sent.append(json["text"])
        if self.exc is not None:
            raise self.exc
        import httpx as _httpx
        audio = (np.ones(self.samples, dtype=np.float32) * 0.3).tobytes()
        return _httpx.Response(200, content=audio,
                               request=_httpx.Request("POST", url))


def _remote(http):
    from rtvoice.tts import RemoteTTS
    tts = object.__new__(RemoteTTS)
    tts.base_url = "http://x"
    tts._client = http
    tts._generation = 0
    tts.failures = 0
    return tts


@pytest.mark.asyncio
async def test_a_remote_engine_streams_like_a_local_one():
    http = _FakeHTTP(samples=FRAME_SAMPLES * 3)
    frames = await _drain(_remote(http).stream("Okay, hold on. I'm checking that now."))

    assert len(frames) > 1
    assert all(len(f) <= FRAME_SAMPLES for f in frames)
    assert np.allclose(np.concatenate(frames), 0.3)


@pytest.mark.asyncio
async def test_the_remote_engine_is_asked_one_clause_at_a_time():
    """Not the whole utterance: the caller runs a clause ahead of playback, so
    every round trip after the first is overlapped with speaking the previous
    clause and only the FIRST is ever waited on."""
    http = _FakeHTTP()
    text = "Okay, hold on a moment. I am checking that for you right now."
    await _drain(_remote(http).stream(text))

    assert http.sent == split_for_streaming(text)
    assert len(http.sent) > 1


@pytest.mark.asyncio
async def test_a_remote_failure_is_silence_not_an_exception():
    """The caller is _speak_facts, speaking a result someone is waiting on --
    inside the audio path. An exception there is worse than a quiet turn."""
    import httpx as _httpx
    tts = _remote(_FakeHTTP(exc=_httpx.ConnectError("refused")))

    assert await _drain(tts.stream("Anything at all.")) == []
    assert tts.failures > 0


@pytest.mark.asyncio
async def test_barge_in_works_across_the_wire_too():
    http = _FakeHTTP(samples=FRAME_SAMPLES * 10)
    tts = _remote(http)

    frames = []
    async for frame in tts.stream("A long spoken sentence that keeps going."):
        frames.append(frame)
        if len(frames) == 2:
            tts.stop()

    assert len(frames) == 2
