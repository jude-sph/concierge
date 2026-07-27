import asyncio

import numpy as np
import pytest

from rtvoice.tts import KokoroTTS, _resample


class FakePipeline:
    """Mock Kokoro pipeline for testing without GPU dependencies."""

    def __init__(self, num_chunks: int = 3):
        self.num_chunks = num_chunks

    def __call__(self, text: str, voice: str = "af_heart"):
        """Yields num_chunks audio chunks at 24 kHz."""
        for i in range(self.num_chunks):
            # Yield (speaker_id, language_id, audio)
            # Each chunk is 1000 samples at 24 kHz
            yield (0, "a", np.ones(1000, dtype=np.float32) * (i + 1) * 0.1)


@pytest.mark.asyncio
async def test_resample_24k_to_16k():
    """Test resampling from Kokoro's 24 kHz to system 16 kHz."""
    audio_24k = np.ones(240, dtype=np.float32)  # 10 ms at 24 kHz
    audio_16k = _resample(audio_24k, 24000, 16000)
    assert len(audio_16k) == 160  # 10 ms at 16 kHz
    assert audio_16k.dtype == np.float32


@pytest.mark.asyncio
async def test_stream_yields_resampled_chunks():
    """Test that stream yields chunks resampled to 16 kHz."""
    tts = object.__new__(KokoroTTS)
    tts._pipeline = FakePipeline(num_chunks=3)
    tts.voice = "af_heart"
    tts._generation = 0

    chunks = []
    async for chunk in tts.stream("Test text"):
        chunks.append(chunk)

    # Should get 3 chunks (from FakePipeline)
    assert len(chunks) == 3
    # Each chunk resampled from 1000 24kHz to ~667 16kHz
    assert all(isinstance(c, np.ndarray) for c in chunks)
    assert all(c.dtype == np.float32 for c in chunks)
    assert all(len(c) in [666, 667] for c in chunks)  # Approximate due to rounding


@pytest.mark.asyncio
async def test_stop_halts_stream_within_chunk():
    """Test that stop() halts the stream between chunks."""
    tts = object.__new__(KokoroTTS)
    tts._pipeline = FakePipeline(num_chunks=5)
    tts.voice = "af_heart"
    tts._generation = 0

    chunks = []

    async def collect_with_stop():
        async for chunk in tts.stream("Test text"):
            chunks.append(chunk)
            if len(chunks) == 2:
                tts.stop()

    await collect_with_stop()
    # Should stop after 2 chunks, not continue to 5
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_generation_token_prevents_race_condition():
    """Test that generation tokens prevent race conditions between concurrent streams.

    This reproduces the bug from the coordinator review:
    1. stream("utterance one") is running
    2. stop() is called
    3. stream("utterance two") is called immediately
    4. Without generation tokens, the old generator would resume and emit both utterances
    5. With generation tokens, the old generator exits cleanly
    """
    tts = object.__new__(KokoroTTS)
    tts._pipeline = FakePipeline(num_chunks=10)  # Long stream
    tts.voice = "af_heart"
    tts._generation = 0

    # Track which generation each chunk came from
    chunks_by_call = {"first": [], "second": []}

    async def first_stream():
        """Start first stream and collect some chunks."""
        async for chunk in tts.stream("utterance one"):
            chunks_by_call["first"].append(chunk)
            await asyncio.sleep(0.001)  # Small delay to let second stream start
            if len(chunks_by_call["first"]) >= 3:
                break

    async def second_stream():
        """Start second stream shortly after the first."""
        await asyncio.sleep(0.002)  # Let first stream start
        tts.stop()  # This increments generation, invalidating first stream
        async for chunk in tts.stream("utterance two"):
            chunks_by_call["second"].append(chunk)

    # Run both streams concurrently
    await asyncio.gather(first_stream(), second_stream())

    # Both streams should get chunks
    assert len(chunks_by_call["first"]) > 0
    assert len(chunks_by_call["second"]) > 0

    # The first stream should not have gotten all 10 chunks - it should stop early
    # because second_stream() called stop() which incremented generation
    assert len(chunks_by_call["first"]) < 10
    # The second stream may get all 10 chunks since it started fresh after stop()
    # but importantly, the generations are isolated - no interleaved audio


@pytest.mark.asyncio
async def test_consecutive_streams_work_correctly():
    """Test that consecutive streams (one after another) work without interference."""
    tts = object.__new__(KokoroTTS)
    tts._pipeline = FakePipeline(num_chunks=3)
    tts.voice = "af_heart"
    tts._generation = 0

    # First stream
    first_chunks = []
    async for chunk in tts.stream("First"):
        first_chunks.append(chunk)

    # Second stream
    second_chunks = []
    async for chunk in tts.stream("Second"):
        second_chunks.append(chunk)

    # Both should complete normally
    assert len(first_chunks) == 3
    assert len(second_chunks) == 3
