import asyncio

import numpy as np
import pytest

from rtvoice.tts import KokoroTTS, _resample


class FakePipeline:
    """Mock Kokoro pipeline for testing without GPU dependencies."""

    def __init__(self, num_chunks: int = 3, base_value: float = 0.1):
        self.num_chunks = num_chunks
        self.base_value = base_value

    def __call__(self, text: str, voice: str = "af_heart"):
        """Yields num_chunks audio chunks at 24 kHz."""
        for i in range(self.num_chunks):
            # Yield (speaker_id, language_id, audio)
            # Each chunk is 1000 samples at 24 kHz
            # Use base_value to distinguish between different streams
            yield (0, "a", np.ones(1000, dtype=np.float32) * self.base_value)


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

    This reproduces the exact bug from the coordinator review:
    1. Start stream(A) and pull ONE chunk (do not break)
    2. Call stop() to invalidate stream A
    3. Start stream(B) and collect all its chunks
    4. Try to resume generator A - must raise StopAsyncIteration
    5. Assert stream B contains ONLY B's values, none of A's (no interleaving)

    The test uses distinguishable chunk values (0.5 for A, 0.7 for B) to detect
    any interleaving. With the old _stopped flag implementation, this test would
    FAIL because the old generator would resume after stop() sets the flag, then
    stream(B) resets it to False, allowing A to continue yielding its 0.5 values.
    """
    tts = object.__new__(KokoroTTS)
    # Pipeline for stream A yields chunks with value 0.5
    pipeline_a = FakePipeline(num_chunks=10, base_value=0.5)
    # Pipeline for stream B yields chunks with value 0.7
    pipeline_b = FakePipeline(num_chunks=5, base_value=0.7)
    tts.voice = "af_heart"
    tts._generation = 0

    # Start stream A and pull one chunk
    generator_a = tts.stream("utterance A")
    tts._pipeline = pipeline_a
    chunk_a = await generator_a.__anext__()
    # Verify it's from A
    assert np.allclose(chunk_a, 0.5, atol=1e-6), "First chunk should be from stream A (0.5)"

    # Now call stop() to invalidate this generator
    tts.stop()

    # Start stream B and collect all its chunks
    chunks_b = []
    tts._pipeline = pipeline_b
    generator_b = tts.stream("utterance B")
    async for chunk in generator_b:
        chunks_b.append(chunk)

    # Stream B should have gotten all 5 chunks
    assert len(chunks_b) == 5, f"Stream B should get 5 chunks, got {len(chunks_b)}"

    # Critical assertion: all of B's chunks should be 0.7 (from pipeline_b)
    # If the old _stopped flag implementation were used, A's chunks (0.5) would
    # interleave with B's chunks when we resume A's generator
    for chunk in chunks_b:
        assert np.allclose(
            chunk, 0.7, atol=1e-6
        ), f"Stream B chunk should be 0.7, got {chunk[0]}"

    # Now try to resume generator A - it must raise StopAsyncIteration
    # With the old implementation (shared flag), it would instead yield more 0.5 chunks
    with pytest.raises(StopAsyncIteration):
        await generator_a.__anext__()


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
