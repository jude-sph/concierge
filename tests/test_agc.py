"""Automatic gain control on the path into `VoiceService.feed_audio()`.

Why this exists: 14 real voice clips were run through the upstream turn-taking
model. One -- a bare "Yes." confirming a destructive device write -- was
silently discarded: zero turn events, not even speech-detected. Its RMS was
0.017. The same clip peak-normalised to 0.70 produced the correct turn. The
upstream model gates out anything below ~0.02 RMS as far-field noise, and
short quiet utterances get reset before they can accumulate. Quiet speech
("yes" said softly, trailing off, sitting back) is normal, and a user
confirming a destructive write deserves better than being ignored without a
trace.

These tests exercise gain normalisation through the real `VoiceService`, with
`SoulXClient.feed` replaced by a fake that records exactly the audio it was
handed (this is the "sent upstream" boundary AGC must sit in front of) and
returns a `blank` wire state so no turn machinery is exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from rtvoice.soulx_client import CHUNK_SAMPLES
from rtvoice.voice_service import VoiceService


def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk))))


def make_chunk(level: float, seed: int, n: int = CHUNK_SAMPLES) -> np.ndarray:
    """A chunk of pseudo-speech noise with (approximately) the given RMS."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n).astype(np.float32)
    current = rms(raw)
    if current == 0:
        return raw
    return (raw * (level / current)).astype(np.float32)


class FakeClient:
    """Stands in for SoulXClient: records the exact audio it was fed upstream
    and returns a `blank` state so StateAdapter emits no events."""

    def __init__(self):
        self.fed: list[np.ndarray] = []

    async def feed(self, chunk: np.ndarray) -> dict:
        self.fed.append(np.array(chunk, dtype=np.float32, copy=True))
        return {"state": "blank"}


def make_voice(tmp_path, **kwargs) -> tuple[VoiceService, FakeClient]:
    voice = VoiceService(tmp_path, **kwargs)
    fake = FakeClient()
    voice.client = fake
    return voice, fake


async def feed_many(voice: VoiceService, chunks: list[np.ndarray]) -> None:
    for chunk in chunks:
        await voice.feed_audio(chunk)


# --- quiet speech gets brought up ---------------------------------------------

@pytest.mark.asyncio
async def test_quiet_speech_is_brought_up_toward_the_target(tmp_path):
    """RMS ~0.017 is exactly the level of the clip that vanished upstream.
    After AGC settles, the audio actually sent upstream should sit much
    closer to the 0.05 target than the original 0.017."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        chunks = [make_chunk(0.017, seed=i) for i in range(60)]
        await feed_many(voice, chunks)

        sent_rms = [rms(c) for c in fake.fed[-10:]]  # after the estimate settles
        assert all(r > 0.03 for r in sent_rms), sent_rms
        assert min(sent_rms) > rms(chunks[0]) * 1.5
    finally:
        voice.close()


# --- loud speech is left alone -------------------------------------------------

@pytest.mark.asyncio
async def test_loud_speech_is_not_amplified_further_and_does_not_clip(tmp_path):
    """Levels like the clips that already worked (RMS 0.04-0.075) and louder
    (peak-normalised clip08N) must not be pushed up further, and must never
    exceed [-1, 1]."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        chunks = [make_chunk(0.3, seed=100 + i) for i in range(30)]
        await feed_many(voice, chunks)

        for original, sent in zip(chunks, fake.fed):
            assert rms(sent) <= rms(original) + 1e-6
            assert np.max(np.abs(sent)) <= 1.0
    finally:
        voice.close()


# --- silence must never be amplified into noise --------------------------------

@pytest.mark.asyncio
async def test_silence_stays_silence(tmp_path):
    """The most important property in this task: a run of near-zero chunks
    must not be amplified into fake speech. Getting this wrong destroys the
    silence detection the turn-taking model depends on."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        rng = np.random.default_rng(7)
        chunks = [
            (rng.standard_normal(CHUNK_SAMPLES).astype(np.float32) * 1e-5)
            for _ in range(40)
        ]
        await feed_many(voice, chunks)

        for original, sent in zip(chunks, fake.fed):
            assert rms(sent) < 0.001
            np.testing.assert_allclose(sent, original, atol=1e-7)
    finally:
        voice.close()


@pytest.mark.asyncio
async def test_silence_after_loud_speech_is_not_amplified_by_a_stale_estimate(tmp_path):
    """A stale high running-level estimate from earlier loud speech must not
    cause a subsequent silent chunk to be boosted."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        loud = [make_chunk(0.3, seed=200 + i) for i in range(20)]
        await feed_many(voice, loud)

        silent = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        await voice.feed_audio(silent)

        sent = fake.fed[-1]
        assert not sent.any()
    finally:
        voice.close()


# --- gain is bounded -------------------------------------------------------------

@pytest.mark.asyncio
async def test_gain_is_bounded_by_the_cap(tmp_path):
    """A chunk that is quiet but not silent (above the noise floor, far below
    target) must not be blown up arbitrarily -- the applied gain is capped."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        chunks = [make_chunk(0.001, seed=300 + i) for i in range(80)]
        await feed_many(voice, chunks)

        for original, sent in zip(chunks[-20:], fake.fed[-20:]):
            in_rms = rms(original)
            out_rms = rms(sent)
            if in_rms > 1e-9:
                assert out_rms / in_rms <= 10.0 + 1e-6
        assert voice.last_agc_gain <= 10.0 + 1e-6
    finally:
        voice.close()


# --- never clip -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_never_exceeds_unit_range_even_with_an_impulsive_peak(tmp_path):
    """A chunk with low RMS but one near-full-scale sample must not be pushed
    past [-1, 1] by the gain that a low RMS would otherwise justify."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        # Warm the estimate up to something that would justify a large gain.
        for i in range(30):
            await voice.feed_audio(make_chunk(0.017, seed=400 + i))

        spiky = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        spiky[0] = 0.95
        spiky[1] = -0.9
        await voice.feed_audio(spiky)

        sent = fake.fed[-1]
        assert np.max(np.abs(sent)) <= 1.0 + 1e-6
    finally:
        voice.close()


# --- agc=False is a bit-identical passthrough -------------------------------------

@pytest.mark.asyncio
async def test_agc_false_passes_audio_through_bit_identically(tmp_path):
    voice, fake = make_voice(tmp_path, agc=False)
    try:
        chunks = [make_chunk(0.017, seed=i) for i in range(10)] + [
            np.zeros(CHUNK_SAMPLES, dtype=np.float32),
            make_chunk(0.3, seed=999),
        ]
        await feed_many(voice, chunks)

        for original, sent in zip(chunks, fake.fed):
            np.testing.assert_array_equal(sent, original)
    finally:
        voice.close()


# --- the estimate adapts across a sequence, not per chunk --------------------------

@pytest.mark.asyncio
async def test_gain_estimate_adapts_across_a_sequence_rather_than_per_chunk(tmp_path):
    """A single quiet chunk right after silence/startup should not already be
    boosted all the way to the target -- the running estimate must build up
    over several chunks, exactly so a single loud or quiet outlier chunk
    can't swing the gain wildly (which would itself reintroduce clipping and
    fake-silence risk)."""
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        chunks = [make_chunk(0.017, seed=500 + i) for i in range(40)]
        await feed_many(voice, chunks)

        first_gain = rms(fake.fed[0]) / rms(chunks[0])
        late_gains = [rms(s) / rms(o) for s, o in zip(fake.fed[-5:], chunks[-5:])]
        target_gain = 0.05 / 0.017

        assert first_gain < target_gain * 0.9
        assert min(late_gains) > first_gain
    finally:
        voice.close()


# --- what it did is visible ---------------------------------------------------------

@pytest.mark.asyncio
async def test_applied_gain_and_input_rms_are_recorded_per_chunk(tmp_path):
    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        quiet = make_chunk(0.017, seed=600)
        await voice.feed_audio(quiet)
        assert voice.last_agc_input_rms == pytest.approx(rms(quiet), rel=0.05)
        assert voice.last_agc_gain >= 1.0

        loud = make_chunk(0.3, seed=601)
        await voice.feed_audio(loud)
        assert voice.last_agc_input_rms == pytest.approx(rms(loud), rel=0.05)
    finally:
        voice.close()


# --- recording captures what was actually sent upstream -----------------------------

@pytest.mark.asyncio
async def test_recorder_captures_post_agc_audio(tmp_path):
    """A replay of a session must reproduce what the model actually saw, so
    the recorded user audio must be the post-AGC signal, not the raw input."""
    import soundfile as sf

    voice, fake = make_voice(tmp_path, agc=True, agc_target_rms=0.05)
    try:
        chunks = [make_chunk(0.017, seed=700 + i) for i in range(30)]
        await feed_many(voice, chunks)
    finally:
        voice.close()

    recorded, _ = sf.read(tmp_path / "user.wav", dtype="float32")
    sent_concat = np.concatenate(fake.fed)
    np.testing.assert_allclose(recorded, sent_concat, atol=1e-6)

    raw_concat = np.concatenate(chunks)
    assert not np.allclose(recorded, raw_concat, atol=1e-3)
