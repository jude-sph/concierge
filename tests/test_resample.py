"""Anti-aliasing regression test for `rtvoice.resample`.

The bug this guards against: `tts.py` used to resample Kokoro's 24 kHz
output down to 16 kHz with plain `np.interp` and no low-pass filter first.
Every frequency above the new 8 kHz Nyquist folded back down into the
audible band as aliasing -- a harsh, metallic artifact reported as "the
synthesized voice sounds bad", which had nothing to do with the TTS model
itself.

This test feeds a tone above the destination Nyquist through the resampler
and checks the output does not contain a strong spurious *low* frequency
component at the alias location (src_rate - dst_rate folds an above-Nyquist
tone down to dst_rate - freq, e.g. 9 kHz aliases to 7 kHz at 24k->16k). A
correct anti-aliasing low-pass suppresses the tone instead of letting it fold
down.

`test_naive_interp_resampling_does_alias` reproduces the old implementation
inline (it no longer exists in the codebase) specifically to demonstrate that
the new assertion is not vacuous: it fails against the old approach and
passes against the new one.
"""
from __future__ import annotations

import numpy as np

from rtvoice.resample import resample_audio

SRC_RATE = 24000
DST_RATE = 16000


def _tone(freq: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _band_energy_fraction(signal: np.ndarray, rate: int, target_freq: float, bw: float = 200.0) -> float:
    """Fraction of `signal`'s total spectral energy sitting within `bw` Hz of
    `target_freq`."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / rate)
    total = float(np.sum(np.abs(spectrum) ** 2))
    if total == 0.0:
        return 0.0
    mask = (freqs >= target_freq - bw) & (freqs <= target_freq + bw)
    return float(np.sum(np.abs(spectrum[mask]) ** 2)) / total


def _naive_interp_resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """The old, buggy implementation this test file exists to keep fixed --
    plain linear interpolation, no anti-aliasing filter. Reproduced here
    rather than imported: it no longer exists anywhere in the codebase."""
    n = int(round(len(audio) * dst / src))
    idx = np.linspace(0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def test_naive_interp_resampling_does_alias():
    """Sanity check on the test itself: confirms a tone above the new
    Nyquist, run through the old naive approach, really does show up as a
    strong low-frequency alias -- so the assertion below is not vacuous."""
    tone = _tone(freq=9000.0, seconds=0.2, rate=SRC_RATE)  # > 8 kHz Nyquist at 16k
    aliased = _naive_interp_resample(tone, SRC_RATE, DST_RATE)
    alias_freq = DST_RATE - 9000.0  # 7000 Hz: where 9 kHz folds to at 16 kHz
    fraction = _band_energy_fraction(aliased, DST_RATE, alias_freq)
    assert fraction > 0.5, (
        f"expected the naive implementation to alias strongly, got only "
        f"{fraction:.2%} of energy at the alias frequency"
    )


def test_resample_audio_does_not_alias_a_tone_above_the_destination_nyquist():
    """The actual regression test: the real resampler in use today must not
    show this same aliasing artifact."""
    tone = _tone(freq=9000.0, seconds=0.2, rate=SRC_RATE)
    out = resample_audio(tone, SRC_RATE, DST_RATE)
    alias_freq = DST_RATE - 9000.0
    fraction = _band_energy_fraction(out, DST_RATE, alias_freq)
    assert fraction < 0.3, (
        f"resample_audio aliased a 9 kHz tone down to ~{alias_freq:.0f} Hz: "
        f"{fraction:.2%} of output energy sits there (should be filtered out)"
    )


def test_resample_audio_is_a_no_op_when_rates_match():
    audio = np.linspace(-1, 1, 500, dtype=np.float32)
    out = resample_audio(audio, 16000, 16000)
    assert np.array_equal(out, audio)
    assert out.dtype == np.float32


def test_resample_audio_handles_empty_input():
    out = resample_audio(np.zeros(0, dtype=np.float32), 24000, 16000)
    assert len(out) == 0
    assert out.dtype == np.float32
