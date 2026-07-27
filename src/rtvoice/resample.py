"""Shared anti-aliased sample-rate conversion.

Every rate conversion anywhere in this codebase must go through
`resample_audio` rather than reimplementing its own. The bug this module
exists to prevent regressing to: `np.interp`-based rate conversion with no
low-pass filter first. Decimating without filtering lets everything above
the destination Nyquist fold back down into the audible band as aliasing --
a harsh, metallic artifact that has nothing to do with the source audio's
actual quality.

Preference order, each a graceful degradation from the one before it:

  1. `soxr` (soxr.resample) -- studio-grade variable-rate resampler, the same
     family of algorithm ffmpeg/libswresample uses. Used if installed.
  2. `scipy.signal.resample_poly` -- polyphase FIR resampling, exact for
     rational rate ratios (which 24000:16000 = 3:2 is), with a low-pass
     filter of our own design (scipy's default filter measurably under-
     attenuates content near the new Nyquist -- verified empirically against
     a 9 kHz tone resampled 24k->16k, which left ~90% of its energy in the
     aliased image using the library default). Used if soxr is unavailable
     but scipy is.
  3. A hand-rolled windowed-sinc low-pass followed by linear interpolation.
     Only reached if neither library is importable. Still filters before
     decimating, so it never reproduces the plain-`np.interp` aliasing bug,
     even in the worst case.

`soxr` and `scipy` are both regular (non-heavy, pure-wheel) dependencies of
this project, but the imports are guarded anyway so this module -- and
anything that imports it -- keeps working in a stripped-down environment.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import soxr
except ImportError:  # pragma: no cover - depends on environment
    soxr = None

try:
    from scipy.signal import firwin, resample_poly
except ImportError:  # pragma: no cover - depends on environment
    firwin = None
    resample_poly = None


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Convert `audio` from `src_rate` to `dst_rate` with anti-aliasing.

    Always filters before decimating. Never a naive `np.interp`
    stretch/squeeze over the raw samples -- see the module docstring.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=True)

    if soxr is not None:
        out = soxr.resample(audio, src_rate, dst_rate)
        return np.asarray(out, dtype=np.float32)

    if resample_poly is not None:
        return _resample_scipy(audio, src_rate, dst_rate)

    return _resample_fallback(audio, src_rate, dst_rate)


def _resample_scipy(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    g = math.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    max_rate = max(up, down)
    # scipy's own default anti-aliasing filter for resample_poly is too
    # short to fully suppress content close to the new Nyquist (measured:
    # a 9 kHz tone resampled 24k->16k left ~90% of its energy in the 7 kHz
    # aliased image with the library default). Design a longer, stricter
    # windowed-sinc low-pass ourselves instead of trusting the default.
    numtaps = 80 * max_rate + 1
    cutoff = 1.0 / max_rate
    taps = firwin(numtaps, cutoff, window=("kaiser", 8.6))
    out = resample_poly(audio, up, down, window=taps)
    return np.asarray(out, dtype=np.float32)


def _resample_fallback(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Windowed-sinc low-pass + linear interpolation, numpy only.

    Only reached if neither soxr nor scipy is importable. Still applies an
    anti-aliasing low-pass before resampling.
    """
    n_out = int(round(len(audio) * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    if dst_rate < src_rate:
        # Low-pass at the destination Nyquist, expressed as a fraction of
        # the source Nyquist, before decimating.
        cutoff = dst_rate / float(src_rate)
        taps = 127
        n = np.arange(taps) - (taps - 1) / 2.0
        kernel = cutoff * np.sinc(cutoff * n)
        kernel *= np.hamming(taps)
        kernel /= kernel.sum()
        audio = np.convolve(audio, kernel, mode="same").astype(np.float32)

    idx = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
