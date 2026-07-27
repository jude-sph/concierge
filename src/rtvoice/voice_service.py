"""Owns all audio: SoulX-Duplug in, Kokoro out, both channels recorded.

This is the relocatable boundary. If the tunnel measures badly (Task 1), this
service moves to the Mac and only text crosses the wire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .recorder import SessionRecorder
from .soulx_client import CHUNK_SAMPLES, SAMPLE_RATE, SoulXClient
from .states import StateAdapter, TurnEvent

CHUNK_MS = int(CHUNK_SAMPLES / SAMPLE_RATE * 1000)


def _rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))


class AutoGainControl:
    """Smoothed, speech-level gain control for the audio sent upstream.

    Why this exists: the upstream turn-taking model gates out chunks below
    ~0.02 RMS as far-field noise, and short quiet utterances get reset before
    they can accumulate. A real "Yes." confirming a destructive write was
    silently dropped -- zero turn events -- at RMS 0.017 / peak 0.29.

    This is the SECOND design, replacing an RMS-targeted one that overshot.
    A sweep of gain levels against the real upstream model (on a GPU, not
    available in this dev environment) showed the true picture:

        variant                       peak    model result
        raw (0.29)                    0.290   DROPPED (too quiet)
        uniform 2.0x-3.0x             0.58-0.87  OK
        uniform 4.0x                  1.000   DROPPED (clipped)
        peak-normalised to 0.70        0.700   OK, cleanest
        prior AGC (RMS-target, snap)   1.000   1 turn but EMPTY TRANSCRIPT

    Every success sat in output peak ~0.55-0.87; every failure was either
    too quiet (peak <= 0.435) or clipped (peak == 1.000, worst of all --
    clipping produced a turn with a blank transcript, which is worse than
    being dropped because downstream code then acts on an empty utterance).
    The RMS-targeted predecessor drove all four real test clips to peak
    1.000, i.e. it was optimizing the wrong variable: RMS headroom does not
    predict peak headroom for real speech, whose crest factor varies chunk
    to chunk. Peak is therefore the control variable here, not RMS.

    Design, each point load-bearing:

    - The gain is driven by a running estimate of *speech-like* chunk peak
      (each chunk's own `max(abs(chunk))`), smoothed with a single EMA
      coefficient (`smoothing`) across the sequence -- not applied per chunk
      from that chunk's own peak. A single stray sample (a click, a brief
      clip-like transient) is damped by the EMA rather than immediately
      redefining "the speech level" for every subsequent chunk; the running
      estimate moves by only `(1 - smoothing)` of the way toward any one
      chunk's observed peak. This is what keeps the gain from chasing a
      one-off transient spike up and back down.
    - Any chunk whose OWN rms is below `floor_rms` is passed through
      untouched and does not perturb the running estimate (silence
      detection stays RMS-based -- a peak-based floor would be fooled by a
      single noise sample). This check is per-chunk and independent of the
      running estimate on purpose: a silent chunk arriving right after loud
      speech must not be boosted by a stale high estimate.
    - The first speech-like chunk after a silent gap snaps the running peak
      estimate straight to that chunk's own peak instead of easing into it
      from whatever the estimate was before the gap (fast attack; ordinary
      EMA smoothing resumes on the chunks after that, i.e. slower release).
      Without this, a short utterance -- a one-word confirmation being the
      exact case that matters -- can be over before a slow EMA ramp from a
      neutral start ever reaches a useful gain (this was already true, and
      already fixed, under the old RMS framing; the mechanism carries over
      unchanged, just measuring peak instead of RMS).
    - The gain is clamped to [1.0, max_gain]: it only tries to boost, never
      attenuate, based on the target -- but see the ceiling below, which
      can and does override this when the input is already loud enough on
      its own to risk breaching the ceiling.
    - After scaling, the result is limited to a hard ceiling strictly below
      1.0 (`ceiling`, default 0.9) rather than allowed to approach or touch
      1.0. This is a correctness requirement, not a nicety: the measured
      evidence above shows output peak 1.000 corrupting the transcript even
      when a turn was still detected. If the scaled chunk's true peak would
      exceed the ceiling -- whether because the gain overshot or because the
      raw input was already loud enough by itself -- gain is reduced
      (attenuating if necessary) so the ceiling is never crossed.
    """

    def __init__(
        self,
        *,
        target_peak: float = 0.65,
        ceiling: float = 0.9,
        max_gain: float = 10.0,
        floor_rms: float = 0.003,
        smoothing: float = 0.9,
    ):
        if not 0.0 < ceiling < 1.0:
            raise ValueError("ceiling must be strictly between 0 and 1")
        self.target_peak = target_peak
        self.ceiling = ceiling
        self.max_gain = max_gain
        self.floor_rms = floor_rms
        self.smoothing = smoothing
        self.peak_level = target_peak  # neutral start: gain 1.0 until speech is seen
        self.last_gain = 1.0
        self.last_input_rms = 0.0
        # Set on every silent chunk, cleared on the next speech-like one.
        # Marks that the running estimate is stale from before a gap, so the
        # next real speech should snap the estimate rather than ease into it.
        self._after_silence = False

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        input_rms = _rms(chunk)
        self.last_input_rms = input_rms

        if input_rms < self.floor_rms:
            # Silence (or near enough): pass through untouched. Do not let
            # this chunk perturb the running level estimate -- a burst of
            # silence must not average itself into "recent speech level".
            self.last_gain = 1.0
            self._after_silence = True
            return chunk

        chunk_peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0

        if self._after_silence:
            # First speech-like chunk after a gap: snap straight to its
            # measured peak instead of easing in from the stale estimate.
            # A short utterance (e.g. a one-word confirmation) can be over
            # before a slow EMA ramp ever gets there -- fast attack here,
            # ordinary smoothing (slower release) below once speech is
            # already established.
            self.peak_level = max(chunk_peak, 1e-6)
            self._after_silence = False
        else:
            self.peak_level = (
                self.smoothing * self.peak_level + (1 - self.smoothing) * chunk_peak
            )

        raw_gain = self.target_peak / max(self.peak_level, 1e-6)
        gain = min(max(raw_gain, 1.0), self.max_gain)

        out = chunk * np.float32(gain)
        true_peak = float(np.max(np.abs(out))) if out.size else 0.0
        if true_peak > self.ceiling:
            # Hard ceiling: never let output approach 1.0, whether that's
            # because the target-based gain overshot or because the raw
            # input was already loud enough on its own. This can reduce
            # gain below 1.0 (attenuate) -- deliberately overriding the
            # "only ever boost" rule above, because breaching the ceiling
            # is worse than leaving already-loud audio untouched.
            gain *= self.ceiling / true_peak
            out = chunk * np.float32(gain)

        self.last_gain = gain
        return out


class VoiceService:
    def __init__(
        self,
        session_dir: str | Path,
        soulx_url: str = "ws://localhost:8000/turn",
        *,
        tts_factory: Callable[[], object] | None = None,
        # OFF by default: measured net-harmful against the real model. See below.
        agc: bool = False,
        agc_target_rms: float = 0.05,
        agc_target_peak: float = 0.65,
    ):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.recorder = SessionRecorder(session_dir)
        self._tts = None
        self._tts_factory = tts_factory
        self._t_ms = 0
        self.agc = agc
        # `agc_target_rms` predates the switch to peak-targeted gain (see
        # AutoGainControl's docstring for why RMS turned out to be the wrong
        # control variable). It is kept as an accepted keyword purely for
        # backward compatibility with existing callers/tests -- it is no
        # longer wired into the gain calculation. Use `agc_target_peak`.
        self.agc_target_rms = agc_target_rms
        self.agc_target_peak = agc_target_peak
        self._agc = AutoGainControl(target_peak=agc_target_peak) if agc else None
        # Visible per-chunk record of what AGC did, so a session can be
        # inspected after the fact to see whether AGC was active and how
        # hard it was working.
        self.last_agc_gain = 1.0
        self.last_agc_input_rms = 0.0

    # -- text to speech -------------------------------------------------------
    #
    # `tts` used to be a plain attribute set to None and never assigned by
    # anything, so the first spoken reply raised AttributeError -- swallowed
    # by the orchestrator's return_exceptions=True gather and logged as
    # turn_task_error, i.e. the system failed to speak invisibly. It is now
    # built on FIRST USE. Lazily, because Kokoro pulls in GPU dependencies
    # that must not be imported at module scope (and are not installed in the
    # test environment), and because constructing a VoiceService must stay
    # cheap and side-effect-free.

    @property
    def tts(self):
        if self._tts is None:
            self._tts = self._make_tts()
        return self._tts

    @tts.setter
    def tts(self, value) -> None:
        self._tts = value

    def _make_tts(self):
        if self._tts_factory is not None:
            return self._tts_factory()
        from . import tts as tts_module

        try:
            return tts_module.KokoroTTS()
        except Exception as exc:  # ImportError, missing model, no GPU, ...
            raise RuntimeError(
                "text-to-speech is unavailable: KokoroTTS could not be "
                "constructed. The `kokoro` package and its GPU dependencies "
                "are not installed in every environment this runs in. "
                "Install kokoro, or supply your own: "
                "VoiceService(session_dir, tts_factory=...) or voice.tts = <obj>."
            ) from exc

    @property
    def stream_ms(self) -> int:
        """Milliseconds of audio fed so far.

        This is the clock TurnEvents are stamped with, so it is also the clock
        Orchestrator.on_tick must be driven on -- wall-clock time would make
        the silence timeout meaningless whenever audio is not being played at
        exactly real time.
        """
        return self._t_ms

    async def feed_audio(self, chunk: np.ndarray) -> list[TurnEvent]:
        if self._agc is not None:
            chunk = self._agc.apply(chunk)
            self.last_agc_gain = self._agc.last_gain
            self.last_agc_input_rms = self._agc.last_input_rms
        else:
            self.last_agc_gain = 1.0
            self.last_agc_input_rms = _rms(np.asarray(chunk, dtype=np.float32))
        # The recorder must capture what was actually sent upstream (post
        # AGC), so a replay of a session reproduces what the model saw --
        # hence this runs on `chunk` only after AGC has (maybe) replaced it.
        self.recorder.write_user(chunk)
        wire = await self.client.feed(chunk)
        events = self.adapter.feed(wire, self._t_ms)
        self._t_ms += CHUNK_MS
        return events

    async def speak(self, text: str, utterance_id: str) -> None:
        async for chunk in self.tts.stream(text):
            self.recorder.write_model(chunk)

    async def stop(self) -> None:
        # Deliberately reads the private slot: stopping speech must never be
        # the thing that constructs a (possibly unavailable) TTS engine.
        if self._tts is not None:
            self._tts.stop()

    def close(self) -> None:
        self.recorder.close()
