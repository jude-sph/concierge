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
    silently dropped -- zero turn events -- at RMS 0.017; the identical clip
    peak-normalised to RMS ~0.07 was transcribed correctly. Quiet speech is
    normal (trailing off, sitting back), so this exists to bring it up to a
    level the model reliably hears, without turning room tone into fake
    speech.

    Design, each point load-bearing:

    - The gain is driven by a running EMA of *speech-like* chunk RMS, not by
      each chunk's own level. Per-chunk normalisation would amplify silence
      into noise and destroy the very silence detection the turn-taking
      model depends on (a chunk near zero divided by its own near-zero RMS
      is undefined/huge). Smoothing over a sequence is also what makes the
      gain applied to any one chunk not swing wildly on an outlier.
    - Any chunk whose OWN rms is below `floor_rms` is passed through
      untouched and does not perturb the running estimate. This check is
      per-chunk and independent of the running estimate on purpose: a
      silent chunk arriving right after loud speech must not be boosted by
      a stale high estimate.
    - The first speech-like chunk after a silent gap snaps the running
      estimate straight to that chunk's own level instead of easing into it
      from whatever the estimate was before the gap (fast attack; ordinary
      EMA smoothing resumes on the chunks after that, i.e. slower release).
      Without this, a short utterance -- a one-word confirmation being the
      exact case that matters -- can end before a slow EMA ramp from a
      neutral start ever reaches a useful gain: measured on real hardware,
      a "Yes." whose *final*-chunk gain looked fine (2.83x) was still
      silently dropped, because the OVERALL audio actually sent upstream
      averaged out at only ~1.36x effective gain.
    - The gain is clamped to [1.0, max_gain]: it only ever boosts (audio
      that is already at or above target is left alone rather than
      attenuated -- levels that already worked upstream must not be
      touched) and never boosts more than `max_gain`, so a near-silent
      chunk that barely clears the floor can't be blown up.
    - After scaling, the result is peak-limited back into [-1, 1] rather
      than allowed to clip, because an isolated loud sample can exceed unit
      scale even when the chunk's RMS looked safely boostable.
    """

    def __init__(
        self,
        target_rms: float,
        *,
        max_gain: float = 10.0,
        floor_rms: float = 0.003,
        smoothing: float = 0.9,
    ):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.floor_rms = floor_rms
        self.smoothing = smoothing
        self.level = target_rms  # neutral start: gain 1.0 until speech is seen
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

        if self._after_silence:
            # First speech-like chunk after a gap: snap straight to its
            # measured level instead of easing in from the stale estimate.
            # A short utterance (e.g. a one-word confirmation) can be over
            # before a slow EMA ramp ever gets there -- fast attack here,
            # ordinary smoothing (slower release) below once speech is
            # already established.
            self.level = input_rms
            self._after_silence = False
        else:
            self.level = self.smoothing * self.level + (1 - self.smoothing) * input_rms

        raw_gain = self.target_rms / max(self.level, self.floor_rms)
        gain = min(max(raw_gain, 1.0), self.max_gain)

        out = chunk * np.float32(gain)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            gain *= 1.0 / peak
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
        agc: bool = True,
        agc_target_rms: float = 0.05,
    ):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.recorder = SessionRecorder(session_dir)
        self._tts = None
        self._tts_factory = tts_factory
        self._t_ms = 0
        self.agc = agc
        self.agc_target_rms = agc_target_rms
        self._agc = AutoGainControl(agc_target_rms) if agc else None
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
