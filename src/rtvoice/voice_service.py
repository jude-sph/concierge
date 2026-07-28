"""Owns all audio: SoulX-Duplug in, Kokoro out, both channels recorded.

This is the relocatable boundary. If the tunnel measures badly (Task 1), this
service moves to the Mac and only text crosses the wire.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from .asr import RollingAudio, transcribe_span
from .recorder import SessionRecorder
from .soulx_client import CHUNK_SAMPLES, SAMPLE_RATE, SoulXClient
from .states import StateAdapter, TurnEvent, UserState, is_backchannel
from .tts import OUTPUT_SAMPLE_RATE as TTS_OUTPUT_SAMPLE_RATE

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
        # The rate synthesized speech is actually produced at. Defaults to
        # Kokoro's native rate (see tts.OUTPUT_SAMPLE_RATE) because that is
        # now what gets played back -- TTS output is no longer downsampled
        # to the 16 kHz the *input* path needs. Recorded here, at
        # construction, rather than read off `self.tts` later: the recorder
        # is created eagerly and `tts` is built lazily on first use (it must
        # not require a GPU just to construct a VoiceService), so the rate
        # the recorder tags model.wav with has to be known up front. Override
        # this if a `tts_factory` produces audio at some other rate.
        tts_sample_rate: int = TTS_OUTPUT_SAMPLE_RATE,
        # Built on first use, for the same reason as `tts`: loading a Whisper
        # model onto a GPU must not be a side effect of constructing a
        # VoiceService. Left as None (by passing a factory that returns None,
        # or by leaving `asr` unset in an environment without the package)
        # every transcript falls back to whatever the turn-taking model said,
        # which is the previous behaviour exactly.
        asr_factory: Callable[[], object] | None = None,
        # Above this per-chunk RMS the person is taken to be talking. Well
        # under the upstream far-field threshold on purpose: the whole point
        # is to hear the quiet speech that gets gated out up there.
        # Absolute floor: below this nothing counts as speech however quiet
        # the room is, so a silent line can never be mistaken for talking.
        speech_floor_rms: float = 0.008,
        # How far above the measured background a chunk must sit to count as
        # speech. 3x is roughly 10dB -- comfortably above the variation of
        # steady room tone, comfortably below the level of someone talking.
        speech_margin: float = 3.0,
    ):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.tts_sample_rate = tts_sample_rate
        self.recorder = SessionRecorder(
            session_dir, user_sample_rate=SAMPLE_RATE, model_sample_rate=tts_sample_rate
        )
        self._tts = None
        self._tts_factory = tts_factory
        self._t_ms = 0
        # --- speech recognition -----------------------------------------------
        #
        # SoulX-Duplug ships a transcript alongside its turn state, and using
        # it is what produced "toju" for "to Jude" and "s." for "Yes." -- a
        # Chinese-first ASR, reading a buffer whose first 160-320ms its own
        # far-field gate had already discarded. See asr.py for the full
        # diagnosis. Its TURN decisions are kept and its words are replaced.
        self._asr = None
        self._asr_factory = asr_factory
        self._asr_ready = asr_factory is None  # nothing to build if unset
        self.audio_log = RollingAudio(seconds=30.0, sample_rate=SAMPLE_RATE)
        # Recent (turn-taking model's text, what was actually said) pairs, for
        # after-the-fact inspection of a session. Bounded; purely diagnostic.
        self.asr_swaps: deque[tuple[str, str]] = deque(maxlen=20)
        self.asr_failures = 0
        # --- our own ears -----------------------------------------------------
        #
        # Position on the audio clock of the last chunk that sounded like
        # someone talking, measured HERE rather than taken from the turn-taking
        # model. That model resets its speech flag every time it finalises a
        # turn, after which a quiet continuation trips its far-field gate and
        # is reported as `idle` -- so its account of "is the person still
        # speaking" is exactly wrong in the case that matters, when a command
        # was split and the second half is on its way. This is the audio it
        # discarded, and it is what holds the merge window open (see
        # Orchestrator._merge_expired).
        # The threshold ADAPTS to the room. A fixed floor was the mistake: at
        # 0.008 RMS, ordinary room tone through a laptop microphone counts as
        # speech, so `quiet_ms` never grew, the merge window thought the
        # person was still talking, and it held on. Measured live, dispatch to
        # the reasoner had a p90 of 5.2s against an intended ceiling of 2.5s,
        # and that overshoot is most of the delay before an answer is spoken.
        #
        # A fixed floor cannot be right for both a quiet room and a noisy one,
        # and the quantity that matters is not loudness but whether this chunk
        # stands out from the background. So the background is measured
        # continuously and speech is what exceeds it by a margin.
        self.speech_floor_rms = speech_floor_rms
        self.speech_margin = speech_margin
        # Starts at the floor and falls fast/rises slow (see feed_audio), so a
        # room is characterised within a second or two of silence and a long
        # utterance cannot drag the estimate up into its own level.
        self.noise_rms = speech_floor_rms
        self.last_speech_ms: int | None = None
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
        # Optional sink for outgoing speech audio, set by whatever is playing
        # it back live (e.g. the /audio websocket). None by default so
        # constructing or using a VoiceService without a listener attached
        # (every existing caller, every existing test) behaves exactly as
        # before -- this is purely additive.
        self.on_audio_chunk: Optional[Callable[[np.ndarray], Awaitable[None]]] = None
        # Optional sink fired whenever speech is stopped (barge-in or reset),
        # set by the same listener that owns on_audio_chunk (the /audio
        # websocket). Halting generation server-side is not enough on its
        # own: audio already sent for this utterance is already sitting in
        # the browser's WebAudio queue and would otherwise keep playing to
        # completion regardless of what the server does next. This hook is
        # the server's half of telling the browser "stop and discard
        # whatever you're playing, right now" -- None by default, so every
        # existing caller/test that never attaches a listener is unaffected.
        self.on_playback_cancel: Optional[Callable[[], Awaitable[None]]] = None
        # Set by stop(), cleared by speak(). Lets an interruption cut short
        # the wait for already-sent audio to finish playing.
        self._stopped = asyncio.Event()

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
        import os

        from . import tts as tts_module

        # TTS_URL selects a remote engine (see scripts/tts_server.py):
        # Chatterbox sounds far more human than Kokoro and cannot run in this
        # process -- it needs numpy<2, which the turn-taking model's
        # environment cannot have. Unset, everything behaves exactly as
        # before, which is also the fallback if that machine is not up.
        remote = os.environ.get("TTS_URL", "").strip()
        if remote:
            return tts_module.RemoteTTS(remote)

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

    # -- speech recognition ---------------------------------------------------

    @property
    def asr(self):
        if not self._asr_ready:
            # Marked ready BEFORE the call, and failure is swallowed: this
            # runs on the audio path, from feed_audio, once per session. A
            # missing CUDA library or an unavailable model would otherwise
            # raise out of feed_audio and take the microphone down entirely
            # -- trading a worse transcript for no conversation at all. It is
            # also retried never rather than once per chunk, which is what
            # setting the flag first buys.
            self._asr_ready = True
            try:
                self._asr = self._asr_factory()
            except Exception:
                self._asr = None
                self.asr_failures += 1
        return self._asr

    @asr.setter
    def asr(self, value) -> None:
        self._asr = value
        self._asr_ready = True

    async def _retranscribe(self, events: list[TurnEvent]) -> list[TurnEvent]:
        """Replace SoulX's words with Whisper's, on our own audio.

        Two kinds of event carry a transcript that this system acts on, and
        they need different treatment of the read mark:

        * COMPLETE / BACKCHANNEL are finalisations. The span they cover is
          over, so the mark advances past it -- unconditionally, even if the
          transcription came back empty, because the mark tracks turn
          boundaries and re-offering a failed span as part of the NEXT
          utterance would splice two turns together.
        * INCOMPLETE is a pause, not an ending. The orchestrator holds it and
          force-dispatches it only if the turn never completes (the silence
          timeout), so it needs good words too -- but the person may simply be
          mid-sentence, so the mark must NOT move or the rest of the utterance
          would be transcribed without its beginning.

        The state is re-derived from the new text rather than carried over:
        "Yes." is a backchannel and SoulX's "s." is not, and it is the
        backchannel classification that lets a one-word confirmation answer a
        pending destructive write.
        """
        if self.asr is None or not events:
            return events

        out: list[TurnEvent] = []
        for ev in events:
            if ev.state in (UserState.COMPLETE, UserState.BACKCHANNEL):
                audio = self.audio_log.take()
            elif ev.state is UserState.INCOMPLETE:
                audio = self.audio_log.peek()
                # A pause this short cannot contain a dispatchable utterance;
                # not worth a GPU call on every gap between words.
                if len(audio) < SAMPLE_RATE * 0.4:
                    out.append(ev)
                    continue
            else:
                out.append(ev)
                continue

            try:
                text = await transcribe_span(self.asr, audio, SAMPLE_RATE)
            except Exception:
                # Never let recognition take down the audio path: a turn with
                # the upstream model's (poor) words still beats no turn.
                self.asr_failures += 1
                out.append(ev)
                continue

            self.asr_swaps.append((ev.transcript, text))
            state = ev.state
            if state in (UserState.COMPLETE, UserState.BACKCHANNEL):
                state = (UserState.BACKCHANNEL if is_backchannel(text)
                         else UserState.COMPLETE)
            out.append(TurnEvent(state, text, ev.t_ms))
        return out

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
        # Kept BEFORE the upstream call, so the audio a turn event refers to is
        # already buffered by the time that event comes back and is asked to
        # be transcribed. This is the copy that still has the utterance's
        # onset in it -- the thing SoulX's far-field gate throws away.
        self.audio_log.append(chunk)
        # Track the room: fall fast toward a new quiet level, rise slowly.
        # Asymmetric on purpose -- a room that goes quiet should be recognised
        # as quiet within a second, but a long utterance must not drag the
        # background estimate up into its own level, which would make the
        # speaker's own voice stop counting as speech partway through.
        rms = self.last_agc_input_rms
        if rms < self.noise_rms:
            self.noise_rms = 0.85 * self.noise_rms + 0.15 * rms
        else:
            self.noise_rms = 0.999 * self.noise_rms + 0.001 * rms
        threshold = max(self.speech_floor_rms, self.noise_rms * self.speech_margin)
        if rms >= threshold:
            # The END of this chunk: the speech in it runs right up to there,
            # so that is the last moment we can say we heard anyone. Stamping
            # the start instead makes `quiet_ms` report a chunk of silence
            # that has not happened yet.
            self.last_speech_ms = self._t_ms + CHUNK_MS
        wire = await self.client.feed(chunk)
        events = self.adapter.feed(wire, self._t_ms)
        self._t_ms += CHUNK_MS
        return await self._retranscribe(events)

    def quiet_ms(self, now_ms: int) -> Optional[int]:
        """How long since we last heard the person, or None if never.

        `None` means no speech-like audio has arrived at all this session, and
        callers must treat that as "no information" rather than "silent
        forever" -- at startup it would otherwise read as an infinitely long
        silence and defeat every timer that consults it.
        """
        if self.last_speech_ms is None:
            return None
        return max(0, now_ms - self.last_speech_ms)

    async def speak(self, text: str, utterance_id: str) -> None:
        """Speak, and do not return until the audio would have finished.

        The waiting at the end is load-bearing, not politeness. Synthesis runs
        far faster than real time -- Kokoro produced 6.95s of speech in 208ms
        on the demo machine -- and frames are pushed to the listener as fast as
        they are made. So the moment the last frame is SENT, seconds of audio
        are still queued in the browser and still playing.

        Everything upstream keys off this call returning:

          * `policy_state.speaking` goes false, and barge-in is gated on it.
            Measured live, the reply was sent by 259s and the user spoke at
            262s -- four seconds into audio they could still hear -- and no
            Stop was emitted, because as far as the server was concerned it
            had finished talking. Barge-in could essentially never fire.
          * `_speak_lock` is released, so the next utterance would start
            streaming on top of one the listener is still hearing.

        Returning on real playback time makes both correct. `stop()` cuts the
        wait short, so an interruption is still immediate.
        """
        self._stopped.clear()
        started = time.monotonic()
        sent = 0
        async for chunk in self.tts.stream(text):
            self.recorder.write_model(chunk)
            sent += len(chunk)
            if self.on_audio_chunk is not None:
                await self.on_audio_chunk(chunk)

        remaining = sent / self.tts_sample_rate - (time.monotonic() - started)
        if remaining > 0:
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass  # played to the end, uninterrupted

    async def stop(self) -> None:
        # Wakes `speak` out of its wait for playback to drain, so an
        # interruption is immediate rather than lasting until the audio would
        # have finished on its own.
        self._stopped.set()
        # Deliberately reads the private slot: stopping speech must never be
        # the thing that constructs a (possibly unavailable) TTS engine.
        if self._tts is not None:
            self._tts.stop()
        # Tell whatever is playing this back live to discard anything already
        # queued -- halting generation here only stops chunks not yet sent;
        # it does nothing about ones already on the wire. Fired unconditionally
        # (even with no TTS constructed yet, even with nothing currently
        # speaking): a cheap, idempotent "there is nothing to hear" signal is
        # harmless, and this is also the single choke point every barge-in and
        # every reset already routes through.
        if self.on_playback_cancel is not None:
            await self.on_playback_cancel()

    def close(self) -> None:
        self.recorder.close()
