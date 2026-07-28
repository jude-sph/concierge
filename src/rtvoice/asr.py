"""Words come from Whisper. Turn-taking comes from SoulX-Duplug.

WHY THIS EXISTS
---------------
SoulX-Duplug's wire protocol carries a transcript, and we used it. That was a
mistake, for two reasons visible in its own source (`service/model.py`):

1. The transcript is produced by `cascade_asr`, which `config.yaml` sets to
   `sensevoice` -- SenseVoice-Small, a Chinese-first FunASR model. On English
   it renders "to Jude" as "toju". Its job in that project is to caption a
   Mandarin conversation; transcription accuracy in English was never a design
   goal, and no amount of configuration changes what the weights know.

2. The audio the transcript is made from has its head cut off. Before any
   state is emitted, `state_predict` applies a far-field gate:

       if rms(process_chunk) < far_field_threshold and not speech_detected
               and state == "<|user_nonidle|>":
           self.reset()          # <-- this wipes buffer_for_asr

   A quiet onset chunk therefore discards the utterance buffer and starts
   over. The onset of an utterance is *reliably* quiet -- that is what an
   onset is. Measured against real speech, "please change Sarah Chen to Jude"
   came back as "Change Sarah.", and "Yes." came back as "s.": the first
   160-320ms, gated out before the ASR ever saw it.

Neither is a tuning problem, and both are upstream of us. But the second one
is only a problem because SoulX is being asked for something it never needed
to provide: we are already sending it every audio chunk, so we already hold
the whole utterance, onset included. What SoulX is genuinely good at -- and
what it was chosen for -- is deciding *when* a turn ends, semantically, at
240ms latency. That signal is kept. Only the words are taken elsewhere.

So the split is: SoulX says WHEN, Whisper says WHAT.

The helpers below (`trim_to_speech`, `RollingAudio`) are deliberately pure and
GPU-free, so the framing logic that decides which audio gets transcribed is
testable without a model, a GPU, or a turn-taking server.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import numpy as np

# Whisper's own training data is 16 kHz, which is also what the microphone
# path already produces for SoulX. No resampling is needed anywhere here.
SAMPLE_RATE = 16000

# A span is cut to this before transcription. Long enough for any plausible
# spoken command; short enough that a runaway buffer cannot turn one turn into
# a multi-second GPU call. If someone really did talk for longer than this, the
# tail is what they most recently said, so the tail is what we keep.
MAX_SPAN_S = 20.0

# How much audio before the first speech-like frame survives trimming. This is
# the whole point of the exercise: the fricative or stop that opens a word
# ("P-lease", "Y-es") sits below any energy threshold you could pick, so the
# threshold must never be what decides where the utterance starts.
PRE_ROLL_MS = 300


def _frame_rms(audio: np.ndarray, frame: int) -> np.ndarray:
    """RMS per fixed-size frame; trailing partial frame dropped."""
    usable = (len(audio) // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[:usable].reshape(-1, frame).astype(np.float64)
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)


def trim_to_speech(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    pre_roll_ms: int = PRE_ROLL_MS,
    frame_ms: int = 30,
    absolute_floor: float = 0.004,
    relative_floor: float = 0.15,
) -> np.ndarray:
    """Drop the silence in front of an utterance, keeping a pre-roll.

    A span runs from the end of the previous turn, so it usually opens with
    however long the person spent not talking. Handing that to Whisper is slow
    (it is charged for the silence) and unsafe: on near-silent input Whisper
    emits its training-set filler -- "Thank you.", "you", a subtitle credit --
    with high confidence, and downstream this system acts on transcripts.

    The threshold is relative to the span's own loudest frame rather than
    absolute, so it does not need to know the microphone's level, with an
    absolute floor underneath so that a span containing *only* room tone finds
    no speech at all and returns empty -- which the caller reads as "nothing
    was said", the correct answer, rather than transcribing the room.

    Only the FRONT is trimmed. The end of the span is the turn boundary SoulX
    just reported, and trailing quiet there is a real part of how the
    utterance ended.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    if len(audio) < frame:
        return audio[:0]

    rms = _frame_rms(audio, frame)
    if rms.size == 0:
        return audio[:0]

    threshold = max(absolute_floor, relative_floor * float(rms.max()))
    speech = np.flatnonzero(rms >= threshold)
    if speech.size == 0:
        return audio[:0]

    pre_roll = int(sample_rate * pre_roll_ms / 1000)
    start = max(0, int(speech[0]) * frame - pre_roll)
    return audio[start:]


class RollingAudio:
    """The last `seconds` of microphone audio, with a movable read mark.

    The mark is where the previous utterance was finalised. Everything after
    it is what has been said since -- which is precisely the span to
    transcribe when SoulX reports a turn boundary, and which (unlike SoulX's
    own buffer) still contains the onset.

    Positions are ABSOLUTE sample counts since the session began, not indices
    into the deque, so the mark stays meaningful as old audio ages out. If the
    mark falls off the back of the buffer -- someone spoke for longer than the
    buffer holds -- `take()` returns what is still there rather than failing.
    """

    def __init__(self, seconds: float = 30.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.capacity = int(seconds * sample_rate)
        self._chunks: deque[np.ndarray] = deque()
        self._held = 0        # samples currently in _chunks
        self.total = 0        # samples ever appended
        self.mark = 0         # absolute position of the last finalisation

    def append(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        self._chunks.append(chunk)
        self._held += len(chunk)
        self.total += len(chunk)
        while self._held - len(self._chunks[0]) >= self.capacity:
            self._held -= len(self._chunks.popleft())

    @property
    def _oldest(self) -> int:
        """Absolute position of the first sample still buffered."""
        return self.total - self._held

    def take(self, max_span_s: float = MAX_SPAN_S) -> np.ndarray:
        """Audio from the mark to now, and move the mark to now.

        The mark advances whether or not the span turned out to contain
        speech: it tracks turn boundaries, not successful transcriptions, and
        a span that failed to transcribe must not be re-offered as part of the
        next utterance.
        """
        start = max(self.mark, self._oldest, self.total - int(max_span_s * self.sample_rate))
        self.mark = self.total
        if start >= self.total or not self._chunks:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate(list(self._chunks))
        return joined[start - self._oldest:]

    def peek(self, max_span_s: float = MAX_SPAN_S) -> np.ndarray:
        """Audio from the mark to now, WITHOUT moving the mark.

        For a pause that may not be an ending: the words so far are worth
        transcribing (the silence timeout may have to dispatch them), but the
        person may simply still be mid-sentence, and consuming the span here
        would leave the rest of the utterance with no beginning.
        """
        saved = self.mark
        audio = self.take(max_span_s)
        self.mark = saved
        return audio

    def reset(self) -> None:
        self._chunks.clear()
        self._held = 0
        self.mark = self.total


class WhisperASR:
    """faster-whisper, called off the event loop.

    Constructed lazily by whoever owns it: this imports and loads a model onto
    a GPU, which must not happen merely because something imported this module.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel  # heavy; GPU deps

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        self.beam_size = beam_size

    def transcribe_sync(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            # Each utterance is transcribed on its own. Carrying the previous
            # one in as context is what makes Whisper fall into repetition
            # loops, and here it would also let a misheard earlier turn bias
            # the words this system acts on.
            condition_on_previous_text=False,
            vad_filter=True,
            # Whisper's fallback when it hears nothing is a confident piece of
            # training-set filler. Raising this makes it prefer to say nothing.
            no_speech_threshold=0.6,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe without blocking the audio websocket.

        The synthesis and microphone paths share this event loop, so a
        multi-hundred-millisecond synchronous GPU call here would stall
        playback and stop new chunks being read for its whole duration.
        """
        return await asyncio.to_thread(self.transcribe_sync, audio, sample_rate)


async def transcribe_span(
    asr: Optional[object],
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Trim, then transcribe, or return "" if there is nothing to transcribe.

    Kept as a free function so the trim-then-transcribe policy is one thing,
    shared by the live path and by offline replay of a recorded session.
    """
    if asr is None:
        return ""
    speech = trim_to_speech(audio, sample_rate)
    if speech.size == 0:
        return ""
    return await asr.transcribe(speech, sample_rate)
