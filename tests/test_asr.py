"""Whisper says what was said; SoulX-Duplug only says when.

The transcript used to come from SoulX's own `cascade_asr`, which its config
sets to SenseVoice -- a Chinese-first model -- reading a buffer its far-field
gate had already trimmed the front off. Live, that produced "toju" for "to
Jude" and "s." for "Yes.".

These tests cover the part of the fix that does not need a GPU: which audio
gets transcribed, and what happens to the turn event afterwards. The model
itself is faked throughout.
"""
import numpy as np
import pytest

from rtvoice.asr import RollingAudio, transcribe_span, trim_to_speech
from rtvoice.states import UserState

SR = 16000


def _speech(seconds: float, level: float = 0.3) -> np.ndarray:
    """Something loud and varying enough to read as speech."""
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float, level: float = 0.0) -> np.ndarray:
    return np.full(int(SR * seconds), level, dtype=np.float32)


# --- trim_to_speech --------------------------------------------------------

def test_leading_silence_is_dropped():
    audio = np.concatenate([_silence(2.0), _speech(1.0)])
    assert len(trim_to_speech(audio, SR)) < len(audio)


def test_a_pre_roll_survives_so_the_onset_is_never_clipped():
    """The whole point of the exercise.

    The consonant that opens a word sits below any energy threshold you could
    pick -- which is exactly how "Please change Sarah" became "Change Sarah"
    upstream. So the threshold must not be what decides where the utterance
    starts: audio before the first loud frame is kept.
    """
    audio = np.concatenate([_silence(2.0), _speech(1.0)])
    trimmed = trim_to_speech(audio, SR, pre_roll_ms=300)

    # 1.0s of speech, plus roughly the pre-roll, and definitely more than the
    # speech alone.
    assert len(trimmed) > SR * 1.0
    assert len(trimmed) < SR * 1.6


def test_trailing_quiet_is_kept():
    """The end of the span is the turn boundary the model just reported, and
    how an utterance trails off is part of it."""
    audio = np.concatenate([_speech(1.0), _silence(1.0)])
    assert len(trim_to_speech(audio, SR)) == len(audio)


def test_a_span_of_pure_silence_finds_nothing():
    """Which the caller reads as "nothing was said" -- the correct answer.

    Handed near-silence, Whisper confidently emits training-set filler
    ("Thank you.", a subtitle credit), and this system acts on transcripts.
    """
    assert trim_to_speech(_silence(3.0), SR).size == 0


def test_quiet_room_tone_is_not_speech():
    room = (np.random.default_rng(0).normal(0, 0.0005, SR * 2)).astype(np.float32)
    assert trim_to_speech(room, SR).size == 0


def test_quiet_speech_is_still_found():
    """Quiet speech is the case the upstream far-field gate drops outright.

    The threshold is relative to the span's own loudest frame, so it does not
    need to know the microphone's level.
    """
    audio = np.concatenate([_silence(1.0), _speech(1.0, level=0.02)])
    assert trim_to_speech(audio, SR).size > 0


def test_audio_shorter_than_a_frame_is_not_speech():
    assert trim_to_speech(np.ones(10, dtype=np.float32), SR).size == 0


# --- RollingAudio ----------------------------------------------------------

def test_take_returns_everything_since_the_mark():
    log = RollingAudio(seconds=30.0, sample_rate=SR)
    log.append(_speech(1.0))
    assert len(log.take()) == SR

    log.append(_speech(0.5))
    assert len(log.take()) == SR // 2


def test_the_mark_advances_even_when_nothing_was_transcribed():
    """It tracks TURN BOUNDARIES, not successful transcriptions.

    A span that failed to transcribe must not be re-offered as part of the
    next utterance, or one turn's audio gets spliced onto the next one's.
    """
    log = RollingAudio(seconds=30.0, sample_rate=SR)
    log.append(_silence(1.0))
    log.take()

    log.append(_speech(0.5))
    assert len(log.take()) == SR // 2


def test_peek_leaves_the_mark_alone():
    """A pause may not be an ending: the person can still be mid-sentence, and
    consuming the span would leave the rest of it with no beginning."""
    log = RollingAudio(seconds=30.0, sample_rate=SR)
    log.append(_speech(1.0))

    assert len(log.peek()) == SR
    log.append(_speech(1.0))
    assert len(log.take()) == 2 * SR


def test_old_audio_ages_out_but_recent_audio_survives():
    log = RollingAudio(seconds=2.0, sample_rate=SR)
    for _ in range(10):
        log.append(_speech(1.0))

    span = log.take()
    assert 0 < len(span) <= SR * 3


def test_a_span_longer_than_the_cap_keeps_its_tail():
    """If someone really talked for that long, the tail is what they most
    recently said, so the tail is what survives."""
    log = RollingAudio(seconds=60.0, sample_rate=SR)
    log.append(_speech(30.0))
    assert len(log.take(max_span_s=5.0)) == 5 * SR


def test_taking_twice_in_a_row_yields_nothing_the_second_time():
    log = RollingAudio(seconds=30.0, sample_rate=SR)
    log.append(_speech(1.0))
    log.take()
    assert log.take().size == 0


# --- transcribe_span -------------------------------------------------------

class FakeASR:
    def __init__(self, text: str = "hello there"):
        self.text = text
        self.spans: list[int] = []

    async def transcribe(self, audio, sample_rate=SR):
        self.spans.append(len(audio))
        return self.text


@pytest.mark.asyncio
async def test_no_asr_means_no_transcript():
    """Which the caller reads as "keep the turn-taking model's words" --
    exactly the old behaviour, for an environment with no GPU budget."""
    assert await transcribe_span(None, _speech(1.0), SR) == ""


@pytest.mark.asyncio
async def test_silence_never_reaches_the_model():
    asr = FakeASR()
    assert await transcribe_span(asr, _silence(2.0), SR) == ""
    assert asr.spans == []


@pytest.mark.asyncio
async def test_speech_is_trimmed_before_it_reaches_the_model():
    asr = FakeASR()
    audio = np.concatenate([_silence(5.0), _speech(1.0)])

    assert await transcribe_span(asr, audio, SR) == "hello there"
    assert asr.spans[0] < len(audio)
