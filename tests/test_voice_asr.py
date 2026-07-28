"""VoiceService replaces the turn-taking model's words with its own.

SoulX-Duplug's turn DECISIONS are kept -- they are why it was chosen, and they
arrive at 240ms. Its transcript is not: it comes from a Chinese-first ASR
reading a buffer whose onset its far-field gate already discarded. See asr.py.
"""
import numpy as np
import pytest

from rtvoice.states import UserState
from rtvoice.voice_service import VoiceService

SR = 16000
CHUNK = 2560


def _speech(n: int = CHUNK, level: float = 0.3) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(n: int = CHUNK) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


class FakeSoulX:
    """Replays a scripted sequence of wire states, one per chunk."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    async def feed(self, chunk):
        state = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return state


class FakeASR:
    def __init__(self, text="please change Sarah Chen to Jude"):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio, sample_rate=SR):
        self.calls += 1
        return self.text


def _voice(tmp_path, script, asr=None):
    voice = VoiceService(tmp_path / "session")
    voice.client = FakeSoulX(script)
    voice.asr = asr
    return voice


@pytest.mark.asyncio
async def test_the_transcript_is_replaced_on_a_completed_turn(tmp_path):
    """SoulX said "Change Sarah."; the person said "please change Sarah Chen
    to Jude". The audio we kept still has the beginning in it."""
    asr = FakeASR()
    voice = _voice(tmp_path, [{"state": "nonidle", "asr_buffer": "Change"},
                              {"state": "speak", "text": "Change Sarah."}], asr)

    await voice.feed_audio(_speech())
    events = await voice.feed_audio(_speech())

    assert [e.transcript for e in events] == ["please change Sarah Chen to Jude"]
    assert events[0].state is UserState.COMPLETE


@pytest.mark.asyncio
async def test_a_confirmation_misheard_as_a_word_becomes_a_backchannel(tmp_path):
    """"Yes." came through as "s." -- and the state is re-derived from the new
    text, not carried over, because it is the BACKCHANNEL classification that
    lets a one-word confirmation answer a pending destructive write."""
    voice = _voice(tmp_path, [{"state": "speak", "text": "s."}], FakeASR("Yes."))

    events = await voice.feed_audio(_speech())

    assert events[0].transcript == "Yes."
    assert events[0].state is UserState.BACKCHANNEL


@pytest.mark.asyncio
async def test_a_real_command_stays_a_completed_turn(tmp_path):
    voice = _voice(tmp_path, [{"state": "speak", "text": "ok"}],
                   FakeASR("delete all my messages"))

    events = await voice.feed_audio(_speech())

    assert events[0].state is UserState.COMPLETE


@pytest.mark.asyncio
async def test_without_an_asr_the_upstream_transcript_is_used_unchanged(tmp_path):
    """The previous behaviour exactly, for an environment with no GPU budget."""
    voice = _voice(tmp_path, [{"state": "speak", "text": "Change Sarah."}], None)

    events = await voice.feed_audio(_speech())

    assert events[0].transcript == "Change Sarah."


@pytest.mark.asyncio
async def test_a_failing_asr_falls_back_rather_than_losing_the_turn(tmp_path):
    """A turn with the upstream model's poor words beats no turn at all."""

    class Broken:
        async def transcribe(self, audio, sample_rate=SR):
            raise RuntimeError("cuda is on fire")

    voice = _voice(tmp_path, [{"state": "speak", "text": "Change Sarah."}], Broken())

    events = await voice.feed_audio(_speech())

    assert events[0].transcript == "Change Sarah."
    assert voice.asr_failures == 1


@pytest.mark.asyncio
async def test_idle_and_nonidle_are_not_transcribed(tmp_path):
    """They carry no transcript this system acts on; a GPU call per chunk of
    silence would be pure cost."""
    asr = FakeASR()
    voice = _voice(tmp_path, [{"state": "idle"}, {"state": "nonidle",
                                                  "asr_buffer": "hm"}], asr)

    await voice.feed_audio(_silence())
    await voice.feed_audio(_speech())

    assert asr.calls == 0


@pytest.mark.asyncio
async def test_a_pause_is_transcribed_without_consuming_the_utterance(tmp_path):
    """INCOMPLETE is a pause, not an ending.

    The orchestrator holds it and force-dispatches it only if the turn never
    completes -- so it needs good words. But the person may simply be
    mid-sentence, so the read mark must not move, or the rest of the utterance
    would be transcribed without its beginning.
    """
    asr = FakeASR()
    voice = _voice(tmp_path, [
        {"state": "nonidle", "asr_buffer": "please change"},
        {"state": "idle"},          # -> INCOMPLETE, the model declined the turn
        {"state": "nonidle", "asr_buffer": "sarah"},
        {"state": "speak", "text": "Sarah."},
    ], asr)

    for _ in range(3):
        await voice.feed_audio(_speech())
    before_complete = voice.audio_log.mark
    await voice.feed_audio(_speech())

    assert before_complete == 0, "a pause must not consume the utterance"
    assert voice.audio_log.mark == 4 * CHUNK, "a completed turn must"


@pytest.mark.asyncio
async def test_the_audio_kept_is_the_audio_that_was_heard(tmp_path):
    voice = _voice(tmp_path, [{"state": "idle"}], None)

    await voice.feed_audio(_speech())
    await voice.feed_audio(_speech())

    assert voice.audio_log.total == 2 * CHUNK


# --- our own ears ----------------------------------------------------------

@pytest.mark.asyncio
async def test_speech_is_noticed_independently_of_the_turn_model(tmp_path):
    """The turn model clears its speech flag every time it finalises a turn,
    after which a quiet continuation trips its far-field gate and is reported
    as `idle`. So "are they still talking" is measured here instead."""
    voice = _voice(tmp_path, [{"state": "idle"}], None)

    await voice.feed_audio(_speech())
    assert voice.quiet_ms(voice.stream_ms) == 0

    for _ in range(5):
        await voice.feed_audio(_silence())
    assert voice.quiet_ms(voice.stream_ms) == 5 * 160


@pytest.mark.asyncio
async def test_before_any_audio_there_is_no_answer(tmp_path):
    """None means "no information", not "silent forever" -- at startup the
    latter would read as an infinitely long silence and defeat every timer
    that consults it."""
    voice = _voice(tmp_path, [{"state": "idle"}], None)
    assert voice.quiet_ms(0) is None
