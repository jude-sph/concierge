"""IMPORTANT 2 regressions: there was no audio path at all.

`VoiceService.tts` was never assigned by anything, so the first spoken reply
raised -- and the exception was absorbed by the `return_exceptions=True`
gather and logged as `turn_task_error`, failing invisibly. Nothing called
`feed_audio()`. Nothing called `on_tick()` outside tests, so the silence
timeout was dead code in production. And importing the orchestrator module
constructed a whole app at import time, with filesystem side effects.

None of this can be verified end to end here (no GPU, `kokoro` is not
installed, the remote turn-taking server is unreachable), so these tests
exercise the seams with fakes.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rtvoice.audio_driver import AudioDriver
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.soulx_client import CHUNK_SAMPLES
from rtvoice.states import TurnEvent, UserState
from rtvoice.voice_service import VoiceService

from fakes import FakeConcierge

REPO = Path(__file__).resolve().parents[1]


# --- import-time side effects -----------------------------------------------

def test_importing_the_orchestrator_module_has_no_side_effects(tmp_path):
    """`app = _default_app()` at module scope meant that merely importing the
    orchestrator built a DeviceState, opened an event log and created session
    directories -- in whatever the current working directory happened to be."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import rtvoice.orchestrator as m; print(hasattr(m, 'app'))"],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
    assert not (tmp_path / "sessions").exists()
    assert list(tmp_path.iterdir()) == []


def test_create_default_app_is_a_callable_factory(tmp_path, monkeypatch):
    from rtvoice.orchestrator import create_default_app

    (tmp_path / "fixtures").mkdir()
    shutil.copy(REPO / "fixtures" / "device_state.json",
                tmp_path / "fixtures" / "device_state.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SESSION_ID", "unit-test")

    app = create_default_app()
    assert app.state.orch is not None
    app.state.orch.voice.close()


# --- lazily constructed TTS --------------------------------------------------

class FakeTTS:
    constructed = 0

    def __init__(self):
        type(self).constructed += 1
        self.stopped = 0

    def stop(self):
        self.stopped += 1

    async def stream(self, text):
        yield np.zeros(160, dtype=np.float32)


def test_tts_is_not_constructed_until_it_is_actually_needed(tmp_path):
    made = []

    def factory():
        made.append(1)
        return FakeTTS()

    voice = VoiceService(tmp_path, tts_factory=factory)
    try:
        assert made == []          # constructing the service must not need a GPU
        assert voice.tts is not None
        assert made == [1]
        assert voice.tts is voice.tts
        assert made == [1]         # constructed once, cached
    finally:
        voice.close()


@pytest.mark.asyncio
async def test_stop_never_constructs_a_tts(tmp_path):
    made = []
    voice = VoiceService(tmp_path, tts_factory=lambda: made.append(1) or FakeTTS())
    try:
        await voice.stop()
        assert made == []
    finally:
        voice.close()


@pytest.mark.asyncio
async def test_stop_fires_the_playback_cancel_hook_when_one_is_attached(tmp_path):
    """The barge-in fix: halting TTS generation (stopping the underlying
    engine) only stops chunks not yet produced -- it does nothing about
    audio already sent and sitting in a browser's playback queue.
    VoiceService.stop() must also fire on_playback_cancel, the hook whatever
    is relaying audio live (the /audio websocket) attaches, so that side can
    tell the browser to actually stop making sound."""
    voice = VoiceService(tmp_path, tts_factory=FakeTTS)
    calls = []

    async def hook():
        calls.append(1)

    voice.on_playback_cancel = hook
    try:
        await voice.stop()
        assert calls == [1]
    finally:
        voice.close()


@pytest.mark.asyncio
async def test_stop_without_a_playback_cancel_hook_still_works(tmp_path):
    """No listener attached (every existing caller before this fix, and every
    other test in this file) must not raise -- the hook is optional, same
    pattern as on_audio_chunk."""
    voice = VoiceService(tmp_path, tts_factory=FakeTTS)
    try:
        await voice.stop()  # must not raise
    finally:
        voice.close()


@pytest.mark.asyncio
async def test_speak_uses_the_lazily_built_tts_and_records_it(tmp_path):
    voice = VoiceService(tmp_path, tts_factory=FakeTTS)
    try:
        await voice.speak("hello", "u1")
        await voice.stop()
        assert voice.tts.stopped == 1
    finally:
        voice.close()


def test_model_wav_is_recorded_at_the_rate_tts_actually_produces(tmp_path):
    """Pins that the recorder's declared rate for model.wav matches the rate
    synthesized speech is actually produced at (Kokoro's native 24 kHz, now
    that TTS output is played back unresampled -- see tts.OUTPUT_SAMPLE_RATE)
    rather than a hardcoded assumption left over from when it was downsampled
    to 16 kHz. A mismatch here means every recorded reply replays pitch-
    shifted."""
    import soundfile as sf

    from rtvoice.tts import OUTPUT_SAMPLE_RATE

    voice = VoiceService(tmp_path, tts_factory=FakeTTS)
    try:
        assert voice.recorder.model_sample_rate == OUTPUT_SAMPLE_RATE
    finally:
        voice.close()

    _, model_sr = sf.read(tmp_path / "model.wav", dtype="float32")
    assert model_sr == OUTPUT_SAMPLE_RATE


@pytest.mark.asyncio
async def test_model_wav_rate_follows_an_explicit_tts_sample_rate_override(tmp_path):
    """A caller supplying a `tts_factory` that produces audio at some other
    rate must be able to say so, and have the recorder honor it -- the
    recorder is constructed eagerly (before `tts` is ever built), so this
    has to be settable at construction time rather than inferred later."""
    import soundfile as sf

    voice = VoiceService(tmp_path, tts_factory=FakeTTS, tts_sample_rate=8000)
    try:
        assert voice.recorder.model_sample_rate == 8000
        await voice.speak("hello", "u1")
    finally:
        voice.close()

    _, model_sr = sf.read(tmp_path / "model.wav", dtype="float32")
    assert model_sr == 8000


def test_missing_kokoro_raises_a_clear_error_rather_than_an_obscure_one(tmp_path,
                                                                       monkeypatch):
    """`kokoro` is a GPU dependency that is deliberately not installed here and
    must never be imported at module scope. Asking for TTS without it must say
    so plainly instead of failing deep inside an audio call."""
    import rtvoice.tts as tts_mod

    class Unavailable:
        def __init__(self, *a, **kw):
            raise ImportError("No module named 'kokoro'")

    monkeypatch.setattr(tts_mod, "KokoroTTS", Unavailable)
    voice = VoiceService(tmp_path)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            voice.tts
        assert "kokoro" in str(excinfo.value).lower()
    finally:
        voice.close()


# --- the driver --------------------------------------------------------------

class ScriptedVoice:
    """Stands in for VoiceService: hands back canned turn events per chunk and
    advances an audio clock exactly as the real service does (event timestamps
    are taken before the clock advances)."""

    CHUNK_MS = 160

    def __init__(self, script=()):
        self.script = [list(s) for s in script]
        self.fed = []
        self.spoken = []
        self.stops = 0
        self._t_ms = 0

    @property
    def stream_ms(self) -> int:
        return self._t_ms

    async def feed_audio(self, chunk):
        self.fed.append(chunk)
        pairs = self.script.pop(0) if self.script else []
        events = [TurnEvent(state, text, self._t_ms) for state, text in pairs]
        self._t_ms += self.CHUNK_MS
        return events

    async def speak(self, text, utterance_id):
        self.spoken.append(text)

    async def stop(self):
        self.stops += 1


class RecordingOrch:
    def __init__(self):
        self.events = []
        self.ticks = []

    async def on_turn_event(self, ev):
        self.events.append(ev)

    async def on_tick(self, now_ms):
        self.ticks.append(now_ms)


@pytest.mark.asyncio
async def test_driver_feeds_audio_and_forwards_turn_events(tmp_path):
    voice = ScriptedVoice([[(UserState.COMPLETE, "find pizza")], []])
    orch = RecordingOrch()
    driver = AudioDriver(voice, orch)

    await driver.run([np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 2)

    assert len(voice.fed) == 2
    assert [e.transcript for e in orch.events] == ["find pizza"]


@pytest.mark.asyncio
async def test_driver_ticks_on_the_audio_clock_not_the_wall_clock(tmp_path):
    """on_tick's now_ms is compared against turn-event timestamps, which are
    positions in the audio stream. Ticking on wall-clock time would make the
    silence timeout meaningless whenever audio is not played back in real
    time (a WAV file streamed as fast as the network allows, for instance)."""
    voice = ScriptedVoice()
    orch = RecordingOrch()
    driver = AudioDriver(voice, orch, tick_interval_ms=160)

    await driver.run([np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 4)

    assert orch.ticks == [160, 320, 480, 640]


@pytest.mark.asyncio
async def test_driver_makes_the_silence_timeout_actually_fire(tmp_path):
    """The whole point of on_tick: with nothing driving it, a turn the
    turn-taking model never finalises hangs forever."""
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": [{"id": 1, "first_name": "Sarah"}]}))
    device = DeviceState(state, tmp_path / "j.jsonl")
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=ScriptedVoice(),
        log=EventLog(tmp_path / "e.jsonl"), device=device,
        silence_timeout_ms=2000,
    )
    voice = ScriptedVoice([[(UserState.INCOMPLETE, "rename my contacts to Hans")]])
    driver = AudioDriver(voice, orch)

    await driver.run([np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 20)

    assert len(orch.registry.all()) == 1
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "silence_timeout" in kinds


@pytest.mark.asyncio
async def test_driver_drain_feeds_trailing_silence(tmp_path):
    voice = ScriptedVoice()
    orch = RecordingOrch()
    driver = AudioDriver(voice, orch)

    await driver.drain(800)

    assert len(voice.fed) == 5
    assert all(c.shape == (CHUNK_SAMPLES,) and not c.any() for c in voice.fed)


# --- the CLI -----------------------------------------------------------------

def test_wav_is_chunked_with_the_final_chunk_padded():
    from tools.audio_client import chunk_audio

    audio = np.ones(CHUNK_SAMPLES + 5, dtype=np.float32)
    chunks = list(chunk_audio(audio))
    assert len(chunks) == 2
    assert all(c.shape == (CHUNK_SAMPLES,) for c in chunks)
    assert chunks[1][:5].tolist() == [1.0] * 5
    assert not chunks[1][5:].any()


def test_wav_reader_rejects_the_wrong_sample_rate(tmp_path):
    import soundfile as sf

    from tools.audio_client import read_wav

    path = tmp_path / "wrong.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 8000)
    with pytest.raises(ValueError) as excinfo:
        read_wav(path)
    assert "16000" in str(excinfo.value)


def test_wav_reader_downmixes_stereo_to_mono(tmp_path):
    import soundfile as sf

    from tools.audio_client import read_wav

    path = tmp_path / "stereo.wav"
    stereo = np.stack([np.ones(1600), np.zeros(1600)], axis=1).astype(np.float32)
    sf.write(path, stereo, 16000)
    audio = read_wav(path)
    assert audio.ndim == 1
    assert audio.shape == (1600,)
    assert audio[0] == pytest.approx(0.5, abs=1e-3)   # PCM_16 quantisation
