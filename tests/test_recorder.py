import numpy as np
import soundfile as sf
from rtvoice.recorder import SessionRecorder


def test_writes_three_wavs(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.zeros(1600, dtype=np.float32))
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.1)
    rec.close()
    for name in ("user.wav", "model.wav", "mix.wav"):
        assert (tmp_path / name).exists(), name


def test_channels_are_kept_separate(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)
    rec.write_model(np.zeros(1600, dtype=np.float32))
    rec.close()
    user, sr = sf.read(tmp_path / "user.wav", dtype="float32")
    assert sr == 16000
    assert np.allclose(user, 0.5, atol=1e-3)


def test_mix_is_stereo_user_left_model_right(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.25)
    rec.close()
    mix, sr = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert mix.ndim == 2 and mix.shape[1] == 2
    assert np.allclose(mix[:, 0], 0.5, atol=1e-3)
    assert np.allclose(mix[:, 1], 0.25, atol=1e-3)


def test_unequal_lengths_are_zero_padded(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(3200, dtype=np.float32) * 0.5)
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.25)
    rec.close()
    mix, _ = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert mix.shape[0] == 3200
    assert np.allclose(mix[1600:, 1], 0.0, atol=1e-6)


# --- user/model channels at different rates -----------------------------------
#
# TTS output is no longer downsampled to 16 kHz for playback (see tts.py's
# OUTPUT_SAMPLE_RATE), so model.wav's true rate can now differ from
# user.wav's. Each file must be tagged with the rate it actually contains --
# otherwise a replay of model.wav plays back pitch-shifted -- and the mix
# must not silently combine mismatched rates as if they were the same.


def test_model_file_is_tagged_with_the_rate_it_was_given(tmp_path):
    """Pins that the recorder writes model.wav at the rate it is actually
    told the model audio is produced at, not a hardcoded assumption."""
    rec = SessionRecorder(tmp_path, user_sample_rate=16000, model_sample_rate=24000)
    rec.write_user(np.zeros(1600, dtype=np.float32))
    rec.write_model(np.ones(2400, dtype=np.float32) * 0.1)
    rec.close()

    user, user_sr = sf.read(tmp_path / "user.wav", dtype="float32")
    model, model_sr = sf.read(tmp_path / "model.wav", dtype="float32")
    assert user_sr == 16000
    assert model_sr == 24000
    assert len(model) == 2400


def test_mix_uses_a_common_rate_when_channels_differ(tmp_path):
    """The mix can't carry two rates in one WAV file, so when the channels
    differ, it must be built at a single explicit rate (the higher of the
    two, so no bandwidth is thrown away) rather than naively zero-padding
    mismatched-rate arrays together as if they lined up sample-for-sample."""
    rec = SessionRecorder(tmp_path, user_sample_rate=16000, model_sample_rate=24000)
    rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)  # 0.1s @ 16kHz
    rec.write_model(np.ones(2400, dtype=np.float32) * 0.25)  # 0.1s @ 24kHz
    rec.close()

    mix, mix_sr = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert mix_sr == 24000
    assert mix.ndim == 2 and mix.shape[1] == 2
    # Both channels represent the same 0.1s duration at the mix rate.
    assert mix.shape[0] == 2400


def test_context_manager_support(tmp_path):
    """Test that SessionRecorder works as a context manager."""
    with SessionRecorder(tmp_path) as rec:
        rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)
        rec.write_model(np.ones(1600, dtype=np.float32) * 0.25)
    # Files should exist after exiting context
    assert (tmp_path / "user.wav").exists()
    assert (tmp_path / "model.wav").exists()
    assert (tmp_path / "mix.wav").exists()
    # Verify contents
    user, sr = sf.read(tmp_path / "user.wav", dtype="float32")
    assert sr == 16000
    assert np.allclose(user, 0.5, atol=1e-3)
