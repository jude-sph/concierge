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
