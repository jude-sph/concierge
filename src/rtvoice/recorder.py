"""Dual-channel session recorder.

Channels are kept separate as well as mixed so user and model audio can be
analysed independently - which is how acoustic echo gets diagnosed rather
than guessed at. Recording is always on; sessions are curated afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class SessionRecorder:
    def __init__(self, session_dir: str | Path) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._user: list[np.ndarray] = []
        self._model: list[np.ndarray] = []

    def write_user(self, chunk: np.ndarray) -> None:
        self._user.append(np.asarray(chunk, dtype=np.float32).copy())

    def write_model(self, chunk: np.ndarray) -> None:
        self._model.append(np.asarray(chunk, dtype=np.float32).copy())

    def close(self) -> None:
        user = np.concatenate(self._user) if self._user else np.zeros(0, dtype=np.float32)
        model = np.concatenate(self._model) if self._model else np.zeros(0, dtype=np.float32)

        sf.write(self.dir / "user.wav", user, SAMPLE_RATE)
        sf.write(self.dir / "model.wav", model, SAMPLE_RATE)

        n = max(len(user), len(model))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[: len(user), 0] = user
        mix[: len(model), 1] = model
        sf.write(self.dir / "mix.wav", mix, SAMPLE_RATE)
