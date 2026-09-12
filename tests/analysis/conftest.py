"""Shared fixtures: a synthetic tone on disk, and Essentia-availability skips."""
from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def write_tone_wav(path: Path, seconds: float = 3.0, hz: float = 440.0,
                   sr: int = 44100) -> Path:
    """A 440 Hz sine at 44.1 kHz, so decode() has to resample."""
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.5 * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    return write_tone_wav(tmp_path / "tone.wav")


HAVE_ESSENTIA = importlib.util.find_spec("essentia") is not None
needs_essentia = pytest.mark.skipif(
    not HAVE_ESSENTIA, reason="parity tests need essentia (Mac only)"
)

MODELS = Path(__file__).resolve().parents[2] / "models"
HAVE_EFFNET = (MODELS / "discogs-effnet-bs64-1.pb").exists()
needs_effnet = pytest.mark.skipif(
    not HAVE_EFFNET, reason="run scripts/fetch_models.py first"
)
