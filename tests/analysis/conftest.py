"""Shared fixtures: synthetic audio on disk, and the v2-availability skips."""
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


def write_stereo_tone_wav(path: Path, seconds: float = 3.0, hz: float = 440.0,
                          sr: int = 44100) -> Path:
    """The same 440 Hz sine on both channels — identical L/R, so a correct
    downmix must reproduce the mono version's peak exactly."""
    t = np.arange(int(seconds * sr)) / sr
    ch = (0.5 * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
    pcm = np.repeat(ch[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def stereo_tone_wav(tmp_path: Path) -> Path:
    return write_stereo_tone_wav(tmp_path / "stereo_tone.wav")


MODELS = Path(__file__).resolve().parents[2] / "models"

# The analysis stack (torch, msclap, librosa, pyloudnorm, beat_this) is a
# heavy optional extra, so a checkout that only runs the server tests does
# not install it and everything that needs a model skips. Both the plain and
# the analysis interpreter must be green.
HAVE_V2 = importlib.util.find_spec("msclap") is not None
needs_v2 = pytest.mark.skipif(
    not HAVE_V2, reason="these tests need the analysis extra (torch + msclap)"
)

V2_MODELS = MODELS / "v2"
HAVE_CLAP_WEIGHTS = (V2_MODELS / "CLAP_weights_2023.pth").exists()
needs_clap_weights = pytest.mark.skipif(
    not HAVE_CLAP_WEIGHTS, reason="run scripts/fetch_models.py first"
)
