"""Shared fixtures: a synthetic tone on disk, and Essentia-availability skips."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def run_essentia(script: str, out: Path) -> np.ndarray:
    """Run `script` in a subprocess (it must np.save its result to `out`).

    Essentia and TensorFlow cannot both be loaded in one process (they
    deadlock/abort — verified locally), and this test suite also exercises
    embedding.py's TensorFlow graph. So the essentia reference is always
    computed out of process, and only numpy arrays cross back over.
    """
    try:
        subprocess.run(
            [sys.executable, "-c", script], check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"essentia subprocess failed:\n{e.stderr}") from e
    return np.load(out)


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


HAVE_ESSENTIA = importlib.util.find_spec("essentia") is not None
needs_essentia = pytest.mark.skipif(
    not HAVE_ESSENTIA, reason="parity tests need essentia (Mac only)"
)

MODELS = Path(__file__).resolve().parents[2] / "models"
HAVE_EFFNET = (MODELS / "discogs-effnet-bs64-1.pb").exists()
needs_effnet = pytest.mark.skipif(
    not HAVE_EFFNET, reason="run scripts/fetch_models.py first"
)
