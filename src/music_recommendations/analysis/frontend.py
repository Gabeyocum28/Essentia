"""Audio -> EffNet input, without Essentia.

Essentia has no Linux aarch64 wheels, so the VM cannot import it. What
Essentia contributed to analysis was decoding (MonoLoader) and the MusiCNN
mel front-end (TensorflowInputMusiCNN + the framing inside
TensorflowPredictEffnetDiscogs). Both are reproduced here in numpy; the
parameters were measured against Essentia and are recorded in
docs/superpowers/plans/2026-09-11-analysis-on-arm.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class DecodeError(RuntimeError):
    """ffmpeg could not decode the file."""


def decode(path: Path | str) -> np.ndarray:
    """Decode any audio file to mono float32 at 16 kHz via the ffmpeg CLI."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:  # ffmpeg binary itself missing
        raise DecodeError("ffmpeg not installed") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(proc.stderr.decode(errors="replace").strip()
                          or f"ffmpeg produced no audio for {path}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
