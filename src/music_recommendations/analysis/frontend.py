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
        # NOT "-ac", "1": ffmpeg's built-in downmix applies a sqrt(2) gain
        # (identical L/R at peak 0.5 comes out at peak ~0.707), unlike
        # Essentia's MonoLoader, which is a plain (L+R)/2 average (peak
        # stays 0.5). The mel front-end's log10(1 + 10000x) is not
        # gain-invariant, so that mismatch alone was enough to knock some
        # fixture tracks below 0.99 cosine parity. `pan=mono|c0<c0+c1`
        # renormalizes by the number of input channels present, giving the
        # same (L+R)/2 mean Essentia uses (and passing mono through
        # unchanged).
        "-af", "pan=mono|c0<c0+c1",
        "-ar", str(SAMPLE_RATE),
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


from . import registry  # noqa: E402

FRAME_SIZE = 512
HOP_SIZE = 256
N_MELS = 96
_N_BINS = FRAME_SIZE // 2 + 1
_LOG_SCALE = 10000.0


def _slaney_hz2mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    log_step = np.log(6.4) / 27.0
    with np.errstate(divide="ignore"):
        return np.where(f < 1000.0, f / (200.0 / 3.0),
                        15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / log_step)


def _slaney_mel2hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    log_step = np.log(6.4) / 27.0
    return np.where(m < 15.0, m * (200.0 / 3.0),
                    1000.0 * np.exp(log_step * (m - 15.0)))


def mel_filterbank() -> np.ndarray:
    """(96, 257) Slaney-mel triangles over 0-8000 Hz, each with unit area.

    This is Essentia MelBands(warpingFormula="slaneyMel", normalize="unit_tri",
    weighting="linear"), which is what TensorflowInputMusiCNN uses.
    """
    edges = _slaney_mel2hz(np.linspace(_slaney_hz2mel(0.0),
                                       _slaney_hz2mel(SAMPLE_RATE / 2),
                                       N_MELS + 2))
    freqs = np.arange(_N_BINS) * SAMPLE_RATE / FRAME_SIZE
    fb = np.zeros((N_MELS, _N_BINS))
    for i in range(N_MELS):
        left, centre, right = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None) * (2.0 / (right - left))
    return fb.astype(np.float32)


_FILTERBANK = None


def _filterbank() -> np.ndarray:
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = mel_filterbank()
    return _FILTERBANK


# Essentia Windowing(type="hann", normalized=False): symmetric Hann, then the
# frame is rolled by half its length (zero-phase). The roll changes nothing
# about the magnitude spectrum but is kept so intermediate values match.
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_SIZE) / (FRAME_SIZE - 1))).astype(np.float32)


def _frames(audio: np.ndarray) -> np.ndarray:
    """Essentia FrameCutter(startFromZero=True): first frame at 0, last frame
    zero-padded to a full 512."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))
    n_frames = -(-(len(audio) - FRAME_SIZE) // HOP_SIZE) + 1
    padded_len = (n_frames - 1) * HOP_SIZE + FRAME_SIZE
    audio = np.pad(audio, (0, padded_len - len(audio)))
    idx = np.arange(FRAME_SIZE)[None, :] + HOP_SIZE * np.arange(n_frames)[:, None]
    return audio[idx]


def mel_frames(audio: np.ndarray) -> np.ndarray:
    """(n_frames, 96) log-compressed mel bands, identical to Essentia's
    TensorflowInputMusiCNN applied to FrameCutter(512, 256) frames."""
    frames = _frames(audio) * _WINDOW
    frames = np.roll(frames, FRAME_SIZE // 2, axis=1)
    power = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) ** 2
    bands = power @ _filterbank().T
    return np.log10(1.0 + _LOG_SCALE * bands).astype(np.float32)


def patches(mel: np.ndarray) -> np.ndarray:
    """(n_patches, 128, 96) windows of mel frames with hop 62; a trailing
    partial patch is discarded (Essentia lastPatchMode="discard")."""
    size, hop = registry.PATCH_SIZE, registry.PATCH_HOP
    n = len(mel)
    if n < size:
        return np.zeros((0, size, mel.shape[1]), dtype=np.float32)
    count = 1 + (n - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return np.ascontiguousarray(mel[idx], dtype=np.float32)
