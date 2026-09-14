"""Tempo, beat confidence, loudness and key -- the numbers CLAP cannot give.

A contrastive embedding knows what a track sounds *like*; it does not know
that it is at 92 BPM in F minor at -9 LUFS. Those are the numbers a listener
can name, so they are the ones worth showing in the math panel and worth
ranking on ("same feel, similar tempo"). All four come from small, exact
DSP rather than from a model:

  tempo/beats  Beat This! (MIT, transformer beat tracker), falling back to
               librosa's onset-autocorrelation tracker when its checkpoint
               is not installed.
  loudness     pyloudnorm, ITU-R BS.1770-4 -- the same integrated LUFS a
               mastering engineer would read.
  key          librosa CQT chroma against the Krumhansl-Schmuckler profiles.

Every import is inside a function: `RHYTHM_KEYS` and this docstring must be
readable without librosa or torch installed.
"""
from __future__ import annotations

import numpy as np

from . import registry

__all__ = ["RHYTHM_KEYS", "SILENCE_LUFS", "rhythm_features"]

# The dict `rhythm_features` returns, in contract order (contract/features.py
# RHYTHM_KEYS must equal this tuple).
RHYTHM_KEYS = (
    "tempo_bpm", "beat_strength", "loudness_lufs", "loudness_range",
    "key", "mode", "key_strength",
)

# BS.1770 integrated loudness of true silence is -inf. A float -inf poisons
# every mean, every distance and every JSON encoder downstream, so silence is
# reported as -70 LUFS -- the standard's own absolute gate, i.e. "below the
# quietest thing the measurement is defined for".
SILENCE_LUFS = -70.0

# Krumhansl-Kessler key profiles: the perceived stability of each of the
# twelve pitch classes in a major and a minor key, measured by probe-tone
# experiments. Correlating a track's chroma against all 24 rotations is the
# textbook key finder, and it is honest about uncertainty -- the correlation
# it maximizes IS key_strength.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_beat_tracker = None


def rhythm_features(waveform: np.ndarray, sr: int) -> dict:
    """The seven RHYTHM_KEYS for one decoded, mono track."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    tempo_bpm, beat_strength = _tempo(y, sr)
    lufs, lra = _loudness(y, sr)
    key, mode, strength = _key(y, sr)
    return {
        "tempo_bpm": float(tempo_bpm),
        "beat_strength": float(beat_strength),
        "loudness_lufs": float(lufs),
        "loudness_range": float(lra),
        "key": int(key),
        "mode": mode,
        "key_strength": float(strength),
    }


# ---- tempo ------------------------------------------------------------------

def _load_beat_tracker():
    """Beat This! frame model + minimal postprocessor, or None.

    None is a supported state, not an error: the checkpoint is 81 MB and a
    host that skipped fetch_models.py, or a test run, should still get a
    tempo. Cached (including the None) so a missing checkpoint costs one
    import attempt per process rather than one per track.
    """
    global _beat_tracker
    if _beat_tracker is not None:
        return _beat_tracker[0]
    try:
        if not registry.BEAT_THIS_CKPT.exists():
            raise FileNotFoundError(registry.BEAT_THIS_CKPT)
        import torch
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor

        torch.set_num_threads(2)
        frames = Audio2Frames(
            checkpoint_path=str(registry.BEAT_THIS_CKPT), device="cpu"
        )
        _beat_tracker = ((frames, Postprocessor(type="minimal")),)
    except Exception:  # noqa: BLE001 - any failure means "use librosa"
        _beat_tracker = (None,)
    return _beat_tracker[0]


def _tempo(y: np.ndarray, sr: int) -> tuple[float, float]:
    """(BPM, beat_strength in [0, 1]).

    Tempo is the *median* inter-beat interval, not the mean: a tracker that
    drops or doubles one beat in a 30 s preview would drag a mean several BPM
    off, and the median ignores it.
    """
    tracker = _load_beat_tracker()
    if tracker is not None:
        try:
            return _tempo_beat_this(tracker, y, sr)
        except Exception:  # noqa: BLE001 - fall through to librosa
            pass
    return _tempo_librosa(y, sr)


def _tempo_beat_this(tracker, y: np.ndarray, sr: int) -> tuple[float, float]:
    import torch

    frames, post = tracker
    beat_logits, downbeat_logits = frames(y, sr)
    beats, _downbeats = post(beat_logits, downbeat_logits)
    if len(beats) < 3:
        raise ValueError("too few beats")
    bpm = 60.0 / float(np.median(np.diff(np.asarray(beats))))
    # Beat strength is the model's own confidence where it placed the beats:
    # the mean activation over all frames would just measure how sparse beats
    # are (~0.02 for everything), which says nothing about the groove.
    act = torch.sigmoid(beat_logits).cpu().numpy()
    idx = np.clip((np.asarray(beats) * 50).astype(int), 0, len(act) - 1)
    return bpm, float(np.mean(act[idx]))


def _tempo_librosa(y: np.ndarray, sr: int) -> tuple[float, float]:
    import librosa

    if not np.any(y):
        return 0.0, 0.0
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0])
    if len(beats) >= 3:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    peak = float(onset.max()) or 1.0
    return bpm, float(np.clip(onset.mean() / peak, 0.0, 1.0))


# ---- loudness ---------------------------------------------------------------

def _loudness(y: np.ndarray, sr: int) -> tuple[float, float]:
    """(integrated LUFS, loudness range in LU), both floored at silence.

    Loudness range is EBU R128's LRA read on short-term blocks: the spread
    between the 10th and 95th percentile of 3 s windows. It separates a
    dynamic live recording from a brickwalled master, which integrated
    loudness alone cannot.
    """
    import pyloudnorm as pyln

    block = 3.0
    if y.size < int(block * sr) or not np.any(y):
        return SILENCE_LUFS, 0.0
    meter = pyln.Meter(sr, block_size=block)
    integrated = _floor(meter.integrated_loudness(y))

    step = int(sr)  # 1 s hop over 3 s windows
    win = int(block * sr)
    shorts = [
        _floor(meter.integrated_loudness(y[s:s + win]))
        for s in range(0, y.size - win + 1, step)
    ]
    loud = [v for v in shorts if v > SILENCE_LUFS]
    if len(loud) < 2:
        return integrated, 0.0
    lo, hi = np.percentile(loud, [10, 95])
    return integrated, float(hi - lo)


def _floor(value: float) -> float:
    v = float(value)
    return SILENCE_LUFS if not np.isfinite(v) or v < SILENCE_LUFS else v


# ---- key --------------------------------------------------------------------

def _key(y: np.ndarray, sr: int) -> tuple[int, str, float]:
    """(pitch class 0-11 where 0 = C, "major"|"minor", correlation in [0, 1]).

    Silence and pure noise give a flat chroma, whose correlation with every
    profile is ~0: key_strength is the caller's signal that the key field is
    meaningless, so it must not be faked up to something confident.
    """
    import librosa

    if not np.any(y):
        return 0, "major", 0.0
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if not np.isfinite(chroma).all() or chroma.sum() <= 0:
        return 0, "major", 0.0
    chroma = chroma - chroma.mean()
    if not np.any(chroma):
        return 0, "major", 0.0

    best = (0, "major", -1.0)
    for name, profile in (("major", _MAJOR), ("minor", _MINOR)):
        centered = profile - profile.mean()
        for tonic in range(12):
            rotated = np.roll(centered, tonic)
            denom = np.linalg.norm(chroma) * np.linalg.norm(rotated)
            corr = float(np.dot(chroma, rotated) / denom) if denom else 0.0
            if corr > best[2]:
                best = (tonic, name, corr)
    return best[0], best[1], max(0.0, best[2])
