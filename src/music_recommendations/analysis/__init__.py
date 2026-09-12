"""analyze_track: MP3 path in, dict of named feature vectors out (spec §2.1).

Pure — no cache, no HTTP, no Redis. Callers (server, corpus/ingest)
decide when to run it and where results live. Returned dict keys must
match contract/features.py FEATURE_KEYS.

Callers that persist or compare these vectors also want FEATURES_VERSION
and METRICS from .schema — the version to know when cached vectors went
stale, the metrics to know how to compare them. Import them from
music_recommendations.analysis.schema directly to skip loading TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import FEATURES_VERSION, METRICS

__all__ = ["analyze_track", "as_json", "FEATURES_VERSION", "METRICS"]


def analyze_track(mp3_path: Path | str) -> dict:
    """Run EffNet; returns {"embedding": (1280,) float32}."""
    # Imported here, not at module scope: importing TensorFlow costs ~1 s,
    # and a caller that only wants FEATURES_VERSION or METRICS should not pay.
    from . import embedding

    mp3_path = Path(mp3_path)
    if not mp3_path.exists():
        raise FileNotFoundError(mp3_path)

    frames = embedding.effnet_frames(mp3_path)
    if len(frames) == 0:
        raise ValueError(f"{mp3_path}: too short for one EffNet patch")
    return {"embedding": frames.mean(axis=0).astype(np.float32)}


def as_json(features: dict) -> dict:
    """Feature dict with ndarrays flattened to lists, for storage or transport."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else float(v)
        for k, v in features.items()
    }
