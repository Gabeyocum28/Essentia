"""analyze_track: MP3 path in, dict of named feature vectors out (spec §2.1).

analyze_tracks takes several paths and runs them through one packed EffNet
pass, returning a dict or an exception per path; analyze_track is the
one-path case of it.

Pure — no cache, no HTTP, no store access. Callers (server, corpus/ingest)
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

__all__ = ["analyze_track", "analyze_tracks", "as_json", "FEATURES_VERSION",
           "METRICS"]


def analyze_tracks(paths: list[Path | str]) -> list[dict | Exception]:
    """Analyze several files in one inference pass; one slot per input path.

    A slot holds either the feature dict or the exception that path raised,
    so one unreadable or too-short file never costs the rest of the group
    its analysis. Decoding and the mel front end still run per file (they
    are cheap and independent); only the EffNet pass is shared, and it is
    shared because a single preview's 28 patches would otherwise leave more
    than half of the graph's fixed 64-patch batch filled with zero padding.
    """
    # Imported here, not at module scope: importing TensorFlow costs ~1 s,
    # and a caller that only wants FEATURES_VERSION or METRICS should not pay.
    from . import embedding, frontend

    results: list[dict | Exception | None] = [None] * len(paths)
    ready: list[int] = []
    patch_groups: list[np.ndarray] = []
    for i, raw in enumerate(paths):
        path = Path(raw)
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            patches = frontend.patches(
                frontend.mel_frames(embedding.load_audio(path))
            )
            if len(patches) == 0:
                raise ValueError(f"{path}: too short for one EffNet patch")
            ready.append(i)
            patch_groups.append(patches)
        except Exception as exc:  # noqa: BLE001 - reported in this path's slot
            results[i] = exc

    if ready:
        for i, frames in zip(ready, embedding.embed_patch_groups(patch_groups)):
            results[i] = {"embedding": frames.mean(axis=0).astype(np.float32)}
    return results  # type: ignore[return-value]


def analyze_track(mp3_path: Path | str) -> dict:
    """Run EffNet; returns {"embedding": (1280,) float32}."""
    result = analyze_tracks([mp3_path])[0]
    if isinstance(result, Exception):
        raise result
    return result


def as_json(features: dict) -> dict:
    """Feature dict with ndarrays flattened to lists, for storage or transport."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else float(v)
        for k, v in features.items()
    }
