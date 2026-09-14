"""analyze_track: MP3 path in, dict of named feature vectors out (spec §2.1).

analyze_tracks takes several paths and runs them through one batched pass,
returning a dict or an exception per path; analyze_track is the one-path
case of it.

Pure — no cache, no HTTP, no store access. Callers (server, corpus/ingest)
decide when to run it and where results live. Returned dict keys must
match contract/features.py FEATURE_KEYS.

The implementation is analysis/v2.py: a Microsoft CLAP embedding, eight
zero-shot feel axes on top of it, and rhythm/loudness/key. Nothing here
imports TensorFlow or Essentia any more — the Discogs-EffNet pipeline that
used to live beside this one was non-commercial, and removing it is what
the clean-room work was for.

Callers that persist or compare these vectors also want FEATURES_VERSION
and METRICS from .schema — the version to know when cached vectors went
stale, the metrics to know how to compare them. Import them from
music_recommendations.analysis.schema directly to skip loading torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import FEATURES_VERSION, METRICS

__all__ = ["analyze_track", "analyze_tracks", "as_json", "FEATURES_VERSION",
           "METRICS"]


def analyze_tracks(paths: list[Path | str]) -> list[dict | Exception]:
    """One slot per path, holding features or that path's exception.

    A thin forwarder rather than `analyze_tracks = v2.analyze_tracks`, so
    that importing this package stays free: v2 pulls in torch, librosa and
    ~700 MB of CLAP weights the first time it actually runs, and callers
    that only want FEATURES_VERSION (the store deciding whether a cached row
    is stale) must not pay for that.
    """
    from . import v2

    return v2.analyze_tracks(paths)


def analyze_track(path: Path | str) -> dict:
    """Features for one file: `embedding` (1024,), `feel` (8,),
    `rhythm` (a dict of the seven RHYTHM_KEYS), `_features_version`."""
    from . import v2

    return v2.analyze_track(path)


def as_json(features: dict) -> dict:
    """Feature dict with ndarrays flattened to lists, for storage or transport.

    `rhythm` is already a dict of plain numbers and strings and passes
    through as-is; only the vectors need converting.
    """
    out: dict = {}
    for k, v in features.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {kk: (vv if isinstance(vv, str) else
                           (int(vv) if isinstance(vv, (int, np.integer))
                            else float(vv)))
                      for kk, vv in v.items()}
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out
