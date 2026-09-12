"""Prove the Atlas connection works: indexes, a write, a read, a queue round trip.

    MONGODB_URI="mongodb+srv://..." python3 scripts/atlas_check.py

Writes one throwaway document under _id "atlas-check" and removes it again.
Exit code 0 on success, 1 on any failure (message on stderr).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.server import store  # noqa: E402

_URI_RE = re.compile(r"mongodb(?:\+srv)?://[^\s\"']*")


def _redact(text: str) -> str:
    """pymongo errors can echo the connection string (credentials and all)
    back at us; scrub it before it hits stderr/logs."""
    return _URI_RE.sub("mongodb://<redacted>", text)

TRACK = {"track_id": "atlas-check", "title": "check", "artist": "check",
         "album": "check", "artwork_url": "", "preview_url": ""}


def _cleanup() -> None:
    """Best-effort: never let a cleanup failure hide the original error."""
    try:
        store.clear_embed_marker("atlas-check")
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        print(f"atlas check cleanup failed: {type(exc).__name__}: {_redact(str(exc))}", file=sys.stderr)
    try:
        store.db().tracks.delete_one({"_id": "atlas-check"})
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        print(f"atlas check cleanup failed: {type(exc).__name__}: {_redact(str(exc))}", file=sys.stderr)


def run() -> int:
    try:
        store.ensure_indexes()
        before = store.corpus_size()
        # A prior failed run can leave "embed:atlas-check" queued, which would
        # make enqueue_embed below return False through no fault of this run.
        store.clear_embed_marker("atlas-check")
        try:
            vec = np.linspace(-1, 1, 1280, dtype=np.float32)
            store.put_track(TRACK, {"embedding": vec, "_features_version": 3})
            got = store.get_features("atlas-check")
            assert got and np.allclose(got["embedding"], vec, atol=0.01), "embedding mismatch"
            assert (store.enqueue_embed("atlas-check")
                    and store.dequeue_embed(timeout=0) == "atlas-check")
        finally:
            _cleanup()
        jobs = store.db().jobs.count_documents({})  # read before reset() drops the client
        store.reset()
        print(f"tracks: {before}  jobs: {jobs}  round-trip ok")
        return 0
    except Exception as exc:  # noqa: BLE001 - this script's job is to report any failure
        print(f"atlas check failed: {type(exc).__name__}: {_redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
