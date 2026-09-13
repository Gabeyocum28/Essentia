"""Score the corpus that was analyzed before the feel heads existed.

The eleven heads read the STORED embedding, so this needs no audio, no
downloads and no EffNet pass -- it dequantizes what is already in Mongo,
runs eleven small matmuls over it, and writes eleven floats back. Measured
on this Mac: ~60k rows a second through the heads, so the whole corpus is
bounded by the database round trips, not the model.

    python3 scripts/feel_backfill.py          # only rows that have no feel
    python3 scripts/feel_backfill.py --all    # rescore everything

Exit code 0 unless the store itself fails.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis.feel import feel_vectors  # noqa: E402
from music_recommendations.analysis.quantize import from_int8  # noqa: E402
from music_recommendations.server import store  # noqa: E402

# Rows per inference + write batch. The heads would happily take the whole
# corpus at once, but the embeddings alone are 5 MB per thousand rows once
# dequantized, and a chunk this size keeps peak memory flat while still
# amortizing the per-call cost to nothing.
CHUNK = 2000

FEEL_DP = store.FEEL_DP


def _query(everything: bool) -> dict:
    """Analyzed rows to score. Retired duplicates are included on purpose:
    they are still legitimate seeds (store.LIVE's docstring), and a seed with
    no feel vector would silently disable the blend for whoever played it."""
    query: dict = {"embedding": {"$exists": True}}
    if not everything:
        query["feel"] = {"$exists": False}
    return query


def _write(rows: list[tuple[str, list[float]]]) -> None:
    """Write one chunk's vectors.

    Every row gets a DIFFERENT vector, so there is no `$in` grouping to be
    had (the trick dedupe_corpus.py uses). A pymongo bulk_write is the right
    shape and is tried first; mongomock cannot execute the UpdateOne objects
    this pymongo builds, and the test suite runs on mongomock, so a per-doc
    fallback stands behind it.
    """
    tracks = store.db().tracks
    try:
        from pymongo import UpdateOne

        tracks.bulk_write([UpdateOne({"_id": track_id},
                                     {"$set": {"feel": vector}})
                           for track_id, vector in rows], ordered=False)
        return
    except Exception:  # noqa: BLE001 - any driver that refuses the bulk form
        pass
    for track_id, vector in rows:
        tracks.update_one({"_id": track_id}, {"$set": {"feel": vector}})


def _score(docs: list[dict]) -> list[tuple[str, list[float]]]:
    """One inference pass over a whole chunk -> (id, rounded vector) pairs."""
    embeddings = np.stack([from_int8(d["embedding"], d["scale"]) for d in docs])
    vectors = feel_vectors(embeddings)
    return [(doc["_id"], [round(float(v), FEEL_DP) for v in vector])
            for doc, vector in zip(docs, vectors)]


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="rescore every analyzed row, not just the "
                             "rows that have no feel vector yet")
    parser.add_argument("--chunk", type=int, default=CHUNK,
                        help=f"rows per inference + write batch "
                             f"(default: {CHUNK})")
    args = parser.parse_args(argv)

    cursor = store.db().tracks.find(_query(args.all),
                                    {"embedding": 1, "scale": 1})
    started = time.monotonic()
    scored = 0
    chunk: list[dict] = []
    for doc in cursor:
        if doc.get("embedding") is None or doc.get("scale") is None:
            continue
        chunk.append(doc)
        if len(chunk) >= args.chunk:
            _write(_score(chunk))
            scored += len(chunk)
            chunk = []
    if chunk:
        _write(_score(chunk))
        scored += len(chunk)

    elapsed = time.monotonic() - started
    rate = scored / elapsed if elapsed > 0 else 0.0
    print(f"scored {scored} in {elapsed:.1f} s ({rate:.0f} tracks/s)")
    if scored:
        print("the API holds its matrix in memory: restart it to pick this up")
    return 0


if __name__ == "__main__":
    sys.exit(run())
