"""Retire the re-releases already sitting in the corpus.

The crawler only stopped admitting duplicates from the day the dedupe key
shipped; everything analyzed before that is still there -- the 2009
remaster, the live take, the "(feat. X)" credit, each as its own rankable
row. This walks the whole live corpus once and marks the extras as
`duplicate_of` the earliest analyzed edition.

    python3 scripts/dedupe_corpus.py            # dry run, writes nothing
    python3 scripts/dedupe_corpus.py --apply    # marks them

Two passes, both conservative:

  1. key  -- group on `dedupe_key`. Exact string match, so it only ever
     folds editions of one recording by one primary artist.
  2. embedding -- among the pass-1 survivors of ONE normalized artist,
     anything whose cosine to an earlier survivor exceeds the threshold
     (0.995 by default). Never compared across artists, so a cover keeps
     its own row no matter how close it sounds.

Exit code 0 unless the store itself fails.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis.quantize import from_int8  # noqa: E402
from music_recommendations.server import store  # noqa: E402
from music_recommendations.server.dedupe import dedupe_key, normalize_artist  # noqa: E402

# A group larger than this is not a re-release cluster, it is a data
# problem (a whole artist mis-tagged onto one name). An N x N cosine
# matrix for it would also stop being free.
MAX_GROUP = 500

_EPOCH = datetime(1970, 1, 1)


def _order(doc: dict) -> tuple[datetime, str]:
    """Oldest first; ties broken by id so the choice is deterministic."""
    return (doc.get("analyzed_at") or _EPOCH, doc["_id"])


def _load() -> list[dict]:
    """Every live analyzed doc, oldest first.

    `dedupe_key` is recomputed when absent: docs analyzed before the key
    shipped do not carry one, and those are exactly the docs this script
    exists for.
    """
    docs = list(store.db().tracks.find(
        store.LIVE,
        {"title": 1, "artist": 1, "analyzed_at": 1,
         "embedding": 1, "scale": 1, "dedupe_key": 1},
    ))
    for doc in docs:
        if not doc.get("dedupe_key"):
            doc["dedupe_key"] = dedupe_key(doc.get("title"), doc.get("artist"))
    docs.sort(key=_order)
    return docs


def _key_pass(docs: list[dict]) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """{duplicate id: primary id} plus the groups, for the printout."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        groups[doc["dedupe_key"]].append(doc)

    marks: dict[str, str] = {}
    for members in groups.values():
        primary = members[0]["_id"]  # docs came in sorted, so [0] is earliest
        for doc in members[1:]:
            marks[doc["_id"]] = primary
    return marks, groups


def _unit(doc: dict) -> np.ndarray | None:
    """The stored embedding, unit length. None if the row has no vector."""
    if doc.get("embedding") is None or doc.get("scale") is None:
        return None
    vec = from_int8(doc["embedding"], doc["scale"])
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


def _embedding_pass(survivors: list[dict], threshold: float) -> dict[str, str]:
    """Near-identical vectors within one artist -> {duplicate id: primary id}.

    Groups are re-release clusters, so a few rows each: one dense cosine
    matrix per artist is cheaper than any index would be.
    """
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for doc in survivors:
        by_artist[normalize_artist(doc.get("artist"))].append(doc)

    marks: dict[str, str] = {}
    for artist, members in by_artist.items():
        if len(members) < 2:
            continue
        if len(members) > MAX_GROUP:
            print(f"  skipping artist {artist!r}: {len(members)} tracks "
                  f"(over the {MAX_GROUP} cap)", file=sys.stderr)
            continue

        vectors, rows = [], []
        for doc in members:  # already oldest-first
            unit = _unit(doc)
            if unit is not None:
                vectors.append(unit)
                rows.append(doc)
        if len(rows) < 2:
            continue

        cosines = np.asarray(vectors) @ np.asarray(vectors).T
        for i in range(1, len(rows)):
            for j in range(i):  # earliest matching survivor wins
                if rows[j]["_id"] in marks:
                    continue
                if cosines[i, j] > threshold:
                    marks[rows[i]["_id"]] = rows[j]["_id"]
                    break
    return marks


def _report_groups(groups: dict[str, list[dict]], limit: int) -> None:
    biggest = sorted((g for g in groups.values() if len(g) > 1),
                     key=len, reverse=True)[:limit]
    if not biggest:
        return
    print(f"top {len(biggest)} key groups:")
    for members in biggest:
        head = members[0]
        print(f"  {len(members):3d}x  {head.get('title') or '?'} "
              f"-- {head.get('artist') or '?'}")


def _apply(marks: dict[str, str]) -> None:
    """Write `duplicate_of`, one update per PRIMARY rather than per row.

    All the duplicates of one primary take the same value, so they fold
    into a single `$in` update: a couple of thousand rows become a few
    hundred round trips instead of a few thousand. (A pymongo bulk_write
    of UpdateOnes would be fewer still, but mongomock cannot execute the
    UpdateOne objects this pymongo builds, and the test suite runs on
    mongomock.)
    """
    by_primary: dict[str, list[str]] = defaultdict(list)
    for track_id, primary in marks.items():
        by_primary[primary].append(track_id)
    for primary, ids in by_primary.items():
        store.db().tracks.update_many({"_id": {"$in": ids}},
                                      {"$set": {"duplicate_of": primary}})
    # mark_duplicate() drops this cache per write; the bulk path has to do
    # it once itself or corpus_ids() keeps listing the retired rows.
    with store._lock:  # noqa: SLF001 - same module's documented invalidation
        store._ids_cache = None


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write duplicate_of (default: dry run)")
    parser.add_argument("--threshold", type=float, default=0.995,
                        help="cosine above which two tracks by one artist "
                             "are the same recording (default: 0.995)")
    parser.add_argument("--limit-groups", type=int, default=10,
                        help="how many key groups to print (default: 10)")
    args = parser.parse_args(argv)

    docs = _load()
    total = len(docs)

    key_marks, groups = _key_pass(docs)
    survivors = [d for d in docs if d["_id"] not in key_marks]
    embed_marks = _embedding_pass(survivors, args.threshold)
    marks = {**key_marks, **embed_marks}

    print(f"live analyzed tracks: {total}")
    print(f"pass 1 (key):        {len(key_marks)}")
    print(f"pass 2 (embedding):  {len(embed_marks)}")
    print(f"total to mark:       {len(marks)}")
    _report_groups(groups, args.limit_groups)

    share = (100.0 * len(marks) / total) if total else 0.0
    if args.apply:
        _apply(marks)
        print(f"marked {len(marks)} of {total}")
        print("the API holds its matrix in memory: restart it to pick this up")
    else:
        print(f"would mark {len(marks)} of {total} ({share:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
