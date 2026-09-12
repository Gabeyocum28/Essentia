"""Analyze crawled candidates into Atlas (needs MONGODB_URI).

Usage:
  python3 scripts/analyze_corpus.py                 up to 300 tracks
  python3 scripts/analyze_corpus.py 1000            up to 1000
  python3 scripts/analyze_corpus.py --workers 9

Resumable: anything already stored at the current FEATURES_VERSION is
skipped, so re-running after a crash or a sleep picks up where it left off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis import FEATURES_VERSION  # noqa: E402
from music_recommendations.corpus.ingest import ingest  # noqa: E402

CANDIDATES = Path(__file__).resolve().parents[1] / "corpus_candidates.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", nargs="?", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard", default="0/1",
                        help="i/n — take every track whose id hashes to i mod n")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    args = parser.parse_args()

    if not args.candidates.exists():
        sys.exit(f"{args.candidates} not found — run scripts/build_corpus.py first")
    tracks = json.loads(args.candidates.read_text())

    index, total = (int(x) for x in args.shard.split("/"))
    if total > 1:
        tracks = [t for t in tracks if int(t["track_id"]) % total == index]
        print(f"shard {index}/{total}: {len(tracks)} of the candidates")

    print(f"ingesting up to {args.limit} tracks at FEATURES_VERSION {FEATURES_VERSION}")
    started = time.time()
    try:
        stored = ingest(tracks, limit=args.limit, workers=args.workers)
    except Exception as exc:
        sys.exit(f"ingest failed: {exc}\n(is MONGODB_URI set and reachable?)")

    elapsed = time.time() - started
    rate = f"{elapsed / stored:.2f}s/track" if stored else "n/a"
    print(f"\nstored {stored} tracks in {elapsed / 60:.1f} min ({rate})")


if __name__ == "__main__":
    main()
