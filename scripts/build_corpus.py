"""Crawl candidate tracks and save them for analysis.

Usage:
  python3 scripts/build_corpus.py                 charts, all genres  -> corpus_candidates.json
  python3 scripts/build_corpus.py --snowball      artist graph from the 8 jazz roots
  python3 scripts/build_corpus.py --snowball --hops 3
  python3 scripts/build_corpus.py --per-genre 50

Writes contract-shaped Track dicts. Crawling only -- no downloads, no
analysis, no store writes. Run analyze_corpus.py next.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.corpus import crawl  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "corpus_candidates.json"


FLUSH_EVERY = 500


def _load(path: Path) -> dict:
    return ({t["track_id"]: t for t in json.loads(path.read_text())}
            if path.exists() else {})


def _deep_crawl(args) -> None:
    """Album deep cuts off artists already in the pool, saved as it goes.

    Runs for hours and is meant to be killed whenever the analyzer has enough
    to chew on, so it writes every FLUSH_EVERY new tracks instead of at the
    end. Re-running resumes: artists whose tracks are already in the pool cost
    one wasted lookup, not a re-crawl.
    """
    pool = _load(args.out)
    if not pool:
        sys.exit(f"{args.out} is empty — run a chart crawl first, "
                 "--deep expands an existing pool")

    # Spread across the whole pool rather than the first N alphabetically,
    # so the deep cuts span every genre the charts reached.
    names = list(dict.fromkeys(t["artist"] for t in pool.values()))
    random.shuffle(names)
    names = names[: args.deep]
    print(f"resolving {len(names)} artists (~{len(names) // 5}s)...", flush=True)
    artist_ids = crawl.resolve_artists(names)

    print(f"deep-crawling {len(artist_ids)} artists x {args.albums} albums "
          f"-> {args.out}", flush=True)
    before = len(pool)
    since_flush = 0
    try:
        for track in crawl.deep_cuts(artist_ids, albums_per_artist=args.albums):
            if track["track_id"] in pool:
                continue
            pool[track["track_id"]] = track
            since_flush += 1
            if since_flush >= FLUSH_EVERY:
                args.out.write_text(json.dumps(list(pool.values()), indent=1))
                since_flush = 0
                print(f"  {len(pool)} candidates (+{len(pool) - before})", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — saving what we have")
    finally:
        args.out.write_text(json.dumps(list(pool.values()), indent=1))
        print(f"{len(pool)} candidates (+{len(pool) - before} new) -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snowball", action="store_true",
                        help="crawl the artist-relatedness graph instead of charts")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--per-genre", type=int, default=100)
    parser.add_argument("--expand", type=int, default=0, metavar="N",
                        help="after charting, snowball 1 hop off N of the "
                             "artists found — the cheap way to go from ~1.7k "
                             "candidates to tens of thousands")
    parser.add_argument("--deep", type=int, default=0, metavar="N",
                        help="album deep cuts from N artists already in the "
                             "pool — the obscure end of the catalogue, since "
                             "charts and /top only ever return hits")
    parser.add_argument("--albums", type=int, default=6,
                        help="albums per artist for --deep")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.deep:
        _deep_crawl(args)
        return

    if args.snowball:
        print(f"snowballing {args.hops} hops from {len(crawl.ROOTS)} roots...")
        tracks = crawl.snowball(hops=args.hops)
    else:
        print(f"crawling {len(crawl.GENRES)} genre charts...")
        tracks = crawl.from_charts(per_genre=args.per_genre)

    if args.expand:
        # Charts are broad but shallow — ~100 tracks per genre and no depth.
        # One hop of /related off the artists they surfaced multiplies that,
        # while staying anchored to a genuinely genre-spread starting set.
        seeds = list(dict.fromkeys(t["artist"] for t in tracks))[: args.expand]
        print(f"expanding 1 hop from {len(seeds)} chart artists "
              f"(~{len(seeds) * 2} API calls, be patient)...")
        before = len(tracks)
        seen = {t["track_id"] for t in tracks}
        for track in crawl.snowball(root_names=seeds, hops=1, per_artist=20):
            if track["track_id"] not in seen:
                seen.add(track["track_id"])
                tracks.append(track)
        print(f"  +{len(tracks) - before} tracks from the expansion")

    if not tracks:
        sys.exit("no candidates found — Deezer may be rate limiting; try again")

    existing = {}
    if args.out.exists():
        existing = {t["track_id"]: t for t in json.loads(args.out.read_text())}
    before = len(existing)
    for track in tracks:
        existing.setdefault(track["track_id"], track)

    args.out.write_text(json.dumps(list(existing.values()), indent=1))
    artists = len({t["artist"] for t in existing.values()})
    print(f"{len(existing)} candidates ({len(existing) - before} new) "
          f"from {artists} artists -> {args.out}")


if __name__ == "__main__":
    main()
