"""Deezer as a Source: the development catalogue.

Deezer has no licence that lets us ship, so this is the source we build
against and never the one we sell: 30-second previews, no attribution
obligation, and ids that stay bare digits (see base.py).

Nothing here is new API code. Search, single-track lookup and preview
re-signing are `server/deezer.py`; the crawl arms are `corpus/crawl.py`.
What this file adds is the rotation that used to live in
`worker.crawl_step` -- charts, snowball, deep cuts -- so the worker no
longer knows anything about Deezer's shape.
"""
from __future__ import annotations

from music_recommendations.corpus import crawl
from music_recommendations.corpus.sources.base import BaseSource, Track
from music_recommendations.server import deezer as api
from music_recommendations.server import store

# How many newly-discovered artist names one chart/snowball step may add to
# the crawl roots, and how many roots are kept in total.
GROW_PER_STEP = 5
MAX_ROOTS = 200

ARMS = 3          # charts, snowball, deep cuts


def _grow_roots(roots: list[str], tracks: list[Track]) -> None:
    """Add up to GROW_PER_STEP newly-discovered artist names to the
    `crawl_roots` state, so the snowball/deep-cuts graph expands from what
    the corpus actually contains instead of staying pinned to the 8 seed
    roots forever."""
    discovered: list[str] = []
    for track in tracks:
        name = track.get("artist")
        if name and name not in roots and name not in discovered:
            discovered.append(name)
        if len(discovered) >= GROW_PER_STEP:
            break
    if not discovered:
        return
    state = store.get_state("crawl_roots") or {"names": []}
    names = list(state.get("names", []))
    for name in discovered:
        if name not in names:
            names.append(name)
    store.put_state("crawl_roots", {"names": names[:MAX_ROOTS]})


class DeezerSource(BaseSource):
    name = "deezer"
    prefixed = False          # bare digits; "no prefix" means Deezer

    def search(self, q: str, limit: int = 25) -> list[Track]:
        return [self._tag(t) for t in api.search(q, limit)]

    def track(self, track_id: str) -> Track | None:
        found = api.get_track(self.local_id(track_id))
        return self._tag(found) if found else None

    def preview_url(self, track_id: str) -> str | None:
        return api.fresh_preview_url(self.local_id(track_id))

    def attribution(self, track: Track) -> dict | None:
        """None: Deezer previews carry no attribution obligation, and this
        source never ships. Returning a Deezer link here would put a "via
        Deezer" credit on every row of a development corpus."""
        return None

    def candidates(self, step: int) -> list[Track]:
        """One bounded slice, rotating over three arms.

        Breadth (charts across genres), depth (the artist-relatedness
        graph), and obscurity (album tracks that never show up in a /top or
        /related call) all keep growing this way. `step // ARMS` indexes
        within each arm, so every arm walks its own list.
        """
        genres = list(crawl.GENRES)
        roots_state = store.get_state("crawl_roots")
        roots = crawl.ROOTS + (roots_state["names"] if roots_state else [])
        arm = step % ARMS
        index = step // ARMS
        grow_from: list[Track] | None = None

        if arm == 0:
            genre = genres[index % len(genres)]
            tracks = crawl.from_charts([genre], per_genre=100)
            self._label = f"chart {crawl.GENRES[genre]}"
            grow_from = tracks
        elif arm == 1:
            root = roots[index % len(roots)]
            tracks = crawl.snowball([root], hops=1, per_artist=10)
            self._label = f"snowball {root}"
            grow_from = tracks
        else:
            root = roots[index % len(roots)]
            ids = crawl.resolve_artists([root])
            tracks = list(crawl.deep_cuts(ids, albums_per_artist=3))
            self._label = f"deep cuts {root}"

        if grow_from is not None:
            _grow_roots(roots, grow_from)
        return [self._tag(t) for t in tracks]

    def _tag(self, track: Track) -> Track:
        return {**track, "source": self.name}
