"""Jamendo as a Source: the first catalogue we can actually ship.

Every track on Jamendo is Creative Commons licensed and the API hands back
the full audio file, not a 30-second preview -- so playback is legal and the
analysis pipeline gets whatever it asks for (it still reads the first 30 s,
which keeps embeddings comparable with the Deezer-era corpus).

The licence is the point, so attribution is not optional: every track
carries `shareurl` (the human-readable page) and `license_ccurl` (the deed),
and both travel with the track into the store and out to the clients.

Needs JAMENDO_CLIENT_ID (free, from developer.jamendo.com). Without it the
source disables itself rather than failing every call -- unless it is the
only source configured, in which case there is nothing to fall back to and
starting up would be a lie.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from music_recommendations.corpus.sources.base import BaseSource, Track

API = "https://api.jamendo.com/v3.0/"
SLEEP = 0.2          # politeness delay before every call
TIMEOUT = 20

# The rotation the crawler walks. Chosen to spread the corpus across the
# space the embedding has to tell apart, not to mirror Jamendo's own
# taxonomy; `featured` is appended as an eleventh arm because it surfaces
# tracks no tag query reaches.
TAGS = ["jazz", "blues", "soul", "funk", "electronic",
        "rock", "folk", "classical", "hiphop", "ambient"]
PER_PAGE = 100

_last_call = 0.0
_call_lock = threading.Lock()


def _get(path: str, **params) -> dict:
    """One GET against the v3 API, rate-limited. Returns {} rather than raising.

    The single HTTP choke point for this module: `client_id` and
    `format=json` are added here, and tests stub exactly this function.
    """
    global _last_call
    query = {"client_id": os.environ.get("JAMENDO_CLIENT_ID", ""),
             "format": "json", **params}
    url = f"{API}{path}?{urllib.parse.urlencode(query)}"
    with _call_lock:
        elapsed = time.monotonic() - _last_call
        if elapsed < SLEEP:
            time.sleep(SLEEP - elapsed)
        _last_call = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _results(payload: dict) -> list[dict]:
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _attribution(raw: dict) -> dict | None:
    """The credit line the CC licence obliges us to show, or None.

    `shareurl` is the load-bearing half -- a credit with no link back is not
    attribution -- so a track without one gets nothing rather than a
    licence badge pointing nowhere.
    """
    url = raw.get("shareurl")
    if not url:
        return None
    return {"source": "jamendo", "url": url,
            "license": raw.get("license_ccurl") or None}


def _to_track(raw: dict) -> Track | None:
    """A Jamendo track payload -> contract Track, or None if unplayable."""
    track_id = raw.get("id")
    audio = raw.get("audio")
    if not track_id or not audio:
        return None
    track: Track = {
        "track_id": f"jamendo:{track_id}",
        "title": raw.get("name") or "",
        "artist": raw.get("artist_name") or "",
        "album": raw.get("album_name") or "",
        "artwork_url": raw.get("image") or raw.get("album_image") or "",
        "preview_url": audio,
        "source": "jamendo",
    }
    attribution = _attribution(raw)
    if attribution:
        track["attribution"] = attribution
    return track


def _tracks(payload: dict, limit: int | None = None) -> list[Track]:
    out = []
    for raw in _results(payload):
        track = _to_track(raw)
        if track:
            out.append(track)
        if limit is not None and len(out) >= limit:
            break
    return out


class JamendoSource(BaseSource):
    name = "jamendo"

    def enabled(self) -> bool:
        return bool(os.environ.get("JAMENDO_CLIENT_ID"))

    def disabled_reason(self) -> str:
        return "JAMENDO_CLIENT_ID is not set"

    def search(self, q: str, limit: int = 25) -> list[Track]:
        return _tracks(_get("tracks/", search=q, limit=limit), limit)

    def track(self, track_id: str) -> Track | None:
        found = _tracks(_get("tracks/", id=self.local_id(track_id)), 1)
        return found[0] if found else None

    def preview_url(self, track_id: str) -> str | None:
        """The `audio` URL.

        Unsigned and stable, unlike Deezer's -- but it is still fetched
        fresh rather than read back from the store, because the store never
        persists a preview URL for anyone.
        """
        found = self.track(track_id)
        return found["preview_url"] if found else None

    def attribution(self, track: Track) -> dict | None:
        return track.get("attribution")

    def candidates(self, step: int) -> list[Track]:
        """One tag's most popular tracks, or the featured list every 11th step."""
        index = step % (len(TAGS) + 1)
        if index < len(TAGS):
            tag = TAGS[index]
            self._label = f"jamendo tag {tag}"
            payload = _get("tracks/", tags=tag, order="popularity_total",
                           limit=PER_PAGE)
        else:
            self._label = "jamendo featured"
            payload = _get("tracks/", featured=1, limit=PER_PAGE)
        return _tracks(payload)
