"""MongoDB Atlas read/write. Database `essentia`, three collections:

  tracks  _id=track_id, title, artist, album, artwork_url,
          embedding (int8 bytes), scale (float), features_version, analyzed_at
          -- the last four are absent until the track is analyzed.
  jobs    _id="embed:{id}" | "attr:{seed}|{rec}", kind, state, claimed_at,
          attempts, error, created_at  (+ track_id or seed_id/rec_id)
  cache   _id="preview:{id}" | "attr:{seed}:{rec}", value, expires_at (TTL)

Signed preview URLs are never stored on a track (they expire in minutes);
GET /preview re-signs and caches them in `cache`.

Every function here is the same name app.py, corpus/ingest.py and the
worker called when this was Redis. Only the backend changed.

Timestamps are naive UTC throughout (not timezone-aware): mongomock strips
tzinfo off datetimes on round-trip, so comparing an aware `_now()` against a
value read back from mongomock raises. Using naive UTC everywhere avoids the
mismatch in both the mongomock-backed tests and the real Atlas driver.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pymongo
from pymongo import ReturnDocument

from music_recommendations.analysis.quantize import from_int8, to_int8

TRACK_FIELDS = ("title", "artist", "album", "artwork_url")
VERSION_KEY = "_features_version"   # what corpus/ingest.py expects to read back

_client: pymongo.MongoClient | None = None
_indexes_ready = False
_lock = threading.Lock()

_now_lock = threading.Lock()
_last_now: datetime | None = None


def _now() -> datetime:
    """Naive UTC, monotonically increasing within this process.

    MongoDB (mongomock included) stores datetimes at millisecond
    resolution. Two calls close enough together would otherwise tie, and
    corpus_ids()/tracks_since() rely on strict "$gt newest" to find only
    newly-analyzed rows -- a tie would silently drop a row. Rather than
    fabricating a value ahead of the real clock (which would make a
    newly-written row look newer than a `datetime.now()` taken by a caller
    moments later), spin until the wall clock itself ticks over to the next
    millisecond. That costs at most ~1ms and only when two writes race.
    """
    global _last_now
    with _now_lock:
        while True:
            now = datetime.utcnow()
            # Round to millisecond precision up front: that is what a round
            # trip through Mongo storage will truncate it to anyway, and
            # the monotonic check below has to compare like with like.
            now = now.replace(microsecond=(now.microsecond // 1000) * 1000)
            if _last_now is None or now > _last_now:
                _last_now = now
                return now
            time.sleep(0.0002)


def db():
    """Lazily connect using MONGODB_URI / MONGODB_DB; one client per process."""
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is not set")
        _client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client[os.environ.get("MONGODB_DB", "essentia")]


def reset() -> None:
    """Forget the client and caches (tests, and after a fork)."""
    global _client, _indexes_ready, _ids_cache, _last_now
    _client = None
    _indexes_ready = False
    _ids_cache = None
    _last_now = None


def ensure_indexes() -> None:
    """Idempotent; called once per process before the first write."""
    global _indexes_ready
    if _indexes_ready:
        return
    d = db()
    d.tracks.create_index("analyzed_at")
    d.jobs.create_index([("state", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)])
    d.cache.create_index("expires_at", expireAfterSeconds=0)
    _indexes_ready = True


# ---- tracks ----

def _contract(doc: dict) -> dict:
    return {"track_id": doc["_id"], **{k: doc.get(k) for k in TRACK_FIELDS},
            "preview_url": ""}


def _meta(track: dict) -> dict:
    return {k: track.get(k) for k in TRACK_FIELDS}


def put_track(track: dict, features: dict) -> None:
    """Upsert contract fields plus the analyzed embedding."""
    ensure_indexes()
    data, scale = to_int8(np.asarray(features["embedding"], dtype=np.float32))
    db().tracks.update_one(
        {"_id": track["track_id"]},
        {"$set": {**_meta(track), "embedding": data, "scale": float(scale),
                  "features_version": int(features.get(VERSION_KEY, 0)),
                  "analyzed_at": _now()}},
        upsert=True,
    )


def put_track_meta(track: dict) -> None:
    """Upsert contract fields only; leaves any embedding in place."""
    ensure_indexes()
    db().tracks.update_one({"_id": track["track_id"]}, {"$set": _meta(track)}, upsert=True)


def get_track(track_id: str) -> dict | None:
    doc = db().tracks.find_one({"_id": track_id}, {k: 1 for k in TRACK_FIELDS})
    return _contract(doc) if doc else None


def get_many_tracks(track_ids: list[str]) -> list[dict | None]:
    if not track_ids:
        return []
    found = {d["_id"]: _contract(d) for d in
             db().tracks.find({"_id": {"$in": track_ids}}, {k: 1 for k in TRACK_FIELDS})}
    return [found.get(t) for t in track_ids]


def _features(doc: dict | None) -> dict | None:
    if not doc or doc.get("embedding") is None:
        return None
    return {"embedding": from_int8(doc["embedding"], doc["scale"]).tolist(),
            VERSION_KEY: doc.get("features_version", 0)}


def get_features(track_id: str) -> dict | None:
    return _features(db().tracks.find_one({"_id": track_id},
                                          {"embedding": 1, "scale": 1, "features_version": 1}))


def get_many_features(track_ids: list[str]) -> list[dict | None]:
    if not track_ids:
        return []
    found = {d["_id"]: _features(d) for d in
             db().tracks.find({"_id": {"$in": track_ids}},
                              {"embedding": 1, "scale": 1, "features_version": 1})}
    return [found.get(t) for t in track_ids]


def get_analyzed_at(track_id: str) -> datetime | None:
    doc = db().tracks.find_one({"_id": track_id}, {"analyzed_at": 1})
    return doc.get("analyzed_at") if doc else None


def tracks_since(stamp: datetime) -> list[tuple[str, np.ndarray]]:
    """(id, float32 vector) for every track analyzed strictly after `stamp`."""
    cursor = db().tracks.find({"analyzed_at": {"$gt": stamp}},
                              {"embedding": 1, "scale": 1}).sort("analyzed_at", 1)
    return [(d["_id"], from_int8(d["embedding"], d["scale"])) for d in cursor]


# corpus_ids() is called on every /recommend. Instead of shipping every id
# each time, remember the ids seen so far and the newest analyzed_at, and ask
# only for tracks newer than that. Tracks are only ever added.
_ids_cache: tuple[datetime, list[str]] | None = None


def corpus_ids() -> list[str]:
    """All analyzed track ids, sorted."""
    global _ids_cache
    with _lock:
        if _ids_cache is None:
            docs = list(db().tracks.find({"embedding": {"$exists": True}},
                                         {"analyzed_at": 1}))
            newest = max((d["analyzed_at"] for d in docs), default=datetime(1970, 1, 1))
            _ids_cache = (newest, sorted(d["_id"] for d in docs))
            return list(_ids_cache[1])
        newest, ids = _ids_cache
        fresh = list(db().tracks.find({"analyzed_at": {"$gt": newest}}, {"analyzed_at": 1}))
        if fresh:
            newest = max(d["analyzed_at"] for d in fresh)
            ids = sorted(set(ids) | {d["_id"] for d in fresh})
            _ids_cache = (newest, ids)
        return list(ids)


def corpus_size() -> int:
    return len(corpus_ids())


def base_matrix() -> tuple[list[str], np.ndarray]:
    """Every analyzed embedding as one float32 matrix, ids in row order."""
    ids, rows = [], []
    for d in db().tracks.find({"embedding": {"$exists": True}},
                              {"embedding": 1, "scale": 1}).sort("_id", 1):
        ids.append(d["_id"])
        rows.append(from_int8(d["embedding"], d["scale"]))
    if not rows:
        return [], np.empty((0, 1), dtype=np.float32)
    return ids, np.stack(rows).astype(np.float32)
