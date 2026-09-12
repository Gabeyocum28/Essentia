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


def _now() -> datetime:
    return datetime.utcnow()


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
    global _client, _indexes_ready, _ids_cache
    _client = None
    _indexes_ready = False
    _ids_cache = None


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
    """(id, float32 vector) for every track analyzed at or after `stamp`.

    Inclusive, not strict: MongoDB stores datetimes at millisecond
    resolution and the API and the embed worker are separate processes, so
    two tracks analyzed within the same millisecond (by the same process or
    different ones) can legitimately tie on `analyzed_at`. A strict `$gt`
    would then silently drop whichever of the tied rows a caller already
    has the watermark for. Using `$gte` means the boundary row (the one
    exactly at `stamp`) can come back again; callers that page through this
    by re-using the last-seen `analyzed_at` as the next `stamp` must dedupe
    by track id.
    """
    cursor = db().tracks.find({"analyzed_at": {"$gte": stamp}},
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
        # $gte, not $gt: millisecond-resolution timestamps mean a track
        # analyzed in another process within the same millisecond as the
        # watermark would otherwise be missed. Re-fetching the watermark
        # row itself is harmless -- the set union below dedupes it.
        fresh = list(db().tracks.find({"analyzed_at": {"$gte": newest}}, {"analyzed_at": 1}))
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


# ---- jobs (internal; not part of the HTTP contract) ----
#
# One document per pending unit of work. `state` moves queued -> running ->
# (deleted on success | failed). A queued or running document is the dedup
# guard the Redis sets used to be; requeue_stale() is the TTL.

_POLL_S = 0.5


def _attr_pair(seed_id: str, rec_id: str) -> str:
    return f"{seed_id}|{rec_id}"


def _enqueue(job_id: str, fields: dict) -> bool:
    ensure_indexes()
    existing = db().jobs.find_one({"_id": job_id}, {"state": 1})
    if existing and existing["state"] in ("queued", "running"):
        return False
    db().jobs.replace_one(
        {"_id": job_id},
        {"_id": job_id, **fields, "state": "queued", "claimed_at": None,
         "attempts": 0, "error": None, "created_at": _now()},
        upsert=True,
    )
    return True


def enqueue_embed(track_id: str) -> bool:
    """Queue a track for analysis. False if already queued or running."""
    return _enqueue(f"embed:{track_id}", {"kind": "embed", "track_id": track_id})


def enqueue_attribution(seed_id: str, rec_id: str) -> bool:
    return _enqueue(f"attr:{_attr_pair(seed_id, rec_id)}",
                    {"kind": "attr", "seed_id": seed_id, "rec_id": rec_id})


def _claim(kind: str) -> dict | None:
    return db().jobs.find_one_and_update(
        {"kind": kind, "state": "queued"},
        {"$set": {"state": "running", "claimed_at": _now()}, "$inc": {"attempts": 1}},
        sort=[("created_at", pymongo.ASCENDING)],
        return_document=ReturnDocument.AFTER,
    )


def dequeue_job(timeout: int = 5) -> tuple[str, str] | None:
    """Next job as (kind, payload), embed jobs first; polls up to `timeout` s."""
    deadline = time.monotonic() + timeout
    while True:
        job = _claim("embed")
        if job:
            return "embed", job["track_id"]
        job = _claim("attr")
        if job:
            return "attribution", _attr_pair(job["seed_id"], job["rec_id"])
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_S)


def dequeue_embed(timeout: int = 5) -> str | None:
    deadline = time.monotonic() + timeout
    while True:
        job = _claim("embed")
        if job:
            return job["track_id"]
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_S)


def clear_embed_marker(track_id: str) -> None:
    """The job finished (or the caller gave up): drop it so it can be requeued."""
    db().jobs.delete_one({"_id": f"embed:{track_id}"})


def clear_attribution_marker(seed_id: str, rec_id: str) -> None:
    db().jobs.delete_one({"_id": f"attr:{_attr_pair(seed_id, rec_id)}"})


def fail_job(job_id: str, error: str) -> None:
    db().jobs.update_one({"_id": job_id},
                         {"$set": {"state": "failed", "error": error[:500]}})


def requeue_stale(max_age_s: int = 600, max_attempts: int = 3) -> int:
    """Return jobs stuck in `running` to `queued`, or fail them past the cap.

    `attempts` is incremented by `_claim()` on every dequeue, so the cap
    counts total claims (including the one currently stuck), not retries.
    """
    cutoff = _now() - timedelta(seconds=max_age_s)
    stale = list(db().jobs.find({"state": "running", "claimed_at": {"$lt": cutoff}}))
    for job in stale:
        if job["attempts"] >= max_attempts:
            fail_job(job["_id"], f"gave up after {job['attempts']} attempts")
        else:
            db().jobs.update_one({"_id": job["_id"]},
                                 {"$set": {"state": "queued", "claimed_at": None}})
    return len(stale)


# ---- cache: attribution results and signed preview URLs ----

_PREVIEW_TTL_S = 600


def _cache_get(key: str):
    doc = db().cache.find_one({"_id": key})
    if not doc:
        return None
    if doc.get("expires_at") is not None and doc["expires_at"] <= _now():
        return None
    return doc["value"]


def _cache_put(key: str, value, ttl: int | None) -> None:
    ensure_indexes()
    expires = _now() + timedelta(seconds=ttl) if ttl else None
    db().cache.replace_one({"_id": key}, {"_id": key, "value": value, "expires_at": expires},
                           upsert=True)


def get_attribution(seed_id: str, rec_id: str) -> dict | None:
    return _cache_get(f"attr:{seed_id}:{rec_id}")


def put_attribution(seed_id: str, rec_id: str, payload: dict, ttl: int | None = None) -> None:
    """A ready result is kept forever; a failure gets a TTL so it is retried."""
    _cache_put(f"attr:{seed_id}:{rec_id}", payload, ttl)


def get_cached_preview(track_id: str) -> str | None:
    return _cache_get(f"preview:{track_id}")


def put_cached_preview(track_id: str, url: str, ttl: int = _PREVIEW_TTL_S) -> None:
    _cache_put(f"preview:{track_id}", url, ttl)
