# Atlas Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Redis and the file snapshot with MongoDB Atlas as the only data store, behind the existing `store` function names, so the API and worker run against Atlas with no Mac-side processes.

**Architecture:** `server/store.py` becomes a pymongo implementation of the same functions `app.py`, `corpus/ingest.py`, and `scripts/embed_worker.py` already call. Three collections: `tracks` (metadata plus int8 embedding), `jobs` (the embed and attribution queues with claim timestamps), `cache` (signed preview URLs and attribution results, TTL-indexed). The recommender keeps its in-process matrix; the store hands it the whole dequantized matrix once, then only tracks analyzed after a watermark. Tests swap the real client for `mongomock`.

**Tech Stack:** Python 3.11, pymongo 4.x, mongomock 4.x (tests), numpy, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md`, section 5 (and section 8 for error handling).

## Global Constraints

- Python `>=3.11,<3.12`. `pyproject.toml` runtime deps become `fastapi`, `uvicorn`, `numpy`, `pymongo>=4.6`; `redis` is removed. Dev extra gains `mongomock>=4.3`.
- Every existing `store` function keeps its name and call signature: `put_track`, `put_track_meta`, `get_track`, `get_many_tracks`, `get_features`, `get_many_features`, `corpus_size`, `corpus_ids`, `enqueue_embed`, `dequeue_embed`, `clear_embed_marker`, `get_attribution`, `put_attribution`, `enqueue_attribution`, `dequeue_job`, `clear_attribution_marker`, `get_cached_preview`, `put_cached_preview`. Two additions: `base_matrix()` and `tracks_since(ts)`.
- Embeddings are stored as int8 bytes plus a float scale via `analysis.quantize.to_int8` / `from_int8`. `get_features` returns `{"embedding": list[float], "_features_version": int}` (dequantized), never the raw bytes.
- Env: `MONGODB_URI` (required at runtime, no default), `MONGODB_DB` (default `essentia`). No `REDIS_URL`, no `CORPUS_SNAPSHOT` anywhere after this plan.
- Collections and document shapes exactly as in spec section 5.1: `tracks` (`_id`, title, artist, album, artwork_url, embedding BinData, scale, features_version, analyzed_at), `jobs` (`_id`, kind, track_id / seed_id + rec_id, state, claimed_at, attempts, error, created_at), `cache` (`_id`, value, expires_at).
- Branch `atlas-store` off `main`. Run `python3 -m pytest -q` before every commit (parity test stays opt-in and is untouched).
- Do not touch `ios/`, `web/`, or `deploy/` in this plan.

## Facts about the current code (do not re-derive)

- `app.py` calls `store.*` at 26 sites; the only `snapshot` uses are in `_cold_matrix` (`snapshot.active()`, `snapshot.KEY`, `snapshot.base_matrix()`) and the import. `_MATRIX_CACHE` is keyed by feature key and appends rows for ids in `corpus` that it has not seen (`_build`). It treats `corpus_ids()` as the growth signal, called on every `/recommend` and `/viz/*`.
- `corpus/ingest.py` writes `store.put_track(track, {**features, "_features_version": FEATURES_VERSION})` and reads `store.get_features(id).get("_features_version")` to decide re-analysis.
- `scripts/embed_worker.py` uses `store.dequeue_job`, `clear_embed_marker`, `clear_attribution_marker`, `put_track`, `put_attribution`, `get_features`, `get_track`. Its job payloads are `("embed", track_id)` and `("attribution", "seed|rec")`. It is rewritten in sub-project 3; it must keep working here.
- Tests: `tests/server/conftest.py` defines `FakeRedis` and a `fake_redis` fixture used by `test_app.py` (17 uses), `test_viz.py` (11), `test_embed_worker.py` (12), `test_store.py`. Eight assertions inspect Redis internals (`fake_redis.lists[...]`, `fake_redis.mget_calls`). Test feature dicts use short vectors like `{"embedding": [0.1, 0.2], "genre": [...]}` and compare with `==`.
- `scripts/push_tracks.py`, `export_snapshot.py`, `export_corpus.py` are Redis-only (RESP piping, RDB moves, snapshot export). `tests/server/test_push_tracks.py` tests the first.
- Neither `pymongo` nor `mongomock` is installed on the Mac yet.

## File structure

- Rewrite `src/music_recommendations/server/store.py` (~250 lines: connection, tracks, jobs, cache).
- Delete `src/music_recommendations/server/snapshot.py`.
- Modify `src/music_recommendations/server/app.py` (`_cold_matrix`, `/seed` error handling, imports).
- Rewrite `tests/server/conftest.py` fixture (`fake_mongo`); update `test_store.py`, `test_app.py`, `test_viz.py`, `test_embed_worker.py`.
- Delete `scripts/push_tracks.py`, `scripts/export_snapshot.py`, `scripts/export_corpus.py`, `tests/server/test_push_tracks.py`.
- Create `scripts/atlas_check.py` (live connectivity check, run manually).
- Modify `pyproject.toml`, `uv.lock`, `README.md`, `.gitignore`, `src/music_recommendations/server/CLAUDE.md`.

---

### Task 1: Dependencies, connection, and the tracks collection

**Files:**
- Modify: `pyproject.toml`
- Rewrite: `src/music_recommendations/server/store.py`
- Rewrite: `tests/server/conftest.py`
- Rewrite: `tests/server/test_store.py`

**Interfaces:**
- Consumes: `analysis.quantize.to_int8(vec) -> (bytes, float)`, `from_int8(data, scale) -> np.ndarray`.
- Produces: `store.db() -> pymongo.database.Database`; `store.reset()` (clears the per-process client and caches; tests call it); `store.put_track(track, features)`, `put_track_meta(track)`, `get_track(id)`, `get_many_tracks(ids)`, `get_features(id)`, `get_many_features(ids)`, `corpus_ids() -> list[str]` (sorted), `corpus_size() -> int`, `base_matrix() -> (list[str], np.ndarray float32)`, `tracks_since(ts: datetime) -> list[tuple[str, np.ndarray]]`. The `fake_mongo` pytest fixture.

- [ ] **Step 1: Branch and dependencies**

```bash
git checkout main && git pull -q && git checkout -b atlas-store
```

In `pyproject.toml` replace the `dependencies` list with:

```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pymongo>=4.6",
    "numpy>=1.26",
]
```

and the dev extra with `dev = ["pytest>=8.0", "httpx>=0.27", "mongomock>=4.3"]`. Then:

```bash
python3 -m pip install -q -e ".[dev]" && python3 -c "import pymongo, mongomock; print(pymongo.__version__, mongomock.__version__)"
```

- [ ] **Step 2: Write the fixture**

Replace `tests/server/conftest.py` with:

```python
"""Shared fakes: an in-memory MongoDB (mongomock) standing in for Atlas."""
import mongomock
import pytest

from music_recommendations.server import store


@pytest.fixture(autouse=True)
def clear_matrix_cache():
    """/recommend caches the corpus matrix in module state; tests must not share it."""
    from music_recommendations.server import app, viz

    app._MATRIX_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    viz.clear_geometry_cache()
    yield
    app._MATRIX_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    viz.clear_geometry_cache()


@pytest.fixture
def fake_mongo(monkeypatch):
    """A fresh in-memory database wired into store.db() for one test."""
    database = mongomock.MongoClient().get_database("essentia_test")
    store.reset()
    monkeypatch.setattr(store, "db", lambda: database)
    yield database
    store.reset()
```

- [ ] **Step 3: Write the failing track tests**

Replace `tests/server/test_store.py` with (queue and cache tests are added in Task 2):

```python
"""store.py: the Atlas document layout from the module docstring."""
from datetime import datetime, timezone

import numpy as np

from music_recommendations.analysis import FEATURES_VERSION
from music_recommendations.server import store

TRACK = {
    "track_id": "42",
    "title": "Blue in Green",
    "artist": "Miles Davis",
    "album": "Kind of Blue",
    "artwork_url": "http://x/a.jpg",
    "preview_url": "http://x/p.mp3",
}
FEATURES = {"embedding": [0.1, 0.2, -0.3], "_features_version": FEATURES_VERSION}


def test_put_then_get_track_roundtrips(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.get_track("42") == TRACK


def test_put_then_get_features_dequantizes(fake_mongo):
    store.put_track(TRACK, FEATURES)
    got = store.get_features("42")
    assert set(got) == {"embedding", "_features_version"}
    assert got["_features_version"] == FEATURES_VERSION
    assert np.allclose(got["embedding"], FEATURES["embedding"], atol=0.01)


def test_track_document_shape(fake_mongo):
    store.put_track(TRACK, FEATURES)
    doc = fake_mongo.tracks.find_one({"_id": "42"})
    assert doc["title"] == "Blue in Green"
    assert "preview_url" not in doc            # signed URLs are never stored
    assert isinstance(doc["embedding"], bytes) and len(doc["embedding"]) == 3
    assert doc["scale"] > 0
    assert doc["features_version"] == FEATURES_VERSION
    assert isinstance(doc["analyzed_at"], datetime)


def test_put_registers_corpus_id(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.corpus_ids() == ["42"]
    assert store.corpus_size() == 1


def test_get_missing_track_returns_none(fake_mongo):
    assert store.get_track("nope") is None
    assert store.get_features("nope") is None


def test_get_many_tracks_preserves_requested_order_and_missing_values(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.get_many_tracks(["nope", "42", "also-nope"]) == [None, TRACK, None]


def test_get_many_features_preserves_order(fake_mongo):
    store.put_track(TRACK, FEATURES)
    got = store.get_many_features(["nope", "42"])
    assert got[0] is None
    assert np.allclose(got[1]["embedding"], FEATURES["embedding"], atol=0.01)


def test_corpus_ids_empty_when_no_tracks(fake_mongo):
    assert store.corpus_ids() == []


def test_put_track_meta_writes_track_only(fake_mongo):
    store.put_track_meta(TRACK)
    assert store.get_track("42") == TRACK
    assert store.get_features("42") is None
    assert "42" not in store.corpus_ids()


def test_put_track_meta_does_not_erase_an_embedding(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.put_track_meta({**TRACK, "album": "Renamed"})
    assert store.get_track("42")["album"] == "Renamed"
    assert store.get_features("42") is not None


def test_corpus_ids_is_sorted_and_incremental(fake_mongo):
    for tid in ("b", "a", "c"):
        store.put_track({**TRACK, "track_id": tid}, FEATURES)
    assert store.corpus_ids() == ["a", "b", "c"]
    store.put_track({**TRACK, "track_id": "0"}, FEATURES)
    assert store.corpus_ids() == ["0", "a", "b", "c"]   # watermark picked up the new one


def test_base_matrix_rows_match_ids(fake_mongo):
    store.put_track({**TRACK, "track_id": "a"}, {"embedding": [1.0, 0.0]})
    store.put_track({**TRACK, "track_id": "b"}, {"embedding": [0.0, 1.0]})
    store.put_track_meta({**TRACK, "track_id": "meta-only"})
    ids, matrix = store.base_matrix()
    assert ids == ["a", "b"]
    assert matrix.dtype == np.float32 and matrix.shape == (2, 2)
    assert np.allclose(matrix, [[1.0, 0.0], [0.0, 1.0]], atol=0.01)


def test_tracks_since_returns_only_newer(fake_mongo):
    store.put_track({**TRACK, "track_id": "old"}, {"embedding": [1.0]})
    stamp = store.get_analyzed_at("old")
    store.put_track({**TRACK, "track_id": "new"}, {"embedding": [2.0]})
    got = store.tracks_since(stamp)
    assert [tid for tid, _ in got] == ["new"]
    assert store.tracks_since(datetime.now(timezone.utc)) == []
```

- [ ] **Step 4: Run to verify they fail**

Run: `python3 -m pytest tests/server/test_store.py -q`
Expected: errors (`fixture 'fake_mongo' not found` before Step 2 lands; `AttributeError` for the new functions after).

- [ ] **Step 5: Rewrite store.py (tracks half)**

Replace `src/music_recommendations/server/store.py` with:

```python
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
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

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
    return datetime.now(timezone.utc)


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
            newest = max((d["analyzed_at"] for d in docs), default=datetime(1970, 1, 1, tzinfo=timezone.utc))
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
```

Note: mongomock stores `datetime` values naive (it drops tzinfo) in some versions. If `test_tracks_since_returns_only_newer` fails on a naive/aware comparison, normalise in `tracks_since` and `corpus_ids` by converting `stamp`/`newest` with `.replace(tzinfo=None)` when `db().client` is a mongomock client is NOT acceptable; instead store and compare naive UTC everywhere: change `_now()` to `datetime.utcnow()` and the 1970 sentinel to naive, and have `get_analyzed_at` return what is stored. Pick naive UTC throughout and say so in the docstring.

- [ ] **Step 6: Run to verify they pass**

Run: `python3 -m pytest tests/server/test_store.py -q`
Expected: 13 PASS. The rest of `tests/server` still imports `fake_redis` and fails; that is fixed in Tasks 2 and 3.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/music_recommendations/server/store.py tests/server/conftest.py tests/server/test_store.py
git commit -m "feat(store): MongoDB tracks collection with int8 embeddings replaces Redis track keys"
```

---

### Task 2: Jobs and cache collections

**Files:**
- Modify: `src/music_recommendations/server/store.py`
- Modify: `tests/server/test_store.py`

**Interfaces:**
- Produces: `enqueue_embed(id) -> bool`, `dequeue_embed(timeout=5) -> str | None`, `clear_embed_marker(id)`, `enqueue_attribution(seed, rec) -> bool`, `dequeue_job(timeout=5) -> ("embed", id) | ("attribution", "seed|rec") | None`, `clear_attribution_marker(seed, rec)`, `fail_job(job_id, error)`, `requeue_stale(max_age_s=600, max_attempts=3) -> int`, `get_attribution(seed, rec)`, `put_attribution(seed, rec, payload, ttl=None)`, `get_cached_preview(id)`, `put_cached_preview(id, url, ttl=600)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/server/test_store.py`:

```python
# ---- jobs ----

def test_enqueue_embed_creates_queued_job_and_guards(fake_mongo):
    assert store.enqueue_embed("9") is True
    assert store.enqueue_embed("9") is False
    job = fake_mongo.jobs.find_one({"_id": "embed:9"})
    assert job["kind"] == "embed" and job["state"] == "queued" and job["track_id"] == "9"
    assert job["attempts"] == 0 and job["claimed_at"] is None


def test_dequeue_embed_claims_oldest_then_none(fake_mongo):
    store.enqueue_embed("1")
    store.enqueue_embed("2")
    assert store.dequeue_embed(timeout=0) == "1"
    assert fake_mongo.jobs.find_one({"_id": "embed:1"})["state"] == "running"
    assert store.dequeue_embed(timeout=0) == "2"
    assert store.dequeue_embed(timeout=0) is None


def test_clear_embed_marker_deletes_job_so_it_can_requeue(fake_mongo):
    store.enqueue_embed("9")
    store.dequeue_embed(timeout=0)
    store.clear_embed_marker("9")
    assert fake_mongo.jobs.find_one({"_id": "embed:9"}) is None
    assert store.enqueue_embed("9") is True


def test_dequeue_job_prefers_embed_over_attribution(fake_mongo):
    store.enqueue_attribution("s", "r")
    store.enqueue_embed("9")
    assert store.dequeue_job(timeout=0) == ("embed", "9")
    assert store.dequeue_job(timeout=0) == ("attribution", "s|r")
    assert store.dequeue_job(timeout=0) is None


def test_enqueue_attribution_guards_pending_pair(fake_mongo):
    assert store.enqueue_attribution("s", "r") is True
    assert store.enqueue_attribution("s", "r") is False
    store.dequeue_job(timeout=0)
    store.clear_attribution_marker("s", "r")
    assert store.enqueue_attribution("s", "r") is True


def test_fail_job_records_error_and_blocks_requeue_until_cleared(fake_mongo):
    store.enqueue_embed("9")
    store.dequeue_embed(timeout=0)
    store.fail_job("embed:9", "boom")
    job = fake_mongo.jobs.find_one({"_id": "embed:9"})
    assert job["state"] == "failed" and job["error"] == "boom"
    assert store.dequeue_embed(timeout=0) is None


def test_requeue_stale_returns_running_jobs_to_queue_then_fails_them(fake_mongo):
    from datetime import timedelta
    store.enqueue_embed("9")
    store.dequeue_embed(timeout=0)
    old = store._now() - timedelta(seconds=700)
    fake_mongo.jobs.update_one({"_id": "embed:9"}, {"$set": {"claimed_at": old, "attempts": 2}})
    assert store.requeue_stale(max_age_s=600, max_attempts=3) == 1
    assert fake_mongo.jobs.find_one({"_id": "embed:9"})["state"] == "queued"
    store.dequeue_embed(timeout=0)
    fake_mongo.jobs.update_one({"_id": "embed:9"}, {"$set": {"claimed_at": old}})
    assert store.requeue_stale(max_age_s=600, max_attempts=3) == 1
    assert fake_mongo.jobs.find_one({"_id": "embed:9"})["state"] == "failed"


# ---- cache ----

def test_attribution_cache_roundtrip_and_ttl_field(fake_mongo):
    assert store.get_attribution("s", "r") is None
    store.put_attribution("s", "r", {"status": "ready", "bands": [1, 2]})
    assert store.get_attribution("s", "r") == {"status": "ready", "bands": [1, 2]}
    assert fake_mongo.cache.find_one({"_id": "attr:s:r"})["expires_at"] is None
    store.put_attribution("s", "r", {"status": "failed"}, ttl=60)
    assert fake_mongo.cache.find_one({"_id": "attr:s:r"})["expires_at"] is not None


def test_preview_cache_roundtrip_and_expiry(fake_mongo):
    from datetime import timedelta
    assert store.get_cached_preview("42") is None
    store.put_cached_preview("42", "http://signed", ttl=600)
    assert store.get_cached_preview("42") == "http://signed"
    fake_mongo.cache.update_one({"_id": "preview:42"},
                                {"$set": {"expires_at": store._now() - timedelta(seconds=1)}})
    assert store.get_cached_preview("42") is None   # expired entries are ignored even before TTL reaps them
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/server/test_store.py -q`
Expected: the 9 new tests FAIL with `AttributeError`.

- [ ] **Step 3: Implement jobs and cache**

Append to `src/music_recommendations/server/store.py`:

```python
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
    """Return jobs stuck in `running` to `queued`, or fail them past the cap."""
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/server/test_store.py -q`
Expected: 22 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/music_recommendations/server/store.py tests/server/test_store.py
git commit -m "feat(store): jobs and cache collections replace the Redis queues"
```

---

### Task 3: App and worker on the new store; delete the snapshot

**Files:**
- Modify: `src/music_recommendations/server/app.py`
- Delete: `src/music_recommendations/server/snapshot.py`
- Modify: `tests/server/test_app.py`, `tests/server/test_viz.py`, `tests/server/test_embed_worker.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2. `frontend.DecodeError` from `analysis.frontend`.

- [ ] **Step 1: Migrate the tests (fixture rename and internals)**

Mechanical edits across the three test files:
1. `fake_redis` → `fake_mongo` everywhere (fixture parameter names and uses).
2. Replace each `assert fake_redis.lists["embed:queue"] == [tid]` with `assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"`; replace `assert fake_redis.lists["attr:queue"] == [f"{seed}|{rec}"]` with `assert fake_mongo.jobs.find_one({"_id": f"attr:{seed}|{rec}"})["state"] == "queued"`.
3. In `test_viz.py` around line 219, the assertion on `fake_redis.mget_calls` verified that tracks are read in one batched call. Replace it with a monkeypatch counter: `calls = []; monkeypatch.setattr(app.store, "get_many_tracks", lambda ids, _f=store.get_many_tracks: (calls.append(list(ids)), _f(ids))[1])` and assert `len(calls) == 1` (keep whatever the original assertion checked about which ids were requested).
4. Feature dicts: the store now round-trips only `embedding` (plus `_features_version`) through int8. Every test that does `store.put_track(TRACK, FEATURES)` and later compares `store.get_features(...) == FEATURES` must instead assert `np.allclose(got["embedding"], FEATURES["embedding"], atol=0.01)`. Remove `genre`/`groove` keys from test feature dicts. Where a test feeds vectors to `/recommend` and asserts an ordering, keep the values; int8 quantization of a handful of 2-3 element vectors preserves cosine ordering unless two vectors are near-identical (check any that differ by less than 2% and spread them out).
5. `test_embed_worker.py` loads `scripts/embed_worker.py`; it uses `store.enqueue_embed`, `dequeue_job`, etc. and needs only the fixture rename plus item 4.

Run: `python3 -m pytest tests/server -q`
Expected: `test_app.py`/`test_viz.py` seed tests that hit `_cold_matrix` still pass (the Redis path of `_cold_matrix` uses `get_many_features`, which works). Any failure now should be in the assertions you just rewrote; fix those before moving on. Do not commit yet.

- [ ] **Step 2: Write the failing app tests**

Append to `tests/server/test_app.py` (adapt the client/fixture names to what the file already uses; `client` is a `TestClient(app.app)`):

```python
def test_cold_matrix_uses_base_matrix(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod, store
    for tid, vec in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        store.put_track({**TRACK, "track_id": tid}, {"embedding": vec})
    calls = []
    real = store.get_many_features
    monkeypatch.setattr(store, "get_many_features", lambda ids: (calls.append(ids), real(ids))[1])
    ids, matrix = app_mod._cold_matrix(("a", "b"), "embedding")
    assert ids == ["a", "b"] and matrix.shape == (2, 2)
    assert calls == []          # one matrix read, no per-track fetches


def test_seed_returns_502_when_analysis_fails(fake_mongo, monkeypatch, tmp_path):
    from music_recommendations.analysis import frontend
    from music_recommendations.server import app as app_mod
    mp3 = tmp_path / "p.mp3"; mp3.write_bytes(b"x")
    monkeypatch.setattr(app_mod, "_fetch_preview_audio", lambda tid, t: mp3)
    monkeypatch.setattr(app_mod.deezer, "get_track", lambda t: dict(TRACK))
    def boom(path): raise frontend.DecodeError("ffmpeg: bad file")
    monkeypatch.setattr(app_mod, "analyze_track", boom)
    r = client.post("/seed", json={"track_id": "42"})
    assert r.status_code == 502
    assert r.json()["detail"] == "analysis failed"
```

Run: `python3 -m pytest tests/server/test_app.py -q -k "cold_matrix or 502"`
Expected: both FAIL.

- [ ] **Step 3: Change app.py**

1. Remove `from music_recommendations.server import snapshot` (and any `snapshot` mention). Add `from music_recommendations.analysis import frontend`.
2. Replace `_cold_matrix` with:

```python
def _cold_matrix(corpus: tuple[str, ...],
                 feature_key: str) -> tuple[list[str], np.ndarray]:
    """Every row, for a cache that has nothing yet.

    The store hands back the whole embedding matrix in one query instead of
    one fetch per track. Anything in `corpus` that the matrix does not yet
    contain (analyzed between the two reads) is appended the usual way.
    """
    if feature_key == "embedding":
        ids, matrix = store.base_matrix()
        known_rows = set(ids)
        extra = [t for t in corpus if t not in known_rows]
        if extra:
            more_ids, rows = _rows_for(extra, feature_key)
            if rows:
                ids = ids + more_ids
                matrix = np.vstack([matrix, np.stack(rows).astype(matrix.dtype)])
        return ids, matrix
    ids, rows = _rows_for(list(corpus), feature_key)
    return ids, (np.stack(rows) if rows else np.empty((0, 1)))
```

3. In `/seed`, replace the `except (NotImplementedError, ImportError):` block's neighbourhood so that:

```python
    try:
        features = _to_plain(analyze_track(mp3))
        _safe(store.put_track, track, features)
    except (NotImplementedError, ImportError):
        # No TensorFlow on this host: hand the job to the worker and wait.
        return _seed_via_worker(req.track_id, track)
    except (frontend.DecodeError, ValueError) as exc:
        # The preview itself is bad (undecodable, too short). Not transient,
        # so do not queue it; tell the client (spec §8).
        raise HTTPException(502, "analysis failed") from exc
    finally:
        mp3.unlink(missing_ok=True)
```

4. Update comments that mention "Redis" or "the Mac embed worker" in `app.py` to say "the store" / "the worker" (comments only; grep `Redis|Mac` in the file).
5. `git rm src/music_recommendations/server/snapshot.py`.

- [ ] **Step 4: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: all PASS. Then grep for stragglers: `grep -rn "snapshot\|REDIS_URL\|CORPUS_SNAPSHOT\|redis" src tests --include='*.py'` must return only `scripts`-free, comment-free results (i.e. nothing under `src/` and `tests/`).

- [ ] **Step 5: Commit**

```bash
git add -A src/music_recommendations/server tests/server
git commit -m "feat(server): app and worker on the Atlas store; snapshot mode removed; 502 on bad previews"
```

---

### Task 4: Scripts, docs, lock file

**Files:**
- Delete: `scripts/push_tracks.py`, `scripts/export_snapshot.py`, `scripts/export_corpus.py`, `tests/server/test_push_tracks.py`
- Modify: `scripts/embed_worker.py` (docstring), `scripts/analyze_corpus.py` (docstring and the "is redis-server running?" message), `src/music_recommendations/corpus/ingest.py` (docstring), `src/music_recommendations/server/CLAUDE.md`, `README.md`, `.gitignore`, `uv.lock`

- [ ] **Step 1: Delete the Redis-only scripts and their test**

```bash
git rm scripts/push_tracks.py scripts/export_snapshot.py scripts/export_corpus.py tests/server/test_push_tracks.py
```

- [ ] **Step 2: Docstrings and docs**

- `scripts/embed_worker.py` docstring: replace the first paragraph with `"""Pop embed jobs from the Atlas jobs collection, analyze, write back.` and the run line with `    MONGODB_URI=... python3 scripts/embed_worker.py`. Remove the sentence about the ARM VM not importing essentia.
- `scripts/analyze_corpus.py`: first line `"""Analyze crawled candidates into Atlas (needs MONGODB_URI).`; delete the Machine A / Machine B Redis paragraph; change the exit message to `(is MONGODB_URI set and reachable?)`.
- `src/music_recommendations/corpus/ingest.py` docstring: `-> write Redis` becomes `-> write the store`; any "Redis" in comments becomes "the store".
- `src/music_recommendations/server/CLAUDE.md`: replace any Redis key description with a pointer: `Storage is MongoDB Atlas; see store.py's docstring for the three collections. Set MONGODB_URI (and optionally MONGODB_DB) to run.`
- `README.md`: replace the `redis-server &  # storage` line with `export MONGODB_URI="mongodb+srv://..."   # Atlas connection string, never committed`; remove any mention of snapshots or `CORPUS_SNAPSHOT`; in the layout section describe `server/` as "FastAPI backend on MongoDB Atlas".
- `.gitignore`: delete the `dump.rdb`, `dump*.rdb`, `corpus_snapshot/` entries and their comments; keep `*.npy`.

- [ ] **Step 3: Lock file, suite, commit**

```bash
uv lock && grep -c '"redis"' uv.lock   # expect 0
python3 -m pytest -q
git add -A scripts tests src README.md .gitignore uv.lock
git commit -m "chore: drop Redis-only scripts and docs; Atlas is the only store"
```

---

### Task 5: Live Atlas check script

**Files:**
- Create: `scripts/atlas_check.py`
- Test: `tests/server/test_atlas_check.py`

**Interfaces:**
- Consumes: `store.ensure_indexes`, `put_track`, `get_features`, `corpus_size`, `enqueue_embed`, `dequeue_embed`, `clear_embed_marker`.

- [ ] **Step 1: Write the test (runs against mongomock)**

`tests/server/test_atlas_check.py`:

```python
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "atlas_check.py"
spec = importlib.util.spec_from_file_location("atlas_check", SCRIPT)
atlas_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas_check)


def test_check_round_trips_a_fixture_track_and_cleans_up(fake_mongo, capsys):
    assert atlas_check.run() == 0
    out = capsys.readouterr().out
    assert "tracks:" in out and "round-trip ok" in out
    assert fake_mongo.tracks.find_one({"_id": "atlas-check"}) is None
    assert fake_mongo.jobs.find_one({"_id": "embed:atlas-check"}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/server/test_atlas_check.py -q`
Expected: FAIL (file not found).

- [ ] **Step 3: Write the script**

`scripts/atlas_check.py`:

```python
"""Prove the Atlas connection works: indexes, a write, a read, a queue round trip.

    MONGODB_URI="mongodb+srv://..." python3 scripts/atlas_check.py

Writes one throwaway document under _id "atlas-check" and removes it again.
Exit code 0 on success, 1 on any failure (message on stderr).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.server import store  # noqa: E402

TRACK = {"track_id": "atlas-check", "title": "check", "artist": "check",
         "album": "check", "artwork_url": "", "preview_url": ""}


def run() -> int:
    try:
        store.ensure_indexes()
        before = store.corpus_size()
        vec = np.linspace(-1, 1, 1280, dtype=np.float32)
        store.put_track(TRACK, {"embedding": vec, "_features_version": 3})
        got = store.get_features("atlas-check")
        assert got and np.allclose(got["embedding"], vec, atol=0.01), "embedding mismatch"
        assert store.enqueue_embed("atlas-check") and store.dequeue_embed(timeout=0) == "atlas-check"
        store.clear_embed_marker("atlas-check")
        store.db().tracks.delete_one({"_id": "atlas-check"})
        store.reset()
        print(f"tracks: {before}  jobs: {store.db().jobs.count_documents({})}  round-trip ok")
        return 0
    except Exception as exc:  # noqa: BLE001 - this script's job is to report any failure
        print(f"atlas check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
```

- [ ] **Step 4: Run, commit, push, PR**

```bash
python3 -m pytest -q
git add scripts/atlas_check.py tests/server/test_atlas_check.py
git commit -m "feat(scripts): atlas_check.py proves the live connection"
git push -u origin atlas-store
gh pr create --base main --title "Atlas store: MongoDB replaces Redis and the file snapshot" --body "Implements sub-project 2 of docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md."
```

The live run (`MONGODB_URI=... python3 scripts/atlas_check.py`) is the user's step, since only they hold the connection string. It is the spec's gate for this sub-project together with the green suite.

---

## Self-review

- **Spec coverage.** §5.1 collections and indexes (Tasks 1-2), §5.2 store API with `base_matrix` and `tracks_since` and atomic claim + stale requeue (Tasks 1-2), Redis and snapshot deleted (Tasks 3-4), `MONGODB_URI` lazy client (Task 1), §5.3 mongomock tests with the job state machine and watermark (Tasks 1-2), §8 analysis-failure 502 (Task 3). The 30-second background top-up from §5.2 is replaced by a per-request watermark inside `corpus_ids()`, which is cheaper and needs no thread; the ledger should record this as a ruling. The "Atlas unreachable at startup → 503" line of §8 is not implemented: `app.py`'s existing `_safe` degradation (fixture fallback) is kept unchanged; sub-project 3 revisits it when the deployment shape is real.
- **Placeholders.** None.
- **Type consistency.** `get_features` returns `{"embedding": list, "_features_version": int}` (Task 1) and `ingest.py` reads `VERSION_KEY` from it unchanged; `dequeue_job` payloads match what `scripts/embed_worker.py` parses (`"seed|rec"`); `fake_mongo` is the fixture name in every test file; `store._now()` is used by tests in Task 2 and defined in Task 1.
