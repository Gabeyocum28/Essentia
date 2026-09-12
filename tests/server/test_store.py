"""store.py: the Atlas document layout from the module docstring."""
from datetime import datetime, timedelta

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
    # preview_url is never persisted (it's a signed URL that expires in
    # minutes) -- get_track always answers "" for it; app.py re-signs.
    assert store.get_track("42") == {**TRACK, "preview_url": ""}


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
    assert store.get_many_tracks(["nope", "42", "also-nope"]) == [
        None, {**TRACK, "preview_url": ""}, None,
    ]


def test_get_many_features_preserves_order(fake_mongo):
    store.put_track(TRACK, FEATURES)
    got = store.get_many_features(["nope", "42"])
    assert got[0] is None
    assert np.allclose(got[1]["embedding"], FEATURES["embedding"], atol=0.01)


def test_corpus_ids_empty_when_no_tracks(fake_mongo):
    assert store.corpus_ids() == []


def test_put_track_meta_writes_track_only(fake_mongo):
    store.put_track_meta(TRACK)
    assert store.get_track("42") == {**TRACK, "preview_url": ""}
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
    # Inclusive of the boundary stamp: millisecond-resolution timestamps
    # mean "old" itself may legitimately come back when queried with its
    # own analyzed_at (see tracks_since's docstring) -- callers dedupe by
    # id. What must hold is that "new" is present, and that a stamp
    # strictly after both returns nothing.
    store.put_track({**TRACK, "track_id": "old"}, {"embedding": [1.0]})
    stamp = store.get_analyzed_at("old")
    store.put_track({**TRACK, "track_id": "new"}, {"embedding": [2.0]})
    got = [tid for tid, _ in store.tracks_since(stamp)]
    assert "new" in got
    # A stamp built to be unambiguously later than both writes, rather than
    # a fresh `_now()` call racing "new" for the same millisecond (which
    # $gte would then legitimately include) -- deterministic without a
    # sleep. Naive, matching store.py's own naive-UTC timestamps: an aware
    # datetime compared against mongomock's stored naive values is not
    # reliable (mongomock's $gte mishandles aware-vs-naive).
    assert store.tracks_since(store._now() + timedelta(seconds=1)) == []



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


# ---- durable state + queue depth ----

def test_state_roundtrip_never_expires(fake_mongo):
    assert store.get_state("crawl") is None
    store.put_state("crawl", {"genre_index": 3})
    assert store.get_state("crawl") == {"genre_index": 3}
    assert fake_mongo.cache.find_one({"_id": "state:crawl"})["expires_at"] is None


def test_queued_count_counts_only_queued(fake_mongo):
    store.enqueue_embed("1")
    store.enqueue_embed("2")
    store.dequeue_embed(timeout=0)
    assert store.queued_count() == 1


def test_failed_ids(fake_mongo):
    store.enqueue_embed("1")
    store.enqueue_embed("2")
    store.enqueue_embed("3")
    store.fail_job("embed:1", "boom")
    store.fail_job("embed:2", "boom")
    # "3" stays queued, "4" was never enqueued at all.
    assert store.failed_ids(["1", "2", "3", "4"]) == {"1", "2"}
    assert store.failed_ids([]) == set()
