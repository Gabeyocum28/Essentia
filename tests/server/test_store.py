"""store.py: the Atlas document layout from the module docstring."""
from datetime import datetime, timedelta

import numpy as np
import pymongo
import pytest

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


# ---- dedupe ----

def test_put_stores_the_dedupe_key(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert fake_mongo.tracks.find_one({"_id": "42"})["dedupe_key"] == \
        "miles davis|blue in green"
    # ... and it never leaks into a contract response.
    assert "dedupe_key" not in store.get_track("42")


def test_put_track_meta_stores_the_dedupe_key(fake_mongo):
    store.put_track_meta({**TRACK, "track_id": "43",
                          "title": "Blue in Green (2005 Remaster)"})
    assert fake_mongo.tracks.find_one({"_id": "43"})["dedupe_key"] == \
        "miles davis|blue in green"


def test_existing_keys_reports_only_keys_already_stored(fake_mongo):
    store.put_track(TRACK, FEATURES)
    got = store.existing_keys(["miles davis|blue in green", "miles davis|so what"])
    assert got == {"miles davis|blue in green"}
    assert store.existing_keys([]) == set()


def test_existing_keys_ignores_retired_duplicates(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.mark_duplicate("42", "primary")
    assert store.existing_keys(["miles davis|blue in green"]) == set()


def test_marked_duplicate_leaves_the_live_corpus(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.put_track({**TRACK, "track_id": "43",
                     "title": "Blue in Green (2005 Remaster)"}, FEATURES)
    stamp = store.get_analyzed_at("42")
    assert sorted(store.corpus_ids()) == ["42", "43"]

    store.mark_duplicate("43", "42")

    assert store.corpus_ids() == ["42"]
    assert store.base_matrix()[0] == ["42"]
    assert [tid for tid, _ in store.tracks_since(stamp)] == ["42"]
    assert fake_mongo.tracks.find_one({"_id": "43"})["duplicate_of"] == "42"


def test_retired_duplicate_can_still_be_seeded(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.mark_duplicate("42", "primary")
    # An explicit seed of a retired id still ranks: only the corpus
    # ENUMERATION drops it.
    assert store.get_features("42") is not None
    assert store.get_many_features(["42"])[0] is not None
    assert store.get_track("42") is not None


def test_ensure_indexes_covers_the_dedupe_key(fake_mongo):
    store.ensure_indexes()
    indexed = {entry[0] if isinstance(entry, (list, tuple)) else entry
               for info in fake_mongo.tracks.index_information().values()
               for entry in info["key"]}
    assert {"analyzed_at", "dedupe_key"} <= indexed


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


def test_data_size_bytes_falls_back_when_dbstats_is_unsupported(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.data_size_bytes() >= 1


def test_data_size_bytes_uses_own_dbstats_when_the_cluster_cannot_be_listed(
        fake_mongo, monkeypatch):
    """mongomock lists no databases, so this exercises the middle fallback:
    dbStats on the database we are connected to."""
    monkeypatch.setattr(fake_mongo, "command", lambda name: {"dataSize": 1000, "indexSize": 24})
    assert store.data_size_bytes() == 1024


class _FakeDatabase:
    def __init__(self, stats):
        self._stats = stats

    def command(self, name):
        assert name == "dbStats"
        return self._stats


class _FakeClient:
    def __init__(self, databases):
        self._databases = databases

    def list_database_names(self):
        return list(self._databases)

    def __getitem__(self, name):
        return self._databases[name]


def test_data_size_bytes_sums_every_user_database(monkeypatch):
    """Atlas bills the cluster, not one database: a leftover database beside
    `essentia` eats the same 512 MB, so the cap has to see it."""
    databases = {
        "essentia": _FakeDatabase({"dataSize": 1000, "indexSize": 24}),
        "leftover": _FakeDatabase({"dataSize": 500, "indexSize": 0}),
        "admin": _FakeDatabase({"dataSize": 10 ** 9, "indexSize": 10 ** 9}),
        "local": _FakeDatabase({"dataSize": 10 ** 9, "indexSize": 0}),
        "config": _FakeDatabase({"dataSize": 10 ** 9, "indexSize": 0}),
    }
    own = databases["essentia"]
    own.client = _FakeClient(databases)
    monkeypatch.setattr(store, "db", lambda: own)
    assert store.data_size_bytes() == 1524


def test_data_size_bytes_falls_back_when_the_cluster_listing_fails(monkeypatch):
    class _AngryClient:
        def list_database_names(self):
            raise pymongo.errors.OperationFailure("not authorized")

    own = _FakeDatabase({"dataSize": 7, "indexSize": 1})
    own.client = _AngryClient()
    monkeypatch.setattr(store, "db", lambda: own)
    assert store.data_size_bytes() == 8


# ---- feel: eleven probabilities per track, optional on the way in ----

FEEL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.95]


def test_put_track_stores_the_feel_vector(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "feel": FEEL})
    doc = fake_mongo.tracks.find_one({"_id": "42"})
    assert doc["feel"] == pytest.approx(FEEL, abs=1e-4)


def test_stored_feel_is_rounded_to_four_places(fake_mongo):
    """Eleven full doubles a row is the kind of thing that quietly doubles a
    free-tier cluster; four places is far finer than the heads resolve."""
    store.put_track(TRACK, {**FEATURES, "feel": [0.123456789] * 11})
    assert fake_mongo.tracks.find_one({"_id": "42"})["feel"] == [0.1235] * 11


def test_put_track_without_feel_writes_no_field(fake_mongo):
    """Rows analyzed before the heads shipped, and the corpus the backfill
    has not reached yet: the key is absent, not null."""
    store.put_track(TRACK, FEATURES)
    assert "feel" not in fake_mongo.tracks.find_one({"_id": "42"})


def test_put_track_without_feel_leaves_an_existing_vector_alone(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "feel": FEEL})
    store.put_track(TRACK, FEATURES)
    assert fake_mongo.tracks.find_one({"_id": "42"})["feel"] is not None


def test_get_features_returns_the_feel_vector(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "feel": FEEL})
    got = store.get_features("42")
    assert set(got) == {"embedding", "feel", "_features_version"}
    assert got["feel"] == pytest.approx(FEEL, abs=1e-4)


def test_get_features_omits_feel_when_the_row_has_none(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert "feel" not in store.get_features("42")


def test_get_many_features_carries_feel_per_row(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "feel": FEEL})
    store.put_track({**TRACK, "track_id": "43"}, FEATURES)
    got = store.get_many_features(["42", "43", "nope"])
    assert got[0]["feel"] == pytest.approx(FEEL, abs=1e-4)
    assert "feel" not in got[1]
    assert got[2] is None


def test_feel_accepts_a_numpy_vector(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "feel": np.asarray(FEEL, dtype=np.float32)})
    assert store.get_features("42")["feel"] == pytest.approx(FEEL, abs=1e-4)
