"""store.py: the Atlas document layout from the module docstring."""
from datetime import datetime, timedelta

import numpy as np
import pymongo
import pytest

from contract.features import RHYTHM_KEYS
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

# A bare-digit id means Deezer (corpus/sources/base.py), so every Track read
# back carries `source` even though nothing wrote one.
STORED = {**TRACK, "preview_url": "", "source": "deezer"}


def test_put_then_get_track_roundtrips(fake_mongo):
    store.put_track(TRACK, FEATURES)
    # preview_url is never persisted (it's a signed URL that expires in
    # minutes) -- get_track always answers "" for it; app.py re-signs.
    assert store.get_track("42") == STORED


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
        None, STORED, None,
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
    assert store.get_track("42") == STORED
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
    store.put_track({**TRACK, "track_id": "a"}, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track({**TRACK, "track_id": "b"}, {"embedding": [0.0, 1.0], "_features_version": FEATURES_VERSION})
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
    store.put_track({**TRACK, "track_id": "old"}, {"embedding": [1.0], "_features_version": FEATURES_VERSION})
    stamp = store.get_analyzed_at("old")
    store.put_track({**TRACK, "track_id": "new"}, {"embedding": [2.0], "_features_version": FEATURES_VERSION})
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


# ---- source and attribution (contract.features.TRACK_OPTIONAL_FIELDS) ----

JAMENDO = {
    "track_id": "jamendo:168",
    "title": "Sunrise",
    "artist": "Dee Yan-Key",
    "album": "Morning",
    "artwork_url": "http://x/a.jpg",
    "preview_url": "http://x/full.mp3",
    "source": "jamendo",
    "attribution": {"source": "jamendo",
                    "url": "https://www.jamendo.com/track/168/sunrise",
                    "license": "http://creativecommons.org/licenses/by-sa/3.0/"},
}


def test_attribution_round_trips_as_a_backlink(fake_mongo):
    store.put_track(JAMENDO, FEATURES)
    got = store.get_track("jamendo:168")
    assert got["source"] == "jamendo"
    assert got["attribution_url"] == JAMENDO["attribution"]["url"]
    # The licence deed is kept on the row but is not a contract field.
    assert "attribution" not in got
    assert "license" not in got


def test_the_stored_attribution_keeps_the_licence(fake_mongo):
    """Published narrow, stored whole: if the clients ever need the deed,
    the row already has it and no re-crawl is needed."""
    store.put_track(JAMENDO, FEATURES)
    doc = fake_mongo.tracks.find_one({"_id": "jamendo:168"})
    assert doc["attribution"] == JAMENDO["attribution"]


def test_a_bare_id_defaults_to_deezer_and_has_no_attribution(fake_mongo):
    store.put_track(TRACK, FEATURES)
    got = store.get_track("42")
    assert got["source"] == "deezer"
    assert "attribution_url" not in got


def test_source_is_read_off_a_namespaced_id_when_unstated(fake_mongo):
    store.put_track_meta({k: v for k, v in JAMENDO.items()
                          if k not in ("source", "attribution")})
    got = store.get_track("jamendo:168")
    assert got["source"] == "jamendo"
    assert "attribution_url" not in got


def test_get_many_tracks_carries_the_optional_fields(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.put_track(JAMENDO, FEATURES)
    bare, cc = store.get_many_tracks(["42", "jamendo:168"])
    assert "attribution_url" not in bare
    assert cc["attribution_url"] == JAMENDO["attribution"]["url"]


# ---- rhythm: the seven named numbers, optional the same way feel is ----

RHYTHM = {"tempo_bpm": 136.4, "beat_strength": 0.97, "loudness_lufs": -27.6,
          "loudness_range": 6.0, "key": 2, "mode": "minor",
          "key_strength": 0.73}


def test_put_track_stores_the_rhythm_dict(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "rhythm": RHYTHM})
    assert fake_mongo.tracks.find_one({"_id": "42"})["rhythm"] == RHYTHM


def test_stored_rhythm_is_narrowed_to_the_contract_keys(fake_mongo):
    """A future extractor must not silently widen every document."""
    store.put_track(TRACK, {**FEATURES, "rhythm": {**RHYTHM, "swing": 0.5}})
    assert set(fake_mongo.tracks.find_one({"_id": "42"})["rhythm"]) == set(RHYTHM_KEYS)


def test_put_track_without_rhythm_writes_no_field(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert "rhythm" not in fake_mongo.tracks.find_one({"_id": "42"})


def test_get_features_returns_the_rhythm_dict(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "rhythm": RHYTHM})
    got = store.get_features("42")
    assert got["rhythm"] == RHYTHM
    assert set(got) == {"embedding", "rhythm", "_features_version"}


def test_get_features_omits_rhythm_when_the_row_has_none(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert "rhythm" not in store.get_features("42")


def test_get_many_rhythm_projects_only_rhythm_in_request_order(fake_mongo):
    store.put_track(TRACK, {**FEATURES, "rhythm": RHYTHM})
    store.put_track({**TRACK, "track_id": "43"}, FEATURES)
    assert store.get_many_rhythm(["43", "42", "nope"]) == [None, RHYTHM, None]


def test_get_many_rhythm_of_nothing_is_nothing(fake_mongo):
    assert store.get_many_rhythm([]) == []


# ---- the corpus is version-4 only ----

def test_a_row_from_an_older_feature_version_is_not_in_the_corpus(fake_mongo):
    """A v3 EffNet vector and a v4 CLAP vector share no space at all, so a
    corpus that mixes them ranks noise and nothing in the numbers says so."""
    store.put_track(TRACK, {**FEATURES, "_features_version": FEATURES_VERSION - 1})
    assert store.corpus_ids() == []
    assert store.base_matrix()[0] == []
    assert store.tracks_since(datetime(1970, 1, 1)) == []


def test_a_row_with_no_version_at_all_is_not_in_the_corpus(fake_mongo):
    """A writer that does not say which stack produced a vector has not
    earned a place in the ranking."""
    store.put_track(TRACK, {"embedding": [0.1, 0.2, -0.3]})
    assert store.corpus_ids() == []


def test_the_current_version_is_in_the_corpus(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.corpus_ids() == ["42"]
    assert store.base_matrix()[0] == ["42"]


def test_an_old_row_is_still_readable_by_name(fake_mongo):
    """Leaving the corpus is not disappearing: a stale id is still a seed a
    user can search up and press play on, and the backfill has to read it."""
    store.put_track(TRACK, {**FEATURES, "_features_version": FEATURES_VERSION - 1})
    assert store.get_track("42") == STORED
    assert store.get_features("42")["_features_version"] == FEATURES_VERSION - 1


def test_the_incremental_corpus_cache_also_honours_the_version(fake_mongo):
    """corpus_ids' warm path is a different query from its cold one, so the
    filter has to be in both -- _live_since is why it can only be in one."""
    store.put_track(TRACK, FEATURES)
    assert store.corpus_ids() == ["42"]          # cold: fills the cache
    store.put_track({**TRACK, "track_id": "43"},
                    {**FEATURES, "_features_version": FEATURES_VERSION - 1})
    store.put_track({**TRACK, "track_id": "44"}, FEATURES)
    assert store.corpus_ids() == ["42", "44"]    # warm: watermark query


# ---- what the re-analysis backfill (Task 4) reads ----

def test_stale_count_counts_only_superseded_live_rows(fake_mongo):
    store.put_track(TRACK, FEATURES)
    store.put_track({**TRACK, "track_id": "43"},
                    {**FEATURES, "_features_version": FEATURES_VERSION - 1})
    store.put_track({**TRACK, "track_id": "44"},
                    {**FEATURES, "_features_version": 1})
    assert store.stale_count() == 2


def test_a_retired_duplicate_is_not_stale_work(fake_mongo):
    """Re-analyzing a row that will never rank again is wasted download."""
    store.put_track(TRACK, {**FEATURES, "_features_version": 1})
    store.mark_duplicate("42", "99")
    assert store.stale_count() == 0
    assert store.stale_ids() == []


def test_stale_ids_are_oldest_analyzed_first_and_limited(fake_mongo):
    """Oldest first so an interrupted backfill makes forward progress: a
    re-analyzed row gets a fresh analyzed_at and sorts to the back."""
    for track_id in ("a", "b", "c"):
        store.put_track({**TRACK, "track_id": track_id},
                        {**FEATURES, "_features_version": 1})
        fake_mongo.tracks.update_one(
            {"_id": track_id},
            {"$set": {"analyzed_at": datetime(2026, 9, 1 + ord(track_id) - ord("a"))}},
        )
    assert store.stale_ids(2) == ["a", "b"]
    assert store.stale_ids() == ["a", "b", "c"]
    assert store.stale_ids(0) == []


def test_a_row_leaves_the_queue_only_after_three_classifiable_failures(fake_mongo):
    """One bad afternoon at a source must not retire a third of the corpus,
    and one unfixable row must not be retried forever. Three attempts."""
    store.put_track(TRACK, {**FEATURES, "_features_version": 1})
    assert store.stale_count() == 1

    for expected in (1, 2):
        assert store.record_reanalysis_failure(
            "42", "UnfetchableTrack: no preview", classifiable=True) == expected
        assert store.stale_count() == 1        # still work to do

    assert store.record_reanalysis_failure(
        "42", "UnfetchableTrack: no preview",
        classifiable=True) == store.REANALYSIS_MAX_ATTEMPTS
    assert store.stale_count() == 0
    assert store.stale_ids() == []
    assert store.reanalysis_failed_count() == 1
    assert fake_mongo.tracks.find_one({"_id": "42"})["reanalysis_error"] == \
        "UnfetchableTrack: no preview"


def test_an_unclassifiable_failure_never_retires_a_row(fake_mongo):
    """A CDN 500 or a socket timeout is evidence about the world, not about
    this recording. It still counts an attempt -- that is the queue's backoff
    -- but it can never take the row out."""
    store.put_track(TRACK, {**FEATURES, "_features_version": 1})

    for _ in range(store.REANALYSIS_MAX_ATTEMPTS + 3):
        store.record_reanalysis_failure("42", "OSError: 500", classifiable=False)

    assert store.stale_count() == 1
    assert store.reanalysis_failed_count() == 0
    assert "reanalysis_failed_at" not in fake_mongo.tracks.find_one({"_id": "42"})


def test_a_failed_row_sorts_behind_one_that_has_not_been_tried(fake_mongo):
    """Otherwise the row that just failed keeps the head of an oldest-first
    queue and nothing behind it is ever reached."""
    for track_id in ("a", "b"):
        store.put_track({**TRACK, "track_id": track_id},
                        {**FEATURES, "_features_version": 1})
        fake_mongo.tracks.update_one(
            {"_id": track_id},
            {"$set": {"analyzed_at": datetime(2026, 9, 1 + ord(track_id) - ord("a"))}},
        )
    assert store.stale_ids(1) == ["a"]

    store.record_reanalysis_failure("a", "OSError: 500", classifiable=False)

    assert store.stale_ids(2) == ["b", "a"]


def test_a_prioritized_row_beats_the_whole_backlog(fake_mongo):
    """/seed on a superseded row: a client is blocked on that one track, and
    19,000 older rows are in front of it."""
    for track_id in ("a", "b", "c"):
        store.put_track({**TRACK, "track_id": track_id},
                        {**FEATURES, "_features_version": 1})
        fake_mongo.tracks.update_one(
            {"_id": track_id},
            {"$set": {"analyzed_at": datetime(2026, 9, 1 + ord(track_id) - ord("a"))}},
        )
    assert store.stale_ids(1) == ["a"]

    store.prioritize_reanalysis("c")

    assert store.stale_ids(3)[0] == "c"


def test_a_given_up_row_is_still_a_playable_seed(fake_mongo):
    """It keeps its old vectors and its metadata: a user can still search it
    up and press play. It is only out of the RANKING."""
    store.put_track(TRACK, {**FEATURES, "_features_version": 1})
    for _ in range(store.REANALYSIS_MAX_ATTEMPTS):
        store.record_reanalysis_failure("42", "boom", classifiable=True)

    assert store.get_track("42")["title"] == TRACK["title"]
    assert store.get_features("42") is not None
    assert store.corpus_ids() == []


def test_a_successful_analysis_clears_every_reanalysis_field(fake_mongo):
    """Otherwise the row stays hidden from the NEXT version bump's backfill
    as well, forever -- and a stale attempt count would retire it early."""
    store.put_track(TRACK, {**FEATURES, "_features_version": 1})
    store.prioritize_reanalysis("42")
    for _ in range(store.REANALYSIS_MAX_ATTEMPTS):
        store.record_reanalysis_failure("42", "boom", classifiable=True)

    store.put_track(TRACK, FEATURES)

    assert store.reanalysis_failed_count() == 0
    assert store.corpus_ids() == ["42"]
    row = fake_mongo.tracks.find_one({"_id": "42"})
    for field in ("reanalysis_failed_at", "reanalysis_error",
                  "reanalysis_attempts", "reanalysis_attempted_at",
                  "reanalyze_priority"):
        assert field not in row
