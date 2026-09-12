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
    store.put_track({**TRACK, "track_id": "old"}, {"embedding": [1.0]})
    stamp = store.get_analyzed_at("old")
    store.put_track({**TRACK, "track_id": "new"}, {"embedding": [2.0]})
    got = store.tracks_since(stamp)
    assert [tid for tid, _ in got] == ["new"]
    assert store.tracks_since(datetime.now(timezone.utc)) == []

