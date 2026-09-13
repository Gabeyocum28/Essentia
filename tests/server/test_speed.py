"""Server speed work: the viz snapshot policy, covariance PCA, the track
metadata cache, gzip, and /viz/map's compact points.

Every cache here is a correctness risk as much as a speed one, so each test
pins BOTH halves: no recompute/refetch inside the window, and a recompute
after it.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from music_recommendations.server import app as app_module
from music_recommendations.server import store, viz

TRACK = {
    "track_id": "x",
    "title": "Track",
    "artist": "Artist",
    "album": "Album",
    "artwork_url": None,
}


@pytest.fixture
def client(fake_mongo):
    return TestClient(app_module.app)


# ---- covariance PCA ----

def _svd_reference(matrix):
    m = np.asarray(matrix, dtype=float)
    unit = m / np.linalg.norm(m, axis=1, keepdims=True)
    c = unit - unit.mean(axis=0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    coords = u[:, :8] * s[:8]
    for col in range(8):
        if coords[np.argmax(np.abs(coords[:, col])), col] < 0:
            coords[:, col] = -coords[:, col]
    return coords, (s[:8] ** 2) / float(np.sum(s ** 2))


def test_project_top8_matches_the_svd_reference():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((300, 40)).astype(np.float32)
    coords, var = viz.project_top8(m)
    ref, ref_var = _svd_reference(m)
    assert np.allclose(var, ref_var, atol=1e-6)
    for col in range(8):
        assert np.allclose(coords[:, col], ref[:, col], atol=1e-4)


def test_project_top8_small_and_degenerate():
    coords, var = viz.project_top8(np.ones((2, 3), np.float32))
    assert coords.shape == (2, 8) and var.shape == (8,)
    coords, var = viz.project_top8(np.zeros((1, 5), np.float32))
    assert coords.shape == (1, 8)


def test_project_top8_survives_more_rows_than_one_block(monkeypatch):
    """The covariance accumulates in row blocks; the seam must not change
    the answer."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((60, 12)).astype(np.float32)
    ref, ref_var = _svd_reference(m)
    monkeypatch.setattr(viz, "_PCA_BLOCK", 7)      # nine ragged blocks
    coords, var = viz.project_top8(m)
    assert np.allclose(var, ref_var, atol=1e-6)
    for col in range(8):
        assert np.allclose(coords[:, col], ref[:, col], atol=1e-4)


# ---- viz snapshot policy ----

def test_viz_snapshot_is_reused_inside_the_window(fake_mongo):
    """Sub-VIZ_GROWTH_PCT growth inside the window must not rebuild: the
    derived caches (subset/top8/mst) all key on id(matrix), so a new matrix
    object is a cold Insights screen."""
    for i in range(30):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 29.0]})
    ids1, m1 = app_module._viz_snapshot()
    assert len(ids1) == 30
    store.put_track({**TRACK, "track_id": "t30"}, {"embedding": [0.5, 0.5]})
    ids2, m2 = app_module._viz_snapshot()
    assert m2 is m1 and ids2 == ids1          # +1 of 30 is under 5%: no rebuild
    assert "t30" not in ids2


def test_viz_snapshot_refreshes_after_the_window_or_growth(fake_mongo,
                                                           monkeypatch):
    for tid in ("a", "b"):
        store.put_track({**TRACK, "track_id": tid}, {"embedding": [1.0, 0.0]})
    ids1, m1 = app_module._viz_snapshot()
    monkeypatch.setattr(app_module, "VIZ_REFRESH_S", 0.0)
    store.put_track({**TRACK, "track_id": "c"}, {"embedding": [0.0, 1.0]})
    ids2, m2 = app_module._viz_snapshot()
    assert "c" in ids2 and m2 is not m1
    monkeypatch.setattr(app_module, "VIZ_REFRESH_S", 3600.0)
    for i in range(10):                        # +>5% growth forces a refresh too
        store.put_track({**TRACK, "track_id": f"g{i}"},
                        {"embedding": [1.0, 1.0]})
    ids3, _ = app_module._viz_snapshot()
    assert len(ids3) == 13


def test_viz_snapshot_refreshes_for_a_seed_it_does_not_hold(fake_mongo):
    """A track analyzed by /seed right now must appear on its own Insights
    screen, whatever the refresh window says."""
    for i in range(30):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 29.0]})
    ids1, m1 = app_module._viz_snapshot()
    store.put_track({**TRACK, "track_id": "fresh"}, {"embedding": [0.0, 1.0]})
    ids2, m2 = app_module._viz_snapshot()
    assert "fresh" not in ids2 and m2 is m1
    ids3, m3 = app_module._viz_snapshot(require="fresh")
    assert "fresh" in ids3 and m3 is not m1
    # An id that is nowhere in the corpus is not a reason to rebuild.
    ids4, m4 = app_module._viz_snapshot(require="nonexistent")
    assert m4 is m3


def test_viz_subset_stays_warm_across_requests_inside_the_window(client,
                                                                 fake_mongo,
                                                                 monkeypatch):
    """The point of the snapshot: two Insights calls while a crawl adds
    tracks reuse one subset, one PCA, one MST."""
    monkeypatch.setattr(app_module, "VIZ_MAX", 50)
    for i in range(30):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 29.0]})
    first = client.get("/viz/tour?track_id=t0").json()
    subsets = len(app_module._SUBSET_CACHE)
    store.put_track({**TRACK, "track_id": "t30"}, {"embedding": [0.5, 0.5]})
    second = client.get("/viz/tour?track_id=t0").json()
    assert first == second
    assert len(app_module._SUBSET_CACHE) == subsets
    assert len(app_module._TOP8_CACHE) == 1


def test_viz_map_draws_a_seed_analyzed_after_the_snapshot(client, fake_mongo,
                                                          monkeypatch):
    monkeypatch.setattr(app_module, "VIZ_MAX", 50)
    for i in range(30):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 29.0]})
    client.get("/viz/tour?track_id=t0")
    store.put_track({**TRACK, "track_id": "fresh"}, {"embedding": [0.0, 1.0]})
    body = client.get("/viz/map?track_id=fresh&axis=sounds_like&limit=3").json()
    assert body["seed"]["track_id"] == "fresh"
    assert "fresh" in body["points"]["ids"]


def test_viz_map_skips_a_rec_analyzed_after_the_snapshot(client, fake_mongo,
                                                         monkeypatch):
    """A rec the snapshot predates has no row to be drawn at, and every
    client reads rec.x/rec.y unconditionally. It is filtered out of the list
    rather than forcing a refresh -- and the list is still `limit` long,
    because the filter runs before the truncation."""
    monkeypatch.setattr(app_module, "VIZ_MAX", 50)
    for i in range(30):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, 5.0 + i]})
    client.get("/viz/tour?track_id=t0")
    before = app_module._VIZ_SNAPSHOT
    # A near-duplicate of the seed, so it would otherwise rank first.
    store.put_track({**TRACK, "track_id": "twin"}, {"embedding": [1.0, 5.0]})

    recommended = client.get(
        "/recommend?track_id=t0&axis=sounds_like&limit=3").json()["results"]
    assert recommended[0]["track_id"] == "twin"    # /recommend is unaffected

    body = client.get("/viz/map?track_id=t0&axis=sounds_like&limit=3").json()
    assert "twin" not in {rec["track_id"] for rec in body["recs"]}
    assert len(body["recs"]) == 3                  # still a full list
    assert all(isinstance(rec["x"], float) and isinstance(rec["y"], float)
               for rec in body["recs"])
    assert "twin" not in body["points"]["ids"]
    # ... and no refresh happened: the snapshot is the same matrix object.
    assert app_module._VIZ_SNAPSHOT[2] is before[2]


def test_viz_subset_does_not_touch_the_unit_cache(fake_mongo, monkeypatch):
    """_UNIT_CACHE is a single slot holding the LIVE matrix for /recommend.
    Normalizing the snapshot matrix through it evicted the live one, so the
    two paths took turns re-normalizing a whole corpus for each other."""
    monkeypatch.setattr(app_module, "VIZ_MAX", 3)
    vecs = {"a": [1, 0], "b": [0.9, 0.1], "c": [0.7, 0.3],
            "d": [0.5, 0.5], "e": [0, 1]}
    for tid, v in vecs.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    monkeypatch.setattr(app_module, "_similarity", lambda *a, **k: (
        _ for _ in ()).throw(AssertionError("_viz_subset used _UNIT_CACHE")))
    ids, matrix = app_module._viz_subset("c")
    assert set(ids) == {"b", "c", "d"}             # the same nearest set
    assert matrix.shape == (3, 2)
    assert app_module._UNIT_CACHE == {}


def test_seed_cosine_matches_a_normalized_dot_and_caches_row_norms(fake_mongo):
    rng = np.random.default_rng(3)
    m = rng.standard_normal((12, 5)).astype(np.float32)
    unit = m / np.linalg.norm(m, axis=1, keepdims=True)
    got = app_module._seed_cosine(m, 4)
    assert np.allclose(got, unit @ unit[4], atol=1e-6)
    assert app_module._ROW_NORMS[0] is m
    assert app_module._ROW_NORMS[1].shape == (12,)


def test_seed_cosine_survives_a_zero_row(fake_mongo):
    m = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    got = app_module._seed_cosine(m, 0)
    assert np.all(np.isfinite(got))
    assert got[0] == pytest.approx(1.0) and got[1] == pytest.approx(0.0)


# ---- track metadata cache ----

def test_tracks_cached_batches_misses_and_reuses_hits(fake_mongo, monkeypatch):
    for tid in ("a", "b", "c"):
        store.put_track_meta({**TRACK, "track_id": tid})
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks",
                        lambda ids, _f=real: (calls.append(list(ids)), _f(ids))[1])
    got = app_module._tracks_cached(["a", "b"])
    assert [t["track_id"] for t in got] == ["a", "b"] and calls == [["a", "b"]]
    got = app_module._tracks_cached(["b", "c", "zzz"])
    assert calls[-1] == ["c", "zzz"] and got[2] is None
    assert [t["track_id"] for t in got[:2]] == ["b", "c"]
    # A miss is not cached, so the next lookup asks again.
    app_module._tracks_cached(["zzz"])
    assert calls[-1] == ["zzz"]


def test_tracks_cached_is_bounded(fake_mongo, monkeypatch):
    monkeypatch.setattr(app_module, "_TRACK_META_MAX", 3)
    for i in range(5):
        app_module._remember_track({**TRACK, "track_id": f"t{i}"})
    assert len(app_module._TRACK_META) == 3
    assert list(app_module._TRACK_META) == ["t2", "t3", "t4"]


def test_recommend_fetches_tracks_in_one_batch(client, fake_mongo, monkeypatch):
    for i in range(4):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 3.0]})
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks",
                        lambda ids, _f=real: (calls.append(list(ids)), _f(ids))[1])
    monkeypatch.setattr(store, "get_track", lambda tid: (_ for _ in ()).throw(
        AssertionError("per-track fetch")))
    r = client.get("/recommend?track_id=t0&axis=sounds_like&limit=3")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert all(t["title"] == TRACK["title"] for t in results)   # real metadata
    assert all(t["artist"] == TRACK["artist"] for t in results)
    assert len(calls) == 1


def test_viz_hubs_fetches_its_rows_in_one_batch(client, fake_mongo, monkeypatch):
    for i in range(6):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 5.0]})
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks",
                        lambda ids, _f=real: (calls.append(list(ids)), _f(ids))[1])
    monkeypatch.setattr(store, "get_track", lambda tid: (_ for _ in ()).throw(
        AssertionError("per-track fetch")))
    body = client.get("/viz/hubs?track_id=t0&limit=2").json()
    assert body["hubs"] and len(calls) == 1


# ---- gzip and compact points ----

def test_responses_are_gzipped_for_clients_that_ask(client, fake_mongo):
    for i in range(40):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 39.0]})
    r = client.get("/viz/map?track_id=t0&axis=sounds_like&limit=5",
                   headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert r.json()["seed"]["track_id"] == "t0"


def test_viz_map_compact_points_omit_the_track_dicts(client, fake_mongo):
    for i in range(10):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 9.0]})
    params = "track_id=t0&axis=sounds_like&limit=3"
    full = client.get(f"/viz/map?{params}").json()["points"]
    compact = client.get(f"/viz/map?{params}&points=compact").json()["points"]
    assert set(full) == {"ids", "x", "y", "tracks"}
    assert set(compact) == {"ids", "x", "y"}
    assert compact["ids"] == full["ids"]
    assert compact["x"] == full["x"] and compact["y"] == full["y"]


def test_viz_map_compact_reads_only_the_tracks_it_names(client, fake_mongo,
                                                        monkeypatch):
    """Compact is a speed mode, not just a smaller payload: it must not read
    the whole subset's metadata either."""
    for i in range(10):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 9.0]})
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks",
                        lambda ids, _f=real: (calls.append(list(ids)), _f(ids))[1])
    body = client.get(
        "/viz/map?track_id=t0&axis=sounds_like&limit=3&points=compact").json()
    assert len(calls) == 1
    assert set(calls[0]) == {"t0", *(r["track_id"] for r in body["recs"])}


def test_viz_map_rejects_an_unknown_points_mode(client, fake_mongo):
    for i in range(3):
        store.put_track({**TRACK, "track_id": f"t{i}"},
                        {"embedding": [1.0, i / 2.0]})
    r = client.get("/viz/map?track_id=t0&axis=sounds_like&points=sparse")
    assert r.status_code == 422
