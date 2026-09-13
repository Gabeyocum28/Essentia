"""viz.py + GET /viz/map: 2D projection of the corpus and per-rec score math.

Non-contract debug/demo endpoint. The galaxy map is always a projection of
EMBEDDING space (that's "where the music lies"); the axis only changes which
scores and math are attached to the recommendations.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from music_recommendations.server import app as app_module
from music_recommendations.server import store, viz

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "contract" / "fixture.json").read_text()
)["tracks"]

TRACK = {
    "track_id": "x",
    "title": "Track",
    "artist": "Artist",
    "album": "Album",
    "artwork_url": None,
}


# ---- project_2d (pure math) ----

def test_project_2d_shape_and_determinism():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(30, 8))
    xy = viz.project_2d(matrix)
    assert xy.shape == (30, 2)
    assert np.allclose(xy, viz.project_2d(matrix))


def test_project_2d_first_axis_captures_dominant_direction():
    # Points spread 100x wider along one direction: that spread must land on x.
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(50, 5))
    matrix[:, 2] *= 100.0
    xy = viz.project_2d(matrix)
    assert xy[:, 0].std() > xy[:, 1].std()


def test_project_2d_handles_single_row():
    xy = viz.project_2d(np.ones((1, 4)))
    assert xy.shape == (1, 2)
    assert np.all(np.isfinite(xy))


# ---- project_top8 (pure math) ----

def test_project_top8_shape_and_determinism():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(30, 10))
    coords8, variance = viz.project_top8(matrix)
    assert coords8.shape == (30, 8)
    assert variance.shape == (8,)
    coords8_again, variance_again = viz.project_top8(matrix)
    assert np.allclose(coords8, coords8_again)
    assert np.allclose(variance, variance_again)


def test_project_top8_matches_project_2d_first_two_columns():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(25, 6))
    xy = viz.project_2d(matrix)
    coords8, _ = viz.project_top8(matrix)
    assert np.allclose(xy, coords8[:, :2])


def test_project_top8_variance_non_increasing_and_bounded():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 12))
    _, variance = viz.project_top8(matrix)
    assert np.all(variance >= 0.0) and np.all(variance <= 1.0)
    assert all(variance[i] >= variance[i + 1] for i in range(7))


def test_project_top8_pads_low_dimensional_corpora():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(6, 3))
    coords8, variance = viz.project_top8(matrix)
    assert coords8.shape == (6, 8)
    assert np.allclose(coords8[:, 3:], 0.0)
    assert np.allclose(variance[3:], 0.0)


def test_project_top8_handles_single_row():
    coords8, variance = viz.project_top8(np.ones((1, 4)))
    assert coords8.shape == (1, 8)
    assert variance.shape == (8,)
    assert np.all(np.isfinite(coords8))


# ---- minimum_spanning_tree (pure math) ----

def test_minimum_spanning_tree_known_by_inspection():
    # Four points on a line: 0, 1, 3, 6. Cosine distance won't reproduce a
    # literal line, so build a matrix whose cosine geometry has an obvious
    # MST instead: unit vectors at increasing angles on a quarter circle,
    # so the cheapest tree is the path 0-1-2-3 (each hop the same, adjacent
    # angle).
    theta = np.array([0.0, 0.2, 0.4, 0.6])
    matrix = np.column_stack([np.cos(theta), np.sin(theta)])
    edges = viz.minimum_spanning_tree(matrix)

    assert len(edges) == 3
    assert {(i, j) for i, j, _ in edges} == {(0, 1), (1, 2), (2, 3)}
    ds = [d for _, _, d in edges]
    assert ds == sorted(ds)


def test_minimum_spanning_tree_forms_spanning_tree():
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(20, 5))
    ids, matrix = list(range(20)), matrix
    edges = viz.minimum_spanning_tree(matrix)
    assert len(edges) == len(ids) - 1

    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = len(ids)
    for i, j, _ in edges:
        assert i < j
        ri, rj = find(i), find(j)
        assert ri != rj  # no cycle
        parent[ri] = rj
        components -= 1
    assert components == 1


def test_minimum_spanning_tree_sorted_ascending_and_deterministic():
    rng = np.random.default_rng(6)
    matrix = rng.normal(size=(15, 4))
    edges = viz.minimum_spanning_tree(matrix)
    ds = [d for _, _, d in edges]
    assert ds == sorted(ds)
    assert edges == viz.minimum_spanning_tree(matrix)


def test_minimum_spanning_tree_empty_for_single_row():
    assert viz.minimum_spanning_tree(np.ones((1, 4))) == []


# ---- GET /viz/map ----

def fake_features(i: int, n: int) -> dict:
    v = [0.0] * 1280
    v[0] = 1.0
    v[1] = i / max(n - 1, 1)
    v[2] = float(i % 3)
    return {"embedding": v}


@pytest.fixture
def client(fake_mongo):
    return TestClient(app_module.app)


@pytest.fixture
def seeded_corpus(fake_mongo):
    tracks = FIXTURE[:8]
    for i, t in enumerate(tracks):
        store.put_track(t, fake_features(i, len(tracks)))
    return tracks


# One recording per entry except where the title (or the embedding) says
# otherwise: d1 is the seed again, a2 is a1 again, e2 is e1 again, and c1 is
# a genuine cover of a1 by another artist.
VIZ_DUPLICATES = [
    ("s",  "So What",                               "Miles Davis", 0.0),
    ("d1", "So What (2009 Remaster)",               "Miles Davis", 0.02),
    ("a1", "Blue in Green",                         "Miles Davis", 0.4),
    ("a2", "Blue in Green - Live at the Blackhawk", "Miles Davis", 0.6),
    ("c1", "Blue in Green",                         "Bill Evans",  0.8),
    ("e1", "Flamenco Sketches",                     "Miles Davis", 1.2),
    ("e2", "Sketches of Flamenco",                  "Nobody Else", 1.22),
    ("f1", "All Blues",                             "Miles Davis", 1.6),
]


@pytest.fixture
def duplicate_corpus(fake_mongo):
    for track_id, title, artist, theta in VIZ_DUPLICATES:
        v = [0.0] * 1280
        v[0], v[1] = float(np.cos(theta)), float(np.sin(theta))
        store.put_track({**TRACK, "track_id": track_id, "title": title,
                         "artist": artist}, {"embedding": v})
    return VIZ_DUPLICATES


def _viz_rec_ids(client, **params):
    body = client.get("/viz/map", params={"track_id": "s", "axis": "sounds_like",
                                          **params}).json()
    return [rec["track_id"] for rec in body["recs"]]


def test_viz_map_returns_each_recording_once(client, duplicate_corpus):
    """Insights explains the list /recommend served, so it collapses the same
    duplicates -- by key, and by near-identical embedding."""
    assert _viz_rec_ids(client, limit=10) == ["a1", "c1", "e1", "f1"]


def test_viz_map_widens_the_scan_to_stay_limit_long(client, duplicate_corpus):
    assert _viz_rec_ids(client, limit=3) == ["a1", "c1", "e1"]


def test_viz_map_404_when_seed_unanalyzed(client, fake_mongo):
    resp = client.get("/viz/map", params={"track_id": "nope", "axis": "sounds_like"})
    assert resp.status_code == 404


def test_viz_map_400_on_unknown_axis(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    resp = client.get("/viz/map", params={"track_id": tid, "axis": "vibes"})
    assert resp.status_code == 400


def test_viz_map_points_cover_corpus(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like", "limit": 3}
    ).json()
    points = body["points"]
    assert set(points) == {"ids", "x", "y", "tracks"}
    assert sorted(points["ids"]) == sorted(t["track_id"] for t in seeded_corpus)
    assert len(points["x"]) == len(points["y"]) == len(points["ids"])
    assert all(np.isfinite(points["x"])) and all(np.isfinite(points["y"]))


def test_viz_map_point_metadata_stays_aligned_with_coordinates(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    points = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like", "limit": 3}
    ).json()["points"]

    assert len(points["tracks"]) == len(points["ids"])
    assert [track["track_id"] for track in points["tracks"]] == points["ids"]
    assert all(track["title"] and track["artist"] for track in points["tracks"])


def test_viz_map_reads_all_point_metadata_in_one_bulk_request(
    client, seeded_corpus, monkeypatch
):
    tid = seeded_corpus[0]["track_id"]
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks",
                        lambda ids, _f=real: (calls.append(list(ids)), _f(ids))[1])
    client.get("/viz/map", params={"track_id": tid, "axis": "sounds_like"})

    assert len(calls) == 1
    assert len(calls[0]) == len(seeded_corpus)


def test_viz_map_seed_has_position(client, seeded_corpus):
    """Pins the position math on the seed entry."""
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like"}
    ).json()
    seed = body["seed"]
    assert seed["track_id"] == tid
    assert seed["title"] == seeded_corpus[0]["title"]
    assert isinstance(seed["x"], float) and isinstance(seed["y"], float)


def test_viz_map_recs_match_recommend_scores(client, seeded_corpus):
    """The wow screen must show the SAME ranking the user just saw."""
    tid = seeded_corpus[0]["track_id"]
    rec_body = client.get(
        "/recommend", params={"track_id": tid, "axis": "sounds_like", "limit": 3}
    ).json()
    viz_body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like", "limit": 3}
    ).json()
    assert [r["track_id"] for r in viz_body["recs"]] == \
        [r["track_id"] for r in rec_body["results"]]
    for vr, rr in zip(viz_body["recs"], rec_body["results"]):
        assert vr["score"] == pytest.approx(rr["score"])


def test_viz_map_rec_math_reconstructs_cosine(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like", "limit": 3}
    ).json()
    assert body["axis"] == {"id": "sounds_like", "metric": "cosine", "direction": 1}
    for rec in body["recs"]:
        math = rec["math"]
        cosine = math["dot"] / (math["seed_norm"] * math["rec_norm"])
        assert cosine == pytest.approx(rec["score"])
        assert math["centrality"] is None


def test_viz_map_surprise_includes_centrality(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "surprise", "limit": 3}
    ).json()
    assert body["axis"]["direction"] == -1
    for rec in body["recs"]:
        assert isinstance(rec["math"]["centrality"], float)


def test_viz_map_can_disable_surprise_correction(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/viz/map",
        params={"track_id": tid, "axis": "surprise", "correction": "off"},
    ).json()
    assert body["axis"]["correction"] == "off"
    assert all(rec["math"]["centrality"] is None for rec in body["recs"])


def test_viz_map_rejects_unknown_axis(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    assert client.get(
        "/viz/map", params={"track_id": tid, "axis": "groove"}
    ).status_code == 400


@pytest.fixture
def blended_axis():
    """No blended axis ships today -- best_match was the only one and it was
    cut. The blending path stays in app.py/viz.py for the next one, so register
    a throwaway blend here rather than leave that path untested."""
    from music_recommendations.server.axes import BLENDED_AXES
    BLENDED_AXES["blend_test"] = {"embedding": 0.5, "genre": 0.5}
    yield "blend_test"
    del BLENDED_AXES["blend_test"]


def test_viz_map_serves_blended_axis(client, seeded_corpus, blended_axis):
    """Insights opens on whichever axis the user picked, a blend included:
    before, a blended axis 400d here and the screen showed its error state."""
    tid = seeded_corpus[0]["track_id"]
    r = client.get(
        "/viz/map", params={"track_id": tid, "axis": blended_axis, "limit": 3}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["axis"]["metric"] == "blend"
    assert body["axis"]["weights"] == {"embedding": 0.5, "genre": 0.5}
    assert len(body["recs"]) == 3
    # The score is the fused percentile, so it is bounded and ordered ...
    scores = [rec["score"] for rec in body["recs"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    for rec in body["recs"]:
        # ... and every weighted key that actually produced data reports its
        # own percentile, so the panel can show the blend rather than a
        # formula that isn't the score. "genre" is no longer a feature the
        # store round-trips (embedding-only contract), so it contributes no
        # rows and is silently absent from parts rather than blowing up.
        assert set(rec["math"]["parts"]) == {"embedding"}


def test_viz_map_blended_axis_matches_recommend_order(client, seeded_corpus,
                                                      blended_axis):
    """Insights must explain the list the user saw, not re-rank it."""
    tid = seeded_corpus[0]["track_id"]
    params = {"track_id": tid, "axis": blended_axis, "limit": 4}
    recs = client.get("/recommend", params=params).json()["results"]
    viz_recs = client.get("/viz/map", params=params).json()["recs"]
    assert [r["track_id"] for r in viz_recs] == [r["track_id"] for r in recs]
    assert [r["score"] for r in viz_recs] == [r["score"] for r in recs]


# ---- T1: walk, histogram, and hubs ----

def test_shortest_walk_uses_knn_edges_and_preserves_endpoints():
    # Points on a quarter circle: the graph walk follows adjacent samples and
    # is longer than the direct chord through empty ambient space.
    theta = np.linspace(0, np.pi / 2, 7)
    matrix = np.column_stack([np.cos(theta), np.sin(theta)])

    path, geodesic, ambient = viz.shortest_walk(matrix, 0, 6, k=2)

    assert path[0] == 0 and path[-1] == 6
    assert len(path) > 2
    expected_ambient = 1.0 - float(matrix[0] @ matrix[6])
    assert geodesic > 0
    assert ambient == pytest.approx(expected_ambient)


def test_shortest_walk_graph_cache_pins_the_matrix_it_was_built_from():
    # Every entry must hold the matrix it was built from: a bare id() key let
    # a freed matrix's address be reused by a successor, silently serving
    # stale node indices after the corpus grew mid-session.
    theta = np.linspace(0, np.pi / 2, 7)
    first = np.column_stack([np.cos(theta), np.sin(theta)])
    viz.shortest_walk(first, 0, 6, k=2)
    assert viz._GRAPH_CACHE[(id(first), 2)][0] is first

    second = np.column_stack([np.cos(theta[:5]), np.sin(theta[:5])])
    path, _, _ = viz.shortest_walk(second, 0, 4, k=2)
    assert viz._GRAPH_CACHE[(id(second), 2)][0] is second
    assert max(path) < len(second)
    # Both subsets stay cached: this used to be a single slot, so alternating
    # between two seeds' subsets rebuilt the graph on every request.
    assert viz._GRAPH_CACHE[(id(first), 2)][0] is first


def test_viz_walk_returns_track_path_and_distance_math(client, seeded_corpus):
    start = seeded_corpus[0]["track_id"]
    end = seeded_corpus[-1]["track_id"]
    response = client.get(
        "/viz/walk", params={"from": start, "to": end, "k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path"][0]["track_id"] == start
    assert body["path"][-1]["track_id"] == end
    assert all("x" in step and "y" in step for step in body["path"])
    assert body["geodesic"] > 0
    assert body["ambient"] > 0
    assert body["detour"] == pytest.approx(body["geodesic"] / body["ambient"])


def test_viz_walk_rejects_unknown_endpoint(client, seeded_corpus):
    response = client.get(
        "/viz/walk",
        params={"from": seeded_corpus[0]["track_id"], "to": "missing"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "track missing not in corpus"


def test_viz_histogram_has_sixty_bins_and_analytic_null(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get("/viz/histogram", params={"track_id": tid}).json()

    assert len(body["bins"]) == len(body["counts"]) == 60
    assert sum(body["counts"]) == len(seeded_corpus) - 1
    assert body["null"]["mean"] == 0
    assert body["null"]["sd"] == pytest.approx(1 / np.sqrt(1280))
    assert body["rec_scores"] == sorted(body["rec_scores"], reverse=True)
    assert 0 <= body["percentile"] <= 100


def test_viz_histogram_404_when_track_missing(client, seeded_corpus):
    response = client.get(
        "/viz/histogram", params={"track_id": "missing"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "track missing not analyzed"


def test_viz_hubs_reports_k_occurrence_central_and_isolated(client, seeded_corpus):
    body = client.get("/viz/hubs", params={"k": 3, "limit": 3}).json()

    assert body["expected_k"] == 3
    assert sum(row["count"] for row in body["all_counts"]) == len(seeded_corpus) * 3
    assert len(body["hubs"]) == len(body["central"]) == len(body["isolated"]) == 3
    assert body["hubs"][0]["count"] >= body["hubs"][-1]["count"]
    assert body["central"][0]["centrality"] >= body["central"][-1]["centrality"]
    assert body["isolated"][0]["centrality"] <= body["isolated"][-1]["centrality"]


# ---- T2: tour, mst, extremes ----

def test_viz_tour_ids_and_coords_shape(client, seeded_corpus):
    body = client.get("/viz/tour").json()
    n = len(seeded_corpus)
    assert sorted(body["ids"]) == sorted(t["track_id"] for t in seeded_corpus)

    raw = base64.b64decode(body["coords8"])
    coords = np.frombuffer(raw, dtype="<f4").reshape(n, 8)
    assert coords.shape == (n, 8)
    assert np.all(np.isfinite(coords))

    assert len(body["variance"]) == 8
    for v in body["variance"]:
        assert 0.0 <= v <= 1.0
    assert all(
        body["variance"][i] >= body["variance"][i + 1] for i in range(7)
    )


def test_viz_tour_first_two_coords_match_viz_map_xy(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    map_body = client.get(
        "/viz/map", params={"track_id": tid, "axis": "sounds_like"}
    ).json()
    tour_body = client.get("/viz/tour").json()

    map_ids = map_body["points"]["ids"]
    tour_ids = tour_body["ids"]
    n = len(tour_ids)
    coords = np.frombuffer(
        base64.b64decode(tour_body["coords8"]), dtype="<f4"
    ).reshape(n, 8)

    tour_pos = {t: i for i, t in enumerate(tour_ids)}
    for i, mid in enumerate(map_ids):
        row = coords[tour_pos[mid]]
        assert row[0] == pytest.approx(map_body["points"]["x"][i], abs=1e-3)
        assert row[1] == pytest.approx(map_body["points"]["y"][i], abs=1e-3)


def test_viz_tour_is_deterministic_across_calls(client, seeded_corpus):
    first = client.get("/viz/tour").json()
    second = client.get("/viz/tour").json()
    assert first == second


def test_viz_tour_404_when_corpus_empty(client, fake_mongo):
    resp = client.get("/viz/tour")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "needs at least two tracks"


def test_viz_mst_has_n_minus_one_edges_sorted_and_spanning(client, seeded_corpus):
    body = client.get("/viz/mst").json()
    ids = body["ids"]
    n = len(ids)
    edges = body["edges"]
    assert len(edges) == n - 1

    ds = [e[2] for e in edges]
    assert ds == sorted(ds)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    for i, j, d in edges:
        assert 0 <= i < n and 0 <= j < n and i < j
        ri, rj = find(i), find(j)
        assert ri != rj  # no cycle
        parent[ri] = rj
        components -= 1
    assert components == 1


def test_viz_mst_is_deterministic_across_calls(client, seeded_corpus):
    first = client.get("/viz/mst").json()
    second = client.get("/viz/mst").json()
    assert first == second


def test_viz_mst_404_when_corpus_empty(client, fake_mongo):
    resp = client.get("/viz/mst")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "needs at least two tracks"


def test_viz_extremes_low_high_disjoint_and_ordered(client, seeded_corpus):
    body = client.get("/viz/extremes", params={"pc": 1, "limit": 3}).json()
    assert body["pc"] == 1
    assert 0.0 < body["variance_pct"] <= 100.0

    low_ids = [t["track_id"] for t in body["low"]]
    high_ids = [t["track_id"] for t in body["high"]]
    assert set(low_ids).isdisjoint(high_ids)

    tour = client.get("/viz/tour").json()
    n = len(tour["ids"])
    coords = np.frombuffer(
        base64.b64decode(tour["coords8"]), dtype="<f4"
    ).reshape(n, 8)
    pos = {t: i for i, t in enumerate(tour["ids"])}
    low_values = [coords[pos[i], 0] for i in low_ids]
    high_values = [coords[pos[i], 0] for i in high_ids]
    assert low_values == sorted(low_values)
    assert high_values == sorted(high_values, reverse=True)


def test_viz_extremes_pc2_differs_from_pc1(client, seeded_corpus):
    pc1 = client.get("/viz/extremes", params={"pc": 1}).json()
    pc2 = client.get("/viz/extremes", params={"pc": 2}).json()
    assert pc1["low"] != pc2["low"] or pc1["high"] != pc2["high"]


def test_viz_extremes_track_shape_matches_contract(client, seeded_corpus):
    body = client.get("/viz/extremes", params={"pc": 1, "limit": 2}).json()
    for track in body["low"] + body["high"]:
        assert set(track) >= {
            "track_id", "title", "artist", "album", "artwork_url", "preview_url",
        }


def test_viz_extremes_422_on_pc_out_of_range(client, seeded_corpus):
    assert client.get("/viz/extremes", params={"pc": 0}).status_code == 422
    assert client.get("/viz/extremes", params={"pc": 9}).status_code == 422


def test_viz_extremes_422_on_limit_out_of_range(client, seeded_corpus):
    assert client.get("/viz/extremes", params={"limit": 0}).status_code == 422
    assert client.get("/viz/extremes", params={"limit": 11}).status_code == 422


def test_viz_extremes_404_when_corpus_empty(client, fake_mongo):
    resp = client.get("/viz/extremes")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "needs at least two tracks"


# ---- T2.6 attribution: band math, band-stop DSP, and the mailbox route ----


def test_band_edges_are_log_spaced_across_the_model_bandwidth():
    bands = viz.band_edges()

    assert len(bands) == viz.ATTRIBUTION_BANDS
    assert bands[0][0] == pytest.approx(viz.ATTRIBUTION_LO_HZ)
    assert bands[-1][1] == pytest.approx(viz.ATTRIBUTION_HI_HZ)
    # EffNet hears 16 kHz audio, so no band may claim energy above Nyquist.
    assert bands[-1][1] < 8000
    # Contiguous, and equal ratios (equal octave widths) rather than linear.
    for (lo, hi), (next_lo, _) in zip(bands, bands[1:]):
        assert hi == pytest.approx(next_lo)
    ratios = [hi / lo for lo, hi in bands]
    assert max(ratios) == pytest.approx(min(ratios))


def _tone(freq, sample_rate=16000, seconds=1.0, amplitude=1.0):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def _energy_at(signal, freq, sample_rate=16000):
    spectrum = np.abs(np.fft.rfft(signal))
    bin_index = int(round(freq * len(signal) / sample_rate))
    return float(spectrum[bin_index - 1:bin_index + 2].max())


def test_band_stop_reconstructs_audio_outside_the_stopped_band():
    # Nothing of the signal lives in the removed band, so overlap-add with
    # the original phase must give the input back (the COLA property).
    audio = _tone(220.0)
    out = viz.band_stop(audio, 16000, 4000.0, 5000.0)

    assert out.shape == audio.shape
    # Interior samples come back to machine precision. The first and last
    # frame legitimately differ: the preview starts abruptly, and that step
    # has energy inside the stopped band, so removing the band must change
    # it. Only the edges — the padding guarantees full window coverage.
    interior = slice(2048, len(audio) - 2048)
    assert np.abs(out[interior] - audio[interior]).max() < 1e-6


def test_band_stop_with_nothing_to_remove_is_exact_overlap_add():
    # A stop band above Nyquist masks no bins at all, so this isolates the
    # COLA property itself: Hann at 50% hop reconstructs every sample,
    # edges included, with no windowing residue.
    audio = _tone(220.0) + _tone(3000.0, amplitude=0.4)
    out = viz.band_stop(audio, 16000, 9000.0, 9500.0)

    assert np.abs(out - audio).max() < 1e-9


def test_band_stop_removes_only_the_selected_band():
    audio = _tone(200.0) + _tone(4000.0)
    out = viz.band_stop(audio, 16000, 3000.0, 5000.0)

    kept_before, kept_after = _energy_at(audio, 200.0), _energy_at(out, 200.0)
    gone_before, gone_after = _energy_at(audio, 4000.0), _energy_at(out, 4000.0)

    assert gone_after < gone_before / 100          # >40 dB down in the band
    assert kept_after == pytest.approx(kept_before, rel=0.02)


def test_viz_attribute_queues_the_pair_once_and_reports_pending(client, seeded_corpus, fake_mongo):
    seed, rec = seeded_corpus[0]["track_id"], seeded_corpus[1]["track_id"]

    first = client.get("/viz/attribute", params={"seed": seed, "rec": rec})
    assert first.status_code == 200
    assert first.json() == {"status": "pending"}
    assert fake_mongo.jobs.find_one({"_id": f"attr:{seed}:{rec}"})["state"] == "queued"

    # Polling must not pile the same job up behind itself.
    assert client.get("/viz/attribute", params={"seed": seed, "rec": rec}).json() == {
        "status": "pending"
    }
    assert fake_mongo.jobs.find_one({"_id": f"attr:{seed}:{rec}"})["state"] == "queued"


def test_viz_attribute_serves_the_cached_result_when_the_worker_is_done(client, seeded_corpus):
    seed, rec = seeded_corpus[0]["track_id"], seeded_corpus[1]["track_id"]
    ready = {
        "status": "ready",
        "base": 0.83,
        "bands": [{"lo_hz": 60.0, "hi_hz": 120.0, "delta": 0.4}],
    }
    store.put_attribution(seed, rec, ready)

    assert client.get("/viz/attribute", params={"seed": seed, "rec": rec}).json() == ready


def test_viz_attribute_passes_a_worker_failure_through(client, seeded_corpus):
    seed, rec = seeded_corpus[0]["track_id"], seeded_corpus[1]["track_id"]
    store.put_attribution(seed, rec, {"status": "failed", "error": "no preview"}, ttl=60)

    body = client.get("/viz/attribute", params={"seed": seed, "rec": rec}).json()
    assert body["status"] == "failed"


def test_viz_attribute_400_when_seed_is_the_rec(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    resp = client.get("/viz/attribute", params={"seed": tid, "rec": tid})
    assert resp.status_code == 400


def test_viz_attribute_404_when_a_track_is_unanalyzed(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    assert client.get("/viz/attribute", params={"seed": tid, "rec": "nope"}).status_code == 404
    assert client.get("/viz/attribute", params={"seed": "nope", "rec": tid}).status_code == 404


def test_neighbouring_low_bands_bleed_at_the_default_window():
    """Why the worker analyzes at 8192. With the default 2048-point window
    the bins are 7.8 Hz and the taper spills past the narrow low bands, so
    stopping band 2 also eats a third of a tone that lives in band 1 — the
    two bands measure overlapping things. This pins the behaviour the worker
    is overriding; if it ever silently fell back, this is what it would get.
    """
    audio = _tone(80.0)                       # in band 1, not band 2
    neighbour = viz.band_stop(audio, 16000, *viz.band_edges()[1])

    assert _energy_at(neighbour, 80.0) < _energy_at(audio, 80.0) * 0.7


def test_adjacent_low_bands_stay_separable_at_the_attribution_window():
    """At the window the worker actually uses, band 1 removes its own tone
    and band 2 leaves it alone — the bands measure different things."""
    bands = viz.band_edges()
    audio = _tone(80.0)

    stopped = viz.band_stop(audio, 16000, *bands[0], fft_size=8192, hop=4096)
    untouched = viz.band_stop(audio, 16000, *bands[1], fft_size=8192, hop=4096)

    before = _energy_at(audio, 80.0)
    assert _energy_at(stopped, 80.0) < before / 100
    assert _energy_at(untouched, 80.0) > before * 0.95


def test_stop_mask_taper_width_does_not_depend_on_band_width():
    """The phone solos a band by keeping exactly what this mask removes, so
    the two must mirror each other bin for bin — including in the tapers. A
    width-adaptive taper here would silently break that pairing."""
    freqs = np.fft.rfftfreq(8192, 1.0 / 16000)

    wide = viz._stop_gain(freqs, *viz.band_edges()[-1], taper_bins=4)
    narrow = viz._stop_gain(freqs, *viz.band_edges(count=40)[0], taper_bins=4)

    assert np.count_nonzero((wide > 0) & (wide < 1)) == 8
    assert np.count_nonzero((narrow > 0) & (narrow < 1)) == 8


# ---- T3: seed-anchored viz subset ----

def test_viz_subset_is_seed_nearest_and_bounded(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 3)
    # five tracks on a line; seed "c" in the middle
    vecs = {"a": [1, 0], "b": [0.9, 0.1], "c": [0.7, 0.3], "d": [0.5, 0.5], "e": [0, 1]}
    for tid, v in vecs.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    ids, matrix = app_mod._viz_subset("c")
    assert "c" in ids and len(ids) == 3
    assert set(ids) == {"b", "c", "d"}            # the two nearest plus the seed
    assert matrix.shape == (3, 2)
    assert ids == sorted(ids)                      # stable order
    ids2, _ = app_mod._viz_subset("c", extra_ids=["e"])
    assert set(ids2) == {"b", "c", "d", "e"}       # extras appended even when far


def test_viz_subset_without_seed_is_the_first_rows(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid in ("a", "b", "c"):
        store.put_track({**TRACK, "track_id": tid}, {"embedding": [1.0, 0.0]})
    ids, matrix = app_mod._viz_subset(None)
    assert ids == ["a", "b"] and matrix.shape == (2, 2)


def test_viz_subset_cache_tracks_a_grown_full_matrix(client, fake_mongo, monkeypatch):
    """The cached subset must not outlive the full matrix it was sliced
    from: id(matrix) can be reused by an unrelated array once _MATRIX_CACHE
    replaces the original during a crawl, so the cache verifies identity on
    the stored full matrix rather than trusting the id-keyed lookup alone."""
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 10)
    for tid, v in {"a": [1, 0], "b": [0.9, 0.1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    ids, _ = app_mod._viz_subset("a")
    assert set(ids) == {"a", "b"}

    store.put_track({**TRACK, "track_id": "c"}, {"embedding": [0.5, 0.5]})
    # Force _MATRIX_CACHE to grow via a fresh request through the client.
    client.get("/viz/hubs", params={"track_id": "a"})

    ids2, matrix2 = app_mod._viz_subset("a")
    assert "c" in ids2

    current_matrix = app_mod._MATRIX_CACHE["embedding"].matrix
    assert matrix2 is not current_matrix               # the subset is a copy, not a view
    cached = app_mod._SUBSET_CACHE[(id(current_matrix), "a", ())]
    assert cached[0] is current_matrix                  # the cache entry tracks the CURRENT full matrix


def test_tour_and_mst_share_the_subset_for_a_seed(client, fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 3)
    for tid, v in {"a": [1, 0], "b": [0.9, 0.1], "c": [0.7, 0.3], "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    tour = client.get("/viz/tour?track_id=a").json()
    mst = client.get("/viz/mst?track_id=a").json()
    assert tour["ids"] == mst["ids"] and len(tour["ids"]) == 3 and "d" not in tour["ids"]
    hubs = client.get("/viz/hubs?track_id=a").json()
    assert {h["track_id"] for h in hubs["all_counts"]} == set(tour["ids"])


def test_map_subset_includes_surprise_recs(client, fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid, v in {"a": [1, 0], "b": [0.99, 0.01], "c": [0.98, 0.02], "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    body = client.get("/viz/map?track_id=a&axis=surprise&limit=1&correction=off").json()
    rec_ids = {r["track_id"] for r in body["recs"]}
    assert rec_ids <= set(body["points"]["ids"])   # far recs are still drawn


def test_viz_tour_includes_requested_recs_even_when_far(client, fake_mongo,
                                                        monkeypatch):
    """The global endpoints take the same recs /viz/map drew, so a `surprise`
    rec outside the seed's nearest ring still has a row to highlight."""
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid, v in {"a": [1, 0], "b": [0.99, 0.01], "c": [0.98, 0.02],
                   "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})

    plain = client.get("/viz/tour", params={"track_id": "a"}).json()
    assert "d" not in plain["ids"]

    for route in ("/viz/tour", "/viz/mst", "/viz/hubs", "/viz/extremes"):
        body = client.get(route, params={"track_id": "a", "recs": "d"}).json()
        ids = (body["ids"] if "ids" in body
               else [row["track_id"] for row in body["all_counts"]]
               if "all_counts" in body
               else [t["track_id"] for t in body["low"] + body["high"]])
        assert "d" in ids, route


def test_viz_tour_ignores_unknown_and_overlong_rec_lists(client, fake_mongo,
                                                         monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid, v in {"a": [1, 0], "b": [0.99, 0.01], "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    body = client.get("/viz/tour",
                      params={"track_id": "a", "recs": "nope, ,d"}).json()
    assert "d" in body["ids"] and "nope" not in body["ids"]
    # Past the cap the tail is dropped rather than widening the subset.
    assert app_mod._rec_ids(",".join(str(i) for i in range(80))) == tuple(
        str(i) for i in range(50)
    )


def test_viz_walk_from_equals_to_is_a_single_step(client, seeded_corpus):
    """Degenerate but reachable from the UI (tap the seed as destination):
    one step, zero length, and a detour of 1.0 rather than a divide by zero."""
    tid = seeded_corpus[0]["track_id"]
    response = client.get("/viz/walk", params={"from": tid, "to": tid, "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert [step["track_id"] for step in body["path"]] == [tid]
    assert body["geodesic"] == 0.0 and body["ambient"] == 0.0
    assert body["detour"] == 1.0


def test_unit_cache_holds_the_matrix_not_a_bare_id(fake_mongo):
    """A bare id() key let the array it named be freed and its address
    reused, serving a unit matrix built from a different corpus."""
    from music_recommendations.server import app as app_mod
    app_mod._UNIT_CACHE.clear()
    first = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    seed = np.array([1.0, 0.0])
    app_mod._similarity("embedding", first, seed, "cosine")
    assert app_mod._UNIT_CACHE["embedding"][0] is first   # the matrix itself

    second = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    got = app_mod._similarity("embedding", second, seed, "cosine")
    assert app_mod._UNIT_CACHE["embedding"][0] is second  # recomputed, not reused
    assert list(got) == [0.0, 1.0]
    app_mod._UNIT_CACHE.clear()


def test_viz_subset_miss_purges_a_superseded_full_matrix(client, fake_mongo,
                                                         monkeypatch):
    """A stale subset entry holds a strong reference to the full matrix it
    was sliced from, so one of them pins a whole corpus matrix that
    _MATRIX_CACHE has already replaced. Every miss purges them."""
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 10)
    for tid, v in {"a": [1, 0], "b": [0.9, 0.1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    app_mod._viz_subset("a")
    app_mod._viz_subset("b")
    old_matrix = app_mod._MATRIX_CACHE["embedding"].matrix
    assert len(app_mod._SUBSET_CACHE) == 2

    store.put_track({**TRACK, "track_id": "c"}, {"embedding": [0.5, 0.5]})
    client.get("/viz/hubs", params={"track_id": "a"})      # grows the matrix

    assert all(value[0] is not old_matrix
               for value in app_mod._SUBSET_CACHE.values())
    live = [value[2] for value in app_mod._SUBSET_CACHE.values()]
    for cache in (app_mod._TOP8_CACHE, app_mod._MST_CACHE, app_mod._HUBS_CACHE):
        assert all(any(value[0] is subset for subset in live)
                   for value in cache.values())
