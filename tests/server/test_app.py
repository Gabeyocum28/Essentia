"""app.py: the four contract routes, mock-first with the store + analysis real path."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from music_recommendations.server import app as app_module
from music_recommendations.server import store
from contract.features import AXES, FEATURE_KEYS

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "contract" / "fixture.json").read_text()
)["tracks"]

TRACK_KEYS = {"track_id", "title", "artist", "album", "artwork_url", "preview_url"}


def fake_features(seed_val: float) -> dict:
    """Synthetic feature dict matching contract FEATURE_KEYS shapes."""
    out = {}
    for key, dim in FEATURE_KEYS.items():
        if dim == 1:
            out[key] = seed_val
        else:
            v = [0.0] * dim
            v[0] = 1.0
            v[1] = seed_val
            out[key] = v
    return out


@pytest.fixture
def client(fake_mongo):
    return TestClient(app_module.app)


@pytest.fixture
def seeded_corpus(fake_mongo):
    """Five analyzed tracks in the fake store, spread across feature space."""
    for i, t in enumerate(FIXTURE[:5]):
        store.put_track(t, fake_features(i / 4.0))
    return FIXTURE[:5]


# ---- /axes ----

def test_axes_serves_contract_list_verbatim(client):
    assert client.get("/axes").json() == {"axes": AXES}


# ---- /search ----

def test_search_proxies_deezer(client, monkeypatch):
    hits = [dict(FIXTURE[0])]
    monkeypatch.setattr(app_module.deezer, "search", lambda q, limit=10: hits)
    body = client.get("/search", params={"q": "so what"}).json()
    # Deezer's fields pass through untouched except preview_url, which is
    # re-pointed at this server so it does not expire in the client's hands
    # (see _playable); test_search_serves_this_servers_preview_urls covers it.
    assert [{k: v for k, v in t.items() if k != "preview_url"}
            for t in body["results"]] == [
        {k: v for k, v in t.items() if k != "preview_url"} for t in hits
    ]


def test_search_falls_back_to_fixture_when_deezer_down(client, monkeypatch):
    def boom(q, limit=10):
        raise OSError("no network")

    monkeypatch.setattr(app_module.deezer, "search", boom)
    body = client.get("/search", params={"q": "miles"}).json()
    assert len(body["results"]) > 0
    assert all("miles" in t["artist"].lower() or "miles" in t["title"].lower()
               for t in body["results"])
    assert all(set(t) == TRACK_KEYS for t in body["results"])


# ---- /seed ----

def test_seed_warm_track_is_instant_and_never_analyzes(client, seeded_corpus, monkeypatch):
    def boom(path):
        raise AssertionError("analyze_track must not run for a warm seed")

    monkeypatch.setattr(app_module, "analyze_track", boom)
    tid = seeded_corpus[0]["track_id"]
    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}


def test_seed_cold_track_downloads_analyzes_and_stores(client, fake_mongo, monkeypatch):
    track = dict(FIXTURE[7])
    tid = track["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))
    monkeypatch.setattr(app_module, "analyze_track", lambda p: fake_features(0.5))

    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}
    assert store.get_features(tid) is not None
    # get_track never round-trips preview_url (a signed URL is never stored).
    assert store.get_track(tid) == {**track, "preview_url": ""}


@pytest.fixture
def analysis_unavailable(monkeypatch):
    """The ARM-VM condition: essentia can't import, plus instant poll timing."""
    def not_implemented(path):
        raise NotImplementedError

    monkeypatch.setattr(app_module, "analyze_track", not_implemented)
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)


def test_seed_enqueues_and_reports_unanalyzed_on_timeout(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    track = dict(FIXTURE[0])
    tid = track["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))

    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}
    # metadata stored for the worker, job queued, but corpus untouched
    assert store.get_track(tid) == {**track, "preview_url": ""}
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"
    assert tid not in store.corpus_ids()


def test_seed_ready_when_worker_delivers_features(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    track = dict(FIXTURE[1])
    tid = track["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))
    # First get_features call (the warm check) misses and plants the
    # features, as if the worker finished during the wait; later polls hit.
    real_get = store.get_features
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 1.0)
    polled = {"n": 0}

    def get_features_then_appear(track_id):
        polled["n"] += 1
        if polled["n"] == 1:
            store.put_track(track, fake_features(0.5))
            return None
        return real_get(track_id)

    monkeypatch.setattr(store, "get_features", get_features_then_appear)
    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}


def test_seed_double_tap_enqueues_once(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    track = dict(FIXTURE[2])
    tid = track["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))
    client.post("/seed", json={"track_id": tid})
    client.post("/seed", json={"track_id": tid})
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"


def test_seed_store_down_degrades_to_ready(client, monkeypatch):
    """No fake_mongo fixture: store.db() raises -> legacy mock-first path."""
    def store_down():
        raise ConnectionError("store down")

    monkeypatch.setattr(store, "db", store_down)
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(FIXTURE[3]))
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))

    def not_implemented(path):
        raise NotImplementedError

    monkeypatch.setattr(app_module, "analyze_track", not_implemented)
    tid = FIXTURE[3]["track_id"]
    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}


def test_seed_download_failure_queues_instead_of_500(client, fake_mongo, monkeypatch):
    """A Deezer preview fetch failure (timeout, flake) must queue for the
    embed worker like a missing-essentia host does, not 500."""
    def boom(url):
        raise OSError("preview fetch failed")

    tid = FIXTURE[0]["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(FIXTURE[0]))
    monkeypatch.setattr(app_module, "_download_preview", boom)
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)

    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"
    assert tid not in store.corpus_ids()


def test_seed_unknown_track_404(client, fake_mongo, monkeypatch):
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: None)
    assert client.post("/seed", json={"track_id": "doesnotexist"}).status_code == 404


# ---- /recommend ----

def test_recommend_returns_scored_tracks_excluding_seed(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get("/recommend", params={"track_id": tid, "axis": "sounds_like"}).json()
    assert body["seed_track_id"] == tid
    assert body["axis"] == "sounds_like"
    ids = [t["track_id"] for t in body["results"]]
    assert tid not in ids
    assert len(ids) == 4
    for t in body["results"]:
        assert set(t) == TRACK_KEYS | {"score"}
        assert isinstance(t["score"], float)


def test_recommend_orders_by_similarity(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]  # seed_val 0.0; nearest is seed_val 0.25
    body = client.get("/recommend", params={"track_id": tid, "axis": "sounds_like"}).json()
    scores = [t["score"] for t in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["results"][0]["track_id"] == seeded_corpus[1]["track_id"]


def test_recommend_surprise_inverts_order(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    near = client.get("/recommend", params={"track_id": tid, "axis": "sounds_like"}).json()
    far = client.get("/recommend", params={"track_id": tid, "axis": "surprise"}).json()
    assert [t["track_id"] for t in far["results"]] == \
        [t["track_id"] for t in near["results"]][::-1]


def test_recommend_respects_limit(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    body = client.get(
        "/recommend", params={"track_id": tid, "axis": "sounds_like", "limit": 2}
    ).json()
    assert len(body["results"]) == 2


def test_recommend_unknown_axis_400(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    r = client.get("/recommend", params={"track_id": tid, "axis": "vibes"})
    assert r.status_code == 400


def test_recommend_unseeded_track_falls_back_to_fixture(client, fake_mongo):
    """Mock-first: empty corpus -> fixture tracks with dummy descending scores."""
    body = client.get(
        "/recommend", params={"track_id": FIXTURE[0]["track_id"], "axis": "sounds_like"}
    ).json()
    ids = [t["track_id"] for t in body["results"]]
    assert len(ids) == 10
    assert FIXTURE[0]["track_id"] not in ids
    scores = [t["score"] for t in body["results"]]
    assert scores == sorted(scores, reverse=True)


# ---- no store at all (mock-first before anything lands) ----

@pytest.fixture
def no_store(monkeypatch):
    class Down:
        def __getattr__(self, name):
            raise ConnectionError("store is down")

    monkeypatch.setattr(store, "db", lambda: Down())


def test_seed_works_without_store(no_store, monkeypatch):
    client = TestClient(app_module.app)
    tid = FIXTURE[0]["track_id"]
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))
    # This test is about the store being down, not about analysis. Stub the
    # analyzer out: it used to be a NotImplementedError stub that app.py
    # swallowed, but now that the lane has landed it raises FileNotFoundError
    # for a path that does not exist, which app.py should NOT swallow.
    monkeypatch.setattr(app_module, "analyze_track", lambda path: {})
    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json()["status"] == "ready"


def test_recommend_works_without_store(no_store):
    client = TestClient(app_module.app)
    body = client.get(
        "/recommend", params={"track_id": FIXTURE[0]["track_id"], "axis": "sounds_like"}
    )
    assert body.status_code == 200
    assert len(body.json()["results"]) == 10


# ---- / signpost ----

def test_root_lists_routes(client):
    body = client.get("/").json()
    assert "/axes" in str(body)
    assert "/recommend" in str(body)


# ---- /recommend corpus matrix cache ----

def test_a_track_analyzed_after_the_first_request_still_appears(client, seeded_corpus):
    """The crawler writes continuously; a cached matrix must not freeze the corpus."""
    seed = seeded_corpus[0]["track_id"]
    first = client.get("/recommend", params={"track_id": seed, "axis": "sounds_like",
                                             "limit": 10}).json()["results"]
    assert len(first) == 4

    late = FIXTURE[5]
    store.put_track(late, fake_features(0.5))

    second = client.get("/recommend", params={"track_id": seed, "axis": "sounds_like",
                                              "limit": 10}).json()["results"]
    assert late["track_id"] in {t["track_id"] for t in second}
    assert len(second) == 5


def test_repeat_requests_do_not_re_read_the_whole_corpus(client, seeded_corpus, monkeypatch):
    """Re-parsing every feature blob per request is what made /recommend 25s.

    The cold embedding matrix now comes from one store.base_matrix() query
    instead of a per-track store.get_many_features() fetch, so the first
    request's "whole matrix" read shows up there; get_many_features is only
    hit for tracks analyzed after that base read.
    """
    base_reads = []
    real_base = store.base_matrix
    monkeypatch.setattr(store, "base_matrix",
                        lambda: (base_reads.append(1), real_base())[1])
    reads = []
    real = store.get_many_features
    monkeypatch.setattr(store, "get_many_features",
                        lambda ids: reads.append(list(ids)) or real(ids))

    seed = seeded_corpus[0]["track_id"]
    params = {"track_id": seed, "axis": "sounds_like", "limit": 10}
    client.get("/recommend", params=params)
    assert len(base_reads) == 1, "the first request builds the whole matrix in one query"
    assert reads == [], "no per-track fetches when the base matrix covers everyone"

    base_reads.clear()
    reads.clear()
    client.get("/recommend", params=params)
    assert base_reads == [] and reads == [], "an unchanged corpus should be read zero times"

    store.put_track(FIXTURE[5], fake_features(0.5))
    reads.clear()
    client.get("/recommend", params=params)
    assert reads == [[FIXTURE[5]["track_id"]]], "only the new track is parsed"


def test_seed_is_never_recommended_to_itself(client, seeded_corpus):
    """The seed is a row in the shared matrix now, so it must be filtered out."""
    for axis in ("sounds_like", "surprise"):
        seed = seeded_corpus[2]["track_id"]
        results = client.get("/recommend", params={"track_id": seed, "axis": axis,
                                                   "limit": 10}).json()["results"]
        assert seed not in {t["track_id"] for t in results}
        assert len(results) == 4


def test_seed_falls_back_when_essentia_unavailable(client, fake_mongo, monkeypatch):
    """ARM VM: essentia has no aarch64 wheels, so analyze_track raises
    ImportError there. Seed must queue for the worker, not 500, and reports
    unanalyzed rather than the old silent-ready fixture fallback."""
    def no_essentia(path):
        raise ImportError("No module named 'essentia'")

    tid = FIXTURE[0]["track_id"]
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(FIXTURE[0]))
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))
    monkeypatch.setattr(app_module, "analyze_track", no_essentia)
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)
    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}


def test_recommend_limit_is_capped(client, seeded_corpus):
    """limit=500 dumped the whole corpus via the public route; cap it."""
    tid = seeded_corpus[0]["track_id"]
    r = client.get("/recommend", params={"track_id": tid, "axis": "sounds_like",
                                         "limit": 500})
    assert r.status_code == 422


def test_recommend_limit_50_is_allowed(client, seeded_corpus):
    tid = seeded_corpus[0]["track_id"]
    r = client.get("/recommend", params={"track_id": tid, "axis": "sounds_like",
                                         "limit": 50})
    assert r.status_code == 200


# ---- the axis list ----

def test_axes_serves_exactly_the_two_buttons(client):
    served = client.get("/axes").json()["axes"]
    assert [a["id"] for a in served] == ["sounds_like", "surprise"]
    assert [a["label"] for a in served] == [
        "More sounds like this", "Nothing like this",
    ]


def test_retired_best_match_axis_is_rejected(client, seeded_corpus):
    """It was a real axis; a stale client pressing it must 400, not 500."""
    r = client.get("/recommend", params={"track_id": seeded_corpus[0]["track_id"],
                                         "axis": "best_match"})
    assert r.status_code == 400


def test_every_served_axis_is_accepted_by_recommend(client):
    """A button /axes advertises must not 400 when the client presses it."""
    served = {a["id"] for a in client.get("/axes").json()["axes"]}
    for axis in served:
        assert client.get("/recommend", params={"track_id": "x", "axis": axis}).status_code == 200


# ---- GET /preview + stable preview URLs ----
#
# The stored preview_url is a ~15-minute Deezer signature, so every one in the
# corpus is dead. Tracks must therefore go out pointing at this server, and
# /preview re-signs at play time.

@pytest.fixture
def deezer_previews(monkeypatch):
    """Count re-signing calls so cache behavior is observable."""
    calls = []

    def fresh(track_id):
        calls.append(track_id)
        return f"https://cdnt-preview.dzcdn.net/{track_id}.mp3?hdnea=exp=999"

    monkeypatch.setattr(app_module.deezer, "fresh_preview_url", fresh)
    return calls


def test_preview_redirects_to_freshly_signed_url(client, deezer_previews):
    r = client.get("/preview/721063", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == (
        "https://cdnt-preview.dzcdn.net/721063.mp3?hdnea=exp=999"
    )
    assert deezer_previews == ["721063"]


def test_preview_is_never_cached_by_the_client(client, deezer_previews):
    # The target dies in ~15 min; a cached 302 outlives what it points at.
    r = client.get("/preview/721063", follow_redirects=False)
    assert r.headers["cache-control"] == "no-store"


def test_preview_reuses_the_cached_signature(client, deezer_previews):
    for _ in range(3):
        client.get("/preview/721063", follow_redirects=False)
    assert deezer_previews == ["721063"], "should re-sign once, then cache"


def test_preview_404s_when_deezer_has_no_preview(client, monkeypatch):
    monkeypatch.setattr(app_module.deezer, "fresh_preview_url", lambda t: None)
    assert client.get("/preview/nope", follow_redirects=False).status_code == 404


def test_preview_survives_store_being_down(monkeypatch, deezer_previews):
    """No cache is a slower /preview, not a broken one."""
    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "db", boom)
    r = TestClient(app_module.app).get("/preview/721063", follow_redirects=False)
    assert r.status_code == 302


def test_recommend_serves_this_servers_preview_urls(client, seeded_corpus):
    body = client.get(
        f"/recommend?track_id={seeded_corpus[0]['track_id']}&axis=sounds_like"
    ).json()
    assert body["results"], "expected recommendations"
    for track in body["results"]:
        assert track["preview_url"] == (
            f"http://testserver/preview/{track['track_id']}"
        )
        assert "dzcdn.net" not in track["preview_url"]


def test_search_serves_this_servers_preview_urls(client, monkeypatch):
    monkeypatch.setattr(app_module.deezer, "search", lambda q: [dict(FIXTURE[0])])
    body = client.get("/search?q=miles").json()
    assert body["results"][0]["preview_url"] == (
        f"http://testserver/preview/{FIXTURE[0]['track_id']}"
    )


def test_track_shape_is_unchanged_by_the_rewrite(client, seeded_corpus):
    body = client.get(
        f"/recommend?track_id={seeded_corpus[0]['track_id']}&axis=sounds_like"
    ).json()
    assert set(body["results"][0]) == TRACK_KEYS | {"score"}


def test_public_base_url_overrides_the_request_host(client, seeded_corpus,
                                                    monkeypatch):
    """Behind a tunnel or proxy the Host header is the internal name."""
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://essencia.example.com/")
    body = client.get(
        f"/recommend?track_id={seeded_corpus[0]['track_id']}&axis=sounds_like"
    ).json()
    assert body["results"][0]["preview_url"].startswith(
        "https://essencia.example.com/preview/"
    )


def test_playable_needs_no_stored_preview_url(monkeypatch):
    """The store never round-trips preview_url, so the rewrite cannot require one.

    Guarding on preview_url being present meant every corpus track went out
    with no way to play it.
    """
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://host.example")
    assert app_module._playable({"track_id": "x"})["preview_url"] == (
        "https://host.example/preview/x"
    )
    assert app_module._playable({"track_id": "x", "preview_url": None})[
        "preview_url"] == "https://host.example/preview/x"


def test_playable_leaves_placeholder_rows_alone(monkeypatch):
    """viz passes None for ids missing from the corpus; those keep a null."""
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://host.example")
    assert app_module._playable(None) is None


def test_recommend_serves_previews_for_corpus_tracks(client, fake_mongo,
                                                     monkeypatch):
    """End to end: a track with no stored preview still gets a playable URL."""
    for i, t in enumerate(FIXTURE[:4]):
        stripped = {k: v for k, v in t.items() if k != "preview_url"}
        store.put_track(stripped, fake_features(i / 3.0))
    body = client.get(
        f"/recommend?track_id={FIXTURE[0]['track_id']}&axis=sounds_like"
    ).json()
    assert body["results"]
    for track in body["results"]:
        assert track["preview_url"] == (
            f"http://testserver/preview/{track['track_id']}"
        )


def test_seed_resigns_when_the_stored_url_is_expired(client, fake_mongo,
                                                     monkeypatch):
    """The 403 on a stored URL is the norm, not a flake -- re-sign, don't punt."""
    track = dict(FIXTURE[7])
    downloaded = []

    def download(url):
        downloaded.append(url)
        if "hdnea" not in url:          # the stored, expired one
            raise OSError("403 Forbidden")
        return Path("/tmp/x.mp3")

    track["preview_url"] = "https://cdnt-preview.dzcdn.net/dead.mp3"
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))
    monkeypatch.setattr(app_module.deezer, "fresh_preview_url",
                        lambda t: "https://cdnt-preview.dzcdn.net/live.mp3?hdnea=1")
    monkeypatch.setattr(app_module, "_download_preview", download)
    monkeypatch.setattr(app_module, "analyze_track", lambda p: fake_features(0.5))

    body = client.post("/seed", json={"track_id": track["track_id"]}).json()
    assert body == {"track_id": track["track_id"], "status": "ready"}
    assert len(downloaded) == 2, "should try stored, then the re-signed URL"


def test_seed_does_not_resign_when_the_stored_url_works(client, fake_mongo,
                                                        monkeypatch):
    """Re-signing costs a Deezer call; a working URL must not trigger one."""
    track = dict(FIXTURE[7])
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(track))
    monkeypatch.setattr(app_module, "_download_preview", lambda url: Path("/tmp/x.mp3"))
    monkeypatch.setattr(app_module, "analyze_track", lambda p: fake_features(0.5))
    monkeypatch.setattr(app_module.deezer, "fresh_preview_url", lambda t: (_ for _ in ()).throw(
        AssertionError("must not re-sign when the stored URL downloads")))

    client.post("/seed", json={"track_id": track["track_id"]})


# ---- _cold_matrix / 502 on bad previews ----

def test_cold_matrix_uses_base_matrix(fake_mongo, monkeypatch):
    for tid, vec in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        store.put_track({**FIXTURE[0], "track_id": tid}, {"embedding": vec})
    calls = []
    real = store.get_many_features
    monkeypatch.setattr(store, "get_many_features",
                        lambda ids: (calls.append(ids), real(ids))[1])
    ids, matrix = app_module._cold_matrix(("a", "b"), "embedding")
    assert ids == ["a", "b"] and matrix.shape == (2, 2)
    assert calls == []          # one matrix read, no per-track fetches


def test_seed_returns_502_when_analysis_fails(client, fake_mongo, monkeypatch, tmp_path):
    from music_recommendations.analysis import frontend

    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"x")
    monkeypatch.setattr(app_module, "_fetch_preview_audio", lambda tid, t: mp3)
    monkeypatch.setattr(app_module.deezer, "get_track", lambda t: dict(FIXTURE[0]))

    def boom(path):
        raise frontend.DecodeError("ffmpeg: bad file")

    monkeypatch.setattr(app_module, "analyze_track", boom)
    r = client.post("/seed", json={"track_id": "42"})
    assert r.status_code == 502
    assert r.json()["detail"] == "analysis failed"


# ---- indexes created at API startup ----

def test_startup_creates_indexes_for_a_read_only_api_process(fake_mongo):
    """A pure-reader API process never calls put_track/enqueue_* itself, so
    without a startup hook it would never call ensure_indexes() -- Atlas
    would run unindexed until some writer happened to start first."""
    with TestClient(app_module.app):
        pass
    assert "expires_at_1" in fake_mongo.cache.index_information()
