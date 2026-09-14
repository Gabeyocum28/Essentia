"""app.py: the four contract routes, mock-first with the store + analysis real path."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from statistics import pstdev

from music_recommendations.analysis.schema import FEATURES_VERSION
from music_recommendations.analysis import clap
from music_recommendations.server import app as app_module
from music_recommendations.server import deezer as deezer_api
from music_recommendations.server import store
from contract.features import AXES, FEATURE_KEYS, TRACK_FIELDS, TRACK_OPTIONAL_FIELDS

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "contract" / "fixture.json").read_text()
)["tracks"]

TRACK_KEYS = {"track_id", "title", "artist", "album", "artwork_url", "preview_url"}


def _contract_shape(track: dict, scored: bool = False) -> bool:
    """Exactly the contract Track (plus `score` on a recommendation), and
    nothing beyond the two optional keys the contract allows.

    Internal fields -- dedupe_key, the full attribution object, feel -- must
    never reach a response, so this is an exact-set check with a named
    allowance rather than a subset check."""
    keys = set(track)
    required = TRACK_KEYS | ({"score"} if scored else set())
    return keys >= required and keys - required <= set(TRACK_OPTIONAL_FIELDS)


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
    # Without it the row is version 0, which store.LIVE no longer serves.
    out["_features_version"] = FEATURES_VERSION
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
    body = client.get("/axes").json()
    assert body["axes"] == AXES
    # The axis list is the contract; `text_search` is this host's capability
    # flag beside it (see get_axes), and is always present as a bool.
    assert set(body) == {"axes", "text_search"}
    assert isinstance(body["text_search"], bool)


# ---- /search ----

def test_search_proxies_deezer(client, monkeypatch):
    hits = [dict(FIXTURE[0])]
    monkeypatch.setattr(deezer_api, "search", lambda q, limit=10: hits)
    body = client.get("/search", params={"q": "so what"}).json()
    # Deezer's fields pass through untouched except preview_url, which is
    # re-pointed at this server so it does not expire in the client's hands
    # (see _playable); test_search_serves_this_servers_preview_urls covers it.
    assert [{k: v for k, v in t.items() if k not in ("preview_url", "source")}
            for t in body["results"]] == [
        {k: v for k, v in t.items() if k != "preview_url"} for t in hits
    ]
    # ...plus the source that answered, which the clients use for attribution.
    assert [t["source"] for t in body["results"]] == ["deezer"]


def test_search_falls_back_to_fixture_when_deezer_down(client, monkeypatch):
    def boom(q, limit=10):
        raise OSError("no network")

    monkeypatch.setattr(deezer_api, "search", boom)
    body = client.get("/search", params={"q": "miles"}).json()
    assert len(body["results"]) > 0
    assert all("miles" in t["artist"].lower() or "miles" in t["title"].lower()
               for t in body["results"])
    assert all(_contract_shape(t) for t in body["results"])


# ---- /seed ----

def test_seed_warm_track_is_instant_and_never_queues(client, seeded_corpus,
                                                     fake_mongo):
    tid = seeded_corpus[0]["track_id"]
    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"}) is None


def test_seed_never_runs_the_model_in_the_api_process(client, fake_mongo,
                                                      analysis_unavailable,
                                                      monkeypatch):
    """The API must not load CLAP. The audio tower is ~700 MB of weights and
    ~2.5 GB resident, the worker on the same box already holds a copy, and a
    second one on an unlucky request is an OOM kill -- so a cold seed is
    handed to the worker instead of analyzed here.

    Guarded structurally, not by stubbing: the module must not even have an
    analyzer bound to call.
    """
    track = dict(FIXTURE[7])
    tid = track["track_id"]
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))

    assert not hasattr(app_module, "analyze_track")
    assert not hasattr(app_module, "_fetch_preview_audio")

    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "unanalyzed"}
    # The metadata IS stored for the worker to analyze against; get_track
    # never round-trips preview_url (a signed URL is never stored).
    assert store.get_track(tid) == {**track, "preview_url": "", "source": "deezer"}
    assert store.get_features(tid) is None
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"


def test_seed_does_not_fetch_the_preview_either(client, fake_mongo,
                                                analysis_unavailable,
                                                monkeypatch):
    """Downloading the preview was only ever the first half of analyzing it.
    The worker re-fetches a fresh one anyway (a Deezer signature is a
    15-minute lease), so a download here is pure duplicated traffic."""
    def boom(url, *a, **k):
        raise AssertionError("/seed must not fetch audio")

    monkeypatch.setattr(app_module.urllib.request, "urlopen", boom)
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(FIXTURE[8]))
    tid = FIXTURE[8]["track_id"]
    assert client.post("/seed", json={"track_id": tid}).json() == {
        "track_id": tid, "status": "unanalyzed"}


@pytest.fixture
def analysis_unavailable(monkeypatch):
    """Instant poll timing: the worker is not running in these tests, so the
    wait for it to deliver features should not cost 20 real seconds."""
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)


def test_seed_enqueues_and_reports_unanalyzed_on_timeout(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    track = dict(FIXTURE[0])
    tid = track["track_id"]
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))

    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}
    # metadata stored for the worker, job queued, but corpus untouched
    assert store.get_track(tid) == {**track, "preview_url": "", "source": "deezer"}
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"
    assert tid not in store.corpus_ids()


def test_seed_ready_when_worker_delivers_features(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    track = dict(FIXTURE[1])
    tid = track["track_id"]
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))
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
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))
    client.post("/seed", json={"track_id": tid})
    client.post("/seed", json={"track_id": tid})
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"


def test_seed_store_down_degrades_to_ready(client, monkeypatch):
    """No fake_mongo fixture: store.db() raises -> legacy mock-first path."""
    def store_down():
        raise ConnectionError("store down")

    monkeypatch.setattr(store, "db", store_down)
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(FIXTURE[3]))
    tid = FIXTURE[3]["track_id"]
    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}


def test_seed_unplayable_track_queues_instead_of_500(client, fake_mongo, monkeypatch):
    """Nothing about the audio is known at /seed time any more, so a dead
    preview is the worker's problem: the response is a 200 saying the track
    is not analyzed yet, not a 500 and not a 502."""
    tid = FIXTURE[0]["track_id"]
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(FIXTURE[0]))
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)

    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"})["state"] == "queued"
    assert tid not in store.corpus_ids()


def test_seed_on_a_superseded_row_is_not_ready(client, fake_mongo,
                                               analysis_unavailable, monkeypatch):
    """A row the PREVIOUS stack analyzed has features, but not ones this
    server can rank: its vector is in a different space entirely. Answering
    "ready" sends the client straight to a /recommend that cannot work."""
    track = dict(FIXTURE[4])
    tid = track["track_id"]
    store.put_track(track, {**fake_features(0.5),
                            "_features_version": FEATURES_VERSION - 1})
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))

    body = client.post("/seed", json={"track_id": tid}).json()

    assert body == {"track_id": tid, "status": "unanalyzed"}


def test_seed_on_a_superseded_row_jumps_the_reanalysis_queue(
        client, fake_mongo, analysis_unavailable, monkeypatch):
    """Somebody is blocked on this one track, and the whole backlog is in
    front of it. It goes to the front, and it is NOT queued as a cold embed:
    the row already has its metadata, only its vectors are out of date."""
    track = dict(FIXTURE[4])
    tid = track["track_id"]
    store.put_track(track, {**fake_features(0.5),
                            "_features_version": FEATURES_VERSION - 1})
    store.put_track({**FIXTURE[3], "track_id": "older"},
                    {**fake_features(0.1), "_features_version": FEATURES_VERSION - 1})
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))

    client.post("/seed", json={"track_id": tid})

    assert store.stale_ids(2)[0] == tid
    assert "reanalyze_priority" in fake_mongo.tracks.find_one({"_id": tid})
    assert fake_mongo.jobs.find_one({"_id": f"embed:{tid}"}) is None


def test_seed_is_ready_once_the_row_reaches_the_current_version(
        client, fake_mongo, monkeypatch):
    """The wait is version-aware too: a poll that stopped at "there are some
    features" would return ready on exactly the rows it just rejected."""
    track = dict(FIXTURE[4])
    tid = track["track_id"]
    store.put_track(track, {**fake_features(0.5),
                            "_features_version": FEATURES_VERSION - 1})
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 1.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(track))
    polled = {"n": 0}
    real_get = store.get_features

    def get_features_then_upgrade(track_id):
        polled["n"] += 1
        if polled["n"] == 2:
            store.put_track(track, fake_features(0.5))
        return real_get(track_id)

    monkeypatch.setattr(store, "get_features", get_features_then_upgrade)

    body = client.post("/seed", json={"track_id": tid}).json()
    assert body == {"track_id": tid, "status": "ready"}


def test_seed_unknown_track_404(client, fake_mongo, monkeypatch):
    monkeypatch.setattr(deezer_api, "get_track", lambda t: None)
    assert client.post("/seed", json={"track_id": "doesnotexist"}).status_code == 404


# ---- /recommend: duplicate collapsing ----
#
# Deezer publishes one recording under several ids; the corpus can hold the
# remaster, the live take and the plain release side by side. The result
# list must show each RECORDING once -- while still showing a genuine cover
# by another artist.

def _angled(theta: float) -> dict:
    """A unit embedding at `theta` radians from the seed: the cosine between
    two of these is exactly cos(theta_a - theta_b), so a test can ask for a
    near-identical pair (0.9998) or a plain neighbour (0.92) by angle."""
    v = [0.0] * FEATURE_KEYS["embedding"]
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return {"embedding": v, "_features_version": FEATURES_VERSION}


# id, title, artist, angle from the seed (radians)
DUPLICATES = [
    ("s",  "So What",                               "Miles Davis", 0.0),
    ("d1", "So What (2009 Remaster)",               "Miles Davis", 0.02),
    ("a1", "Blue in Green",                         "Miles Davis", 0.4),
    ("a2", "Blue in Green - Live at the Blackhawk", "Miles Davis", 0.6),
    ("c1", "Blue in Green",                         "Bill Evans",  0.8),
    ("e1", "Flamenco Sketches",                     "Miles Davis", 1.2),
    ("e2", "Sketches of Flamenco",                  "Nobody Else", 1.22),
    ("f1", "All Blues",                             "Miles Davis", 1.6),
]
# One entry per distinct recording, in ranked order: a1 and c1 share a title
# but not an artist (a cover), a2 is a1 again, d1 is the seed again, and e2
# is e1 again under a title that does not say so.
DISTINCT = ["a1", "c1", "e1", "f1"]


@pytest.fixture
def duplicate_corpus(fake_mongo):
    for track_id, title, artist, theta in DUPLICATES:
        store.put_track({"track_id": track_id, "title": title, "artist": artist,
                         "album": "Kind of Blue", "artwork_url": "u"},
                        _angled(theta))
    return DUPLICATES


def _rec_ids(client, **params):
    body = client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            **params}).json()
    return [t["track_id"] for t in body["results"]]


def test_recommend_returns_each_recording_once(client, duplicate_corpus):
    assert _rec_ids(client) == DISTINCT


def test_recommend_never_returns_the_seeds_own_re_release(client, duplicate_corpus):
    assert "d1" not in _rec_ids(client)
    assert "d1" not in _rec_ids(client, axis="surprise")


def test_recommend_keeps_a_cover_by_another_artist(client, duplicate_corpus):
    # c1 is "Blue in Green" by Bill Evans: same title as a1, different
    # artist, cosine 0.92 -- a real recommendation, not a re-release.
    assert "c1" in _rec_ids(client)


def test_recommend_drops_a_near_identical_embedding(client, duplicate_corpus):
    # e2's title and artist say nothing, but it sits 0.02 rad from e1
    # (cosine 0.9998) -- the same recording under another name.
    ids = _rec_ids(client)
    assert "e1" in ids and "e2" not in ids


def test_recommend_widens_the_scan_to_stay_limit_long(client, duplicate_corpus):
    # Three distinct recordings exist above the duplicates, so a limit of 3
    # must still come back three long rather than short by the skips.
    assert _rec_ids(client, limit=3) == DISTINCT[:3]


def test_recommend_results_carry_exactly_the_contract_fields(client, duplicate_corpus):
    body = client.get("/recommend", params={"track_id": "s", "axis": "sounds_like"}).json()
    # dedupe_key is stored on every track document; it must never reach a
    # response, here or anywhere else.
    for track in body["results"]:
        assert _contract_shape(track, scored=True)


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
        assert _contract_shape(t, scored=True)
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


# ---- /recommend corpus matrix cache ----

def test_a_track_analyzed_after_the_first_request_still_appears(client, seeded_corpus):
    """The crawler writes continuously; a cached matrix must not freeze the corpus."""
    seed = seeded_corpus[0]["track_id"]
    first = client.get("/recommend", params={"track_id": seed, "axis": "sounds_like",
                                             "limit": 10}).json()["results"]
    assert len(first) == 4

    late = FIXTURE[5]
    # 1.5, not 0.5: seeded_corpus already holds a track at 0.5, and an
    # identical embedding is now collapsed out of the result list as the
    # same recording. This test is about the matrix cache, not dedupe.
    store.put_track(late, fake_features(1.5))

    second = client.get("/recommend", params={"track_id": seed, "axis": "sounds_like",
                                              "limit": 10}).json()["results"]
    assert late["track_id"] in {t["track_id"] for t in second}
    assert len(second) == 5


def test_repeat_requests_do_not_re_read_the_whole_corpus(client, seeded_corpus, monkeypatch):
    """Re-parsing every feature blob per request is what made /recommend 25s.

    The cold embedding matrix now comes from one store.base_matrix() query
    instead of a per-track store.get_many_features() fetch, and the feel
    matrix has a projection of its own (store.get_many_feel), so the generic
    per-track feature read is only hit for tracks analyzed after that base
    read.
    """
    base_reads = []
    real_base = store.base_matrix
    monkeypatch.setattr(store, "base_matrix",
                        lambda: (base_reads.append(1), real_base())[1])
    reads = []
    real = store.get_many_features
    monkeypatch.setattr(store, "get_many_features",
                        lambda ids: reads.append(list(ids)) or real(ids))
    feel_reads = []
    real_feel = store.get_many_feel
    monkeypatch.setattr(store, "get_many_feel",
                        lambda ids: feel_reads.append(list(ids)) or real_feel(ids))

    seed = seeded_corpus[0]["track_id"]
    params = {"track_id": seed, "axis": "sounds_like", "limit": 10}
    client.get("/recommend", params=params)
    corpus = sorted(t["track_id"] for t in seeded_corpus)
    assert len(base_reads) == 1, "the first request builds the whole matrix in one query"
    assert reads == [], "the whole matrix is one base_matrix() query, nothing more"
    assert feel_reads == [corpus], "the feel matrix is one bulk read, not one per track"

    base_reads.clear()
    reads.clear()
    feel_reads.clear()
    client.get("/recommend", params=params)
    assert base_reads == [] and reads == [] and feel_reads == [], \
        "an unchanged corpus should be read zero times"

    store.put_track(FIXTURE[5], fake_features(0.5))
    reads.clear()
    feel_reads.clear()
    client.get("/recommend", params=params)
    # One growth read per matrix, each carrying the new id alone.
    assert reads == [[FIXTURE[5]["track_id"]]], "only the new track is parsed"
    assert feel_reads == [[FIXTURE[5]["track_id"]]]


def test_seed_is_never_recommended_to_itself(client, seeded_corpus):
    """The seed is a row in the shared matrix now, so it must be filtered out."""
    for axis in ("sounds_like", "surprise"):
        seed = seeded_corpus[2]["track_id"]
        results = client.get("/recommend", params={"track_id": seed, "axis": axis,
                                                   "limit": 10}).json()["results"]
        assert seed not in {t["track_id"] for t in results}
        assert len(results) == 4


def test_seed_on_a_host_without_the_analysis_extra(client, fake_mongo, monkeypatch):
    """The API image need not carry torch at all -- it never analyzes. A cold
    seed queues for the worker and reports unanalyzed rather than the old
    silent-ready fixture fallback."""
    tid = FIXTURE[0]["track_id"]
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(FIXTURE[0]))
    monkeypatch.setattr(app_module, "_EMBED_WAIT_S", 0.0)
    monkeypatch.setattr(app_module, "_EMBED_POLL_S", 0.0)
    body = client.post("/seed", json={"track_id": tid})
    assert body.status_code == 200
    assert body.json() == {"track_id": tid, "status": "unanalyzed"}


def test_recommend_with_a_seed_from_another_feature_space_is_not_a_500(
        client, seeded_corpus, fake_mongo):
    """A superseded seed (a 1280-d vector against a 1024-d corpus) used to
    reach numpy as a shape error and come back as a 500. It is a known,
    explainable state -- the row is waiting for re-analysis -- so it gets a
    409 that says so."""
    store.put_track({**FIXTURE[6], "track_id": "old"},
                    {"embedding": [0.1] * 3, "feel": [0.5] * 8,
                     "_features_version": FEATURES_VERSION - 1})

    r = client.get("/recommend", params={"track_id": "old",
                                         "axis": "sounds_like"})

    assert r.status_code == 409
    assert r.json()["detail"]["status"] == "unanalyzed"


def test_viz_map_with_a_mismatched_seed_is_not_a_500(client, seeded_corpus,
                                                     fake_mongo):
    """Same guard on the insights path: it ranks the same seed against the
    same matrix, so it fails the same way without it."""
    store.put_track({**FIXTURE[6], "track_id": "old"},
                    {"embedding": [0.1] * 3, "feel": [0.5] * 8,
                     "_features_version": FEATURES_VERSION - 1})

    r = client.get("/viz/map", params={"track_id": "old", "axis": "sounds_like"})

    assert r.status_code == 409
    assert r.json()["detail"]["status"] == "unanalyzed"


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

    monkeypatch.setattr(deezer_api, "fresh_preview_url", fresh)
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
    monkeypatch.setattr(deezer_api, "fresh_preview_url", lambda t: None)
    assert client.get("/preview/nope", follow_redirects=False).status_code == 404


def test_preview_survives_store_being_down(monkeypatch, deezer_previews):
    """No cache is a slower /preview, not a broken one."""
    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "db", boom)
    r = TestClient(app_module.app).get("/preview/721063", follow_redirects=False)
    assert r.status_code == 302


# ---- GET /preview/{id}/audio: the same-origin mp3 stream for web SOUND mode ----


class _FakeUpstream:
    """Minimal stand-in for urlopen's response: chunked reads, closes once."""

    def __init__(self, payload: bytes, length: "int | None" = None):
        self._buf = payload
        self._pos = 0
        self.closed = False
        # urlopen's response exposes the upstream headers here; a CDN that
        # reports a length is what lets the <audio> element seek.
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def read(self, n: int = -1) -> bytes:
        chunk = self._buf[self._pos:self._pos + n] if n and n > 0 else self._buf[self._pos:]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def test_preview_audio_streams_the_mp3_bytes(client, deezer_previews, monkeypatch):
    payload = b"ID3" + bytes(200_000)
    opened = []

    def fake_open(url):
        opened.append(url)
        return _FakeUpstream(payload)

    monkeypatch.setattr(app_module, "_open_upstream", fake_open)

    r = client.get("/preview/721063/audio")
    assert r.status_code == 200
    assert r.content == payload
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.headers["cache-control"] == "private, max-age=600"
    assert opened == ["https://cdnt-preview.dzcdn.net/721063.mp3?hdnea=exp=999"]


def test_preview_audio_is_never_gzipped(client, deezer_previews, monkeypatch):
    """An mp3 is already compressed. Worse, GZipMiddleware would drop the
    Content-Length and the element would lose the ability to seek."""
    payload = b"ID3" + bytes(200_000)  # well over GZipMiddleware's minimum_size
    monkeypatch.setattr(app_module, "_open_upstream",
                        lambda url: _FakeUpstream(payload, length=len(payload)))

    r = client.get("/preview/721063/audio", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "identity"
    # Byte-identical to what the upstream handed us, not a gzip stream.
    assert r.content == payload


def test_preview_audio_forwards_the_length_and_allows_ranges(client, deezer_previews, monkeypatch):
    payload = b"ID3" + bytes(4_000)
    monkeypatch.setattr(app_module, "_open_upstream",
                        lambda url: _FakeUpstream(payload, length=len(payload)))

    r = client.get("/preview/721063/audio")
    assert r.headers["content-length"] == str(len(payload))
    assert r.headers["accept-ranges"] == "bytes"


def test_preview_audio_without_an_upstream_length_omits_accept_ranges(
    client, deezer_previews, monkeypatch,
):
    """No length means no seeking to claim -- advertising ranges we cannot
    serve would be worse than staying quiet."""
    monkeypatch.setattr(app_module, "_open_upstream", lambda url: _FakeUpstream(b"mp3"))

    r = client.get("/preview/721063/audio")
    assert r.status_code == 200
    assert "accept-ranges" not in r.headers
    assert r.headers["content-encoding"] == "identity"


def test_preview_audio_closes_the_upstream_response(client, deezer_previews, monkeypatch):
    upstream = _FakeUpstream(b"mp3")
    monkeypatch.setattr(app_module, "_open_upstream", lambda url: upstream)
    assert client.get("/preview/721063/audio").content == b"mp3"
    assert upstream.closed


def test_preview_audio_asks_for_the_first_megabyte(client, deezer_previews,
                                                   monkeypatch):
    """The proxy reads the upstream into this container's memory, so it must
    bound what it will read. A Jamendo "preview" is the whole track."""
    import urllib.request

    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["headers"] = dict(getattr(request, "headers", {}))
        return _FakeUpstream(b"mp3")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert client.get("/preview/721063/audio").content == b"mp3"
    assert seen["headers"]["Range"] == f"bytes=0-{app_module.PREVIEW_MAX_BYTES - 1}"


def test_preview_audio_truncates_an_upstream_that_ignored_the_range(
        client, deezer_previews, monkeypatch):
    """A CDN may answer 200 with the whole file. The byte count in the
    streamer, not the request header, is what actually holds -- and the
    Content-Length must not then promise bytes that never arrive."""
    monkeypatch.setattr(app_module, "PREVIEW_MAX_BYTES", 32)
    payload = b"ID3" + bytes(5_000)
    monkeypatch.setattr(app_module, "_open_upstream",
                        lambda url: _FakeUpstream(payload, length=len(payload)))

    r = client.get("/preview/721063/audio")
    assert r.content == payload[:32]
    assert "content-length" not in r.headers
    assert "accept-ranges" not in r.headers


def test_preview_audio_404s_when_deezer_has_no_preview(client, monkeypatch):
    monkeypatch.setattr(deezer_api, "fresh_preview_url", lambda t: None)
    assert client.get("/preview/nope/audio").status_code == 404


def test_preview_audio_502s_when_the_upstream_fetch_fails(client, deezer_previews, monkeypatch):
    def boom(url):
        raise OSError("connection reset")

    monkeypatch.setattr(app_module, "_open_upstream", boom)
    r = client.get("/preview/721063/audio")
    assert r.status_code == 502
    # Fixed message: the exception text can carry the signed CDN URL.
    assert r.json()["detail"] == "upstream preview fetch failed"
    assert "connection reset" not in r.text


def test_preview_audio_wins_over_the_spa_fallback(deezer_previews, monkeypatch, tmp_path, fake_mongo):
    """`audio` has no dot, so the SPA catch-all would serve index.html for it
    if the route were not declared first."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>spa</body></html>")
    monkeypatch.setenv("WEB_DIST", str(dist))
    monkeypatch.setattr(app_module, "_open_upstream", lambda url: _FakeUpstream(b"mp3"))

    r = TestClient(app_module.app).get("/preview/721063/audio")
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"mp3"


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
    monkeypatch.setattr(deezer_api, "search", lambda q, limit=25: [dict(FIXTURE[0])])
    body = client.get("/search?q=miles").json()
    assert body["results"][0]["preview_url"] == (
        f"http://testserver/preview/{FIXTURE[0]['track_id']}"
    )


def test_track_shape_is_unchanged_by_the_rewrite(client, seeded_corpus):
    body = client.get(
        f"/recommend?track_id={seeded_corpus[0]['track_id']}&axis=sounds_like"
    ).json()
    assert _contract_shape(body["results"][0], scored=True)


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


# ---- _cold_matrix ----

def test_cold_matrix_uses_base_matrix(fake_mongo, monkeypatch):
    for tid, vec in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        store.put_track({**FIXTURE[0], "track_id": tid}, {"embedding": vec, "_features_version": FEATURES_VERSION})
    calls = []
    real = store.get_many_features
    monkeypatch.setattr(store, "get_many_features",
                        lambda ids: (calls.append(ids), real(ids))[1])
    ids, matrix = app_module._cold_matrix(("a", "b"), "embedding")
    assert ids == ["a", "b"] and matrix.shape == (2, 2)
    assert calls == []          # one matrix read, no per-track fetches


def test_corpus_matrix_stays_float32_as_the_corpus_grows(fake_mongo):
    """store.base_matrix() hands back float32; _vector() parses float64. The
    growth branch vstacked the two, which upcast the WHOLE corpus matrix on
    the first newly-analyzed track and doubled its memory."""
    import numpy as np

    for tid, vec in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        store.put_track({**FIXTURE[0], "track_id": tid}, {"embedding": vec, "_features_version": FEATURES_VERSION})
    app_module._corpus_matrix(("a", "b"), "embedding", "cosine", False)
    assert app_module._MATRIX_CACHE["embedding"].matrix.dtype == np.float32

    # One more track: the growth path, not a cold rebuild.
    store.put_track({**FIXTURE[0], "track_id": "c"}, {"embedding": [0.5, 0.5], "_features_version": FEATURES_VERSION})
    ids, matrix, _ = app_module._corpus_matrix(("a", "b", "c"), "embedding",
                                               "cosine", False)
    assert ids == ["a", "b", "c"]
    assert app_module._MATRIX_CACHE["embedding"].matrix.dtype == np.float32
    assert matrix.dtype == np.float32
    assert np.allclose(matrix[2], [0.5, 0.5], atol=1e-2)


# ---- indexes created at API startup ----

def test_startup_creates_indexes_for_a_read_only_api_process(fake_mongo):
    """A pure-reader API process never calls put_track/enqueue_* itself, so
    without a startup hook it would never call ensure_indexes() -- Atlas
    would run unindexed until some writer happened to start first."""
    with TestClient(app_module.app):
        pass
    assert "expires_at_1" in fake_mongo.cache.index_information()


# ---- serving the web build ----

def test_spa_is_served_at_root_and_unknown_paths(monkeypatch, tmp_path, fake_mongo):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>spa</body></html>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    monkeypatch.setenv("WEB_DIST", str(dist))
    with TestClient(app_module.app) as c:
        assert c.get("/").text.endswith("spa</body></html>")
        assert c.get("/insights/123").text.endswith("spa</body></html>")
        assert c.get("/assets/a.js").text == "console.log(1)"
        assert c.get("/axes").json()["axes"]                        # API routes still win
        assert c.get("/search?q=x").status_code in (200, 502)


def test_root_is_404_without_a_build(monkeypatch, tmp_path, fake_mongo):
    monkeypatch.setenv("WEB_DIST", str(tmp_path / "missing"))
    with TestClient(app_module.app) as c:
        assert c.get("/").status_code == 404


def test_dotdot_path_traversal_cannot_escape_web_dist(monkeypatch, tmp_path, fake_mongo):
    """``..`` segments used to be joined onto WEB_DIST unresolved, so a
    request for a file just outside the dist root would be served straight
    off disk. httpx normalizes ".." before it ever reaches the server, so
    the direct calls below -- not client.get -- are what actually exercise
    the vulnerable join; the client.get call just confirms the end-to-end
    behavior stays a 404 either way."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>spa</body></html>")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")
    monkeypatch.setenv("WEB_DIST", str(dist))

    with pytest.raises(HTTPException) as exc:
        app_module.spa_fallback("../secret.txt")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        app_module.spa_fallback("assets/../../secret.txt")
    assert exc.value.status_code == 404

    with TestClient(app_module.app) as c:
        assert c.get("/../../pyproject.toml").status_code == 404


def test_absolute_and_double_slash_paths_cannot_escape_web_dist(monkeypatch, tmp_path, fake_mongo):
    """Path("/x") / "/etc/passwd" == "/etc/passwd" -- an absolute right
    operand discards the left, so a path segment that is itself absolute
    used to reach straight past WEB_DIST onto the real filesystem."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>spa</body></html>")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")
    monkeypatch.setenv("WEB_DIST", str(dist))

    # An absolute path parameter -- the exact case the join bug affected.
    with pytest.raises(HTTPException) as exc:
        app_module.spa_fallback(str(secret))
    assert exc.value.status_code == 404

    # A leading double slash (e.g. from a proxy that doesn't collapse it)
    # must not bypass containment either.
    with pytest.raises(HTTPException) as exc:
        app_module.spa_fallback("//../secret.txt")
    assert exc.value.status_code == 404

    # A real request for a no-extension, absolute-looking path never reaches
    # disk at all: it falls back to the SPA shell like any other unknown
    # client-side route, so "//etc/hostname" can't leak the real file either.
    with TestClient(app_module.app) as c:
        r = c.get("//etc/hostname")
        assert r.status_code == 200
        assert r.text.endswith("spa</body></html>")


# ---- /recommend: the feel blend ----
#
# The embedding ranks by style; the eight-dimension feel vector carries
# energy, mood and texture. `feel` weights the second against the first:
#
#     blended = cos(embedding) - feel * mean|z(feel_rec) - z(feel_seed)|
#
# The z-score is per dimension over the corpus, which is what stops the one
# axis with the widest spread from deciding every ranking on its own.
#
# The corpus below is built so the two disagree on purpose. "near" is the
# closest thing in embedding space but feels nothing like the seed; "feely"
# sits further out in style and matches the seed's feel exactly. Which of
# them wins is entirely the slider's decision.

FEEL_DIM = FEATURE_KEYS["feel"]


def _feeling(theta: float, feel: float | None) -> dict:
    """A unit embedding at `theta` radians from the seed, plus a flat feel
    vector (or none at all, for a row the backfill has not reached)."""
    features = _angled(theta)
    if feel is not None:
        features["feel"] = [feel] * FEEL_DIM
    return features


# Feel distances are z-scores over the corpus (app._feel_alignment), and
# every scored row below carries one value on all eight dimensions, so the
# divisor is the population std of the three stored values.
FEEL_SPREAD = pstdev([0.5, 1.0, 0.5])

# id, title, angle from the seed, every feel dimension's value
FEEL_CORPUS = [
    ("s",     "So What",           0.00, 0.5),
    ("near",  "Blue in Green",     0.30, 1.0),   # cos 0.955, feel distance 0.5
    ("feely", "All Blues",         0.45, 0.5),   # cos 0.900, feel distance 0.0
    ("blank", "Flamenco Sketches", 0.60, None),  # no feel vector at all
]


@pytest.fixture
def feel_corpus(fake_mongo):
    for i, (track_id, title, theta, feel) in enumerate(FEEL_CORPUS):
        store.put_track({"track_id": track_id, "title": title,
                         "artist": f"Artist {i}", "album": "Kind of Blue",
                         "artwork_url": "u"},
                        _feeling(theta, feel))
    return FEEL_CORPUS


def _feel_ids(client, **params):
    body = client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            **params}).json()
    return [t["track_id"] for t in body["results"]]


def test_feel_zero_is_the_embedding_only_order(client, feel_corpus):
    """The slider must be safe to turn off: at 0 the endpoint answers exactly
    what it answered before the heads existed, closest cosine first."""
    assert _feel_ids(client, feel=0) == ["near", "feely", "blank"]


def test_a_closer_feel_outranks_a_closer_embedding(client, feel_corpus):
    # "near" leads by 0.055 of cosine and trails by 0.5 of feel distance, so
    # any weight above ~0.11 should flip them.
    order = _feel_ids(client, feel=2)
    assert order[0] == "feely"
    assert order.index("feely") < order.index("near")
    assert _feel_ids(client, feel=0).index("near") < _feel_ids(client, feel=0).index("feely")


def test_a_row_without_feel_is_never_penalized(client, feel_corpus):
    """A partial backfill must not hide tracks: no vector means no penalty,
    so "blank" keeps the place its cosine earned it."""
    scores = {t["track_id"]: t["score"] for t in
              client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                               "feel": 2}).json()["results"]}
    # atol covers the int8 round trip the embedding makes through the store.
    assert scores["blank"] == pytest.approx(math.cos(0.60), abs=2e-3)
    assert scores["feely"] == pytest.approx(math.cos(0.45), abs=2e-3)
    # The penalty is a z-score, not the raw 0.5: see FEEL_SPREAD.
    assert scores["near"] == pytest.approx(
        math.cos(0.30) - 2 * (0.5 / FEEL_SPREAD), abs=2e-3)


def test_feel_leaves_surprise_alone(client, feel_corpus):
    """"Nothing like this" is already a request to leave the neighbourhood;
    penalizing a different feel there would pull it back."""
    off = client.get("/recommend", params={"track_id": "s", "axis": "surprise",
                                           "feel": 0}).json()["results"]
    on = client.get("/recommend", params={"track_id": "s", "axis": "surprise",
                                          "feel": 3}).json()["results"]
    assert off == on


def test_the_server_owns_the_default_feel_weight(client, feel_corpus):
    """The web client omits `feel` at its own default so this number can be
    retuned without shipping a bundle; the two must therefore agree."""
    assert app_module.FEEL_DEFAULT == 0.3
    assert _feel_ids(client) == _feel_ids(client, feel=app_module.FEEL_DEFAULT)


def test_feel_never_leaks_into_a_result_track(client, feel_corpus):
    """FEATURE_KEYS gained a key; TRACK_FIELDS did not."""
    body = client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            "feel": 1.0}).json()
    for track in body["results"]:
        assert _contract_shape(track, scored=True)


def test_feel_weight_is_bounded(client, feel_corpus):
    assert client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            "feel": -1}).status_code == 422
    assert client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            "feel": 99}).status_code == 422


def test_the_feel_alignment_is_cached_across_requests(client, feel_corpus, monkeypatch):
    """The id -> row map over two matrices is O(corpus); rebuilding it per
    request would put the corpus back in the request path."""
    builds = []
    real = app_module._feel_alignment
    monkeypatch.setattr(app_module, "_feel_alignment",
                        lambda *a: (builds.append(1), real(*a))[1])
    params = {"track_id": "s", "axis": "sounds_like", "feel": 1.0}
    client.get("/recommend", params=params)
    client.get("/recommend", params=params)
    assert len(builds) == 2, "the helper is still consulted"
    # ...but the expensive half runs once: the second call is a cache hit.
    assert app_module._FEEL_ALIGN_CACHE is not None


def test_a_seed_without_feel_ranks_on_the_embedding_alone(client, feel_corpus):
    """"blank" has no vector of its own, so there is nothing to compare
    against and every candidate must rank unpenalized."""
    body = client.get("/recommend", params={"track_id": "blank",
                                            "axis": "sounds_like",
                                            "feel": 3}).json()
    scores = [t["score"] for t in body["results"]]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


# ---- pluggable sources: /search fan-out, /preview routing, attribution ----

JAMENDO_TRACK = {
    "track_id": "jamendo:168",
    "title": "Sunrise",
    "artist": "Dee Yan-Key",
    "album": "Morning",
    "artwork_url": "http://x/a.jpg",
    "preview_url": "https://prod.jamendo.com/168.mp3",
    "source": "jamendo",
    "attribution": {"source": "jamendo",
                    "url": "https://www.jamendo.com/track/168/sunrise",
                    "license": "http://creativecommons.org/licenses/by-sa/3.0/"},
}


class _FakeSource:
    def __init__(self, name, results=(), preview=None):
        self.name = name
        self.results = list(results)
        self.preview = preview
        self.asked = []

    def search(self, q, limit=25):
        self.asked.append(q)
        return [dict(t) for t in self.results]

    def track(self, track_id):
        return next((dict(t) for t in self.results
                     if t["track_id"] == track_id), None)

    def preview_url(self, track_id):
        self.asked.append(track_id)
        return self.preview


def test_search_concatenates_the_active_sources_in_order(client, monkeypatch):
    first = _FakeSource("jamendo", [JAMENDO_TRACK])
    second = _FakeSource("deezer", [{**FIXTURE[0], "source": "deezer"}])
    monkeypatch.setattr(app_module.sources, "active", lambda: [first, second])

    results = client.get("/search", params={"q": "sun"}).json()["results"]

    assert [t["source"] for t in results] == ["jamendo", "deezer"]
    assert first.asked == ["sun"] and second.asked == ["sun"]


def test_search_carries_the_attribution_backlink(client, monkeypatch):
    monkeypatch.setattr(app_module.sources, "active",
                        lambda: [_FakeSource("jamendo", [JAMENDO_TRACK])])
    [track] = client.get("/search", params={"q": "sun"}).json()["results"]
    assert _contract_shape(track)
    assert track["attribution_url"] == JAMENDO_TRACK["attribution"]["url"]
    # The full attribution object is server-side only.
    assert "attribution" not in track and "license" not in track


def test_search_survives_one_source_being_down(client, monkeypatch):
    class Broken(_FakeSource):
        def search(self, q, limit=25):
            raise OSError("no network")

    monkeypatch.setattr(app_module.sources, "active",
                        lambda: [Broken("jamendo"),
                                 _FakeSource("deezer", [dict(FIXTURE[0])])])
    results = client.get("/search", params={"q": "so what"}).json()["results"]
    assert [t["track_id"] for t in results] == [FIXTURE[0]["track_id"]]


def test_preview_routes_to_the_source_that_owns_the_id(client, fake_mongo,
                                                       monkeypatch):
    jam = _FakeSource("jamendo", [JAMENDO_TRACK],
                      preview="https://prod.jamendo.com/168.mp3")
    monkeypatch.setattr(app_module.sources, "for_id",
                        lambda tid: jam if tid.startswith("jamendo:") else None)

    r = client.get("/preview/jamendo:168", follow_redirects=False)

    assert r.status_code == 302
    assert r.headers["location"] == "https://prod.jamendo.com/168.mp3"
    assert jam.asked == ["jamendo:168"]


def test_preview_404s_for_an_id_from_an_unknown_source(client, fake_mongo,
                                                       monkeypatch):
    monkeypatch.setattr(app_module.sources, "for_id", lambda tid: None)
    assert client.get("/preview/bandcamp:5").status_code == 404


def test_recommend_rows_carry_the_source_and_backlink(client, fake_mongo):
    """The whole point: a CC track's credit survives the store and the
    ranking and reaches the client."""
    store.put_track({**FIXTURE[0], "track_id": "s"}, fake_features(0.0))
    store.put_track(JAMENDO_TRACK, fake_features(0.1))

    body = client.get("/recommend", params={"track_id": "s",
                                            "axis": "sounds_like"}).json()
    [rec] = [t for t in body["results"] if t["track_id"] == "jamendo:168"]
    assert _contract_shape(rec, scored=True)
    assert rec["source"] == "jamendo"
    assert rec["attribution_url"] == JAMENDO_TRACK["attribution"]["url"]


# ---- /recommend: the tempo term ----
#
# CLAP is trained on 7 s windows with a contrastive objective, so it throws
# tempo away almost completely: a ballad and a double-time burner of the same
# idiom sit in the same corner of the space. `tempo` weights an octave-folded
# distance between the two stored BPMs:
#
#     d = log2(bpm_seed / bpm_rec);  tempo_dist = clip(min(|d|,|d-1|,|d+1|), 0, .5)
#
# Folding, because 90 and 180 BPM are the same groove counted differently and
# every beat tracker disagrees with every other about which to report.

def _with_tempo(theta: float, bpm: float | None) -> dict:
    features = _angled(theta)
    if bpm is not None:
        features["rhythm"] = {"tempo_bpm": bpm, "beat_strength": 0.9,
                              "loudness_lufs": -9.0, "loudness_range": 5.0,
                              "key": 2, "mode": "minor", "key_strength": 0.7}
    return features


# id, angle from the seed, stored BPM
TEMPO_CORPUS = [
    ("s",      0.00, 120.0),
    ("near",   0.30, 160.0),   # closest cosine, a third of an octave out
    ("octave", 0.45, 240.0),   # further out in style, the same groove doubled
    ("slow",   0.60, 40.0),    # 1.58 octaves down: past the clip
    ("blank",  0.75, None),    # no rhythm at all
]


@pytest.fixture
def tempo_corpus(fake_mongo):
    for i, (track_id, theta, bpm) in enumerate(TEMPO_CORPUS):
        store.put_track({"track_id": track_id, "title": f"Track {track_id}",
                         "artist": f"Artist {i}", "album": "Kind of Blue",
                         "artwork_url": "u"},
                        _with_tempo(theta, bpm))
    return TEMPO_CORPUS


def _tempo_scores(client, **params) -> dict[str, float]:
    body = client.get("/recommend", params={"track_id": "s", "axis": "sounds_like",
                                            "feel": 0, **params}).json()
    return {t["track_id"]: t["score"] for t in body["results"]}


def test_tempo_zero_is_the_embedding_only_order(client, tempo_corpus):
    scores = _tempo_scores(client, tempo=0)
    assert list(scores) == ["near", "octave", "slow", "blank"]
    assert scores["near"] == pytest.approx(math.cos(0.30), abs=2e-3)


def test_an_octave_apart_is_not_a_tempo_difference(client, tempo_corpus):
    """240 against 120 is the same groove counted in half bars."""
    scores = _tempo_scores(client, tempo=3)
    assert scores["octave"] == pytest.approx(math.cos(0.45), abs=2e-3)


def test_a_closer_tempo_outranks_a_closer_embedding(client, tempo_corpus):
    """"near" leads by 0.055 of cosine and is 0.415 octaves out of tempo."""
    assert list(_tempo_scores(client, tempo=0))[0] == "near"
    assert list(_tempo_scores(client, tempo=2))[0] == "octave"


def test_the_tempo_distance_is_the_folded_log_ratio(client, tempo_corpus):
    scores = _tempo_scores(client, tempo=1)
    expected = abs(math.log2(120.0 / 160.0))          # 0.415, inside the clip
    assert scores["near"] == pytest.approx(math.cos(0.30) - expected, abs=2e-3)


def test_the_tempo_distance_is_clipped(client, tempo_corpus):
    """40 BPM against 120 folds to 0.585 octaves; past half an octave the two
    tracks are simply at different tempi and the penalty stops growing, so it
    can never swamp the cosine it is subtracted from."""
    scores = _tempo_scores(client, tempo=1)
    assert scores["slow"] == pytest.approx(math.cos(0.60) - 0.5, abs=2e-3)


def test_a_row_without_rhythm_is_never_penalized(client, tempo_corpus):
    scores = _tempo_scores(client, tempo=3)
    assert scores["blank"] == pytest.approx(math.cos(0.75), abs=2e-3)


def test_a_seed_without_rhythm_ranks_on_the_embedding_alone(client, tempo_corpus):
    body = client.get("/recommend", params={"track_id": "blank", "feel": 0,
                                            "axis": "sounds_like",
                                            "tempo": 3}).json()
    scores = [t["score"] for t in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_tempo_leaves_surprise_alone(client, tempo_corpus):
    off = client.get("/recommend", params={"track_id": "s", "axis": "surprise",
                                           "tempo": 0}).json()["results"]
    on = client.get("/recommend", params={"track_id": "s", "axis": "surprise",
                                          "tempo": 3}).json()["results"]
    assert off == on


def test_the_server_owns_the_default_tempo_weight(client, tempo_corpus):
    """The web client omits `tempo` at its own default so this number can be
    retuned without shipping a bundle; the two must therefore agree."""
    assert app_module.TEMPO_DEFAULT == 0.2
    assert (_tempo_scores(client)
            == _tempo_scores(client, tempo=app_module.TEMPO_DEFAULT))


def test_tempo_weight_is_bounded(client, tempo_corpus):
    for bad in (-1, 99):
        assert client.get("/recommend", params={"track_id": "s", "tempo": bad,
                                                "axis": "sounds_like"}
                          ).status_code == 422


def test_tempo_never_leaks_into_a_result_track(client, tempo_corpus):
    body = client.get("/recommend", params={"track_id": "s", "tempo": 1,
                                            "axis": "sounds_like"}).json()
    for track in body["results"]:
        assert _contract_shape(track, scored=True)


def test_the_rhythm_alignment_is_cached_across_requests(client, tempo_corpus,
                                                        monkeypatch):
    """One `rhythm` read for the whole corpus per matrix identity, not one
    per request: this is on the /recommend path."""
    reads = []
    real = store.get_many_rhythm
    monkeypatch.setattr(store, "get_many_rhythm",
                        lambda ids: (reads.append(len(ids)), real(ids))[1])
    params = {"track_id": "s", "axis": "sounds_like", "tempo": 1}
    client.get("/recommend", params=params)
    client.get("/recommend", params=params)
    assert len(reads) == 1


# ---- GET /search/text: the corpus, asked in English ----

TEXT_DIM = FEATURE_KEYS["embedding"]


def _text_vector(theta: float) -> np.ndarray:
    """A (1, 1024) unit row, the shape clap.embed_text answers with."""
    v = np.zeros((1, TEXT_DIM), dtype=np.float32)
    v[0, 0], v[0, 1] = math.cos(theta), math.sin(theta)
    return v


@pytest.fixture
def text_corpus(fake_mongo):
    """Four tracks on a circle, so a text vector at an angle has a known
    nearest neighbour."""
    for i, theta in enumerate((0.0, 0.4, 0.8, 1.2)):
        store.put_track({"track_id": f"t{i}", "title": f"Track {i}",
                         "artist": "Miles Davis", "album": "Kind of Blue",
                         "artwork_url": "u"},
                        _angled(theta))
    return ["t0", "t1", "t2", "t3"]


@pytest.fixture(autouse=True)
def text_search_on(monkeypatch):
    """The host default is OFF. Module-wide (autouse), which is harmless
    everywhere except /axes, whose flag the two tests below assert both ways."""
    monkeypatch.setenv(app_module.TEXT_SEARCH_ENV, "1")


def test_axes_reports_whether_this_host_can_answer_a_text_search(client, monkeypatch):
    assert client.get("/axes").json()["text_search"] is True
    monkeypatch.delenv(app_module.TEXT_SEARCH_ENV)
    assert client.get("/axes").json()["text_search"] is False


def test_search_text_is_503_when_the_host_has_it_switched_off(client, text_corpus,
                                                              monkeypatch):
    """It is off by default: CLAP in the API process is ~2.5 GB resident, and
    that is a deployment decision rather than a per-request one."""
    monkeypatch.delenv(app_module.TEXT_SEARCH_ENV)
    res = client.get("/search/text", params={"q": "jazz"})
    assert res.status_code == 503
    assert "switched off" in res.json()["detail"]


def test_search_text_ranks_the_corpus_by_cosine_to_the_phrase(client, text_corpus,
                                                              monkeypatch):
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.8))
    body = client.get("/search/text", params={"q": "hazy late-night trumpet"}).json()
    ids = [t["track_id"] for t in body["results"]]
    assert ids == ["t2", "t1", "t3", "t0"]
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=2e-3)


def test_search_text_passes_the_phrase_to_clap_once(client, text_corpus, monkeypatch):
    asked = []
    monkeypatch.setattr(clap, "embed_text",
                        lambda prompts: (asked.append(prompts), _text_vector(0.0))[1])
    client.get("/search/text", params={"q": "  solo piano  "})
    assert asked == [["solo piano"]]


def test_search_text_results_are_contract_tracks_with_a_score(client, text_corpus,
                                                              monkeypatch):
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.0))
    for track in client.get("/search/text", params={"q": "jazz"}).json()["results"]:
        assert _contract_shape(track, scored=True)


def test_search_text_honours_the_limit(client, text_corpus, monkeypatch):
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.0))
    body = client.get("/search/text", params={"q": "jazz", "limit": 2}).json()
    assert len(body["results"]) == 2


def test_search_text_rejects_an_empty_query(client, text_corpus, monkeypatch):
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.0))
    assert client.get("/search/text", params={"q": "   "}).status_code == 400


def test_search_text_is_503_when_clap_is_not_installed(client, text_corpus,
                                                       monkeypatch):
    """The rest of the API is fine; the client should hide the toggle rather
    than report the service down."""
    def missing(_prompts):
        raise ImportError("No module named 'msclap'")

    monkeypatch.setattr(clap, "embed_text", missing)
    res = client.get("/search/text", params={"q": "jazz"})
    assert res.status_code == 503
    assert "CLAP" in res.json()["detail"]


def test_search_text_is_503_when_the_weights_were_never_fetched(client, text_corpus,
                                                                monkeypatch):
    def missing(_prompts):
        raise FileNotFoundError("models/v2/CLAP_weights_2023.pth missing")

    monkeypatch.setattr(clap, "embed_text", missing)
    assert client.get("/search/text", params={"q": "jazz"}).status_code == 503


def test_search_text_is_503_against_a_corpus_that_is_not_clap_space(client,
                                                                    monkeypatch):
    """A version-3 (or synthetic) corpus cannot be compared with a text
    vector at all; numpy would raise that as a 500."""
    store.put_track({"track_id": "narrow", "title": "t", "artist": "a",
                     "album": "b", "artwork_url": "u"},
                    {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.0))
    res = client.get("/search/text", params={"q": "jazz"})
    assert res.status_code == 503
    assert "CLAP space" in res.json()["detail"]


def test_search_text_on_an_empty_corpus_is_empty_not_an_error(client, fake_mongo,
                                                              monkeypatch):
    monkeypatch.setattr(clap, "embed_text", lambda prompts: _text_vector(0.0))
    assert client.get("/search/text", params={"q": "jazz"}).json() == {"results": []}


def test_only_new_ids_are_read_when_the_corpus_grows(client, tempo_corpus,
                                                     monkeypatch):
    """The alignment is keyed on the matrix, and the matrix is a NEW object
    every time the corpus grows -- which during a re-analysis backfill is
    every couple of seconds. Re-reading `rhythm` for the whole corpus on each
    of those would put an Atlas full-collection read on the /recommend path.
    """
    asked = []
    real = store.get_many_rhythm
    monkeypatch.setattr(store, "get_many_rhythm",
                        lambda ids: (asked.append(list(ids)), real(ids))[1])
    params = {"track_id": "s", "axis": "sounds_like", "tempo": 1}
    client.get("/recommend", params=params)
    assert asked and set(asked[0]) == {t[0] for t in TEMPO_CORPUS}

    store.put_track({"track_id": "late", "title": "Late", "artist": "A",
                     "album": "Kind of Blue", "artwork_url": "u"},
                    _with_tempo(0.9, 128.0))
    asked.clear()
    client.get("/recommend", params=params)

    assert asked == [["late"]], "the whole corpus was re-read for one new row"


def test_a_transient_store_failure_is_not_memoized_as_no_tempo(client,
                                                               tempo_corpus,
                                                               monkeypatch):
    """Memoizing 0.0 for a read that never happened would turn one unreachable
    Atlas into a permanently tempo-less corpus."""
    real = store.get_many_rhythm
    down = [True]

    def maybe(ids):
        if down[0]:
            raise RuntimeError("connection reset")
        return real(ids)

    monkeypatch.setattr(store, "get_many_rhythm", maybe)
    scores = _tempo_scores(client, tempo=3)
    assert scores["near"] == pytest.approx(math.cos(0.30), abs=2e-3)  # no penalty
    assert not app_module._TEMPO_BY_ID

    down[0] = False             # Atlas comes back
    app_module._RHYTHM_ALIGN_CACHE = None
    assert _tempo_scores(client, tempo=1)["near"] == pytest.approx(
        math.cos(0.30) - abs(math.log2(120.0 / 160.0)), abs=2e-3)


def test_surprise_reads_no_rhythm_at_all(client, tempo_corpus, monkeypatch):
    """`tempo` defaults to 0.2, and the term is sounds_like-only, so a
    `surprise` request must not build the column it would then discard."""
    asked = []
    monkeypatch.setattr(store, "get_many_rhythm",
                        lambda ids: (asked.append(list(ids)), [])[1])
    client.get("/recommend", params={"track_id": "s", "axis": "surprise"})
    assert asked == []


def test_a_seed_with_no_beat_reports_no_tempo_distance(client, fake_mongo):
    """0.0 BPM is the contract's "no beat could be found", not a tempo: there
    is no ratio to take, so the panel must say nothing rather than 0."""
    for track_id, theta, bpm in (("s", 0.0, 0.0), ("other", 0.3, 120.0)):
        store.put_track({"track_id": track_id, "title": track_id, "artist": "A",
                         "album": "Kind of Blue", "artwork_url": "u"},
                        _with_tempo(theta, bpm))
    body = client.get("/viz/map", params={"track_id": "s", "axis": "sounds_like",
                                          "tempo": 2}).json()
    math_ = body["recs"][0]["math"]
    assert math_["tempo_dist"] is None
    assert math_["rhythm"] is None
    assert body["recs"][0]["score"] == pytest.approx(math.cos(0.30), abs=2e-3)
