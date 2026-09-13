"""app.py: the four contract routes, mock-first with the store + analysis real path."""
import json
import math
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from music_recommendations.server import app as app_module
from music_recommendations.server import store
from contract.features import AXES, FEATURE_KEYS, TRACK_FIELDS

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
    return {"embedding": v}


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
        assert set(track) == set(TRACK_FIELDS) | {"score"}


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


def test_preview_audio_404s_when_deezer_has_no_preview(client, monkeypatch):
    monkeypatch.setattr(app_module.deezer, "fresh_preview_url", lambda t: None)
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


def test_corpus_matrix_stays_float32_as_the_corpus_grows(fake_mongo):
    """store.base_matrix() hands back float32; _vector() parses float64. The
    growth branch vstacked the two, which upcast the WHOLE corpus matrix on
    the first newly-analyzed track and doubled its memory."""
    import numpy as np

    for tid, vec in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        store.put_track({**FIXTURE[0], "track_id": tid}, {"embedding": vec})
    app_module._corpus_matrix(("a", "b"), "embedding", "cosine", False)
    assert app_module._MATRIX_CACHE["embedding"].matrix.dtype == np.float32

    # One more track: the growth path, not a cold rebuild.
    store.put_track({**FIXTURE[0], "track_id": "c"}, {"embedding": [0.5, 0.5]})
    ids, matrix, _ = app_module._corpus_matrix(("a", "b", "c"), "embedding",
                                               "cosine", False)
    assert ids == ["a", "b", "c"]
    assert app_module._MATRIX_CACHE["embedding"].matrix.dtype == np.float32
    assert matrix.dtype == np.float32
    assert np.allclose(matrix[2], [0.5, 0.5], atol=1e-2)


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
# The embedding ranks by style; the eleven-dimension feel vector carries
# energy, mood and texture. `feel` weights the second against the first:
#
#     blended = cos(embedding) - feel * mean|feel_rec - feel_seed|
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
    assert scores["near"] == pytest.approx(math.cos(0.30) - 2 * 0.5, abs=2e-3)


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
        assert set(track) == set(TRACK_FIELDS) | {"score"}


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
