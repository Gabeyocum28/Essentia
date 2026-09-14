"""corpus/sources/: the registry, id ownership, and the two sources.

No network: Jamendo's single HTTP choke point (`jamendo._get`) and Deezer's
existing API modules are stubbed.
"""
import mongomock
import pytest

from music_recommendations.corpus import crawl, sources
from music_recommendations.corpus.sources import base, jamendo
from music_recommendations.server import deezer as deezer_api
from music_recommendations.server import store


@pytest.fixture
def fake_mongo(monkeypatch):
    """An in-memory Atlas: the Deezer rotation reads and writes crawl state."""
    database = mongomock.MongoClient().get_database("essentia_test")
    store.reset()
    monkeypatch.setattr(store, "db", lambda: database)
    yield database
    store.reset()


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """Sources are cached per process; no test may inherit another's."""
    monkeypatch.delenv("SOURCES", raising=False)
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
    sources.reset()
    yield
    sources.reset()


# ---- the registry ----

def test_default_is_deezer_alone():
    active = sources.active()
    assert [s.name for s in active] == ["deezer"]


def test_sources_env_selects_and_orders(monkeypatch):
    monkeypatch.setenv("SOURCES", "jamendo, deezer")
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "k")
    assert [s.name for s in sources.active()] == ["jamendo", "deezer"]


def test_unknown_source_name_raises(monkeypatch):
    """A typo in SOURCES must not silently serve a smaller catalogue."""
    monkeypatch.setenv("SOURCES", "deezer,spotifyy")
    with pytest.raises(ValueError) as exc:
        sources.active()
    assert "spotifyy" in str(exc.value)


def test_instances_are_reused():
    """Sources hold rate-limiting and crawl-label state; a new object per
    call would reset both."""
    assert sources.get("deezer") is sources.get("deezer")


# ---- for_id ----

def test_bare_ids_belong_to_deezer():
    assert sources.for_id("2711778").name == "deezer"
    assert sources.for_id("2711778").owns("2711778")


def test_namespaced_ids_route_to_their_source():
    assert sources.for_id("jamendo:168").name == "jamendo"
    assert sources.for_id("jamendo:168").owns("jamendo:168")
    assert not sources.for_id("2711778").owns("jamendo:168")


def test_for_id_resolves_a_source_that_is_switched_off(monkeypatch):
    """An id already in the corpus must stay playable even if its source
    has been dropped from SOURCES."""
    monkeypatch.setenv("SOURCES", "deezer")
    assert sources.for_id("jamendo:168").name == "jamendo"


def test_for_id_is_none_for_an_unknown_namespace():
    assert sources.for_id("bandcamp:5") is None


def test_split_id():
    assert base.split_id("jamendo:168") == ("jamendo", "168")
    assert base.split_id("2711778") == ("deezer", "2711778")


# ---- Jamendo: availability ----

def test_jamendo_is_disabled_without_a_client_id(monkeypatch, capsys):
    monkeypatch.setenv("SOURCES", "deezer,jamendo")
    assert [s.name for s in sources.active()] == ["deezer"]
    assert "jamendo disabled" in capsys.readouterr().out


def test_jamendo_alone_and_unconfigured_refuses_to_start(monkeypatch):
    """Dropping the only source would leave the app with no catalogue, so
    "started fine" would be a lie."""
    monkeypatch.setenv("SOURCES", "jamendo")
    with pytest.raises(RuntimeError) as exc:
        sources.active()
    assert "JAMENDO_CLIENT_ID" in str(exc.value)


# ---- Jamendo: mapping ----

RAW = {
    "id": "168",
    "name": "Sunrise",
    "artist_name": "Dee Yan-Key",
    "album_name": "Morning",
    "image": "https://usercontent.jamendo.com/168.jpg",
    "audio": "https://prod-1.storage.jamendo.com/?trackid=168&format=mp31",
    "shareurl": "https://www.jamendo.com/track/168/sunrise",
    "license_ccurl": "http://creativecommons.org/licenses/by-sa/3.0/",
}


@pytest.fixture
def stub_get(monkeypatch):
    """Replace the one HTTP call with a recorder returning a canned page."""
    calls = []
    page = {"results": [RAW]}

    def fake(path, **params):
        calls.append((path, params))
        return page

    monkeypatch.setattr(jamendo, "_get", fake)
    return calls


def test_jamendo_search_maps_the_v3_fields(monkeypatch, stub_get):
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "k")
    [track] = sources.get("jamendo").search("sunrise", limit=5)
    assert track == {
        "track_id": "jamendo:168",
        "title": "Sunrise",
        "artist": "Dee Yan-Key",
        "album": "Morning",
        "artwork_url": RAW["image"],
        "preview_url": RAW["audio"],
        "source": "jamendo",
        "attribution": {
            "source": "jamendo",
            "url": RAW["shareurl"],
            "license": RAW["license_ccurl"],
        },
    }
    assert stub_get == [("tracks/", {"search": "sunrise", "limit": 5})]


def test_jamendo_track_looks_up_by_bare_id(stub_get):
    track = sources.get("jamendo").track("jamendo:168")
    assert track["track_id"] == "jamendo:168"
    assert stub_get == [("tracks/", {"id": "168"})]


def test_jamendo_preview_url_is_the_audio_field(stub_get):
    assert sources.get("jamendo").preview_url("jamendo:168") == RAW["audio"]


def test_jamendo_drops_a_track_with_no_audio(monkeypatch):
    monkeypatch.setattr(jamendo, "_get",
                        lambda path, **p: {"results": [{k: v for k, v in RAW.items()
                                                        if k != "audio"}]})
    assert sources.get("jamendo").search("x") == []


def test_jamendo_track_without_a_share_url_carries_no_attribution(monkeypatch):
    """A credit with no link back is not attribution."""
    monkeypatch.setattr(jamendo, "_get",
                        lambda path, **p: {"results": [{k: v for k, v in RAW.items()
                                                        if k != "shareurl"}]})
    [track] = sources.get("jamendo").search("x")
    assert "attribution" not in track
    assert sources.get("jamendo").attribution(track) is None


def test_jamendo_falls_back_to_the_album_image(monkeypatch):
    raw = {k: v for k, v in RAW.items() if k != "image"}
    raw["album_image"] = "https://usercontent.jamendo.com/album.jpg"
    monkeypatch.setattr(jamendo, "_get", lambda path, **p: {"results": [raw]})
    [track] = sources.get("jamendo").search("x")
    assert track["artwork_url"] == raw["album_image"]


def test_jamendo_candidates_rotate_tags_then_featured(stub_get):
    source = sources.get("jamendo")
    for step, tag in enumerate(jamendo.TAGS):
        source.candidates(step)
        assert stub_get[step] == ("tracks/", {"tags": tag,
                                              "order": "popularity_total",
                                              "limit": jamendo.PER_PAGE,
                                              "offset": 0})
        assert source.candidate_label() == f"jamendo tag {tag} +0"
    source.candidates(len(jamendo.TAGS))
    assert stub_get[-1] == ("tracks/", {"featured": 1,
                                        "limit": jamendo.PER_PAGE,
                                        "offset": 0})
    assert source.candidate_label() == "jamendo featured +0"
    # ...and then round again.
    source.candidates(len(jamendo.TAGS) + 1)
    assert source.candidate_label() == f"jamendo tag {jamendo.TAGS[0]} +{jamendo.PER_PAGE}"


def test_jamendo_candidates_page_deeper_every_lap(stub_get):
    """Without an offset the rotation re-requests the same hundred most
    popular tracks forever, and the corpus stops growing after one lap."""
    source = sources.get("jamendo")
    arms = len(jamendo.TAGS) + 1
    source.candidates(0)
    source.candidates(arms)
    source.candidates(2 * arms)
    offsets = [params["offset"] for _path, params in stub_get]
    assert offsets == [0, jamendo.PER_PAGE, 2 * jamendo.PER_PAGE]
    assert all(params["tags"] == jamendo.TAGS[0] for _path, params in stub_get)


def test_jamendo_pins_the_audio_format(monkeypatch):
    """Analysis is only comparable across tracks encoded the same way, so the
    format is pinned in the one place every call goes through."""
    seen = {}

    class Resp:
        def read(self):
            return b'{"headers": {"status": "success"}, "results": []}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return Resp()

    monkeypatch.setattr(jamendo.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jamendo, "SLEEP", 0)
    jamendo._get("tracks/", id="1")
    assert f"audioformat={jamendo.AUDIO_FORMAT}" in seen["url"]


def test_jamendo_logs_an_error_inside_a_200_envelope(monkeypatch, capsys):
    """A bad client id or an exhausted monthly quota comes back as HTTP 200
    with an empty result list -- indistinguishable from "no tracks" unless
    the envelope's own status is read."""
    class Resp:
        def read(self):
            return (b'{"headers": {"status": "failed", '
                    b'"error_message": "Your credits are exhausted"}, '
                    b'"results": []}')

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(jamendo.urllib.request, "urlopen",
                        lambda url, timeout=None: Resp())
    monkeypatch.setattr(jamendo, "SLEEP", 0)
    assert jamendo._get("tracks/", id="1")["results"] == []
    out = capsys.readouterr().out
    assert "failed" in out and "credits are exhausted" in out


def test_jamendo_get_raises_on_a_broken_response(monkeypatch):
    """A transport failure must propagate (not disappear into {}): callers
    that need to tell "network is down" from "no results" -- app.search's
    per-source `failures` counter, in particular -- can only do that if a
    dead API looks like an exception rather than an empty envelope. The
    error is still logged here, once, before it is re-raised."""
    def boom(url, timeout=None):
        raise OSError("no network")

    monkeypatch.setattr(jamendo.urllib.request, "urlopen", boom)
    monkeypatch.setattr(jamendo, "SLEEP", 0)
    with pytest.raises(OSError):
        jamendo._get("tracks/", id="1")


# ---- Deezer ----

def test_deezer_ids_stay_bare_and_tracks_are_tagged(monkeypatch):
    monkeypatch.setattr(deezer_api, "get_track",
                        lambda tid: {"track_id": tid, "title": "So What",
                                     "artist": "Miles Davis", "album": "Kind of Blue",
                                     "artwork_url": "", "preview_url": "http://p"})
    track = sources.get("deezer").track("2711778")
    assert track["track_id"] == "2711778"
    assert track["source"] == "deezer"


def test_deezer_carries_no_attribution():
    """Deezer never ships, and a "via Deezer" credit on every development
    row would be noise the licence does not ask for."""
    assert sources.get("deezer").attribution({"track_id": "1"}) is None


def _stub_arms(monkeypatch):
    calls = []
    monkeypatch.setattr(crawl, "from_charts",
                        lambda genre_ids, per_genre=100:
                            (calls.append(("charts", genre_ids, per_genre)), [])[1])
    monkeypatch.setattr(crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10:
                            (calls.append(("snowball", root_names, hops, per_artist)), [])[1])
    monkeypatch.setattr(crawl, "resolve_artists",
                        lambda names: (calls.append(("resolve", names)), [7])[1])
    monkeypatch.setattr(crawl, "deep_cuts",
                        lambda artist_ids, albums_per_artist=6:
                            (calls.append(("deep_cuts", artist_ids, albums_per_artist)),
                             iter([]))[1])
    return calls


def test_deezer_candidates_rotate_charts_snowball_deep_cuts(fake_mongo, monkeypatch):
    calls = _stub_arms(monkeypatch)
    source = sources.get("deezer")

    source.candidates(0)
    assert calls[0] == ("charts", [list(crawl.GENRES)[0]], 100)
    assert source.candidate_label() == f"chart {crawl.GENRES[list(crawl.GENRES)[0]]}"

    source.candidates(1)
    assert calls[1] == ("snowball", [crawl.ROOTS[0]], 1, 10)
    assert source.candidate_label() == f"snowball {crawl.ROOTS[0]}"

    source.candidates(2)
    assert calls[2] == ("resolve", [crawl.ROOTS[0]])
    assert calls[3] == ("deep_cuts", [7], 3)
    assert source.candidate_label() == f"deep cuts {crawl.ROOTS[0]}"

    # step // 3 indexes within each arm: step 3 is the SECOND genre chart.
    source.candidates(3)
    assert calls[4] == ("charts", [list(crawl.GENRES)[1]], 100)


def test_deezer_candidates_grow_the_crawl_roots(fake_mongo, monkeypatch):
    found = [{"track_id": str(i), "title": "t", "artist": f"Artist {i}",
              "album": "", "artwork_url": "", "preview_url": "http://p"}
             for i in range(7)]
    monkeypatch.setattr(crawl, "from_charts", lambda genre_ids, per_genre=100: found)
    sources.get("deezer").candidates(0)
    assert store.get_state("crawl_roots")["names"] == [f"Artist {i}" for i in range(5)]
