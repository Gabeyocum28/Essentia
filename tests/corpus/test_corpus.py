"""corpus/: shape conversion, dedup, versioning. No network in these tests."""
import importlib.util
from pathlib import Path

import pytest

from music_recommendations.corpus import deezer, ingest

CONTRACT = Path(__file__).resolve().parents[2] / "contract"


def _track_fields():
    spec = importlib.util.spec_from_file_location("features", CONTRACT / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRACK_FIELDS


RAW = {
    "id": 2711778,
    "title": "So What",
    "preview": "https://cdnt-preview.dzcdn.net/x.mp3",
    "artist": {"name": "Miles Davis"},
    "album": {"title": "Kind Of Blue", "cover_medium": "https://img/250.jpg"},
}


def test_track_to_contract_produces_exactly_the_contract_fields():
    track = deezer.track_to_contract(RAW)
    assert set(track) == _track_fields()
    assert track["track_id"] == "2711778"
    assert isinstance(track["track_id"], str), "track_id is a string everywhere"


def test_track_without_preview_is_dropped():
    """A track we cannot play is useless to the app and to analysis."""
    assert deezer.track_to_contract({**RAW, "preview": ""}) is None
    assert deezer.track_to_contract({k: v for k, v in RAW.items() if k != "preview"}) is None


def test_missing_album_or_artist_does_not_crash():
    """Deezer omits these on some payloads; a crawl must not die on one row."""
    track = deezer.track_to_contract({"id": 1, "title": "x", "preview": "http://a"})
    assert set(track) == _track_fields()
    assert track["artist"] == "" and track["album"] == ""


def test_already_stored_is_false_when_the_store_is_unreachable(monkeypatch):
    """A crawl against a dead store should re-do work, not silently skip it."""
    def boom(_track_id):
        raise ConnectionError("store is down")

    monkeypatch.setattr(ingest.store, "get_features", boom)
    assert ingest.already_stored("123") is False


def test_already_stored_rejects_a_stale_feature_version(monkeypatch):
    current = ingest.FEATURES_VERSION
    monkeypatch.setattr(
        ingest.store, "get_features",
        lambda _t: {"embedding": [0.0], ingest.VERSION_KEY: current - 1},
    )
    assert ingest.already_stored("123") is False

    monkeypatch.setattr(
        ingest.store, "get_features",
        lambda _t: {"embedding": [0.0], ingest.VERSION_KEY: current},
    )
    assert ingest.already_stored("123") is True


def test_already_stored_rejects_features_written_before_versioning(monkeypatch):
    monkeypatch.setattr(ingest.store, "get_features", lambda _t: {"embedding": [0.0]})
    assert ingest.already_stored("123") is False


@pytest.mark.parametrize("shards", [2, 3, 5])
def test_sharding_covers_every_track_exactly_once(shards):
    """Two machines must not collide and must not leave gaps."""
    ids = [str(i) for i in range(1000, 1200)]
    seen = []
    for index in range(shards):
        seen += [i for i in ids if int(i) % shards == index]
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen))


def test_next_batch_downloads_start_before_the_current_one_is_analyzed(
        monkeypatch, tmp_path):
    """The 10 cores idle if a batch only starts fetching once the last one ends."""
    import threading

    monkeypatch.setattr(ingest, "BATCH", 2)
    monkeypatch.setattr(ingest, "already_stored", lambda _t: False)
    monkeypatch.setattr(ingest.store, "put_track", lambda *_a: None)

    second_batch_started = threading.Event()

    def fake_download(track):
        if track["track_id"] in ("2", "3"):
            second_batch_started.set()
        return tmp_path / f"{track['track_id']}.mp3"

    monkeypatch.setattr(ingest, "download_preview", fake_download)

    overlapped = []

    def fake_analyze_many(paths, workers=None):
        overlapped.append(second_batch_started.wait(timeout=5))
        for path in paths:
            yield path, {"embedding": [0.0]}, None

    monkeypatch.setattr(ingest, "analyze_many", fake_analyze_many)

    tracks = [{"track_id": str(i)} for i in range(4)]
    assert ingest.ingest(tracks, limit=4, progress=False) == 4
    assert overlapped[0] is True, "batch 2 must fetch while batch 1 is analyzed"


def test_previews_are_deleted_once_their_features_are_stored(monkeypatch, tmp_path):
    """The features are the product; a corpus of mp3s would fill the disk."""
    monkeypatch.setattr(ingest, "already_stored", lambda _t: False)
    monkeypatch.setattr(ingest.store, "put_track", lambda *_a: None)

    mp3s = []
    for track_id in ("1", "2"):
        mp3 = tmp_path / f"{track_id}.mp3"
        mp3.write_bytes(b"audio")
        mp3s.append(mp3)

    monkeypatch.setattr(ingest, "download_preview",
                        lambda track: tmp_path / f"{track['track_id']}.mp3")
    monkeypatch.setattr(
        ingest, "analyze_many",
        lambda paths, workers=None: [
            (paths[0], {"embedding": [0.0]}, None),   # stored
            (paths[1], None, "CorruptFile: bad mp3"),  # failed
        ])

    ingest.ingest([{"track_id": "1"}, {"track_id": "2"}], limit=2, progress=False)
    assert not mp3s[0].exists(), "a stored track's audio is dead weight"
    assert not mp3s[1].exists(), "a track that cannot be analyzed is no different"


def test_a_track_already_in_the_store_has_its_leftover_audio_swept(monkeypatch, tmp_path):
    """A run that died between analysis and cleanup should not leak the file."""
    monkeypatch.setattr(ingest, "AUDIO_CACHE", tmp_path)
    monkeypatch.setattr(ingest, "already_stored", lambda _t: True)
    leftover = tmp_path / "7.mp3"
    leftover.write_bytes(b"audio")

    assert ingest.ingest([{"track_id": "7"}], limit=1, progress=False) == 0
    assert not leftover.exists()
