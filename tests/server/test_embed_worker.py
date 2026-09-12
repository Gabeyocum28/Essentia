"""embed_worker.py: pop queued tracks, analyze locally, write back to the store."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from music_recommendations.server import store

SCRIPT = Path(__file__).parents[2] / "scripts" / "embed_worker.py"
spec = importlib.util.spec_from_file_location("embed_worker", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

TRACK = {
    "track_id": "42",
    "title": "Blue in Green",
    "artist": "Miles Davis",
    "album": "Kind of Blue",
    "artwork_url": "http://x/a.jpg",
    "preview_url": "http://x/p.mp3",
}
FEATURES = {"embedding": [0.1, 0.2]}


def assert_features_match(track_id: str, expected: dict) -> None:
    """Compare through the int8 round trip: exact equality no longer holds."""
    got = store.get_features(track_id)
    assert got is not None
    assert np.allclose(got["embedding"], expected["embedding"], atol=0.01)


@pytest.fixture
def analysis_ok(monkeypatch, tmp_path):
    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    monkeypatch.setattr(worker, "analyze_track", lambda p: dict(FEATURES))


def test_process_job_analyzes_stores_and_clears_marker(fake_mongo, analysis_ok, monkeypatch):
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.enqueue_embed("42")

    assert worker.process_job("42") is True
    assert_features_match("42", FEATURES)
    assert "42" in store.corpus_ids()
    assert store.enqueue_embed("42") is True   # marker cleared -> re-enqueueable


def test_process_job_prefers_fresh_deezer_preview_url(fake_mongo, monkeypatch, tmp_path):
    """Stored preview URLs expire (~15 min hdnea token): a queued job must
    re-fetch from Deezer rather than download the URL /seed stored."""
    store.put_track_meta({**TRACK, "preview_url": "http://x/stale.mp3"})
    fresh = {**TRACK, "preview_url": "http://x/fresh.mp3"}
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(fresh))
    monkeypatch.setattr(worker, "analyze_track", lambda p: dict(FEATURES))

    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    downloaded = []

    def recording_download(url):
        downloaded.append(url)
        return mp3

    monkeypatch.setattr(worker, "download_preview", recording_download)

    assert worker.process_job("42") is True
    assert downloaded == ["http://x/fresh.mp3"]
    # get_track never round-trips preview_url; the fresh-vs-stale distinction
    # is already pinned by `downloaded` above.
    assert store.get_track("42")["preview_url"] == ""


def test_process_job_falls_back_to_stored_track_when_deezer_down(fake_mongo, analysis_ok, monkeypatch):
    def deezer_down(t):
        raise OSError("no network")

    store.put_track_meta(TRACK)
    monkeypatch.setattr(worker.deezer, "get_track", deezer_down)
    assert worker.process_job("42") is True
    assert_features_match("42", FEATURES)


def test_process_job_failure_logs_clears_marker_never_raises(fake_mongo, monkeypatch):
    def boom(url):
        raise OSError("download failed")

    store.put_track_meta(TRACK)
    store.enqueue_embed("42")
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    monkeypatch.setattr(worker, "download_preview", boom)

    assert worker.process_job("42") is False
    assert store.get_features("42") is None
    assert store.enqueue_embed("42") is True   # marker cleared despite failure


def test_process_job_no_metadata_anywhere_fails_cleanly(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: None)
    assert worker.process_job("42") is False


def test_tick_survives_a_dequeue_connection_error(monkeypatch):
    """A transient store error on the blocking pop must not kill the loop."""
    calls = {"n": 0}

    def flaky_dequeue(timeout=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("connection refused")
        return None

    monkeypatch.setattr(worker.store, "dequeue_job", flaky_dequeue)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker._tick()  # first call: dequeue raises, caught and swallowed
    assert calls["n"] == 1
    worker._tick()  # second call: process continues normally
    assert calls["n"] == 2


# ---- T2.6: attribution jobs share the loop with embed jobs ----

REC = {**TRACK, "track_id": "43", "title": "Flamenco Sketches"}


@pytest.fixture
def embedding_stub(monkeypatch, tmp_path):
    """Stands in for essentia: 'embeds' audio as its per-band energy, so
    deleting a band provably changes the vector without loading a model."""
    import numpy as np

    mp3 = tmp_path / "seed.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)

    class FakeEmbedding:
        SAMPLE_RATE = 16000

        @staticmethod
        def load_audio(path):
            t = np.arange(16000) / 16000.0
            return np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 4000 * t)

        @staticmethod
        def effnet_frames(path):
            import wave as wave_mod

            with wave_mod.open(str(path), "rb") as src:
                raw = src.readframes(src.getnframes())
            audio = np.frombuffer(raw, dtype="<i2").astype(float)
            spectrum = np.abs(np.fft.rfft(audio))
            lows = spectrum[:len(spectrum) // 2].sum()
            highs = spectrum[len(spectrum) // 2:].sum()
            return np.array([[lows, highs]])

    monkeypatch.setitem(
        __import__("sys").modules,
        "music_recommendations.analysis.embedding", FakeEmbedding,
    )
    monkeypatch.setattr(
        "music_recommendations.analysis.embedding", FakeEmbedding, raising=False
    )
    return FakeEmbedding


def test_dequeue_job_gives_embed_work_priority_over_attribution(fake_mongo):
    store.enqueue_attribution("42", "43")
    store.enqueue_embed("42")

    assert store.dequeue_job(timeout=0) == ("embed", "42")
    assert store.dequeue_job(timeout=0) == ("attribution", "42|43")
    assert store.dequeue_job(timeout=0) is None


def test_tick_routes_an_attribution_job(fake_mongo, monkeypatch):
    seen = []
    monkeypatch.setattr(worker, "process_attribution",
                        lambda seed, rec: seen.append((seed, rec)) or True)
    store.enqueue_attribution("42", "43")

    worker._tick()

    assert seen == [("42", "43")]


def test_process_attribution_writes_one_delta_per_band(fake_mongo, embedding_stub,
                                                       monkeypatch):
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0]})
    store.put_track(REC, {"embedding": [1.0, 0.0]})
    store.enqueue_attribution("42", "43")

    assert worker.process_attribution("42", "43") is True

    cached = store.get_attribution("42", "43")
    assert cached["status"] == "ready"
    assert cached["base"] == pytest.approx(1.0)
    assert len(cached["bands"]) == 10
    for band in cached["bands"]:
        assert band["hi_hz"] > band["lo_hz"]
        assert isinstance(band["delta"], float)
    # Deleting audio must move the model's answer for at least one band,
    # otherwise the attribution is measuring nothing.
    assert any(abs(band["delta"]) > 1e-9 for band in cached["bands"])
    # Marker cleared, so a later re-request can queue the pair again.
    assert store.enqueue_attribution("42", "43") is True


def test_process_attribution_caches_failure_so_the_phone_stops_polling(fake_mongo,
                                                                       monkeypatch):
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0]})   # rec never analyzed

    assert worker.process_attribution("42", "43") is False

    cached = store.get_attribution("42", "43")
    assert cached["status"] == "failed"
    assert cached["error"]


def test_write_wav_keeps_the_audio_at_its_own_level(tmp_path):
    """No normalization anywhere: a gain would move the counterfactuals off
    the level the clean track was analyzed at, and a log-mel front end reads
    that as a spectral change."""
    import numpy as np
    import wave as wave_mod

    quiet = 0.25 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    path = worker._write_wav(quiet, 16000)
    try:
        with wave_mod.open(str(path), "rb") as src:
            peak = np.abs(np.frombuffer(src.readframes(src.getnframes()),
                                        dtype="<i2")).max()
    finally:
        path.unlink(missing_ok=True)

    assert peak == pytest.approx(0.25 * 32767, rel=0.01)


def test_attribution_measures_against_an_identically_processed_reference(
        fake_mongo, embedding_stub, monkeypatch):
    """Deltas compare wav-vs-wav. Measuring against the stored mp3 embedding
    would fold the decode difference into all ten bands as a constant."""
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0]})
    store.put_track(REC, {"embedding": [1.0, 0.0]})

    embeds = []
    original = worker._embed_waveform

    def counting(mod, samples):
        embeds.append(len(samples))
        return original(mod, samples)

    monkeypatch.setattr(worker, "_embed_waveform", counting)
    assert worker.process_attribution("42", "43") is True

    # One clean reference pass, then one per band.
    assert len(embeds) == 11
    # base still reports the stored-embedding cosine, so the number on
    # screen matches the score the rest of the app shows.
    assert store.get_attribution("42", "43")["base"] == pytest.approx(1.0)


def test_attribution_analyzes_at_the_long_window(fake_mongo, embedding_stub,
                                                 monkeypatch):
    """The window is the fix for low-band resolution, so it is pinned here:
    falling back to band_stop's defaults would make neighbouring low bands
    measure the same thing again."""
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0]})
    store.put_track(REC, {"embedding": [1.0, 0.0]})

    seen = []
    original = worker.viz.band_stop

    def spy(audio, sample_rate, lo_hz, hi_hz, **kwargs):
        seen.append(kwargs)
        return original(audio, sample_rate, lo_hz, hi_hz, **kwargs)

    monkeypatch.setattr(worker.viz, "band_stop", spy)
    assert worker.process_attribution("42", "43") is True

    assert len(seen) == 10
    assert all(k == {"fft_size": 8192, "hop": 4096} for k in seen)


def test_attribution_never_clips_a_hot_counterfactual(fake_mongo, embedding_stub,
                                                      monkeypatch):
    """Deleting a band that opposed a peak can push the residual above full
    scale. One shared gain keeps every counterfactual inside the rails —
    a clipped sample is broadband distortion charged to that band."""
    import numpy as np
    import wave as wave_mod

    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0]})
    store.put_track(REC, {"embedding": [1.0, 0.0]})
    # A hot master: peaks already at full scale before anything is removed.
    t = np.arange(16000) / 16000.0
    hot = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 4000 * t)
    hot = hot / np.abs(hot).max()
    monkeypatch.setattr(embedding_stub, "load_audio", staticmethod(lambda p: hot))

    peaks = []
    original = worker._write_wav

    def spy(samples, sample_rate):
        path = original(samples, sample_rate)
        with wave_mod.open(str(path), "rb") as src:
            raw = src.readframes(src.getnframes())
        peaks.append(np.abs(np.frombuffer(raw, dtype="<i2")).max())
        return path

    monkeypatch.setattr(worker, "_write_wav", spy)
    assert worker.process_attribution("42", "43") is True

    assert len(peaks) == 11
    assert max(peaks) < 32767          # nothing pinned to the rail
