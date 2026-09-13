"""worker.py: embed queued tracks, run attributions, crawl when idle."""
import numpy as np
import pytest

from music_recommendations import worker
from music_recommendations.server import store

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
    monkeypatch.setattr(worker, "analyze_tracks", lambda paths: [dict(FEATURES) for _ in paths])


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
    monkeypatch.setattr(worker, "analyze_tracks", lambda paths: [dict(FEATURES) for _ in paths])

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


def test_process_job_fails_cleanly_when_deezer_down(fake_mongo, analysis_ok, monkeypatch):
    """store.get_track() never has a usable preview_url (it's always ""), so
    there's no stored fallback left to fall back to -- a Deezer outage just
    fails the job."""
    def deezer_down(t):
        raise OSError("no network")

    store.put_track_meta(TRACK)
    store.enqueue_embed("42")
    monkeypatch.setattr(worker.deezer, "get_track", deezer_down)
    assert worker.process_job("42") is False
    assert store.get_features("42") is None
    job = fake_mongo.jobs.find_one({"_id": "embed:42"})
    assert job["state"] == "failed" and job["error"]


def test_process_job_failure_logs_records_failed_job_never_raises(fake_mongo, monkeypatch):
    def boom(url):
        raise OSError("download failed")

    store.put_track_meta(TRACK)
    store.enqueue_embed("42")
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: dict(TRACK))
    monkeypatch.setattr(worker, "download_preview", boom)

    assert worker.process_job("42") is False
    assert store.get_features("42") is None
    job = fake_mongo.jobs.find_one({"_id": "embed:42"})
    assert job["state"] == "failed"
    assert "OSError" in job["error"] and "download failed" in job["error"]


def test_process_job_no_metadata_anywhere_fails_cleanly(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: None)
    assert worker.process_job("42") is False


def test_tick_sweeps_stale_jobs_before_dequeuing(fake_mongo, monkeypatch):
    calls = {"n": 0}

    def counting_requeue_stale(*args, **kwargs):
        calls["n"] += 1
        return 0

    monkeypatch.setattr(worker.store, "requeue_stale", counting_requeue_stale)
    monkeypatch.setattr(worker.store, "dequeue_job", lambda timeout=5: None)

    worker._tick()

    assert calls["n"] == 1


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


# ---- crawl + first-boot seed ----

def _tracks(ids):
    return [{"track_id": i, "title": f"t{i}", "artist": "a", "album": "b",
             "artwork_url": "u", "preview_url": "p"} for i in ids]


def test_enqueue_new_reads_features_in_one_batch(fake_mongo, monkeypatch):
    calls = {"batch": 0, "single": 0}
    original = store.get_many_features
    monkeypatch.setattr(worker.store, "get_many_features",
                        lambda ids: (calls.__setitem__("batch", calls["batch"] + 1), original(ids))[1])
    monkeypatch.setattr(worker.store, "get_features",
                        lambda tid: (calls.__setitem__("single", calls["single"] + 1), None)[1])

    assert worker._enqueue_new(_tracks(["1", "2", "3", "4", "5"])) == 5

    assert calls["batch"] == 1
    assert calls["single"] == 0


def test_seed_fixture_if_empty_enqueues_all_thirty_once(fake_mongo):
    assert worker.seed_fixture_if_empty() == 30
    assert store.queued_count() == 30
    # Guard is "nothing queued and empty corpus": a second call is a no-op.
    assert worker.seed_fixture_if_empty() == 0
    assert store.queued_count() == 30


def test_seed_fixture_skipped_when_corpus_has_tracks(fake_mongo):
    store.put_track(_tracks(["x"])[0], {"embedding": [1.0]})
    assert worker.seed_fixture_if_empty() == 0


def test_crawl_step_enqueues_unseen_tracks_and_advances_cursor(fake_mongo, monkeypatch):
    calls = []
    monkeypatch.setattr(worker.crawl, "from_charts",
                        lambda genre_ids, per_genre=100: (calls.append(("charts", genre_ids)), _tracks(["1", "2", "3"]))[1])
    monkeypatch.setattr(worker.crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10: (calls.append(("snowball", root_names)), _tracks(["3", "4"]))[1])
    store.put_track(_tracks(["2"])[0], {"embedding": [1.0]})      # already analyzed
    assert worker.crawl_step() == 2                                 # 1 and 3
    assert store.get_state("crawl") == {"step": 1}
    assert worker.crawl_step() == 1                                 # 4 (3 is queued already)
    assert store.get_state("crawl") == {"step": 2}
    assert calls[0][0] == "charts" and calls[1][0] == "snowball"
    assert store.get_track("1")["title"] == "t1"                    # meta stored for the queue


def test_crawl_step_rotates_three_arms(fake_mongo, monkeypatch):
    """step % 3 picks the source: charts, snowball, then deep cuts (resolve
    the root's artist id, then album tracks) -- each indexed by step // 3."""
    calls = []
    monkeypatch.setattr(worker.crawl, "from_charts",
                        lambda genre_ids, per_genre=100:
                            (calls.append(("charts", genre_ids, per_genre)), _tracks(["c1"]))[1])
    monkeypatch.setattr(worker.crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10:
                            (calls.append(("snowball", root_names, hops, per_artist)), _tracks(["s1"]))[1])
    monkeypatch.setattr(worker.crawl, "resolve_artists",
                        lambda names: (calls.append(("resolve", names)), [999])[1])
    monkeypatch.setattr(worker.crawl, "deep_cuts",
                        lambda artist_ids, albums_per_artist=6:
                            (calls.append(("deep_cuts", artist_ids, albums_per_artist)), iter(_tracks(["d1"])))[1])

    assert worker.crawl_step() == 1
    assert store.get_track("c1") is not None
    assert worker.crawl_step() == 1
    assert store.get_track("s1") is not None
    assert worker.crawl_step() == 1
    assert store.get_track("d1") is not None

    assert calls[0][0] == "charts"
    assert calls[1][0] == "snowball"
    assert calls[2] == ("resolve", [calls[1][1][0]])
    assert calls[3][0] == "deep_cuts" and calls[3][1] == [999] and calls[3][2] == 3
    assert store.get_state("crawl") == {"step": 3}


def test_crawl_roots_grow_from_discovered_artists(fake_mongo, monkeypatch):
    """A chart or snowball step's discovered artists (up to 5, deduped
    against the existing roots) join `crawl_roots` so later snowball/deep-cuts
    arms explore artists the corpus actually contains, not just the 8 seeds."""
    discovered = _tracks(["1", "2", "3", "4", "5", "6"])
    for i, track in enumerate(discovered):
        track["artist"] = f"Artist {i}"
    monkeypatch.setattr(worker.crawl, "from_charts",
                        lambda genre_ids, per_genre=100: discovered)

    worker.crawl_step()

    grown = store.get_state("crawl_roots")
    assert grown is not None
    assert grown["names"] == [f"Artist {i}" for i in range(5)]      # capped at 5 new

    # A later snowball arm picks its root from ROOTS + crawl_roots: put the
    # cursor at an index that only exists once the grown name is appended.
    combined_len = len(worker.crawl.ROOTS) + len(grown["names"])
    grown_index = combined_len - 1                    # the newly grown name
    step = grown_index * 3 + 1                         # arm 1 == snowball
    store.put_state("crawl", {"step": step})
    seen_roots = []
    monkeypatch.setattr(worker.crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10:
                            (seen_roots.append(root_names), [])[1] or [])
    worker.crawl_step()
    assert seen_roots[0] == [grown["names"][-1]]


def test_crawl_roots_cap_at_two_hundred(fake_mongo):
    store.put_state("crawl_roots", {"names": [f"R{i}" for i in range(198)]})
    tracks = _tracks(["1", "2", "3", "4", "5"])
    for i, track in enumerate(tracks):
        track["artist"] = f"New {i}"
    worker._grow_roots(list(worker.crawl.ROOTS), tracks)
    assert len(store.get_state("crawl_roots")["names"]) == 200


def test_idle_backoff_doubles_on_empty_crawl_and_resets_on_yield(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker.store, "requeue_stale", lambda: 0)
    monkeypatch.setattr(worker.store, "dequeue_job", lambda timeout=5: None)
    monkeypatch.setattr(worker, "CRAWL_INTERVAL_S", 10)
    monkeypatch.setattr(worker, "_crawl_backoff_s", 10)
    monkeypatch.setattr(worker, "CRAWL_BACKOFF_MAX_S", 40)

    steps = iter([0, 0, 0, 5])

    def fake_crawl_step():
        return next(steps)

    monkeypatch.setattr(worker, "crawl_step", fake_crawl_step)

    worker._last_crawl = 0.0
    worker._tick()
    assert worker._crawl_backoff_s == 20
    worker._last_crawl = 0.0
    worker._tick()
    assert worker._crawl_backoff_s == 40
    worker._last_crawl = 0.0
    worker._tick()
    assert worker._crawl_backoff_s == 40             # capped
    worker._last_crawl = 0.0
    worker._tick()
    assert worker._crawl_backoff_s == 10             # reset after a yield


def test_enqueue_new_skips_failed_tracks(fake_mongo):
    store.enqueue_embed("2")
    store.fail_job("embed:2", "boom")
    assert worker._enqueue_new(_tracks(["1", "2", "3"])) == 2
    assert store.queued_count() == 2
    assert store.get_track("2") is None              # never even wrote metadata


def test_crawl_step_respects_corpus_cap_and_queue_depth(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker, "CORPUS_CAP", 1)
    store.put_track(_tracks(["x"])[0], {"embedding": [1.0]})
    assert worker.crawl_step() == 0
    monkeypatch.setattr(worker, "CORPUS_CAP", 100000)
    monkeypatch.setattr(worker, "MAX_QUEUED", 1)
    store.enqueue_embed("q1")
    assert worker.crawl_step() == 0


def test_tick_crawls_only_when_idle_and_rate_limited(fake_mongo, monkeypatch):
    crawls = []
    monkeypatch.setattr(worker, "crawl_step", lambda: crawls.append(1) or 0)
    monkeypatch.setattr(worker.store, "requeue_stale", lambda: 0)
    monkeypatch.setattr(worker.store, "dequeue_job", lambda timeout=5: None)
    monkeypatch.setattr(worker, "CRAWL_INTERVAL_S", 1000)
    monkeypatch.setattr(worker, "_crawl_backoff_s", 1000)
    worker._last_crawl = 0.0
    worker._tick()
    worker._tick()
    assert crawls == [1]                                            # second tick inside the interval
    store.enqueue_embed("busy")
    monkeypatch.setattr(worker.store, "dequeue_job", lambda timeout=5: ("embed", "busy"))
    monkeypatch.setattr(worker, "process_jobs", lambda tids: len(tids))
    worker._last_crawl = 0.0
    worker._tick()
    assert crawls == [1]                                            # busy tick never crawls


def test_crawl_step_stops_at_the_byte_cap(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker, "CORPUS_BYTES_CAP", 10)
    monkeypatch.setattr(worker.store, "data_size_bytes", lambda: 11)
    monkeypatch.setattr(worker.crawl, "from_charts", lambda *a, **k: _tracks(["1"]))
    assert worker.crawl_step() == 0


# ---- embed jobs are claimed and analyzed in groups ----

@pytest.fixture
def group_ok(monkeypatch, tmp_path):
    """Queue-independent stubs for a grouped tick: a temp mp3 for every
    download, and an analyze_tracks that reports how many paths it got."""
    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    monkeypatch.setattr(worker.deezer, "get_track",
                        lambda t: {**TRACK, "track_id": t})
    calls = []
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: (calls.append(len(paths)),
                                       [dict(FEATURES) for _ in paths])[1])
    return calls


def _queue_embeds(*ids):
    for tid in ids:
        store.put_track_meta({**TRACK, "track_id": tid})
        store.enqueue_embed(tid)


def test_tick_processes_a_group_of_embed_jobs_with_one_analysis_call(fake_mongo,
                                                                     group_ok):
    """Three queued tracks leave the loop as one inference group, so their
    patches can share the fixed 64-patch batch."""
    _queue_embeds("1", "2", "3")

    worker._tick()

    assert group_ok == [3]
    assert store.corpus_size() == 3
    assert store.queued_count() == 0


def test_group_is_capped_at_group_size(fake_mongo, group_ok, monkeypatch):
    """More queued work than GROUP_SIZE: this tick takes exactly GROUP_SIZE
    and leaves the rest for the next one."""
    _queue_embeds("1", "2", "3", "4", "5")
    monkeypatch.setattr(worker, "GROUP_SIZE", 2)

    worker._tick()

    assert group_ok == [2]
    assert store.queued_count() == 3


def test_group_isolates_a_failed_download(fake_mongo, group_ok, monkeypatch,
                                          tmp_path):
    """One dead preview fails only its own job: the other two still reach
    analysis and get stored."""
    _queue_embeds("1", "2", "3")
    mp3 = tmp_path / "p.mp3"

    def flaky_download(url):
        if url.endswith("/2.mp3"):
            raise OSError("download failed")
        return mp3

    monkeypatch.setattr(worker, "download_preview", flaky_download)
    monkeypatch.setattr(
        worker.deezer, "get_track",
        lambda t: {**TRACK, "track_id": t, "preview_url": f"http://x/{t}.mp3"},
    )

    worker._tick()

    assert group_ok == [2]
    assert_features_match("1", FEATURES)
    assert_features_match("3", FEATURES)
    assert store.get_features("2") is None
    job = fake_mongo.jobs.find_one({"_id": "embed:2"})
    assert job["state"] == "failed"
    assert "OSError" in job["error"] and "download failed" in job["error"]


def test_group_isolates_a_failed_analysis(fake_mongo, group_ok, monkeypatch):
    """analyze_tracks reports a per-track failure in that track's slot
    instead of raising, so the rest of the group is still stored."""
    _queue_embeds("1", "2")
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [ValueError("too short"), dict(FEATURES)])

    worker._tick()

    assert store.get_features("1") is None
    assert_features_match("2", FEATURES)
    job = fake_mongo.jobs.find_one({"_id": "embed:1"})
    assert job["state"] == "failed" and "too short" in job["error"]


def test_group_unlinks_every_temp_file(fake_mongo, group_ok, monkeypatch,
                                       tmp_path):
    """Previews land in temp files; a group must not leak them."""
    _queue_embeds("1", "2")
    made = []

    def make_mp3(url):
        path = tmp_path / f"p{len(made)}.mp3"
        path.write_bytes(b"mp3")
        made.append(path)
        return path

    monkeypatch.setattr(worker, "download_preview", make_mp3)

    worker._tick()

    assert len(made) == 2
    assert not any(path.exists() for path in made)


def test_an_attribution_job_ends_the_group_and_is_still_processed(fake_mongo,
                                                                  group_ok,
                                                                  monkeypatch):
    """A non-embed job claimed while filling a group must not be dropped:
    the embed group runs first, then the attribution."""
    _queue_embeds("1", "2")
    store.enqueue_attribution("42", "43")
    seen = []
    monkeypatch.setattr(worker, "process_attribution",
                        lambda seed, rec: seen.append((seed, rec)) or True)

    worker._tick()

    assert group_ok == [2]
    assert seen == [("42", "43")]
    assert store.corpus_size() == 2


def test_downloads_in_a_group_run_in_parallel(fake_mongo, group_ok, monkeypatch,
                                              tmp_path):
    """Downloads are most of a group's wall clock, so they must overlap: a
    barrier every download has to reach only clears if they run at once."""
    import threading

    _queue_embeds("1", "2", "3")
    mp3 = tmp_path / "p.mp3"
    barrier = threading.Barrier(3, timeout=10)

    def blocking_download(url):
        barrier.wait()
        return mp3

    monkeypatch.setattr(worker, "download_preview", blocking_download)

    worker._tick()

    assert group_ok == [3]
    assert store.corpus_size() == 3


def test_download_preview_leaves_no_temp_file_when_the_fetch_fails(monkeypatch,
                                                                   tmp_path):
    """mkstemp creates the file before the fetch, and only a preview handed
    back to the caller is ever unlinked -- so a 403 on an expired preview
    token used to leak an empty temp file per failing job."""
    import tempfile
    import urllib.request

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def boom(url, timeout=None):
        raise OSError("403")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(OSError):
        worker.download_preview("http://x/p.mp3")
    assert list(tmp_path.iterdir()) == []


def test_download_preview_keeps_the_file_it_returns(monkeypatch, tmp_path):
    import contextlib
    import tempfile
    import urllib.request

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    @contextlib.contextmanager
    def fake_urlopen(url, timeout=None):
        class Resp:
            def read(self):
                return b"mp3 bytes"
        yield Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    path = worker.download_preview("http://x/p.mp3")
    assert path.read_bytes() == b"mp3 bytes"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
