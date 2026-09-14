"""worker.py: embed queued tracks, run attributions, crawl when idle."""
from types import SimpleNamespace

import numpy as np
import pytest

from music_recommendations.analysis.schema import FEATURES_VERSION
from music_recommendations import worker
from music_recommendations.corpus import crawl
from music_recommendations.corpus.sources import deezer as deezer_source
from music_recommendations.server import deezer as deezer_api
from music_recommendations.server import store

TRACK = {
    "track_id": "42",
    "title": "Blue in Green",
    "artist": "Miles Davis",
    "album": "Kind of Blue",
    "artwork_url": "http://x/a.jpg",
    "preview_url": "http://x/p.mp3",
}
FEATURES = {"embedding": [0.1, 0.2], "_features_version": FEATURES_VERSION}


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
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
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
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(fresh))
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
    monkeypatch.setattr(deezer_api, "get_track", deezer_down)
    assert worker.process_job("42") is False
    assert store.get_features("42") is None
    job = fake_mongo.jobs.find_one({"_id": "embed:42"})
    assert job["state"] == "failed" and job["error"]


def test_process_job_failure_logs_records_failed_job_never_raises(fake_mongo, monkeypatch):
    def boom(url):
        raise OSError("download failed")

    store.put_track_meta(TRACK)
    store.enqueue_embed("42")
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    monkeypatch.setattr(worker, "download_preview", boom)

    assert worker.process_job("42") is False
    assert store.get_features("42") is None
    job = fake_mongo.jobs.find_one({"_id": "embed:42"})
    assert job["state"] == "failed"
    assert "OSError" in job["error"] and "download failed" in job["error"]


def test_process_job_no_metadata_anywhere_fails_cleanly(fake_mongo, monkeypatch):
    monkeypatch.setattr(deezer_api, "get_track", lambda t: None)
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
def clap_stub(monkeypatch, tmp_path):
    """Stands in for CLAP: 'embeds' a waveform as its per-band energy, so
    deleting a band provably changes the vector without loading 700 MB of
    weights. `stub.audio` is what the decoder hands back, so a test can swap
    in a hot master.

    Attribution runs on the same model the ranking does, which since the
    cutover is CLAP on 44.1 kHz float waveforms -- no temp wav, no 16 kHz
    resample, no model-specific file loader.
    """
    import numpy as np

    from music_recommendations.analysis import clap, v2

    mp3 = tmp_path / "seed.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)

    sr = v2.SAMPLE_RATE
    t = np.arange(sr) / float(sr)
    stub = SimpleNamespace(
        audio=np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 4000 * t),
        embedded=[],
    )
    monkeypatch.setattr(v2, "decode", lambda path: stub.audio)

    def embed_audio(waves, sr=None):
        out = []
        for wave in waves:
            samples = np.asarray(wave, dtype=float)
            stub.embedded.append(samples)
            spectrum = np.abs(np.fft.rfft(samples))
            half = len(spectrum) // 2
            out.append([spectrum[:half].sum(), spectrum[half:].sum()])
        return np.asarray(out, dtype=float)

    monkeypatch.setattr(clap, "embed_audio", embed_audio)
    return stub


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


def test_process_attribution_writes_one_delta_per_band(fake_mongo, clap_stub,
                                                       monkeypatch):
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track(REC, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
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
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})   # rec never analyzed

    assert worker.process_attribution("42", "43") is False

    cached = store.get_attribution("42", "43")
    assert cached["status"] == "failed"
    assert cached["error"]


def test_attribution_does_not_normalize_a_quiet_track(fake_mongo, clap_stub,
                                                      monkeypatch):
    """No make-up gain anywhere: a gain would move the counterfactuals off
    the level the clean track was analyzed at, and a log-mel front end reads
    that as a spectral change. A track that already fits stays untouched."""
    import numpy as np

    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track(REC, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    clap_stub.audio = clap_stub.audio / np.abs(clap_stub.audio).max() * 0.25

    assert worker.process_attribution("42", "43") is True

    # The clean reference is the first thing embedded, and it goes in at the
    # level it was decoded at.
    assert np.abs(clap_stub.embedded[0]).max() == pytest.approx(0.25, rel=1e-6)


def test_attribution_measures_against_an_identically_processed_reference(
        fake_mongo, clap_stub, monkeypatch):
    """Deltas compare waveform against waveform. Measuring against the stored
    mp3 embedding would fold the decode difference into all ten bands as a
    constant."""
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track(REC, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})

    embeds = []
    original = worker._embed_waveform

    def counting(samples):
        embeds.append(len(samples))
        return original(samples)

    monkeypatch.setattr(worker, "_embed_waveform", counting)
    assert worker.process_attribution("42", "43") is True

    # One clean reference pass, then one per band.
    assert len(embeds) == 11
    # base still reports the stored-embedding cosine, so the number on
    # screen matches the score the rest of the app shows.
    assert store.get_attribution("42", "43")["base"] == pytest.approx(1.0)


def test_attribution_analyzes_at_the_long_window(fake_mongo, clap_stub,
                                                 monkeypatch):
    """The window is the fix for low-band resolution, so it is pinned here:
    falling back to band_stop's defaults would make neighbouring low bands
    measure the same thing again."""
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track(REC, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})

    seen = []
    original = worker.viz.band_stop

    def spy(audio, sample_rate, lo_hz, hi_hz, **kwargs):
        seen.append(kwargs)
        return original(audio, sample_rate, lo_hz, hi_hz, **kwargs)

    monkeypatch.setattr(worker.viz, "band_stop", spy)
    assert worker.process_attribution("42", "43") is True

    assert len(seen) == 10
    assert all(k == {"fft_size": 8192, "hop": 4096} for k in seen)


def test_attribution_never_clips_a_hot_counterfactual(fake_mongo, clap_stub,
                                                      monkeypatch):
    """Deleting a band that opposed a peak can push the residual above full
    scale. One shared gain keeps every counterfactual inside the rails —
    a clipped sample is broadband distortion charged to that band."""
    import numpy as np

    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.put_track(TRACK, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    store.put_track(REC, {"embedding": [1.0, 0.0], "_features_version": FEATURES_VERSION})
    # A hot master: peaks already at full scale before anything is removed.
    clap_stub.audio = clap_stub.audio / np.abs(clap_stub.audio).max()

    assert worker.process_attribution("42", "43") is True

    # One clean reference plus one counterfactual per band, none of them
    # pinned to the rail.
    assert len(clap_stub.embedded) == 11
    assert max(np.abs(s).max() for s in clap_stub.embedded) <= 1.0


# ---- the re-analysis arm: bringing a superseded corpus up to version ----

OLD = {"embedding": [0.9, 0.1], "_features_version": FEATURES_VERSION - 1}


def _stale(track_id: str, title: str = "Old Take") -> dict:
    """A row analyzed by the previous stack: live, not retired, out of LIVE."""
    track = {**TRACK, "track_id": track_id, "title": title}
    store.put_track(track, dict(OLD))
    return track


@pytest.fixture
def reanalysis_ok(monkeypatch, tmp_path):
    """A source that still has every preview, and an analyzer that works."""
    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [dict(FEATURES) for _ in paths])
    monkeypatch.setattr(deezer_source.DeezerSource, "preview_url",
                        lambda self, track_id: f"http://x/{track_id}.mp3")


def test_stale_rows_are_invisible_until_the_arm_runs(fake_mongo, reanalysis_ok):
    """The whole reason the arm exists: LIVE demands the current version, so
    at a bump the corpus a user can be recommended drops to nothing and this
    is what refills it."""
    _stale("42")
    assert list(store.corpus_ids()) == []
    assert store.stale_count() == 1

    assert worker.reanalyze_step() == 1

    assert list(store.corpus_ids()) == ["42"]
    assert store.stale_count() == 0


def test_reanalyze_step_takes_at_most_one_group(fake_mongo, reanalysis_ok,
                                                monkeypatch):
    monkeypatch.setattr(worker, "GROUP_SIZE", 2)
    for i in range(5):
        _stale(str(100 + i))

    assert worker.reanalyze_step() == 2
    assert store.stale_count() == 3


def test_reanalyze_keeps_the_metadata_it_already_had(fake_mongo, reanalysis_ok):
    """A re-analysis is not a re-crawl: only the audio comes back over the
    network, and the title/artist/attribution written at crawl time stay."""
    _stale("42")
    fake_mongo.tracks.update_one(
        {"_id": "42"},
        {"$set": {"source": "jamendo",
                  "attribution": {"source": "jamendo", "url": "http://j/42"}}},
    )

    assert worker.reanalyze_step() == 1

    row = store.get_track("42")
    assert row["title"] == "Old Take"
    assert row["attribution_url"] == "http://j/42"
    assert_features_match("42", FEATURES)


def test_reanalyze_progress_is_logged_with_what_is_left(fake_mongo,
                                                        reanalysis_ok,
                                                        monkeypatch, capsys):
    monkeypatch.setattr(worker, "GROUP_SIZE", 1)
    _stale("42")
    _stale("43", "Another")

    worker.reanalyze_step()

    assert "reanalyzed 1, remaining 1" in capsys.readouterr().out


def test_a_dead_preview_is_retried_three_times_then_given_up_on(fake_mongo,
                                                                 monkeypatch):
    """"No preview URL" is a verdict about this track, so it counts against
    the row -- but not on the first try: a source having a bad afternoon must
    not permanently retire a third of the corpus. Three attempts, then out."""
    _stale("42")
    monkeypatch.setattr(deezer_source.DeezerSource, "preview_url",
                        lambda self, track_id: None)

    for attempt in (1, 2):
        assert worker.reanalyze_step() == 0
        row = fake_mongo.tracks.find_one({"_id": "42"})
        assert row["reanalysis_attempts"] == attempt
        assert "reanalysis_failed_at" not in row    # still in the queue
        assert store.stale_count() == 1

    assert worker.reanalyze_step() == 0
    row = fake_mongo.tracks.find_one({"_id": "42"})
    assert row["reanalysis_attempts"] == store.REANALYSIS_MAX_ATTEMPTS
    assert "reanalysis_failed_at" in row
    assert "no preview" in row["reanalysis_error"]
    assert store.stale_count() == 0               # out of the queue for good
    assert store.stale_ids(10) == []
    assert store.reanalysis_failed_count() == 1
    # ...but the track is still a playable seed with its old vectors.
    assert store.get_track("42")["title"] == "Old Take"
    assert store.get_features("42") is not None


def test_a_per_track_decode_error_counts_attempts_and_gives_up_at_three(
        fake_mongo, reanalysis_ok, monkeypatch):
    """A DecodeError in that path's slot is a verdict about the audio, so it
    is classifiable -- three attempts, then the row leaves the queue."""
    from music_recommendations.analysis.v2 import DecodeError

    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [DecodeError("not audio") for _ in paths])
    _stale("42")

    for attempt in (1, 2):
        assert worker.reanalyze_step() == 0
        assert fake_mongo.tracks.find_one({"_id": "42"})["reanalysis_attempts"] == attempt
        assert store.stale_count() == 1

    assert worker.reanalyze_step() == 0
    assert store.stale_count() == 0
    assert store.reanalysis_failed_count() == 1


def test_an_unclassifiable_per_track_failure_never_gives_up(fake_mongo,
                                                            reanalysis_ok,
                                                            monkeypatch):
    """A download that 500s, or a slot holding some unexpected exception, is
    not evidence about the TRACK. It counts as an attempt (so the row moves
    to the back of the queue instead of blocking it) but must never retire
    the row -- that is how a bad afternoon deletes a corpus."""
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [OSError("500 from the CDN") for _ in paths])
    _stale("42")

    for _ in range(store.REANALYSIS_MAX_ATTEMPTS + 2):
        assert worker.reanalyze_step() == 0

    row = fake_mongo.tracks.find_one({"_id": "42"})
    assert row["reanalysis_attempts"] > store.REANALYSIS_MAX_ATTEMPTS
    assert "reanalysis_failed_at" not in row
    assert store.stale_count() == 1
    assert store.reanalysis_failed_count() == 0


def test_a_group_wide_analysis_failure_marks_nothing(fake_mongo, reanalysis_ok,
                                                     monkeypatch, capsys):
    """analyze_tracks raising (rather than filling slots) means the MODEL is
    broken -- a missing checkpoint, an OOM, a bad image. Charging that to the
    three tracks that happened to be in the group would quietly retire the
    whole corpus three rows at a time."""
    def boom(paths):
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(worker, "analyze_tracks", boom)
    _stale("42")
    _stale("43", "Another")

    assert worker.reanalyze_step() == 0

    for track_id in ("42", "43"):
        row = fake_mongo.tracks.find_one({"_id": track_id})
        assert "reanalysis_failed_at" not in row
        assert "reanalysis_attempts" not in row
    assert store.stale_count() == 2
    assert "checkpoint missing" in capsys.readouterr().out


def test_three_group_wide_failures_halt_the_arm_then_it_retries(
        fake_mongo, reanalysis_ok, monkeypatch, capsys, tmp_path):
    """A broken model fails every group instantly, so without a breaker the
    arm spins on the whole corpus at full speed, logging forever."""
    def boom(paths):
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(worker, "analyze_tracks", boom)
    _stale("42")

    for _ in range(worker.REANALYZE_MAX_GROUP_FAILURES):
        assert worker.reanalyze_step() == 0
    assert "halting the re-analysis arm" in capsys.readouterr().out

    downloaded = []
    monkeypatch.setattr(worker, "download_preview",
                        lambda url: downloaded.append(url))
    assert worker.reanalyze_step() == 0
    assert downloaded == []                     # nothing even attempted
    assert "halting" not in capsys.readouterr().out   # said once, not per tick

    # ...and it comes back by itself once the window lapses.
    worker._halted_until = 0.0
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [dict(FEATURES) for _ in paths])
    mp3 = tmp_path / "again.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    assert worker.reanalyze_step() == 1


def test_one_good_group_resets_the_breaker(fake_mongo, reanalysis_ok,
                                           monkeypatch):
    """The count is CONSECUTIVE failures: two blips a day apart are not a
    broken model, and must not add up to a halt."""
    calls = {"n": 0}

    def sometimes(paths):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("blip")
        return [dict(FEATURES) for _ in paths]

    monkeypatch.setattr(worker, "analyze_tracks", sometimes)
    _stale("42")
    _stale("43", "Another")

    assert worker.reanalyze_step() == 0
    assert worker.reanalyze_step() >= 1
    assert worker._group_failures == 0


def test_a_put_track_failure_marks_nothing(fake_mongo, reanalysis_ok,
                                           monkeypatch, capsys):
    """The analysis SUCCEEDED; the store is what failed. Marking the row
    would retire a perfectly good track because Atlas hiccupped."""
    _stale("42")

    def boom(track, features):
        raise ConnectionError("atlas down")

    monkeypatch.setattr(store, "put_track", boom)

    assert worker.reanalyze_step() == 0
    row = fake_mongo.tracks.find_one({"_id": "42"})
    assert "reanalysis_failed_at" not in row
    assert "reanalysis_attempts" not in row
    assert store.stale_count() == 1
    assert "atlas down" in capsys.readouterr().out


def test_a_later_success_clears_the_give_up_mark(fake_mongo, reanalysis_ok):
    """Otherwise the row would be invisible to the NEXT version bump's
    backfill as well, forever."""
    _stale("42")
    for _ in range(store.REANALYSIS_MAX_ATTEMPTS):
        store.record_reanalysis_failure("42", "DecodeError: not audio",
                                        classifiable=True)
    assert store.stale_count() == 0

    store.put_track({**TRACK, "track_id": "42"}, dict(FEATURES))

    assert store.reanalysis_failed_count() == 0
    assert "42" in store.corpus_ids()
    assert "reanalysis_attempts" not in fake_mongo.tracks.find_one({"_id": "42"})


def test_an_unknown_source_does_not_block_the_queue(fake_mongo, reanalysis_ok):
    """A namespaced id whose source is no longer in the registry: nothing can
    ever fetch its audio, so it is classifiable and retires after three."""
    store.put_track({**TRACK, "track_id": "spotify:9"}, dict(OLD))

    for _ in range(store.REANALYSIS_MAX_ATTEMPTS):
        assert worker.reanalyze_step() == 0

    assert store.stale_count() == 0
    assert store.reanalysis_failed_count() == 1


def test_a_failed_row_goes_to_the_back_of_the_queue(fake_mongo, reanalysis_ok,
                                                    monkeypatch):
    """Otherwise a row that fails and is retried keeps the head of an
    oldest-first queue and the rest of the corpus never gets a turn."""
    monkeypatch.setattr(worker, "GROUP_SIZE", 1)
    _stale("42")
    _stale("43", "Another")
    monkeypatch.setattr(deezer_source.DeezerSource, "preview_url",
                        lambda self, track_id: None if track_id == "42" else "http://x/p.mp3")

    assert worker.reanalyze_step() == 0        # took 42, failed
    assert store.stale_ids(1) == ["43"]        # 42 is now behind it
    assert worker.reanalyze_step() == 1


def test_a_prioritized_row_jumps_the_queue(fake_mongo, reanalysis_ok, monkeypatch):
    """/seed on a superseded row asks for it by name: a user is waiting on
    that one track, not on the 19,000 rows in front of it."""
    monkeypatch.setattr(worker, "GROUP_SIZE", 1)
    _stale("42")
    _stale("43", "Another")
    store.prioritize_reanalysis("43")

    assert store.stale_ids(1) == ["43"]
    assert worker.reanalyze_step() == 1
    assert "43" in store.corpus_ids()


def test_the_arm_logs_how_many_it_has_given_up_on(fake_mongo, reanalysis_ok,
                                                  monkeypatch, capsys):
    """A backfill that is "finishing" only because it retired half the corpus
    should be visible in the same line that reports progress."""
    _stale("42")
    _stale("99", "Given up")
    for _ in range(store.REANALYSIS_MAX_ATTEMPTS):
        store.record_reanalysis_failure("99", "DecodeError: not audio",
                                        classifiable=True)

    worker.reanalyze_step()

    assert "gave up on 1" in capsys.readouterr().out


def test_embed_jobs_come_before_the_backfill(fake_mongo, reanalysis_ok,
                                             monkeypatch):
    """A cold /seed waits 20 s for the worker. During a day-long backfill it
    must not spend that wait behind a re-analysis group."""
    monkeypatch.setattr(worker, "GROUP_SIZE", 3)
    for i in range(5):
        _stale(str(200 + i))
    store.enqueue_embed("42")
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))

    worker._tick()

    # The queued embed job was served, and the backfill gave way to it: at
    # most one stale row was taken this tick, not a full group of three.
    assert store.get_features("42") is not None
    assert store.stale_count() >= 4


def test_reanalysis_runs_before_the_crawl(fake_mongo, reanalysis_ok, monkeypatch):
    """A stale row is a track the corpus already paid to discover and cannot
    currently show; a crawl candidate is one it has not."""
    _stale("42")
    crawled = []
    monkeypatch.setattr(worker, "crawl_step", lambda: crawled.append(1) or 0)
    monkeypatch.setattr(worker, "_last_crawl", 0.0)
    # The queue is empty in these tests; don't spend the 5 s poll proving it.
    monkeypatch.setattr(store, "dequeue_job", lambda timeout=5: None)

    worker._tick()

    assert list(store.corpus_ids()) == ["42"]
    assert crawled == []


def test_a_tick_with_nothing_stale_still_crawls(fake_mongo, monkeypatch):
    crawled = []
    monkeypatch.setattr(worker, "crawl_step", lambda: crawled.append(1) or 0)
    monkeypatch.setattr(worker, "_last_crawl", 0.0)
    # The queue is empty in these tests; don't spend the 5 s poll proving it.
    monkeypatch.setattr(store, "dequeue_job", lambda timeout=5: None)

    worker._tick()

    assert crawled == [1]


def test_a_store_blip_in_the_arm_is_not_fatal(fake_mongo, monkeypatch):
    def boom(limit):
        raise ConnectionError("atlas blip")

    monkeypatch.setattr(store, "stale_ids", boom)
    assert worker.reanalyze_step() == 0


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


def test_seed_fixture_if_empty_enqueues_the_distinct_fixture_tracks_once(fake_mongo):
    # 29, not 30: the fixture holds two Deezer ids for Django Reinhardt's
    # "Minor Swing" (79845626 on a compilation, 6431303 on the Rome
    # Sessions), and _enqueue_new now queues one edition per recording.
    assert worker.seed_fixture_if_empty() == 29
    assert store.queued_count() == 29
    # Guard is "nothing queued and empty corpus": a second call is a no-op.
    assert worker.seed_fixture_if_empty() == 0
    assert store.queued_count() == 29


def test_seed_fixture_skipped_when_corpus_has_tracks(fake_mongo):
    store.put_track(_tracks(["x"])[0], {"embedding": [1.0], "_features_version": FEATURES_VERSION})
    assert worker.seed_fixture_if_empty() == 0


def test_crawl_step_enqueues_unseen_tracks_and_advances_cursor(fake_mongo, monkeypatch):
    calls = []
    monkeypatch.setattr(crawl, "from_charts",
                        lambda genre_ids, per_genre=100: (calls.append(("charts", genre_ids)), _tracks(["1", "2", "3"]))[1])
    monkeypatch.setattr(crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10: (calls.append(("snowball", root_names)), _tracks(["3", "4"]))[1])
    store.put_track(_tracks(["2"])[0], {"embedding": [1.0], "_features_version": FEATURES_VERSION})      # already analyzed
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
    monkeypatch.setattr(crawl, "from_charts",
                        lambda genre_ids, per_genre=100:
                            (calls.append(("charts", genre_ids, per_genre)), _tracks(["c1"]))[1])
    monkeypatch.setattr(crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10:
                            (calls.append(("snowball", root_names, hops, per_artist)), _tracks(["s1"]))[1])
    monkeypatch.setattr(crawl, "resolve_artists",
                        lambda names: (calls.append(("resolve", names)), [999])[1])
    monkeypatch.setattr(crawl, "deep_cuts",
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
    monkeypatch.setattr(crawl, "from_charts",
                        lambda genre_ids, per_genre=100: discovered)

    worker.crawl_step()

    grown = store.get_state("crawl_roots")
    assert grown is not None
    assert grown["names"] == [f"Artist {i}" for i in range(5)]      # capped at 5 new

    # A later snowball arm picks its root from ROOTS + crawl_roots: put the
    # cursor at an index that only exists once the grown name is appended.
    combined_len = len(crawl.ROOTS) + len(grown["names"])
    grown_index = combined_len - 1                    # the newly grown name
    step = grown_index * 3 + 1                         # arm 1 == snowball
    store.put_state("crawl", {"step": step})
    seen_roots = []
    monkeypatch.setattr(crawl, "snowball",
                        lambda root_names, hops=1, per_artist=10:
                            (seen_roots.append(root_names), [])[1] or [])
    worker.crawl_step()
    assert seen_roots[0] == [grown["names"][-1]]


def test_crawl_roots_cap_at_two_hundred(fake_mongo):
    store.put_state("crawl_roots", {"names": [f"R{i}" for i in range(198)]})
    tracks = _tracks(["1", "2", "3", "4", "5"])
    for i, track in enumerate(tracks):
        track["artist"] = f"New {i}"
    deezer_source._grow_roots(list(crawl.ROOTS), tracks)
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


def _edition(track_id, title, artist="Miles Davis"):
    return {"track_id": track_id, "title": title, "artist": artist,
            "album": "Kind of Blue", "artwork_url": "u", "preview_url": "p"}


def test_enqueue_new_skips_a_re_release_of_a_stored_track(fake_mongo):
    store.put_track(_edition("1", "So What"), {"embedding": [1.0], "_features_version": FEATURES_VERSION})
    assert worker._enqueue_new([_edition("2", "So What (2009 Remaster)")]) == 0
    assert store.queued_count() == 0
    assert store.get_track("2") is None          # no metadata written either


def test_enqueue_new_queues_one_of_two_editions_in_the_same_batch(fake_mongo):
    queued = worker._enqueue_new([
        _edition("1", "So What"),
        _edition("2", "So What - Live at the Blackhawk"),
        _edition("3", "Blue in Green"),
    ])
    assert queued == 2
    assert sorted(j["track_id"] for j in fake_mongo.jobs.find({})) == ["1", "3"]


def test_enqueue_new_keeps_a_cover_by_another_artist(fake_mongo):
    store.put_track(_edition("1", "So What"), {"embedding": [1.0], "_features_version": FEATURES_VERSION})
    assert worker._enqueue_new([_edition("2", "So What", artist="Ron Carter")]) == 1


def test_enqueue_new_reads_existing_keys_in_one_batch(fake_mongo, monkeypatch):
    calls = []
    original = store.existing_keys
    monkeypatch.setattr(worker.store, "existing_keys",
                        lambda keys: (calls.append(list(keys)), original(keys))[1])
    assert worker._enqueue_new(_tracks(["1", "2", "3", "4", "5"])) == 5
    assert len(calls) == 1                       # one query for the whole batch
    assert len(calls[0]) == 5


def test_crawl_step_logs_duplicates_skipped(fake_mongo, monkeypatch, capsys):
    monkeypatch.setattr(crawl, "from_charts",
                        lambda genre_ids, per_genre=100: [
                            _edition("1", "So What"),
                            _edition("2", "So What (Remastered)"),
                        ])
    worker.crawl_step()
    assert "1 queued, 1 duplicates skipped" in capsys.readouterr().out


def test_crawl_step_respects_corpus_cap_and_queue_depth(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker, "CORPUS_CAP", 1)
    store.put_track(_tracks(["x"])[0], {"embedding": [1.0], "_features_version": FEATURES_VERSION})
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
    monkeypatch.setattr(crawl, "from_charts", lambda *a, **k: _tracks(["1"]))
    assert worker.crawl_step() == 0


# ---- embed jobs are claimed and analyzed in groups ----

@pytest.fixture
def group_ok(monkeypatch, tmp_path):
    """Queue-independent stubs for a grouped tick: a temp mp3 for every
    download, and an analyze_tracks that reports how many paths it got."""
    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    monkeypatch.setattr(deezer_api, "get_track",
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
        deezer_api, "get_track",
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

    seen = {}

    @contextlib.contextmanager
    def fake_urlopen(request, timeout=None):
        seen["headers"] = dict(getattr(request, "headers", {}))

        class Resp:
            def read(self, amount=None):
                return b"mp3 bytes"[:amount]
        yield Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    path = worker.download_preview("http://x/p.mp3")
    assert path.read_bytes() == b"mp3 bytes"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    # Asked for the first megabyte only: a Jamendo "preview" is the whole
    # track, which can be ten minutes and 20 MB, and analysis reads 30 s.
    assert seen["headers"]["Range"] == f"bytes=0-{worker.PREVIEW_MAX_BYTES - 1}"


def test_download_preview_stops_at_the_cap_when_range_is_ignored(monkeypatch,
                                                                 tmp_path):
    """A CDN may answer 200 with the whole file regardless of Range. The read
    cap, not the header, is what actually bounds the download."""
    import contextlib
    import tempfile
    import urllib.request

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(worker, "PREVIEW_MAX_BYTES", 16)

    @contextlib.contextmanager
    def fake_urlopen(request, timeout=None):
        class Resp:
            def read(self, amount=None):
                blob = b"x" * 1000
                return blob if amount is None else blob[:amount]
        yield Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    path = worker.download_preview("http://x/p.mp3")
    assert path.read_bytes() == b"x" * 16


# ---- feel: the worker scores new tracks as it analyzes them ----

FEEL = [0.1] * 11


def test_process_job_stores_the_feel_vector_analysis_returned(fake_mongo, monkeypatch,
                                                              tmp_path):
    """New tracks must not need the backfill: analyze_tracks returns feel
    alongside the embedding and the worker persists both."""
    mp3 = tmp_path / "p.mp3"
    mp3.write_bytes(b"mp3")
    monkeypatch.setattr(worker, "download_preview", lambda url: mp3)
    monkeypatch.setattr(worker, "analyze_tracks",
                        lambda paths: [{**FEATURES, "feel": list(FEEL)}
                                       for _ in paths])
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.enqueue_embed("42")

    assert worker.process_job("42") is True
    assert store.get_features("42")["feel"] == pytest.approx(FEEL, abs=1e-4)


def test_process_job_survives_analysis_without_feel(fake_mongo, analysis_ok,
                                                    monkeypatch):
    """An older analysis path (or a mocked one) that returns the embedding
    alone must still store the track, just without a feel vector."""
    monkeypatch.setattr(deezer_api, "get_track", lambda t: dict(TRACK))
    store.enqueue_embed("42")

    assert worker.process_job("42") is True
    assert "feel" not in store.get_features("42")


# ---- crawling round-robins the active sources ----

class _FakeSource:
    """A source that records the steps it was asked for."""

    def __init__(self, name, tracks=()):
        self.name = name
        self.tracks = list(tracks)
        self.steps = []

    def candidates(self, step):
        self.steps.append(step)
        return self.tracks

    def candidate_label(self):
        return f"{self.name} slice"


def test_crawl_step_round_robins_the_active_sources(fake_mongo, monkeypatch):
    """Each source gets every Nth step, and the cursor it is handed is
    divided by N -- so a source walks its own rotation one step at a time
    instead of skipping N-1 arms of it."""
    a = _FakeSource("a", _tracks(["a1"]))
    b = _FakeSource("b", _tracks(["b1"]))
    monkeypatch.setattr(worker.sources, "active", lambda: [a, b])

    for _ in range(4):
        worker.crawl_step()

    assert a.steps == [0, 1]
    assert b.steps == [0, 1]
    assert store.get_state("crawl") == {"step": 4}


def test_crawl_step_logs_the_source_label(fake_mongo, monkeypatch, capsys):
    source = _FakeSource("jamendo", _tracks(["j1"]))
    source.candidate_label = lambda: "jamendo tag jazz"
    monkeypatch.setattr(worker.sources, "active", lambda: [source])

    worker.crawl_step()

    out = capsys.readouterr().out
    assert "crawl jamendo tag jazz: 1 candidates, 1 queued" in out


def test_crawl_step_does_nothing_without_an_active_source(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker.sources, "active", lambda: [])
    assert worker.crawl_step() == 0
    assert store.get_state("crawl") is None


def test_fresh_track_uses_the_source_that_owns_the_id(monkeypatch):
    """A jamendo id must never be asked of Deezer."""
    asked = []

    class Jam:
        name = "jamendo"

        def track(self, track_id):
            asked.append(track_id)
            return {"track_id": track_id, "preview_url": "http://j/full.mp3"}

    monkeypatch.setattr(worker.sources, "for_id",
                        lambda tid: Jam() if tid.startswith("jamendo:") else None)
    assert worker._fresh_track("jamendo:168")["preview_url"] == "http://j/full.mp3"
    assert asked == ["jamendo:168"]
    assert worker._fresh_track("42") is None
