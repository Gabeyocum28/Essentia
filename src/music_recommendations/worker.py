"""The one background process: re-analyze superseded rows, embed queued
tracks, answer attribution jobs, and grow the corpus by crawling the active
sources whenever the queue is empty.

It is also the ONLY process that loads the CLAP audio tower. The API loads
at most the text tower (GET /search/text) and hands every unknown seed here
instead of analyzing it itself.

    python -m music_recommendations.worker

Runs on the VM next to the API (deploy/docker-compose.yml). Needs MONGODB_URI.

Which catalogues it crawls is SOURCES (default `deezer`); see
corpus/sources/__init__.py.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from music_recommendations.analysis import analyze_tracks
from music_recommendations.corpus import sources
from music_recommendations.server import store, viz
from music_recommendations.server.dedupe import dedupe_key

CORPUS_CAP = int(os.environ.get("CORPUS_CAP", "300000"))
CORPUS_BYTES_CAP = int(os.environ.get("CORPUS_BYTES_CAP", str(450 * 1024 * 1024)))
CRAWL_INTERVAL_S = float(os.environ.get("CRAWL_INTERVAL_S", "60"))
MAX_QUEUED = 200          # don't flood the queue; the worker drains ~12 tracks/min

# Embed jobs are claimed in groups so the whole group goes through CLAP in
# one forward pass rather than one per track, and so their previews download
# in parallel -- the group then costs roughly the slowest download plus one
# batched inference, not the sum of three of each. It is also the size of a
# re-analysis group (reanalyze_step).
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "3"))
DOWNLOAD_THREADS = int(os.environ.get("DOWNLOAD_THREADS", "3"))
FIXTURE = Path(__file__).resolve().parents[2] / "contract" / "fixture.json"
_last_crawl = 0.0

# The reachable universe from the 17 charts + 8 snowball roots is ~5k
# tracks; once every step yields 0 new candidates, idle-time crawling would
# otherwise poll Deezer forever at a fixed rate. Back off exponentially
# instead, and reset the moment a step yields something. Module attributes
# (not locals) so tests can monkeypatch them.
_crawl_backoff_s = CRAWL_INTERVAL_S
CRAWL_BACKOFF_MAX_S = 3600.0


# A 30 s preview is ~960 KB at 256 kbps, so one megabyte is the whole thing
# with room to spare -- and analysis only ever reads the first 30 s anyway
# (Jamendo hands back the FULL track, which can be ten minutes and 20 MB).
# The Range header asks; the read cap below is what actually holds, because a
# CDN may ignore Range and answer 200 with the entire file.
PREVIEW_MAX_BYTES = 1024 * 1024


def download_preview(url: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    path = Path(name)
    try:
        request = urllib.request.Request(
            url, headers={"Range": f"bytes=0-{PREVIEW_MAX_BYTES - 1}"})
        with urllib.request.urlopen(request, timeout=10) as resp:
            path.write_bytes(resp.read(PREVIEW_MAX_BYTES))
    except BaseException:
        # mkstemp already created the file, and the caller only unlinks
        # previews it was handed back — so a failed fetch (expired preview
        # token, timeout) used to leave an empty temp file behind for every
        # failing job, and the crawler retries forever.
        path.unlink(missing_ok=True)
        raise
    return path


def _to_plain(features: dict) -> dict:
    return {
        k: v.tolist() if hasattr(v, "tolist") else v for k, v in features.items()
    }


def _fresh_track(track_id: str) -> dict | None:
    """Fetch straight from the source that owns the id.

    Deezer preview URLs expire in ~15 min (hdnea token), so a job that
    waited in the queue would 403 on download if we trusted a stored URL.
    There is no usable fallback to the store's own record --
    store.get_track() always answers "" for preview_url (it is never
    persisted) -- so a source failure here fails the job outright.

    The track comes back carrying its `source` and, where the licence asks
    for one, its `attribution`; store._meta persists both.
    """
    source = sources.for_id(track_id)
    if source is None:
        return None
    try:
        return source.track(track_id)
    except Exception:
        return None


def _fail_embed(track_id: str, exc: Exception) -> None:
    """Mark one embed job failed and say why.

    A failure marks the job document `state="failed"` with an `error`
    string (store.fail_job) instead of clearing it: a cleared marker looked
    the same as a job that never ran, so a track with a permanently broken
    preview would just get silently re-queued and fail forever.
    """
    print(f"[worker] {track_id}: FAILED  {exc}", flush=True)
    try:
        store.fail_job(f"embed:{track_id}", f"{type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001 - a store blip must not kill the group
        pass


def _prepare(track_id: str) -> tuple[dict, Path]:
    """Fresh metadata plus the downloaded preview. Raises on either failure.

    Runs on a pool thread: it does nothing but Deezer I/O and a file write,
    so several tracks' previews arrive in about the time the slowest one
    takes rather than the sum of all of them.
    """
    track = _fresh_track(track_id)
    if track is None:
        raise ValueError("no metadata in the store or on Deezer")
    return track, download_preview(track["preview_url"])


def _prepare_safe(track_id: str) -> tuple[dict, Path] | Exception:
    try:
        return _prepare(track_id)
    except Exception as exc:  # noqa: BLE001 - reported per track by the caller
        return exc


def _analyze(paths: list[Path]) -> list[dict | Exception]:
    """Features (or the failure) per path, in order.

    One path or many, this is `analyze_tracks`: its one-path case is what
    `analyze_track` already delegates to, so the old single-path branch here
    was a second way to say the same thing (and a second thing to keep in
    step when the group path changed).
    """
    return analyze_tracks(paths)


def process_jobs(track_ids: list[str]) -> int:
    """Analyze a group of queued tracks together; returns how many were
    stored. Logs and swallows every failure, so one bad track costs only its
    own job and never kills the loop.

    The group exists for the inference batch: CLAP embeds the whole group in
    one forward pass, and the per-track fixed cost of a torch forward is what
    a group amortizes. Downloads are issued in parallel for the same reason
    -- the group is only as fast as its slowest step.

    Success clears the embed marker (clear_embed_marker) so the track can be
    re-analyzed later (e.g. a features-version bump).
    """
    if not track_ids:
        return 0
    started = time.monotonic()
    ready: list[tuple[str, dict, Path]] = []
    stored = 0
    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            prepared = list(pool.map(_prepare_safe, track_ids))
        for track_id, outcome in zip(track_ids, prepared):
            if isinstance(outcome, Exception):
                _fail_embed(track_id, outcome)
                continue
            track, mp3 = outcome
            ready.append((track_id, track, mp3))

        paths = [mp3 for _, _, mp3 in ready]
        try:
            results = _analyze(paths) if paths else []
        except Exception as exc:  # noqa: BLE001 - a dead model fails the group
            results = [exc] * len(paths)

        for (track_id, track, _mp3), features in zip(ready, results):
            try:
                if isinstance(features, Exception):
                    raise features
                store.put_track(track, _to_plain(features))
                store.clear_embed_marker(track_id)
            except Exception as exc:  # noqa: BLE001 - one track, one failure
                _fail_embed(track_id, exc)
                continue
            stored += 1
            print(f"[worker] {track_id}: analyzed  {track['artist']} - {track['title']}", flush=True)
        return stored
    finally:
        for _, _, mp3 in ready:
            mp3.unlink(missing_ok=True)
        elapsed = time.monotonic() - started
        # Throughput is what was actually STORED, not what was claimed: a
        # group where two of three downloads 403'd is not doing 12/min.
        rate = stored / elapsed * 60 if elapsed > 0 else 0.0
        print(f"[worker] group of {len(track_ids)}: {stored} stored in {elapsed:.1f}s "
              f"({rate:.1f} tracks/min)", flush=True)


def process_job(track_id: str) -> bool:
    """One queued track: the one-element case of process_jobs."""
    return process_jobs([track_id]) == 1


def _reanalyze_one(track_id: str) -> tuple[dict, Path] | Exception:
    """Stored metadata plus a freshly downloaded preview, or the failure.

    The metadata comes from the STORE, not from the source: this row is
    already in the corpus, its title/artist/attribution were written when it
    was crawled, and re-fetching them would turn a re-analysis into a
    re-crawl. Only the audio has to come back over the network, and only the
    source can produce a URL for it (Deezer's are 15-minute leases and the
    store never persists one).

    Runs on a pool thread: nothing here but HTTP and a file write.
    """
    try:
        track = store.get_track(track_id)
        if track is None:
            raise ValueError("no metadata in the store")
        source = sources.for_id(track_id)
        if source is None:
            raise ValueError(f"no source owns {track_id!r}")
        url = source.preview_url(track_id)
        if not url:
            raise ValueError("source has no preview for this track any more")
        return track, download_preview(url)
    except Exception as exc:  # noqa: BLE001 - reported per id by the caller
        return exc


def _fail_reanalysis(track_id: str, exc: Exception) -> None:
    """Take one id out of the stale queue for good, and say why.

    The arm works oldest-first, so an id that cannot be re-analyzed (a
    delisted track, a preview that 404s, audio that will not decode) would
    otherwise sit at the head of the queue and be retried every tick while
    the rest of the corpus waited behind it. store.fail_reanalysis marks the
    TRACK document (`reanalysis_failed_at` + `reanalysis_error`), which is
    what store._STALE excludes; the row keeps its old vectors, stays
    playable by id, and is simply not in the ranking until someone clears
    the mark. A later successful put_track clears it automatically.
    """
    print(f"[worker] reanalyze {track_id}: FAILED  {exc}", flush=True)
    try:
        store.fail_reanalysis(track_id, f"{type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001 - a store blip must not kill the group
        pass


def reanalyze_step() -> int:
    """Bring up to GROUP_SIZE stale rows up to the current feature version.

    Stale rows are live-analyzed tracks whose `features_version` is below
    the current one (store.stale_ids). They are invisible to ranking until
    this runs -- at a version bump the visible corpus drops to whatever was
    analyzed by the new stack and refills from here -- so this arm runs
    before the crawl every tick: re-earning a track we already hold beats
    discovering one we do not.

    Returns how many rows were brought up to version. Never raises: every
    failure is either marked on the row (so the queue moves on) or logged.
    """
    try:
        ids = store.stale_ids(GROUP_SIZE)
    except Exception as exc:  # noqa: BLE001 - an Atlas blip is not fatal
        print(f"[worker] reanalyze error {exc}", flush=True)
        return 0
    if not ids:
        return 0

    started = time.monotonic()
    ready: list[tuple[str, dict, Path]] = []
    stored = 0
    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            prepared = list(pool.map(_reanalyze_one, ids))
        for track_id, outcome in zip(ids, prepared):
            if isinstance(outcome, Exception):
                _fail_reanalysis(track_id, outcome)
                continue
            track, mp3 = outcome
            ready.append((track_id, track, mp3))

        paths = [mp3 for _, _, mp3 in ready]
        try:
            results = _analyze(paths) if paths else []
        except Exception as exc:  # noqa: BLE001 - a dead model fails the group
            results = [exc] * len(paths)

        for (track_id, track, _mp3), features in zip(ready, results):
            try:
                if isinstance(features, Exception):
                    raise features
                store.put_track(track, _to_plain(features))
            except Exception as exc:  # noqa: BLE001 - one track, one failure
                _fail_reanalysis(track_id, exc)
                continue
            stored += 1
        return stored
    finally:
        for _, _, mp3 in ready:
            mp3.unlink(missing_ok=True)
        try:
            remaining = store.stale_count()
        except Exception:  # noqa: BLE001
            remaining = -1
        elapsed = time.monotonic() - started
        print(f"[worker] reanalyzed {stored}, remaining {remaining} "
              f"({elapsed:.1f}s for {len(ids)})", flush=True)


# Attribution analysis window. Long enough that the narrowest log-spaced
# band still spans plenty of bins (1.95 Hz each at 16 kHz); hop is half the
# window so the Hann overlap-add stays COLA.
ATTRIBUTION_FFT = 8192
ATTRIBUTION_HOP = 4096


def _shared_gain(signals: "list[np.ndarray]", headroom: float = 0.98) -> float:
    """One scale factor for a whole set of waveforms: never clips any of
    them, and never changes their levels relative to each other."""
    import numpy as np

    peak = max((float(np.max(np.abs(s))) for s in signals), default=0.0)
    return headroom / peak if peak > headroom else 1.0


def _embed_waveform(samples: "np.ndarray") -> "np.ndarray":
    """The CLAP embedding of a raw waveform.

    Deliberately no normalization of the samples: any gain here -- even one
    shared by every band -- moves the counterfactuals away from the level the
    clean track was analyzed at, and a log-mel front end reads a level shift
    as a change in the spectrum, folding a constant bias into all ten deltas.
    The one gain that IS applied is `_shared_gain`, chosen once for the whole
    set by the caller for exactly that reason.
    """
    from music_recommendations.analysis import clap, v2

    return clap.embed_audio([samples], sr=v2.SAMPLE_RATE)[0]


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def process_attribution(seed_id: str, rec_id: str) -> bool:
    """Occlusion attribution for one (seed, rec) pair.

    For each band: delete it from the SEED's waveform, push the
    counterfactual through the real frozen model, and measure how far the
    pair's cosine falls. The drops are not additive (bands interact inside
    the network) -- this is the tractable first-order surrogate for Shapley
    values, and what the audience hears removed is exactly what the model
    lost.
    """
    import numpy as np

    from music_recommendations.analysis import v2

    mp3 = None
    try:
        seed_features = store.get_features(seed_id)
        rec_features = store.get_features(rec_id)
        if not seed_features or not rec_features:
            raise ValueError("both tracks must be analyzed")
        seed_vec = np.asarray(seed_features["embedding"], dtype=float)
        rec_vec = np.asarray(rec_features["embedding"], dtype=float)
        # The baseline compares the STORED embeddings: re-embedding the clean
        # seed here would measure decode jitter as if it were attribution.
        base = _cosine(seed_vec, rec_vec)

        track = _fresh_track(seed_id)
        if track is None:
            raise ValueError("no seed metadata in the store or on Deezer")
        mp3 = download_preview(track["preview_url"])
        audio = v2.decode(mp3)                     # mono float32, 44.1 kHz

        edges = viz.band_edges()
        # A long window on purpose: at 2048 the bins are 7.8 Hz and the
        # lowest log-spaced bands are only a few bins wide, so adjacent
        # bands would come out as the same filter and their attributions
        # would be indistinguishable.
        counterfactuals = [
            viz.band_stop(audio, v2.SAMPLE_RATE, lo_hz, hi_hz,
                          fft_size=ATTRIBUTION_FFT, hop=ATTRIBUTION_HOP)
            for lo_hz, hi_hz in edges
        ]

        # ONE gain for the clean reference and every counterfactual. Shared,
        # so no band gets make-up gain the model could read as spectral
        # change; sized against the loudest of them, because deleting a band
        # that opposed a peak can push the residual above full scale, and a
        # clipped sample is broadband distortion charged to that band.
        gain = _shared_gain([audio, *counterfactuals])

        # Deltas are measured against the clean audio pushed through this
        # SAME wav path at the SAME gain, not against the stored embedding:
        # the stored one came from the mp3, and the decode and level
        # differences would land in every delta as a constant. `base` still
        # reports the stored cosine, so the number on screen matches the
        # score the rest of the app shows.
        reference = _embed_waveform(audio * gain)
        reference_similarity = _cosine(reference, rec_vec)

        bands = []
        for (lo_hz, hi_hz), filtered in zip(edges, counterfactuals):
            started = time.monotonic()
            occluded = _embed_waveform(filtered * gain)
            delta = reference_similarity - _cosine(occluded, rec_vec)
            bands.append({"lo_hz": round(lo_hz, 1), "hi_hz": round(hi_hz, 1),
                          "delta": float(delta)})
            print(f"[worker] attr {seed_id}->{rec_id} "
                  f"{lo_hz:.0f}-{hi_hz:.0f}Hz delta={delta:+.4f} "
                  f"({time.monotonic() - started:.1f}s)", flush=True)

        store.put_attribution(seed_id, rec_id, {
            "status": "ready", "base": base, "bands": bands,
        })
        print(f"[worker] attr {seed_id}->{rec_id}: ready "
              f"(base {base:.3f})", flush=True)
        store.clear_attribution_marker(seed_id, rec_id)
        return True
    except Exception as exc:
        print(f"[worker] attr {seed_id}->{rec_id}: FAILED  {exc}", flush=True)
        try:
            # Cache the failure briefly so the phone stops polling, but let
            # the pair be retried once the TTL lapses.
            store.put_attribution(seed_id, rec_id,
                                  {"status": "failed", "error": str(exc)}, ttl=60)
        except Exception:
            pass
        try:
            # Record the failure on the job document itself (state="failed"
            # + error) instead of clearing the marker: a cleared marker is
            # indistinguishable from a job that never ran, so a pair that
            # is permanently broken would just get silently re-queued
            # forever. Colon-joined -- matches the job _id enqueue_attribution
            # actually wrote (attr:{seed}:{rec}); dequeue_job's pipe-joined
            # payload is a different string used only for routing.
            store.fail_job(f"attr:{seed_id}:{rec_id}", f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        return False
    finally:
        if mp3 is not None:
            mp3.unlink(missing_ok=True)


# How many candidates the last _enqueue_new call dropped as duplicates of a
# recording already in the corpus (or of an earlier candidate in the same
# batch). Module state rather than a second return value so the crawler can
# report it without changing _enqueue_new's contract with its other callers.
_last_duplicates_skipped = 0


def _enqueue_new(tracks: list[dict]) -> int:
    """Store metadata and queue every track we have not analyzed; count queued.

    Skips tracks whose embed job already failed permanently -- otherwise a
    crawl just keeps re-discovering and re-enqueueing the same broken
    tracks (dead preview, unsupported codec, ...) forever.

    Also skips a candidate that is another EDITION of a recording we
    already hold -- the 2009 remaster, the live take, the "(feat. X)"
    credit. Deezer publishes those under their own track ids, so the id and
    feature checks above see a brand new track and the corpus grows by a
    copy. One `existing_keys` query covers the whole batch, and keys seen
    earlier in the same batch are remembered locally: charts routinely
    return two editions side by side, and neither is in the store yet.
    """
    global _last_duplicates_skipped
    ids = [track["track_id"] for track in tracks]
    keys = [dedupe_key(track.get("title"), track.get("artist")) for track in tracks]
    features = store.get_many_features(ids)
    failed = store.failed_ids(ids)
    seen = store.existing_keys(keys)
    queued = 0
    duplicates = 0
    for track, track_features, key in zip(tracks, features, keys):
        if track_features is not None or track["track_id"] in failed:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        store.put_track_meta(track)
        if store.enqueue_embed(track["track_id"]):
            queued += 1
    _last_duplicates_skipped = duplicates
    return queued


def seed_fixture_if_empty() -> int:
    """Empty corpus with nothing queued (first boot, or after every job
    failed): queue the 30 fixture tracks so the app has something to rank."""
    if store.corpus_size() > 0 or store.queued_count() > 0:
        return 0
    tracks = json.loads(FIXTURE.read_text())["tracks"]
    n = _enqueue_new(tracks)
    print(f"[worker] empty corpus: queued {n} fixture tracks", flush=True)
    return n


def crawl_step() -> int:
    """One bounded slice of crawling, round-robin across the active sources.

    Each source decides what its own slice is (`Source.candidates`): for
    Deezer that is the chart / snowball / deep-cuts rotation, for Jamendo a
    tag list. The cursor lives in Atlas so a restart continues where it left
    off, and it is divided by the number of sources before being handed
    over, so every source walks its own rotation one step at a time rather
    than skipping N-1 of them.

    Runs inline in `_tick`, so a slow API response (or backoff) delays job
    processing by up to a few minutes -- acceptable for a background crawl,
    not for the embed/attribution queue it shares the loop with.
    """
    size = store.data_size_bytes()
    if size >= CORPUS_BYTES_CAP or store.corpus_size() >= CORPUS_CAP \
            or store.queued_count() >= MAX_QUEUED:
        if size >= CORPUS_BYTES_CAP:
            print(f"[worker] byte cap reached: {size/1e6:.1f} MB of {CORPUS_BYTES_CAP/1e6:.0f} MB", flush=True)
        return 0
    active = sources.active()
    if not active:
        print("[worker] crawl: no active sources", flush=True)
        return 0
    state = store.get_state("crawl") or {"step": 0}
    step = int(state.get("step", 0))
    source = active[step % len(active)]
    tracks = source.candidates(step // len(active))
    n = _enqueue_new(tracks)
    store.put_state("crawl", {"step": step + 1})
    print(f"[worker] crawl {source.candidate_label()}: {len(tracks)} candidates, "
          f"{n} queued, {_last_duplicates_skipped} duplicates skipped"
          f"  db {size/1e6:.1f} MB", flush=True)
    return n


def _claim_group() -> tuple[list[str], tuple[str, str] | None]:
    """Claim up to GROUP_SIZE embed jobs in one go.

    Returns the claimed track ids plus, if the claim that ended the group
    was not an embed job, that job -- it has already been taken off the
    queue, so the caller must run it rather than drop it. Only the first
    claim blocks; the rest are non-blocking, so a lone queued track is never
    delayed waiting for company that isn't coming.
    """
    job = store.dequeue_job(timeout=5)
    if not job:
        return [], None
    kind, payload = job
    if kind != "embed":
        return [], job
    track_ids = [payload]
    while len(track_ids) < GROUP_SIZE:
        try:
            job = store.dequeue_job(timeout=0)
        except Exception as exc:  # noqa: BLE001 - keep what is already claimed
            print(f"[worker] group claim error {exc}", flush=True)
            break
        if not job:
            break
        kind, payload = job
        if kind != "embed":
            return track_ids, job
        track_ids.append(payload)
    return track_ids, None


def _run_job(job: tuple[str, str]) -> None:
    """Route one non-embed job (today: attribution)."""
    kind, payload = job
    if kind == "embed":
        process_job(payload)
        return
    seed_id, _, rec_id = payload.partition("|")
    if seed_id and rec_id:
        process_attribution(seed_id, rec_id)
    else:
        print(f"[worker] bad attribution job {payload!r}", flush=True)


def _tick() -> None:
    """One loop iteration: sweep stale claims, re-analyze a group of
    superseded rows, then claim and process a group of embed jobs (or a
    single attribution job), if there is any work. When there is none, crawl
    for more corpus -- rate-limited so we don't hammer the sources while idle.

    The re-analysis arm goes first because a stale row is a track the corpus
    already paid to discover and currently cannot show; a crawl candidate is
    one it has not. It does ONE group per tick rather than draining the
    backlog in a loop, so a /seed arriving during a 19k-row backfill still
    gets analyzed within its wait window instead of a day later.

    The process_* helpers never raise, but store.dequeue_job (and
    requeue_stale) can (a transient connection error while polling Atlas)
    -- guard each here so main()'s loop survives a connection blip instead
    of dying.
    """
    try:
        store.requeue_stale()
    except Exception as exc:
        print(f"[worker] requeue_stale error {exc}", flush=True)
    reanalyzed = reanalyze_step()
    try:
        group, leftover = _claim_group()
        if not group and leftover is None:
            global _last_crawl, _crawl_backoff_s
            if reanalyzed:
                # A backfill is running. Crawling now would queue tracks the
                # corpus cannot show yet AND compete with it for the same two
                # cores; the rotation cursor is in Atlas, so nothing is lost
                # by not stepping it this tick.
                return
            if time.monotonic() - _last_crawl >= _crawl_backoff_s:
                _last_crawl = time.monotonic()
                try:
                    n = crawl_step()
                    if n > 0:
                        _crawl_backoff_s = CRAWL_INTERVAL_S
                    else:
                        _crawl_backoff_s = min(_crawl_backoff_s * 2, CRAWL_BACKOFF_MAX_S)
                except Exception as exc:  # noqa: BLE001 - Deezer flakes must not kill the loop
                    print(f"[worker] crawl error {exc}", flush=True)
            return
        if group:
            process_jobs(group)
        if leftover is not None:
            _run_job(leftover)
    except Exception as exc:
        print(f"[worker] queue error {exc}, retrying in 5s", flush=True)
        time.sleep(5)


def main() -> None:
    if not os.environ.get("MONGODB_URI"):
        print("[worker] MONGODB_URI is not set; nothing to do", flush=True)
        while True:
            time.sleep(60)
    print("[worker] up: embed + attribution jobs, crawling when idle (Ctrl-C to stop)", flush=True)
    # Deliberately not guarded: a typo in SOURCES, or Jamendo as the only
    # source with no client id, must stop the process here rather than
    # surface as a crawl that silently never yields anything.
    print(f"[worker] sources: {', '.join(s.name for s in sources.active())}",
          flush=True)
    try:
        seed_fixture_if_empty()
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] seed error {exc}", flush=True)
    while True:
        _tick()


if __name__ == "__main__":
    main()
