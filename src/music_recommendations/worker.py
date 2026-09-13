"""The one background process: embed queued tracks, answer attribution jobs,
and grow the corpus by crawling Deezer whenever the queue is empty.

    python -m music_recommendations.worker

Runs on the VM next to the API (deploy/docker-compose.yml). Needs MONGODB_URI.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from music_recommendations.analysis import analyze_tracks
from music_recommendations.corpus import crawl
from music_recommendations.server import deezer, store, viz

CORPUS_CAP = int(os.environ.get("CORPUS_CAP", "300000"))
CORPUS_BYTES_CAP = int(os.environ.get("CORPUS_BYTES_CAP", str(450 * 1024 * 1024)))
CRAWL_INTERVAL_S = float(os.environ.get("CRAWL_INTERVAL_S", "60"))
MAX_QUEUED = 200          # don't flood the queue; the worker drains ~12 tracks/min

# Embed jobs are claimed in groups so their patches can share one 64-patch
# inference batch: a 30 s preview is 28 patches, so three tracks fill a
# batch that one track would have left 56% zero padding. Three is also the
# download fan-out, which keeps the network wait roughly the length of the
# slowest preview instead of the sum of three.
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


def download_preview(url: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    path = Path(name)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            path.write_bytes(resp.read())
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
    """Fetch straight from Deezer: preview URLs expire in ~15 min (hdnea
    token), so a job that waited in the queue would 403 on download if we
    trusted a stored URL. There is no usable fallback to the store's own
    record -- store.get_track() always answers "" for preview_url (it is
    never persisted) -- so a Deezer failure here fails the job outright."""
    try:
        return deezer.get_track(track_id)
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

    The group exists for the inference batch: EffNet's graph is frozen at 64
    patches and one 30 s preview is 28 of them, so tracks analyzed alone pay
    full price for mostly-empty batches. Downloads are issued in parallel for
    the same reason -- the group is only as fast as its slowest step.

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
        except Exception as exc:  # noqa: BLE001 - a dead graph fails the group
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


def _write_wav(samples: "np.ndarray", sample_rate: int) -> Path:
    """16-bit PCM temp file at the audio's OWN level: MonoLoader wants a
    path, not an array.

    Deliberately no normalization. Any gain here — even one shared by every
    band — moves the counterfactuals away from the level the clean track was
    analyzed at, and a log-mel front end reads a level shift as a change in
    the spectrum, folding a constant bias into all ten deltas.
    """
    import numpy as np

    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    path = Path(name)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


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


def _embed_waveform(embed_mod, samples: "np.ndarray") -> "np.ndarray":
    """Mean EffNet embedding of a raw waveform, via a temp wav (the model's
    loader takes a path). Always cleans the file up."""
    wav = _write_wav(samples, embed_mod.SAMPLE_RATE)
    try:
        return embed_mod.effnet_frames(wav).mean(axis=0)
    finally:
        wav.unlink(missing_ok=True)


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

    from music_recommendations.analysis import embedding as embed_mod

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
        audio = embed_mod.load_audio(mp3)          # mono, 16 kHz — model rate

        edges = viz.band_edges()
        # A long window on purpose: at 2048 the bins are 7.8 Hz and the
        # lowest log-spaced bands are only a few bins wide, so adjacent
        # bands would come out as the same filter and their attributions
        # would be indistinguishable.
        counterfactuals = [
            viz.band_stop(audio, embed_mod.SAMPLE_RATE, lo_hz, hi_hz,
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
        reference = _embed_waveform(embed_mod, audio * gain)
        reference_similarity = _cosine(reference, rec_vec)

        bands = []
        for (lo_hz, hi_hz), filtered in zip(edges, counterfactuals):
            started = time.monotonic()
            occluded = _embed_waveform(embed_mod, filtered * gain)
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


def _enqueue_new(tracks: list[dict]) -> int:
    """Store metadata and queue every track we have not analyzed; count queued.

    Skips tracks whose embed job already failed permanently -- otherwise a
    crawl just keeps re-discovering and re-enqueueing the same broken
    tracks (dead preview, unsupported codec, ...) forever.
    """
    ids = [track["track_id"] for track in tracks]
    features = store.get_many_features(ids)
    failed = store.failed_ids(ids)
    queued = 0
    for track, track_features in zip(tracks, features):
        if track_features is not None or track["track_id"] in failed:
            continue
        store.put_track_meta(track)
        if store.enqueue_embed(track["track_id"]):
            queued += 1
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


def _grow_roots(roots: list[str], tracks: list[dict]) -> None:
    """Add up to 5 newly-discovered artist names to the `crawl_roots` state,
    so the snowball/deep-cuts graph expands from what the corpus actually
    contains instead of staying pinned to the 8 seed roots forever."""
    discovered = []
    for track in tracks:
        name = track.get("artist")
        if name and name not in roots and name not in discovered:
            discovered.append(name)
        if len(discovered) >= 5:
            break
    if not discovered:
        return
    state = store.get_state("crawl_roots") or {"names": []}
    names = list(state.get("names", []))
    for name in discovered:
        if name not in names:
            names.append(name)
    store.put_state("crawl_roots", {"names": names[:200]})


def crawl_step() -> int:
    """One bounded slice of crawling, rotating over three sources: a genre
    chart, one root's snowball neighbours, or a root's deep album cuts.

    Breadth (charts across genres), depth (the artist-relatedness graph),
    and obscurity (album tracks that never show up in a /top or /related
    call) all keep growing this way. The cursor lives in Atlas so a restart
    continues where it left off.

    Runs inline in `_tick`, so a slow Deezer response (or backoff) delays
    job processing by up to a few minutes -- acceptable for a background
    crawl, not for the embed/attribution queue it shares the loop with.
    """
    size = store.data_size_bytes()
    if size >= CORPUS_BYTES_CAP or store.corpus_size() >= CORPUS_CAP \
            or store.queued_count() >= MAX_QUEUED:
        if size >= CORPUS_BYTES_CAP:
            print(f"[worker] byte cap reached: {size/1e6:.1f} MB of {CORPUS_BYTES_CAP/1e6:.0f} MB", flush=True)
        return 0
    state = store.get_state("crawl") or {"step": 0}
    step = int(state.get("step", 0))
    genres = list(crawl.GENRES)
    roots_state = store.get_state("crawl_roots")
    roots = crawl.ROOTS + (roots_state["names"] if roots_state else [])
    arm = step % 3
    grow_from = None
    if arm == 0:
        genre = genres[(step // 3) % len(genres)]
        tracks = crawl.from_charts([genre], per_genre=100)
        source = f"chart {crawl.GENRES[genre]}"
        grow_from = tracks
    elif arm == 1:
        root = roots[(step // 3) % len(roots)]
        tracks = crawl.snowball([root], hops=1, per_artist=10)
        source = f"snowball {root}"
        grow_from = tracks
    else:
        root = roots[(step // 3) % len(roots)]
        ids = crawl.resolve_artists([root])
        tracks = list(crawl.deep_cuts(ids, albums_per_artist=3))
        source = f"deep cuts {root}"
    n = _enqueue_new(tracks)
    if grow_from is not None:
        _grow_roots(roots, grow_from)
    store.put_state("crawl", {"step": step + 1})
    print(f"[worker] crawl {source}: {len(tracks)} candidates, {n} queued"
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
    """One loop iteration: sweep stale claims, then claim and process a
    group of embed jobs (or a single attribution job), if there is any work.
    When there is none, crawl for more corpus -- rate-limited so we don't
    hammer Deezer while idle.

    The process_* helpers never raise, but store.dequeue_job (and
    requeue_stale) can (a transient connection error while polling Atlas)
    -- guard each here so main()'s loop survives a connection blip instead
    of dying.
    """
    try:
        store.requeue_stale()
    except Exception as exc:
        print(f"[worker] requeue_stale error {exc}", flush=True)
    try:
        group, leftover = _claim_group()
        if not group and leftover is None:
            global _last_crawl, _crawl_backoff_s
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
    try:
        seed_fixture_if_empty()
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] seed error {exc}", flush=True)
    while True:
        _tick()


if __name__ == "__main__":
    main()
