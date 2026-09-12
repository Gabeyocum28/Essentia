"""Pop embed jobs from the Atlas jobs collection, analyze, write back.

    MONGODB_URI=... python3 scripts/embed_worker.py

One process, one job at a time: on-demand taps trickle in and analysis is
~1 s. Bulk backfill stays analyze_corpus.py's job.
"""
from __future__ import annotations

import os
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

from music_recommendations.analysis import analyze_track
from music_recommendations.server import deezer, store, viz


def download_preview(url: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    path = Path(name)
    with urllib.request.urlopen(url, timeout=10) as resp:
        path.write_bytes(resp.read())
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


def process_job(track_id: str) -> bool:
    """Analyze one queued track. True on success; logs and swallows failures
    so one bad track never kills the loop.

    A failure marks the job document `state="failed"` with an `error`
    string (store.fail_job) instead of clearing it: a cleared marker looked
    the same as a job that never ran, so a track with a permanently broken
    preview would just get silently re-queued and fail forever. Success
    still clears the marker (clear_embed_marker) so the track can be
    re-analyzed later (e.g. a features-version bump).
    """
    try:
        track = _fresh_track(track_id)
        if track is None:
            print(f"[embed_worker] {track_id}: no metadata in the store or on Deezer", flush=True)
            store.fail_job(f"embed:{track_id}", "no metadata in the store or on Deezer")
            return False
        mp3 = download_preview(track["preview_url"])
        try:
            features = _to_plain(analyze_track(mp3))
        finally:
            mp3.unlink(missing_ok=True)
        store.put_track(track, features)
        store.clear_embed_marker(track_id)
        print(f"[embed_worker] {track_id}: analyzed  {track['artist']} - {track['title']}", flush=True)
        return True
    except Exception as exc:
        print(f"[embed_worker] {track_id}: FAILED  {exc}", flush=True)
        try:
            store.fail_job(f"embed:{track_id}", f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        return False


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
            print(f"[embed_worker] attr {seed_id}->{rec_id} "
                  f"{lo_hz:.0f}-{hi_hz:.0f}Hz delta={delta:+.4f} "
                  f"({time.monotonic() - started:.1f}s)", flush=True)

        store.put_attribution(seed_id, rec_id, {
            "status": "ready", "base": base, "bands": bands,
        })
        print(f"[embed_worker] attr {seed_id}->{rec_id}: ready "
              f"(base {base:.3f})", flush=True)
        store.clear_attribution_marker(seed_id, rec_id)
        return True
    except Exception as exc:
        print(f"[embed_worker] attr {seed_id}->{rec_id}: FAILED  {exc}", flush=True)
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


def _tick() -> None:
    """One loop iteration: sweep stale claims, then dequeue and process a
    job, if there is one.

    The process_* helpers never raise, but store.dequeue_job (and
    requeue_stale) can (a transient connection error while polling Atlas)
    -- guard each here so main()'s loop survives a connection blip instead
    of dying.
    """
    try:
        store.requeue_stale()
    except Exception as exc:
        print(f"[embed_worker] requeue_stale error {exc}", flush=True)
    try:
        job = store.dequeue_job(timeout=5)
        if not job:
            return
        kind, payload = job
        if kind == "embed":
            process_job(payload)
        else:
            seed_id, _, rec_id = payload.partition("|")
            if seed_id and rec_id:
                process_attribution(seed_id, rec_id)
            else:
                print(f"[embed_worker] bad attribution job {payload!r}", flush=True)
    except Exception as exc:
        print(f"[embed_worker] queue error {exc}, retrying in 5s", flush=True)
        time.sleep(5)


def main() -> None:
    print("[embed_worker] watching embed:queue + attr:queue (Ctrl-C to stop)",
          flush=True)
    while True:
        _tick()


if __name__ == "__main__":
    main()
