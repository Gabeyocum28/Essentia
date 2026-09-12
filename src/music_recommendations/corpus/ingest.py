"""Batch loop: download preview -> analyze_track -> write the store.

Same analysis function the server calls on demand; different caller
(spec §7). Skips tracks already in the store; one bad track never kills a run.

Downloads and analysis are bounded by different things, so they run
differently: downloads are I/O and go on threads, analysis is a TensorFlow
forward pass and goes on processes (analysis.batch.analyze_many). Working in
batches keeps both busy without holding thousands of embeddings in memory at
once -- 10k of them is ~100 MB of Python floats before anything is written.

The two stages overlap: batch N+1 is already downloading while batch N is
being analyzed. Measured on a 10-core run, downloading a batch and analyzing
it strictly in turn left the CPU idle for most of the wall clock, because a
preview fetch is network latency and nothing else.

Each mp3 leaves the working cache the moment its features reach the store. The
features ARE the product; the audio is scratch, and at ~225 KB a preview a
corpus of tens of thousands would fill the disk for nothing.

Set ARCHIVE_DIR to keep them anyway -- point it at an external disk and the
preview is MOVED there instead of deleted, which turns a FEATURES_VERSION bump
from a re-download of the whole corpus at Deezer's rate limit into a local,
CPU-only re-analysis. Downloads still land on the internal disk first, so an
archive that is full, slow, or unplugged costs the archive, never the run.
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from music_recommendations.analysis import FEATURES_VERSION
from music_recommendations.analysis.batch import analyze_many
from music_recommendations.corpus.download import AUDIO_CACHE, download_preview
from music_recommendations.server import store

ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR")

BATCH = 100
# Previews are ~225 KB each and most need a fresh signed URL first, so this is
# latency-bound, not bandwidth-bound. deezer._get holds the politeness delay
# for all of them, which is the real ceiling on how wide this can usefully go.
DOWNLOAD_THREADS = 24


# Stored beside the feature vectors rather than wrapping them: the server
# indexes the features dict directly (app.py `_vector`), so a nested
# {"v": ..., "f": ...} envelope would mean changing the server to read what
# this lane writes. An extra key it never looks up costs nothing.
VERSION_KEY = "_features_version"


def already_stored(track_id: str) -> bool:
    """True if this track is stored AND was analyzed by the current version.

    The version check is what makes changing a feature safe: when
    FEATURES_VERSION is bumped, stale vectors are re-analyzed instead of
    silently ranking against fresh ones in the same corpus.
    """
    try:
        stored = store.get_features(track_id)
    except Exception:
        return False
    return bool(stored) and stored.get(VERSION_KEY) == FEATURES_VERSION


def _one_download(track: dict) -> tuple[dict, Path | None]:
    """Runs on a download thread. Never raises: a dead preview is expected."""
    try:
        return track, download_preview(track)
    except Exception:  # noqa: BLE001 - an expired or missing preview, skip it
        return track, None


def _start_downloads(pool: ThreadPoolExecutor, tracks: list[dict]) -> list[Future]:
    """Kick off a batch of fetches and return without waiting for them."""
    return [pool.submit(_one_download, t) for t in tracks]


def _collect(futures: list[Future]) -> list[tuple[dict, Path]]:
    """Wait on a started batch; drop the tracks with no working audio."""
    results = (f.result() for f in futures)
    return [(t, p) for t, p in results if p is not None]


def _discard(path) -> None:
    """Retire an analyzed preview: archived if ARCHIVE_DIR is set, else deleted.

    Never raises. A missing file is fine -- the point is that it is no longer
    taking up room in the working cache -- and an archive that cannot be
    written falls back to deleting rather than stalling the run.
    """
    path = Path(path)
    if ARCHIVE_DIR:
        try:
            target = Path(ARCHIVE_DIR)
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            return
        except (OSError, shutil.Error):
            pass  # unplugged, full, or read-only: fall through and delete
    try:
        path.unlink()
    except OSError:
        pass


def ingest(tracks: list[dict], limit: int = 300, workers: int | None = None,
           progress: bool = True) -> int:
    """Download, analyze, and store up to limit tracks; returns count ingested."""
    pending = []
    for track in tracks:
        if already_stored(track["track_id"]):
            # Left over from a run that died between analysis and cleanup.
            _discard(AUDIO_CACHE / f"{track['track_id']}.mp3")
        else:
            pending.append(track)
    pending = pending[:limit]
    if not pending:
        return 0

    batches = [pending[i:i + BATCH] for i in range(0, len(pending), BATCH)]
    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        inflight = _start_downloads(pool, batches[0])
        for index, _batch in enumerate(batches):
            # Start the next batch's fetches before blocking on this one's, so
            # the download threads keep working through the analysis below.
            current, inflight = inflight, (
                _start_downloads(pool, batches[index + 1])
                if index + 1 < len(batches) else []
            )
            by_path = {str(path): track for track, path in _collect(current)}

            for path, features, error in analyze_many(list(by_path), workers=workers):
                track = by_path[path]
                _discard(path)
                if error:
                    if progress:
                        print(f"  skip {track['track_id']}: {error}")
                    continue
                try:
                    store.put_track(track, {**features, VERSION_KEY: FEATURES_VERSION})
                    done += 1
                except Exception as exc:  # noqa: BLE001 - a store blip is not fatal
                    if progress:
                        print(f"  store failed {track['track_id']}: {exc}")
            if progress:
                print(f"  {done}/{len(pending)} stored")
    return done
