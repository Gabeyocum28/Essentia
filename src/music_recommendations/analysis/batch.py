"""Analyze many files at once, across processes.

analyze_track handles one file and is the contract this lane owes everyone
else. This adds throughput on top of it, which matters once a corpus is
thousands of tracks rather than hundreds.

Processes, not threads: the work is a torch forward pass plus DSP, both
CPU-bound and holding the GIL, so threads buy nothing. Each worker loads its
own copy of CLAP (the module-level singleton in clap.py is per-process), so
there is a fixed startup cost of several seconds and ~700 MB per worker --
irrelevant across thousands of files, dominant across ten, and the reason
`default_workers()` is not simply the core count on a small box.

Callers stay in charge of WHEN analysis happens (spec §7). This is only about
how fast it goes once they have decided to.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator


def default_workers() -> int:
    """Leave a core free so the machine stays usable during a long run."""
    return max(1, (os.cpu_count() or 2) - 1)


def _one(path: str) -> tuple[str, dict | None, str | None]:
    """Runs in the worker. Never raises: one bad file must not kill a batch."""
    from music_recommendations.analysis import analyze_track, as_json

    try:
        return path, as_json(analyze_track(path)), None
    except Exception as exc:  # noqa: BLE001 - a corrupt mp3 is expected, not fatal
        return path, None, f"{type(exc).__name__}: {exc}"


def analyze_many(
    paths: Iterable[Path | str],
    workers: int | None = None,
    on_result: Callable[[str, dict | None, str | None], None] | None = None,
) -> Iterator[tuple[str, dict | None, str | None]]:
    """Yield (path, features_as_json, error) per file, as each finishes.

    Results arrive out of order. `features` is None exactly when `error` is
    set, so a caller can persist the good ones and log the rest:

        for path, features, error in analyze_many(paths):
            if features:
                store.put(path, features)

    Yielding rather than returning a dict matters at corpus scale: 10k
    embeddings held at once is ~100 MB of Python floats before anything is
    written anywhere.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return
    workers = workers or default_workers()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, p): p for p in paths}
        for future in as_completed(futures):
            path, features, error = future.result()
            if on_result is not None:
                on_result(path, features, error)
            yield path, features, error
