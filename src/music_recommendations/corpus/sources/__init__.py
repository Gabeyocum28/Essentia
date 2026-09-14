"""The source registry: which catalogues are switched on, and who owns an id.

    SOURCES=deezer            development (the default)
    SOURCES=jamendo           shippable
    SOURCES=jamendo,deezer    both, jamendo first

Two questions, two functions. `active()` answers "where do new tracks come
from", and drives /search fan-out and the crawler. `for_id()` answers "who
does this id belong to", and drives /preview, /seed and the embed worker --
an id already in the corpus must stay resolvable whether or not its source
is still switched on.

Instances are cached per name: a source may hold rate-limiting state (both
do) and the crawl rotation's label, so handing out a fresh object per call
would quietly reset both.
"""
from __future__ import annotations

import os
import threading

from music_recommendations.corpus.sources.base import (
    DEFAULT_SOURCE, BaseSource, Source, split_id,
)
from music_recommendations.corpus.sources.deezer import DeezerSource
from music_recommendations.corpus.sources.jamendo import JamendoSource

REGISTRY: dict[str, type] = {
    DeezerSource.name: DeezerSource,
    JamendoSource.name: JamendoSource,
}

_instances: dict[str, Source] = {}
_lock = threading.Lock()
# Which disabled sources have already been logged, so a worker that calls
# active() every minute says it once rather than forever.
_warned: set[str] = set()


def get(name: str) -> Source:
    """The one instance of this source. Raises KeyError for an unknown name."""
    with _lock:
        if name not in _instances:
            _instances[name] = REGISTRY[name]()
        return _instances[name]


def names() -> list[str]:
    """The configured source names, in order, from SOURCES."""
    raw = os.environ.get("SOURCES", "") or DEFAULT_SOURCE
    configured = [n.strip() for n in raw.split(",") if n.strip()]
    return configured or [DEFAULT_SOURCE]


def active() -> list[Source]:
    """Every switched-on source, in SOURCES order.

    An unknown name raises: a typo in SOURCES silently serving a smaller
    catalogue is worse than refusing to start. A source that is configured
    but unusable (Jamendo with no client id) is dropped with a log line --
    unless it is the only one, where dropping it would leave the app with
    no catalogue at all and "started fine" would be a lie.
    """
    wanted = names()
    unknown = [n for n in wanted if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown source(s) in SOURCES: {', '.join(unknown)}; "
            f"known: {', '.join(sorted(REGISTRY))}"
        )
    out: list[Source] = []
    for name in wanted:
        source = get(name)
        if not _enabled(source):
            reason = getattr(source, "disabled_reason", lambda: "unavailable")()
            if len(wanted) == 1:
                raise RuntimeError(f"source {name!r} is the only one configured "
                                   f"but is unavailable: {reason}")
            if name not in _warned:
                _warned.add(name)
                print(f"[sources] {name} disabled: {reason}", flush=True)
            continue
        out.append(source)
    return out


def for_id(track_id: str) -> Source | None:
    """The source that owns this id, switched on or not, or None.

    Bare ids are Deezer's (see base.py). A namespaced id whose source is not
    in the registry gets None, which callers turn into a 404 rather than
    guessing.
    """
    name, _local = split_id(str(track_id))
    if name not in REGISTRY:
        return None
    return get(name)


def _enabled(source: Source) -> bool:
    return bool(getattr(source, "enabled", lambda: True)())


def reset() -> None:
    """Drop the instance cache (tests, and after an env change)."""
    with _lock:
        _instances.clear()
        _warned.clear()


__all__ = ["REGISTRY", "BaseSource", "DeezerSource", "JamendoSource", "Source",
           "active", "for_id", "get", "names", "reset", "split_id"]
