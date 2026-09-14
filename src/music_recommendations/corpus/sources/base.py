"""What a music source has to be able to do.

A source is everything the app needs to know about one catalogue: how to
search it, how to fetch one track, how to get a currently-playable URL for
it, how to hand the crawler a bounded slice of new candidates, and what
attribution (if any) the licence obliges us to show.

Track dicts are the contract shape (contract/features.TRACK_FIELDS) plus two
optional keys a source may add:

  source        the source's name, e.g. "jamendo"
  attribution   {"source", "url", "license"} -- what the clients must show

`attribution` is the internal, full form; only its "url" reaches a client
(as `attribution_url`, see server/store._contract). Deezer is a development
source: it returns no attribution, so nothing is rendered for it.

Track ids are namespaced `"<source>:<id>"`, with ONE exception: Deezer ids
stay bare digits. Deezer shipped first and its ids are already in Atlas, in
the fixture, and on phones; renaming them would have been a data migration
for no gain, so "no prefix" simply means Deezer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

# The contract Track plus the two optional keys above. Sources produce these.
Track = dict


@runtime_checkable
class Source(Protocol):
    """One catalogue. Implementations live beside this file."""

    name: str

    def owns(self, track_id: str) -> bool:
        """Is this id one of ours?"""

    def search(self, q: str, limit: int = 25) -> list[Track]:
        """Free-text search, most relevant first."""

    def track(self, track_id: str) -> Track | None:
        """One track by id, or None if it is gone or unplayable."""

    def preview_url(self, track_id: str) -> str | None:
        """A URL that plays right now (some sources sign and expire these)."""

    def candidates(self, step: int) -> list[Track]:
        """One bounded slice of the catalogue for the crawler.

        `step` is a monotonically increasing cursor owned by the caller; the
        source decides what rotation it drives. Bounded on purpose: the
        worker runs this inline between jobs.
        """

    def candidate_label(self) -> str:
        """A human label for the slice the last `candidates` call returned."""

    def attribution(self, track: Track) -> dict | None:
        """{"source", "url", "license"} for this track, or None if the
        licence asks for nothing."""


def split_id(track_id: str) -> tuple[str, str]:
    """`"jamendo:42"` -> `("jamendo", "42")`; `"42"` -> `("deezer", "42")`."""
    name, sep, local = str(track_id).partition(":")
    if not sep or not name:
        return DEFAULT_SOURCE, str(track_id)
    return name, local


DEFAULT_SOURCE = "deezer"


class BaseSource:
    """Shared id and label bookkeeping. Sources subclass this for free."""

    name = ""
    prefixed = True          # False for Deezer, whose ids stay bare

    def __init__(self) -> None:
        self._label = self.name

    # ---- ids ----

    def qualify(self, local_id: str | int) -> str:
        return f"{self.name}:{local_id}" if self.prefixed else str(local_id)

    def local_id(self, track_id: str) -> str:
        name, local = split_id(track_id)
        return local if name == self.name else str(track_id)

    def owns(self, track_id: str) -> bool:
        return split_id(track_id)[0] == self.name

    # ---- crawl labelling ----

    def candidate_label(self) -> str:
        return self._label

    # ---- availability ----

    def enabled(self) -> bool:
        """False when the source is configured out (e.g. no API key).

        Not part of the Protocol: a source that is always available simply
        inherits this. `sources.active()` consults it.
        """
        return True

    def disabled_reason(self) -> str:
        return "unavailable"
