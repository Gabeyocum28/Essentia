"""The key two Deezer rows of the SAME recording share.

Deezer re-publishes one recording under many track ids: the 2009 remaster,
the live take, the "(feat. X)" credit, the deluxe-edition bonus disc. They
carry different ids, different previews and near-identical embeddings, so
the corpus accumulates them and a rec list shows the same music three times.

`dedupe_key(title, artist)` folds those editions onto one string by
stripping the edition marker from the title and the guest credits from the
artist. It is pure and cheap: the crawler calls it per candidate, the store
persists it per track (indexed), and the backfill groups on it.

Deliberately conservative in one direction: the key includes the artist, so
a COVER is never collapsed into the original -- only re-releases by the
same primary artist meet. It is deliberately aggressive about parenthetical
groups that name an edition ("(1990 Remaster)", "(Radio Edit)", "(Take 2)",
"- Live at ..."), because those are exactly the strings Deezer varies.
"""
from __future__ import annotations

import re
import unicodedata

# Words that mark an EDITION of a recording rather than a different one.
# Matched as whole tokens anywhere inside a bracketed group or a trailing
# " - ..." segment: "Radio Edit" and "Live at Carnegie Hall" both qualify
# without needing the marker to come first.
_EDITION_WORDS = frozenset({
    "feat", "feats", "featuring", "ft", "with",
    "remaster", "remastered", "remastering", "remasterisé",
    "live", "mono", "stereo", "take", "takes",
    "version", "versions", "edit", "edited", "mix", "mixes",
    "demo", "bonus", "from", "deluxe", "anniversary",
})

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_GROUP = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Everything from the first of these onward is a guest credit or a second
# billed artist, not the primary one.
_ARTIST_CUTS = (" feat", " featuring", " ft.", " ft ", " & ", ", ", " with ")


def _fold(text: str) -> str:
    """Lowercase, strip accents, and collapse every non-alphanumeric run to
    one space. "Björk's  Song!" -> "bjork s song"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", text).strip()


def _is_edition(fragment: str) -> bool:
    """True if this bracketed group / trailing segment names an edition."""
    if _YEAR.search(fragment):
        return True
    return bool(set(_fold(fragment).split()) & _EDITION_WORDS)


def normalize_title(title: str | None) -> str:
    """The title with every edition marker removed."""
    text = (title or "").lower()

    # Bracketed groups first: "(feat. ...)", "[Live]", "(1990 Remaster)".
    text = _GROUP.sub(
        lambda m: "" if _is_edition(m.group(1) or m.group(2) or "") else m.group(0),
        text,
    )

    # Then trailing " - ..." segments, innermost last: Deezer stacks them
    # ("So What - Live - 2001 Remaster"), so peel until one does not match.
    while " - " in text:
        head, _, tail = text.rpartition(" - ")
        if not _is_edition(tail):
            break
        text = head

    return _fold(text)


def normalize_artist(artist: str | None) -> str:
    """The primary artist only: guest credits and co-billing are dropped."""
    text = (artist or "").lower()
    cut = len(text)
    for marker in _ARTIST_CUTS:
        found = text.find(marker)
        if found != -1:
            cut = min(cut, found)
    return _fold(text[:cut])


def dedupe_key(title: str | None, artist: str | None) -> str:
    """`"<normalized artist>|<normalized title>"` -- one string per recording."""
    return f"{normalize_artist(artist)}|{normalize_title(title)}"
