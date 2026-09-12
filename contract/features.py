"""The shared shapes: Track fields, axis list, and what analyze_track returns.

This file is data, not logic. Changing it is cross-cutting (see CLAUDE.md).
"""
from __future__ import annotations

# Every Track object at every endpoint has exactly these keys.
# "score" additionally appears on recommendation results only.
TRACK_FIELDS = frozenset(
    {"track_id", "title", "artist", "album", "artwork_url", "preview_url"}
)

# GET /axes returns exactly this, in this order.
AXES = [
    {"id": "sounds_like", "label": "More sounds like this"},
    {"id": "surprise",    "label": "Nothing like this"},
]

# analyze_track(mp3_path) -> dict with exactly these keys.
FEATURE_KEYS = {
    "embedding": 1280,   # Discogs-EffNet penultimate layer, patch-mean
}
# Arrays are 1-D float lists/ndarrays of the stated length.
