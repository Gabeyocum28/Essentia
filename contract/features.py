"""The shared shapes: Track fields, axis list, and what analyze_track returns.

This file is data, not logic. Changing it is cross-cutting (see CLAUDE.md).
"""
from __future__ import annotations

# Every Track object at every endpoint has exactly these keys.
# "score" additionally appears on recommendation results only.
TRACK_FIELDS = frozenset(
    {"track_id", "title", "artist", "album", "artwork_url", "preview_url"}
)

# Keys a Track MAY additionally carry, and only when the value exists:
#   source           which catalogue it came from ("deezer", "jamendo", ...)
#   attribution_url  the backlink a Creative Commons licence obliges the
#                    client to show; absent for sources that require none
# Absent, not null, when there is nothing to say -- so a client can test for
# the key. Clients must tolerate both keys being missing.
TRACK_OPTIONAL_FIELDS = frozenset({"source", "attribution_url"})

# GET /axes returns exactly this, in this order.
AXES = [
    {"id": "sounds_like", "label": "More sounds like this"},
    {"id": "surprise",    "label": "Nothing like this"},
]

# analyze_track(mp3_path) -> dict with exactly these keys, plus "rhythm"
# (below) and "_features_version".
FEATURE_KEYS = {
    "embedding": 1024,   # Microsoft CLAP audio tower, L2-normalized
    "feel": 8,           # eight zero-shot contrastive axes on that embedding,
                         # each a probability in [0, 1]; see
                         # analysis/feel.FEEL_KEYS for the dimension order
}
# Arrays are 1-D float lists/ndarrays of the stated length.

# analyze_track(...)["rhythm"] -> dict with exactly these keys. Unlike the two
# vectors above these are named, human-readable quantities: the app can show
# "92 BPM · F minor · -9.4 LUFS" and the ranking can prefer a similar tempo.
#   tempo_bpm      float, 0.0 when no beat could be found
#   beat_strength  float in [0, 1], confidence that those beats are real
#   loudness_lufs  float, integrated BS.1770 loudness, floored at -70
#   loudness_range float in LU, 0.0 when undefined
#   key            int 0-11, 0 = C
#   mode           "major" | "minor"
#   key_strength   float in [0, 1]; 0 means the key field means nothing
RHYTHM_KEYS = (
    "tempo_bpm", "beat_strength", "loudness_lufs", "loudness_range",
    "key", "mode", "key_strength",
)
