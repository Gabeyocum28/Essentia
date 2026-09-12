"""Facts about the feature vectors that callers need but cannot infer.

Deliberately free of essentia and numpy imports: a caller checking whether
its cached vectors are stale should not have to load TensorFlow to find out.

Two things live here, both owned by this lane because both depend on how the
vectors are *built*, which is exactly what this lane owns:

  FEATURES_VERSION -- has the meaning of the numbers changed?
  METRICS          -- how should two of these vectors be compared?
"""
from __future__ import annotations

# Bump whenever the CONTENTS of any feature vector change in a way that makes
# previously stored vectors incomparable with new ones -- a different
# normalization constant, a reordered dimension, a new extractor. Do NOT bump
# for changes that leave the numbers alone (docstrings, refactors).
#
# The shape of the dict is pinned by contract/features.py FEATURE_KEYS and
# changing that needs all four people; this version covers everything the
# contract does not see. Callers that persist features (server/store.py,
# corpus/ingest.py) should store it alongside and re-analyze on mismatch --
# otherwise old and new vectors sit in the same corpus and rank against each
# other silently, which looks like "the recommendations got worse" and is
# almost impossible to trace back. The pre-spec MVP hit this and solved it the
# same way (legacy/mvp/analyzer.py CACHE_VERSION).
# 3: Essentia replaced by ffmpeg + numpy mel + TensorFlow (Linux ARM has no
#    Essentia wheels). Same model, same maths, but not bit-identical: cosine
#    with v2 vectors is >0.99, yet a corpus must not mix the two. Genre head
#    and groove dropped.
# 2: groove stopped folding tempo into an octave and its four dimensions were
#    rescaled to their measured ranges so each contributes comparably. Vectors
#    written under v1 are NOT comparable with v2 and must be re-analyzed.
# 1: initial.
FEATURES_VERSION = 3

# Which distance is correct for each feature key.
#
# This is not a free choice per caller: it follows from how each vector is
# constructed, so it belongs next to the construction rather than in the
# server's axis table.
#
#   embedding -- 1280-d EffNet activations, direction carries the meaning and
#                magnitude mostly reflects loudness. Cosine.
METRICS = {
    "embedding": "cosine",
}

# A note for whoever implements ranking, not a value to import:
# `surprise` (most-distant by embedding) concentrates badly on a raw argmin --
# in 1280-d some tracks are far from *everything* and win for every seed. On
# the 150-track pool one track was the top answer for 22% of seeds. Subtracting
# each track's mean similarity to the corpus fixes it (80 -> 113 distinct
# tracks returned, worst offender 22% -> 11%) and costs one precomputed float
# per track. See spec §4, which names this risk and suggests sampling the far
# tail; the centrality correction is the deterministic version.
