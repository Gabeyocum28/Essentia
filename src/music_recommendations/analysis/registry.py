"""One table of classification heads: file, size, verified node names.

Node names are NOT uniform across heads (spec §2.1) — every head added
here must have its names checked against the actual graph, not assumed.
fetch_models.py and heads.py both read this table; adding an axis that
rides on the embedding is a one-row diff here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

EFFNET_FILE = "discogs-effnet-bs64-1.pb"
EFFNET_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    + EFFNET_FILE
)
EFFNET_OUTPUT = "PartitionedCall:1"  # penultimate layer -> (n_frames, 1280)
EFFNET_INPUT = "serving_default_melspectrogram"

# Framing inside Essentia's TensorflowPredictEffnetDiscogs (its defaults).
PATCH_SIZE = 128   # mel frames per inference
PATCH_HOP = 62     # frames between patch starts (~1 prediction per second)
BATCH_SIZE = 64    # the graph was frozen with a fixed batch of 64 patches


@dataclass(frozen=True)
class Head:
    filename: str      # e.g. "genre_discogs400-discogs-effnet-1.pb"
    n_out: int
    input_node: str    # verified per head; TensorflowPredict2D default lies
    output_node: str


# Empty in v1. The three shipping axes — sounds_like, groove, surprise — need
# only the embedding and the rhythm DSP, so no classification head runs.
#
# The table stays because it is the extension point: re-adding an axis that
# rides on the embedding is one row here plus one FEATURE_KEYS entry, and no
# other lane changes. Node names must be verified against the actual graph for
# every head — the TensorflowPredict2D defaults do not hold (spec §2.1). These
# two were verified and are kept here so re-adding them costs nothing:
#
#   "genre":     Head("genre_discogs400-discogs-effnet-1.pb", 400,
#                     "serving_default_model_Placeholder", "PartitionedCall:0"),
#   "moodtheme": Head("mtg_jamendo_moodtheme-discogs-effnet-1.pb", 56,
#                     "model/Placeholder", "model/Sigmoid"),
#
# The four binary mood heads used "model/Placeholder" -> "model/Softmax", and
# disagree on which class index is positive; heads.py resolves that per head
# from the model's own JSON rather than assuming.
HEADS: dict[str, Head] = {
    "genre": Head("genre_discogs400-discogs-effnet-1.pb", 400,
                  "serving_default_model_Placeholder", "PartitionedCall:0"),
}


def model_url(filename: str) -> str:
    family = filename.split("-")[0]
    return f"https://essentia.upf.edu/models/classification-heads/{family}/{filename}"
