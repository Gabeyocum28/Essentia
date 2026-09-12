"""Where the EffNet graph lives and the framing parameters it expects."""
from __future__ import annotations

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
