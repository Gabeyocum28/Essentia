"""Where the EffNet graph lives and the framing parameters it expects."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

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


# ---- classification heads: eleven small graphs on top of the EffNet vector ----
#
# Each head is a two-layer classifier trained on the SAME 1280-d penultimate
# activations EffNet already produces, so they run on a stored embedding and
# never need the audio back. Input `model/Placeholder:0` (None, 1280), output
# `model/Softmax:0` (None, 2).
#
# `positive` is the column of the softmax we keep. It is written out per head
# rather than inferred from the class names because the rule is not uniform:
# most heads name the positive class first and the negation second
# ("danceable"/"not_danceable"), but `mood_sad`, `mood_relaxed`, `mood_party`
# and `tonal_atonal` put it second, and `timbre` (bright/dark) and
# `voice_instrumental` (instrumental/voice) are two-sided pairs where neither
# name is a negation. tests/analysis/test_feel.py cross-checks every entry
# here against `classes[positive]` in the head's own JSON.

HEADS_DIR = MODELS_DIR / "heads"
HEADS_URL_BASE = "https://essentia.upf.edu/models/classification-heads/"
HEAD_INPUT = "model/Placeholder:0"
HEAD_OUTPUT = "model/Softmax:0"
HEAD_SUFFIX = "-discogs-effnet-1"


class Head(NamedTuple):
    stem: str        # file name without extension, under HEADS_DIR
    positive: int    # which softmax column is the probability we keep

    @property
    def graph(self) -> Path:
        return HEADS_DIR / (self.stem + ".pb")

    @property
    def metadata(self) -> Path:
        return HEADS_DIR / (self.stem + ".json")

    @property
    def urls(self) -> tuple[str, str]:
        """(graph, metadata) upstream URLs. The family directory is the stem
        without the `-discogs-effnet-1` suffix."""
        family = self.stem.removesuffix(HEAD_SUFFIX)
        base = f"{HEADS_URL_BASE}{family}/{self.stem}"
        return base + ".pb", base + ".json"


# Insertion order IS the feel-vector dimension order (feel.FEEL_KEYS).
# The key is also the name of the positive class in the head's JSON.
HEADS: dict[str, Head] = {
    "danceable":    Head("danceability" + HEAD_SUFFIX, 0),
    "happy":        Head("mood_happy" + HEAD_SUFFIX, 0),
    "sad":          Head("mood_sad" + HEAD_SUFFIX, 1),
    "aggressive":   Head("mood_aggressive" + HEAD_SUFFIX, 0),
    "relaxed":      Head("mood_relaxed" + HEAD_SUFFIX, 1),
    "party":        Head("mood_party" + HEAD_SUFFIX, 1),
    "acoustic":     Head("mood_acoustic" + HEAD_SUFFIX, 0),
    "electronic":   Head("mood_electronic" + HEAD_SUFFIX, 0),
    "bright":       Head("timbre" + HEAD_SUFFIX, 0),
    "tonal":        Head("tonal_atonal" + HEAD_SUFFIX, 1),
    "instrumental": Head("voice_instrumental" + HEAD_SUFFIX, 0),
}
