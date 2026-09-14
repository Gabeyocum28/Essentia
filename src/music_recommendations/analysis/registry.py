"""Where the model files live and the framing parameters they expect.

Two generations sit here side by side. The v2 block (CLAP + Beat This!) is
what analysis/v2.py uses; the EffNet block below it is v1, kept until the
cutover task deletes it, and nothing in v2 reads it.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


# ---- v2: CLAP audio/text embedding + Beat This! beat tracker ----------------
#
# Both are downloaded by scripts/fetch_models.py and baked into the image.
# msclap and beat_this will each happily fetch their own weights at first use
# (huggingface_hub and torch.hub respectively); we point them at local files
# instead so that a container starts analyzing without network access and so
# that the version in the image is the version we tested.

V2_DIR = MODELS_DIR / "v2"

CLAP_FILE = "CLAP_weights_2023.pth"
CLAP_URL = f"https://huggingface.co/microsoft/msclap/resolve/main/{CLAP_FILE}"
CLAP_WEIGHTS = V2_DIR / CLAP_FILE

# "final0" is the Beat This! checkpoint trained on everything but GTZAN with
# seed 0 -- the package's own default. The name resolves through the authors'
# WebDAV share (beat_this.inference.CHECKPOINT_URL), which is the only
# published location; we copy the file rather than let torch.hub cache it.
BEAT_THIS_FILE = "beat_this-final0.ckpt"
BEAT_THIS_URL = (
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
)
BEAT_THIS_CKPT = V2_DIR / BEAT_THIS_FILE


# ---- v1: Discogs-EffNet and the eleven classification heads -----------------

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
