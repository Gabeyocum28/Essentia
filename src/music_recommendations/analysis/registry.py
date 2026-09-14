"""Where the model files live and the framing parameters they expect.

One generation now: CLAP (audio and text embedding) plus the Beat This!
beat tracker. The Discogs-EffNet graph and its eleven classification heads
used to sit below this block; they were non-commercial, and the cutover
deleted them along with TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

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
