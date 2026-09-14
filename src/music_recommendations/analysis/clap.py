"""Microsoft CLAP: one audio embedding and one text embedding, same space.

CLAP (Contrastive Language-Audio Pretraining, MIT licence) replaces the
Discogs-EffNet embedding. Two things it gives us that EffNet did not:

  * a 1024-d audio vector under a licence we can sell against, and
  * a *text* tower into the same space, so a mood axis is a pair of
    sentences rather than a trained classifier head (see feel_v2.py), and
    "/search/text" is a cosine against a typed phrase.

The msclap wrapper wants file paths and decodes them with torchaudio /
torchcodec. We do not use that path: the caller has already decoded the
preview once (v2.py) and decoding it again -- with a different resampler,
in a library we would then have to ship -- costs a second per track for
nothing. `embed_audio` therefore takes waveforms and calls the wrapper's
tensor entry point (`_get_audio_embeddings`) directly.

Weights come from models/v2/ (scripts/fetch_models.py puts them there), NOT
from the wrapper's own huggingface download: a container must not need the
network on first analysis.
"""
from __future__ import annotations

import threading

import numpy as np

from . import registry

__all__ = ["SAMPLE_RATE", "DIM", "load", "embed_audio", "embed_text"]

# The 2023 config: 44.1 kHz, 7-second training window, 1024-d projection.
SAMPLE_RATE = 44100
WINDOW_SECONDS = 7
DIM = 1024

# A 30 s preview covers 4+ disjoint 7 s windows. Three, evenly spaced across
# the clip, is the compromise: the embedding stops depending on which part of
# the track the random crop happened to land on (the wrapper's own loader
# crops at random, which makes analysis non-reproducible), and three windows
# batch into one forward pass, so the cost is well under 3x one window.
MAX_WINDOWS = 3

# CLAP is the only heavy thing in the worker and the box has two cores.
TORCH_THREADS = 2

_model = None
_lock = threading.Lock()


def load():
    """The CLAPWrapper, loaded once per process (~3 s, ~700 MB of weights).

    Guarded by a lock because the server may call embed_text from a request
    thread while the worker is mid-analysis.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            import torch
            from msclap import CLAP

            torch.set_num_threads(TORCH_THREADS)
            weights = registry.CLAP_WEIGHTS
            if not weights.exists():
                raise FileNotFoundError(
                    f"{weights} missing -- run scripts/fetch_models.py"
                )
            _model = CLAP(model_fp=str(weights), version="2023", use_cuda=False)
    return _model


def _windows(waveform: np.ndarray, sr: int) -> list[np.ndarray]:
    """Up to MAX_WINDOWS windows of WINDOW_SECONDS, evenly spaced.

    Shorter than one window: tile it. The model was trained on 7 s of audio
    and zero-padding a 2 s clip would tell it "this track is mostly silence",
    which is a statement about our framing, not about the track.
    """
    need = int(WINDOW_SECONDS * sr)
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("empty waveform")
    if y.size < need:
        reps = int(np.ceil(need / y.size))
        return [np.tile(y, reps)[:need]]
    n = min(MAX_WINDOWS, int(y.size // need))
    starts = np.linspace(0, y.size - need, n).astype(int)
    return [y[s:s + need] for s in starts]


def embed_audio(waveforms: list[np.ndarray], sr: int = SAMPLE_RATE) -> np.ndarray:
    """(n, 1024) L2-normalized audio embeddings, one row per waveform.

    Waveforms must already be mono float at `sr`; `sr` must be CLAP's rate
    (resampling belongs with the decode, which happens once).

    All windows of all tracks go through the encoder as one batch: on two
    threads the per-window cost is dominated by the transformer, and batching
    a group of previews is most of why analyze_tracks takes a list.
    """
    if sr != SAMPLE_RATE:
        raise ValueError(f"CLAP wants {SAMPLE_RATE} Hz, got {sr}")
    if not waveforms:
        return np.zeros((0, DIM), dtype=np.float32)

    import torch

    model = load()
    groups = [_windows(w, sr) for w in waveforms]
    flat = [w for group in groups for w in group]
    batch = torch.from_numpy(np.stack(flat)).unsqueeze(1)  # (m, 1, samples)
    with torch.no_grad():
        out = model._get_audio_embeddings(batch).cpu().numpy().astype(np.float32)

    rows = []
    at = 0
    for group in groups:
        rows.append(out[at:at + len(group)].mean(axis=0))
        at += len(group)
    return _normalize(np.stack(rows))


def embed_text(prompts: list[str]) -> np.ndarray:
    """(k, 1024) L2-normalized text embeddings, one row per prompt."""
    if not prompts:
        return np.zeros((0, DIM), dtype=np.float32)
    import torch

    model = load()
    with torch.no_grad():
        out = model.get_text_embeddings(list(prompts))
    return _normalize(out.cpu().numpy().astype(np.float32))


def _normalize(x: np.ndarray) -> np.ndarray:
    """Rows to unit length, so every later comparison is a plain dot product.

    Normalizing here rather than at each call site means the stored vector is
    already the one the ranking wants, and int8 quantization (quantize.py)
    sees a fixed range instead of one that drifts with loudness.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)
