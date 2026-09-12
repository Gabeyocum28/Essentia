"""Audio -> Discogs-EffNet per-patch embeddings (n, 1280), via TensorFlow.

The frozen graph is the same file Essentia's TensorflowPredictEffnetDiscogs
loads; we feed it ourselves because Essentia does not ship on Linux ARM.
The graph was frozen with a fixed batch of 64 patches, so inputs are
zero-padded up to a multiple of 64 and the padding rows are dropped again
(Essentia's lastBatchMode="same").
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from . import frontend, registry

SAMPLE_RATE = frontend.SAMPLE_RATE

# TensorFlow logs a wall of INFO on import; silence before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_session = None
_input = None
_output = None
_load_lock = threading.Lock()


def _build_session():
    """Build the frozen graph and a session for it. Only ever called with
    `_load_lock` held."""
    import tensorflow as tf

    path = registry.MODELS_DIR / registry.EFFNET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python scripts/fetch_models.py"
        )
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(path.read_bytes())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    input_ = graph.get_tensor_by_name(registry.EFFNET_INPUT + ":0")
    output = graph.get_tensor_by_name(registry.EFFNET_OUTPUT)
    session = tf.compat.v1.Session(graph=graph)
    return session, input_, output


def _load():
    """Double-checked lazy init: /seed runs analyze_track under a semaphore
    of 2, so two requests can race in here on a cold process. Without the
    lock both would build a full TF graph + session concurrently (wasted
    work, and a `_session`/`_input`/`_output` triple that could end up mixed
    between the two builds)."""
    global _session, _input, _output
    if _session is not None:
        return
    with _load_lock:
        if _session is not None:
            return
        _session, _input, _output = _build_session()


def embed_patches(patches: np.ndarray) -> np.ndarray:
    """(n, 128, 96) mel patches -> (n, 1280) penultimate-layer activations."""
    n = len(patches)
    if n == 0:
        return np.zeros((0, 1280), dtype=np.float32)
    _load()
    bs = registry.BATCH_SIZE
    out = []
    for start in range(0, n, bs):
        chunk = patches[start:start + bs]
        real = len(chunk)
        if real < bs:
            chunk = np.concatenate(
                [chunk, np.zeros((bs - real,) + chunk.shape[1:], np.float32)]
            )
        result = _session.run(_output, {_input: chunk})
        out.append(np.asarray(result, dtype=np.float32)[:real])
    return np.concatenate(out, axis=0)


def load_audio(mp3_path: Path | str) -> np.ndarray:
    """Decode to mono 16 kHz — the only rate EffNet accepts."""
    return frontend.decode(mp3_path)


def effnet_frames(mp3_path: Path | str) -> np.ndarray:
    """The one slow pass. Everything downstream reuses its output."""
    mel = frontend.mel_frames(load_audio(mp3_path))
    return embed_patches(frontend.patches(mel))
