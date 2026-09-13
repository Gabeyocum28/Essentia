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

    # Two cores is what the VM container has. Left at TensorFlow's defaults
    # (0/0 == "as many as there are cores") the intra-op pool and the
    # inter-op pool each size themselves to the whole machine and then
    # oversubscribe it, so a single 64-patch batch spends measurable time in
    # scheduling. One inter-op thread (the graph is one long dependency
    # chain, so there is nothing to run in parallel at that level) and two
    # intra-op threads match the hardware. Only settable before the runtime
    # initializes; a process that already ran a graph keeps what it has.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

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


def _run_batch(chunk: np.ndarray) -> np.ndarray:
    """One inference pass over at most BATCH_SIZE patches -> (len(chunk), 1280).

    The graph's input is frozen at 64, so a short chunk is zero-padded up to
    64 and the padding rows are dropped from the result. Padding rows cost
    exactly as much as real ones, which is why callers should hand this as
    full a chunk as they can.
    """
    real = len(chunk)
    bs = registry.BATCH_SIZE
    if real < bs:
        chunk = np.concatenate(
            [chunk, np.zeros((bs - real,) + chunk.shape[1:], np.float32)]
        )
    result = _session.run(_output, {_input: chunk})
    return np.asarray(result, dtype=np.float32)[:real]


def embed_patches(patches: np.ndarray) -> np.ndarray:
    """(n, 128, 96) mel patches -> (n, 1280) penultimate-layer activations."""
    n = len(patches)
    if n == 0:
        return np.zeros((0, 1280), dtype=np.float32)
    _load()
    bs = registry.BATCH_SIZE
    out = [_run_batch(patches[start:start + bs]) for start in range(0, n, bs)]
    return np.concatenate(out, axis=0)


def embed_patch_groups(groups: list[np.ndarray]) -> list[np.ndarray]:
    """Embed several tracks' patches at once, packed into full batches.

    One 30 s preview yields 28 patches, so embedding it alone wastes 56% of
    every 64-patch batch on zero padding that costs full price. Concatenating
    three tracks fills the batch instead and then splits the rows back out.

    Each group is independent inside the graph (there is no cross-patch
    mixing in EffNet), so `embed_patch_groups([a, b])` returns exactly what
    `[embed_patches(a), embed_patches(b)]` would, up to float noise from a
    different batch composition.
    """
    if not groups:
        return []
    sizes = [len(g) for g in groups]
    total = sum(sizes)
    if total == 0:
        return [np.zeros((0, 1280), dtype=np.float32) for _ in groups]
    _load()
    flat = np.concatenate([np.asarray(g, dtype=np.float32)
                           for g in groups if len(g)], axis=0)
    bs = registry.BATCH_SIZE
    rows = np.concatenate(
        [_run_batch(flat[start:start + bs]) for start in range(0, total, bs)],
        axis=0,
    )
    out = []
    at = 0
    for size in sizes:
        out.append(rows[at:at + size])
        at += size
    return out


def load_audio(mp3_path: Path | str) -> np.ndarray:
    """Decode to mono 16 kHz — the only rate EffNet accepts."""
    return frontend.decode(mp3_path)


def effnet_frames(mp3_path: Path | str) -> np.ndarray:
    """The one slow pass. Everything downstream reuses its output."""
    mel = frontend.mel_frames(load_audio(mp3_path))
    return embed_patches(frontend.patches(mel))
