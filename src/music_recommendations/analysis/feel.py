"""The feel vector: eleven classifier heads over one stored EffNet embedding.

`sounds_like` on the embedding alone ranks by STYLE — it will happily answer a
slow, sparse ballad with a loud uptempo take by the same kind of band, because
both occupy the same corner of Discogs space. The eleven heads below are the
missing half: energy, mood and texture, as probabilities in [0, 1].

They are cheap in a way that matters. Every head is a small classifier trained
on the SAME 1280-d penultimate activations the corpus already stores, so a
feel vector needs no audio, no decode and no EffNet pass — just a matmul over
a vector that is already in the database. The whole corpus rescores in one
`session.run` per head (see scripts/feel_backfill.py).

Dimension order is `registry.HEADS` insertion order, re-exported as FEEL_KEYS
so the store, the API and the web math panel all label the same columns.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from . import registry

# TensorFlow logs a wall of INFO on import; silence before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The eleven dimension names, in vector order. Each is also the name of the
# positive class in that head's own metadata (test_feel.py checks it).
FEEL_KEYS: list[str] = list(registry.HEADS)
FEEL_DIM = len(FEEL_KEYS)

# key -> (session, input tensor, output tensor). Built once per process.
_sessions: dict[str, tuple] = {}
_load_lock = threading.Lock()


def _build_sessions() -> dict[str, tuple]:
    """One graph + session per head. Only ever called with `_load_lock` held."""
    import tensorflow as tf

    # Same reasoning as embedding._build_session: the VM container has two
    # cores, and TensorFlow's "as many threads as the machine has" defaults
    # oversubscribe it. Only settable before the runtime initializes, so a
    # process that already built the EffNet session keeps what that set.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    built: dict[str, tuple] = {}
    for key, head in registry.HEADS.items():
        path = head.graph
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python3 scripts/fetch_models.py"
            )
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(path.read_bytes())
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        built[key] = (
            tf.compat.v1.Session(graph=graph),
            graph.get_tensor_by_name(registry.HEAD_INPUT),
            graph.get_tensor_by_name(registry.HEAD_OUTPUT),
        )
    return built


def _load() -> dict[str, tuple]:
    """Double-checked lazy init, like embedding._load: /seed analyzes under a
    semaphore of 2, so two requests can race in here on a cold process and
    would otherwise each build eleven graphs."""
    global _sessions
    if _sessions:
        return _sessions
    with _load_lock:
        if not _sessions:
            _sessions = _build_sessions()
        return _sessions


def feel_vectors(embeddings: np.ndarray) -> np.ndarray:
    """(n, 1280) EffNet means -> (n, 11) probabilities, columns in FEEL_KEYS order.

    One `session.run` per head over ALL rows rather than per track: the heads
    are two small dense layers, so the per-call overhead dominates and a whole
    20k-row corpus costs about as much as a single row would eleven times.
    """
    rows = np.asarray(embeddings, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if len(rows) == 0:
        return np.zeros((0, FEEL_DIM), dtype=np.float32)

    sessions = _load()
    out = np.empty((len(rows), FEEL_DIM), dtype=np.float32)
    for column, (key, head) in enumerate(registry.HEADS.items()):
        session, input_, output = sessions[key]
        probs = np.asarray(session.run(output, {input_: rows}), dtype=np.float32)
        out[:, column] = probs[:, head.positive]
    return out


def feel_vector(embedding: np.ndarray) -> np.ndarray:
    """The one-row case: (1280,) -> (11,)."""
    return feel_vectors(np.asarray(embedding, dtype=np.float32)[None, :])[0]
