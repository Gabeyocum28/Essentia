"""int8 storage form for embeddings: 1 byte per dimension plus one scale.

Ranking is cosine, so direction is what matters and a per-vector scale
loses nothing that changes an ordering. 1024 floats -> 1024 bytes + 1 float.
"""
from __future__ import annotations

import numpy as np


def to_int8(vec: np.ndarray) -> tuple[bytes, float]:
    vec = np.asarray(vec, dtype=np.float32)
    peak = float(np.abs(vec).max()) if vec.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.rint(vec / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def from_int8(data: bytes, scale: float) -> np.ndarray:
    return (np.frombuffer(data, dtype=np.int8).astype(np.float32) * np.float32(scale))
