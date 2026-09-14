from __future__ import annotations

import numpy as np

from music_recommendations.analysis import analyze_track, quantize
from tests.analysis.conftest import needs_v2


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_round_trip_size_and_similarity():
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(1024).astype(np.float32) * 3.0
    data, scale = quantize.to_int8(vec)
    assert isinstance(data, bytes) and len(data) == 1024
    assert isinstance(scale, float) and scale > 0
    back = quantize.from_int8(data, scale)
    assert back.dtype == np.float32 and back.shape == (1024,)
    assert _cos(vec, back) > 0.999


def test_extremes_survive():
    vec = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    back = quantize.from_int8(*quantize.to_int8(vec))
    assert np.allclose(back, vec, atol=0.05)


def test_zero_vector_does_not_divide_by_zero():
    data, scale = quantize.to_int8(np.zeros(4, np.float32))
    assert np.array_equal(quantize.from_int8(data, scale), np.zeros(4, np.float32))


@needs_v2
def test_round_trip_on_real_embedding(tone_wav):
    vec = analyze_track(tone_wav)["embedding"]
    back = quantize.from_int8(*quantize.to_int8(vec))
    assert _cos(vec, back) > 0.999
