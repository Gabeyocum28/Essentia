from __future__ import annotations

import numpy as np

from music_recommendations.analysis import embedding, frontend, registry
from tests.analysis.conftest import SR, needs_effnet


@needs_effnet
def test_embed_patches_shape_and_batch_padding():
    rng = np.random.default_rng(0)
    # 70 patches: one full batch of 64 plus a partial batch of 6, so the
    # fixed-batch padding path is exercised.
    p = rng.random((70, registry.PATCH_SIZE, frontend.N_MELS), dtype=np.float32)
    out = embedding.embed_patches(p)
    assert out.shape == (70, 1280)
    assert out.dtype == np.float32
    # Padding must not leak: embedding patch i alone equals patch i in a batch.
    solo = embedding.embed_patches(p[65:66])
    assert np.allclose(solo[0], out[65], atol=1e-4)


@needs_effnet
def test_embed_patches_empty():
    out = embedding.embed_patches(np.zeros((0, 128, 96), np.float32))
    assert out.shape == (0, 1280)


@needs_effnet
def test_effnet_frames_on_tone(tone_wav):
    out = embedding.effnet_frames(tone_wav)
    n_frames = frontend.mel_frames(frontend.decode(tone_wav)).shape[0]
    assert out.shape == (1 + (n_frames - 128) // 62, 1280)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0
