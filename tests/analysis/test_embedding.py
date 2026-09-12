from __future__ import annotations

import threading

import numpy as np

from music_recommendations.analysis import embedding, frontend, registry
from tests.analysis.conftest import needs_effnet


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
def test_load_builds_the_session_exactly_once_under_concurrency(monkeypatch):
    """/seed lets up to 2 requests in at once (server/app.py's
    _ANALYZE_SEM), so a cold process can have concurrent callers race into
    _load(). The double-checked lock must still build exactly one session."""
    original_build = embedding._build_session
    calls = {"n": 0}
    lock = threading.Lock()

    def counting_build():
        with lock:
            calls["n"] += 1
        return original_build()

    monkeypatch.setattr(embedding, "_build_session", counting_build)
    embedding._session = None
    embedding._input = None
    embedding._output = None

    sessions = [None] * 4
    barrier = threading.Barrier(4)

    def worker(i):
        barrier.wait()
        embedding._load()
        sessions[i] = embedding._session

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    assert len({id(s) for s in sessions}) == 1


@needs_effnet
def test_effnet_frames_on_tone(tone_wav):
    out = embedding.effnet_frames(tone_wav)
    n_frames = frontend.mel_frames(frontend.decode(tone_wav)).shape[0]
    assert out.shape == (1 + (n_frames - 128) // 62, 1280)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0
