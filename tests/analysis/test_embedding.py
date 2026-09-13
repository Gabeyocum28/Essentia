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


# ---- packing patches from several tracks into the fixed 64-patch batch ----

@needs_effnet
def test_embed_patch_groups_matches_per_group_results():
    """Packing must be arithmetic-neutral: what a track's patches embed to
    does not depend on which other tracks shared the batch with them."""
    rng = np.random.default_rng(1)
    groups = [
        rng.random((28, 128, 96), dtype=np.float32),
        rng.random((30, 128, 96), dtype=np.float32),
        rng.random((70, 128, 96), dtype=np.float32),
    ]
    packed = embedding.embed_patch_groups(groups)
    assert len(packed) == len(groups)
    for g, out in zip(groups, packed):
        assert out.shape == (len(g), 1280)
        assert np.allclose(out, embedding.embed_patches(g), atol=1e-4)


def test_embed_patch_groups_batches_are_packed(monkeypatch):
    """66 patches across three tracks: one full batch of 64 plus a remainder
    of 2 — instead of three mostly-empty batches, as per-track embedding
    would have produced."""
    calls = []
    monkeypatch.setattr(
        embedding, "_run_batch",
        lambda chunk: (calls.append(len(chunk)),
                       np.zeros((len(chunk), 1280), np.float32))[1],
    )
    groups = [np.zeros((28, 128, 96), np.float32),
              np.zeros((28, 128, 96), np.float32),
              np.zeros((10, 128, 96), np.float32)]
    out = embedding.embed_patch_groups(groups)

    assert calls == [64, 2]
    assert [len(o) for o in out] == [28, 28, 10]


def test_embed_patch_groups_handles_empty_groups(monkeypatch):
    """A track too short for one patch still gets a (0, 1280) slot back, in
    position, so the caller can zip groups to tracks."""
    monkeypatch.setattr(
        embedding, "_run_batch",
        lambda chunk: np.zeros((len(chunk), 1280), np.float32),
    )
    out = embedding.embed_patch_groups([np.zeros((0, 128, 96), np.float32),
                                        np.zeros((3, 128, 96), np.float32)])
    assert [o.shape for o in out] == [(0, 1280), (3, 1280)]
    assert embedding.embed_patch_groups([]) == []
