# Analysis on ARM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `analyze_track()` run without Essentia, so the Oracle ARM VM can embed audio itself.

**Architecture:** Essentia is replaced by three small pieces: ffmpeg decodes MP3 to 16 kHz mono float32, a numpy front-end computes the 96-band log-mel patches that the Discogs-EffNet model expects, and TensorFlow runs the existing frozen graph. A per-vector int8 quantizer shrinks the 1280-float result for storage. The genre head and groove code are deleted. Every stage is verified against real Essentia on the developer's Mac, where Essentia is installed, using tests that skip when Essentia is absent.

**Tech Stack:** Python 3.11, numpy, tensorflow (2.16+; Linux aarch64 and macOS arm64 wheels exist), ffmpeg CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md`, section 4.

## Global Constraints

- Python `>=3.11,<3.12` (pyproject). Do not change it.
- `analyze_track(mp3_path) -> dict` keeps its name, signature, and return shape `{"embedding": np.ndarray(1280,)}`. Callers in `server/app.py` and `corpus/ingest.py` must not change in this plan.
- `contract/features.py` `FEATURE_KEYS` becomes exactly `{"embedding": 1280}`.
- `FEATURES_VERSION` becomes `3` because the vectors are no longer bit-identical with Essentia's.
- Parity bar from the spec: cosine similarity > 0.99 between the new pipeline's mean embedding and Essentia's, on fixture tracks.
- No Essentia import may remain anywhere under `src/`.
- Branch: `analysis-arm`, off `solo-migration-spec` (which holds the spec). Run `python3 -m pytest` before every commit.

## Facts measured on 2026-09-11 (do not re-derive)

These were measured against Essentia 2.1-beta6-dev on the Mac and fix the front-end exactly:

- `TensorflowPredictEffnetDiscogs` defaults: `input="serving_default_melspectrogram"`, `output` we use `"PartitionedCall:1"` (1280-d penultimate layer), `patchSize=128`, `patchHopSize=62`, `batchSize=64`, `lastPatchMode="discard"`, `lastBatchMode="same"` (zero-pad the last batch to 64 patches, return only real ones).
- `TensorflowInputMusiCNN(frame512) -> bands96` equals exactly: Hann window `0.5 - 0.5*cos(2*pi*i/(N-1))` (symmetric, unnormalized) applied and then **rolled by N/2** (Essentia's zero-phase windowing), magnitude spectrum via `rfft` (257 bins), **power** (magnitude squared), Slaney mel filterbank 96 bands over 0–8000 Hz with **unit_tri** normalization (each triangle scaled by `2/(right-left)` so its area is 1), then `log10(1 + 10000 * bands)`. Numpy reproduction of this chain matched to relative error 4e-6.
- Essentia's `FrameCutter(frameSize=512, hopSize=256, startFromZero=True)` on N samples yields `ceil((N-512)/256) + 1` frames, first frame = `signal[:512]`, last frame zero-padded.
- Slaney mel: `hz2mel(f) = f/(200/3)` for f < 1000 else `15 + ln(f/1000)/(ln(6.4)/27)`; inverse accordingly.
- The EffNet graph file is `models/discogs-effnet-bs64-1.pb` (18 MB), fetched by `scripts/fetch_models.py`. `models/` is gitignored.

## File structure

- Create `src/music_recommendations/analysis/frontend.py` — decode + mel + patches. Pure numpy and a subprocess. No TensorFlow.
- Rewrite `src/music_recommendations/analysis/embedding.py` — TensorFlow graph loading and batched inference. Exposes `effnet_frames(mp3_path)` as today.
- Create `src/music_recommendations/analysis/quantize.py` — int8 round trip.
- Modify `src/music_recommendations/analysis/__init__.py` — drop heads.
- Modify `src/music_recommendations/analysis/schema.py` — version 3, embedding-only metrics.
- Modify `src/music_recommendations/analysis/registry.py` — drop `HEADS`; add `EFFNET_INPUT`, `PATCH_SIZE`, `PATCH_HOP`, `BATCH_SIZE`.
- Delete `src/music_recommendations/analysis/heads.py`, `groove.py`.
- Modify `contract/features.py`, `scripts/fetch_models.py`, `pyproject.toml`, `README.md`.
- Create `tests/analysis/test_frontend.py`, `test_embedding.py`, `test_quantize.py`, `test_parity.py`, `conftest.py`.
- Create `deploy/Dockerfile` (analysis-capable base image; sub-project 3 extends it).

---

### Task 1: ffmpeg decode

**Files:**
- Create: `src/music_recommendations/analysis/frontend.py`
- Create: `tests/analysis/conftest.py`
- Test: `tests/analysis/test_frontend.py`

**Interfaces:**
- Produces: `frontend.SAMPLE_RATE = 16000`; `frontend.decode(path: Path | str) -> np.ndarray` float32 1-D mono at 16 kHz. Raises `frontend.DecodeError` (subclass of `RuntimeError`) when ffmpeg fails.

- [ ] **Step 1: Create the branch**

```bash
git checkout solo-migration-spec && git checkout -b analysis-arm
```

- [ ] **Step 2: Write the conftest with a synthetic WAV fixture**

`tests/analysis/conftest.py`:

```python
"""Shared fixtures: a synthetic tone on disk, and Essentia-availability skips."""
from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def write_tone_wav(path: Path, seconds: float = 3.0, hz: float = 440.0,
                   sr: int = 44100) -> Path:
    """A 440 Hz sine at 44.1 kHz, so decode() has to resample."""
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.5 * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    return write_tone_wav(tmp_path / "tone.wav")


HAVE_ESSENTIA = importlib.util.find_spec("essentia") is not None
needs_essentia = pytest.mark.skipif(
    not HAVE_ESSENTIA, reason="parity tests need essentia (Mac only)"
)

MODELS = Path(__file__).resolve().parents[2] / "models"
HAVE_EFFNET = (MODELS / "discogs-effnet-bs64-1.pb").exists()
needs_effnet = pytest.mark.skipif(
    not HAVE_EFFNET, reason="run scripts/fetch_models.py first"
)
```

- [ ] **Step 3: Write the failing decode tests**

`tests/analysis/test_frontend.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from music_recommendations.analysis import frontend
from tests.analysis.conftest import SR, needs_essentia


def test_decode_returns_mono_16k_float32(tone_wav):
    audio = frontend.decode(tone_wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 3 * SR) < SR // 100  # within 10 ms of 3 s
    assert 0.4 < np.abs(audio).max() <= 1.0


def test_decode_missing_file_raises(tmp_path):
    with pytest.raises(frontend.DecodeError):
        frontend.decode(tmp_path / "nope.mp3")


@needs_essentia
def test_decode_matches_essentia_monoloader(tone_wav):
    from essentia.standard import MonoLoader

    ours = frontend.decode(tone_wav)
    ref = MonoLoader(filename=str(tone_wav), sampleRate=SR)()
    n = min(len(ours), len(ref))
    # Different resamplers; agree on the waveform to well under 1%.
    assert np.abs(ours[:n] - ref[:n]).max() < 0.01
```

- [ ] **Step 4: Run to verify it fails**

Run: `python3 -m pytest tests/analysis/test_frontend.py -v`
Expected: FAIL with `ImportError: cannot import name 'frontend'`

- [ ] **Step 5: Implement decode**

`src/music_recommendations/analysis/frontend.py`:

```python
"""Audio -> EffNet input, without Essentia.

Essentia has no Linux aarch64 wheels, so the VM cannot import it. What
Essentia contributed to analysis was decoding (MonoLoader) and the MusiCNN
mel front-end (TensorflowInputMusiCNN + the framing inside
TensorflowPredictEffnetDiscogs). Both are reproduced here in numpy; the
parameters were measured against Essentia and are recorded in
docs/superpowers/plans/2026-09-11-analysis-on-arm.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class DecodeError(RuntimeError):
    """ffmpeg could not decode the file."""


def decode(path: Path | str) -> np.ndarray:
    """Decode any audio file to mono float32 at 16 kHz via the ffmpeg CLI."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:  # ffmpeg binary itself missing
        raise DecodeError("ffmpeg not installed") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(proc.stderr.decode(errors="replace").strip()
                          or f"ffmpeg produced no audio for {path}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
```

- [ ] **Step 6: Run to verify it passes**

Run: `python3 -m pytest tests/analysis/test_frontend.py -v`
Expected: 3 PASS (the Essentia parity test runs on the Mac; skips elsewhere).

- [ ] **Step 7: Commit**

```bash
git add src/music_recommendations/analysis/frontend.py tests/analysis/conftest.py tests/analysis/test_frontend.py
git commit -m "feat(analysis): ffmpeg decode replaces MonoLoader"
```

---

### Task 2: Mel front-end and patching

**Files:**
- Modify: `src/music_recommendations/analysis/frontend.py`
- Modify: `src/music_recommendations/analysis/registry.py`
- Test: `tests/analysis/test_frontend.py`

**Interfaces:**
- Consumes: `frontend.decode`.
- Produces:
  - `registry.PATCH_SIZE = 128`, `registry.PATCH_HOP = 62`, `registry.BATCH_SIZE = 64`, `registry.EFFNET_INPUT = "serving_default_melspectrogram"`.
  - `frontend.FRAME_SIZE = 512`, `frontend.HOP_SIZE = 256`, `frontend.N_MELS = 96`.
  - `frontend.mel_frames(audio: np.ndarray) -> np.ndarray` shape `(n_frames, 96)` float32.
  - `frontend.patches(mel: np.ndarray) -> np.ndarray` shape `(n_patches, 128, 96)` float32; `n_patches = 0` if fewer than 128 frames.
  - `frontend.mel_filterbank() -> np.ndarray` shape `(96, 257)` (exposed for tests).

- [ ] **Step 1: Add constants to registry**

In `src/music_recommendations/analysis/registry.py`, after `EFFNET_OUTPUT = ...` add:

```python
EFFNET_INPUT = "serving_default_melspectrogram"

# Framing inside Essentia's TensorflowPredictEffnetDiscogs (its defaults).
PATCH_SIZE = 128   # mel frames per inference
PATCH_HOP = 62     # frames between patch starts (~1 prediction per second)
BATCH_SIZE = 64    # the graph was frozen with a fixed batch of 64 patches
```

- [ ] **Step 2: Write the failing mel tests**

Append to `tests/analysis/test_frontend.py`:

```python
def _tone(seconds=2.0, hz=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_mel_frames_shape_and_count():
    audio = _tone(2.0)
    mel = frontend.mel_frames(audio)
    n = len(audio)
    expected = -(-(n - frontend.FRAME_SIZE) // frontend.HOP_SIZE) + 1  # ceil + 1
    assert mel.shape == (expected, frontend.N_MELS)
    assert mel.dtype == np.float32


def test_mel_frames_tone_peaks_in_expected_band():
    mel = frontend.mel_frames(_tone(1.0, hz=440.0))
    fb = frontend.mel_filterbank()
    centre_hz = np.arange(fb.shape[1]) * SR / frontend.FRAME_SIZE
    band_centres = (fb * centre_hz).sum(axis=1) / fb.sum(axis=1)
    peak_band = mel[5:-5].mean(axis=0).argmax()
    assert abs(band_centres[peak_band] - 440.0) < 60.0


def test_mel_frames_silence_is_zero():
    mel = frontend.mel_frames(np.zeros(SR, dtype=np.float32))
    assert np.allclose(mel, 0.0)


def test_patches_shape_and_hop():
    mel = np.random.default_rng(0).random((400, 96), dtype=np.float32)
    p = frontend.patches(mel)
    assert p.shape == (1 + (400 - 128) // 62, 128, 96)
    assert np.array_equal(p[1], mel[62:62 + 128])


def test_patches_too_short_is_empty():
    assert frontend.patches(np.zeros((100, 96), np.float32)).shape == (0, 128, 96)


@needs_essentia
def test_mel_frames_match_essentia_input_musicnn():
    from essentia.standard import TensorflowInputMusiCNN

    rng = np.random.default_rng(1)
    audio = (_tone(1.0) + 0.1 * rng.standard_normal(SR)).astype(np.float32)
    ours = frontend.mel_frames(audio)
    ref_fn = TensorflowInputMusiCNN()
    for i in range(0, 40, 7):  # a spread of full frames
        frame = audio[i * frontend.HOP_SIZE: i * frontend.HOP_SIZE + frontend.FRAME_SIZE]
        ref = ref_fn(frame)
        assert np.abs(ours[i] - ref).max() / np.abs(ref).max() < 1e-4
```

- [ ] **Step 3: Run to verify they fail**

Run: `python3 -m pytest tests/analysis/test_frontend.py -v -k "mel or patches"`
Expected: FAIL with `AttributeError: module ... has no attribute 'mel_frames'`

- [ ] **Step 4: Implement the front-end**

Append to `src/music_recommendations/analysis/frontend.py`:

```python
from . import registry  # noqa: E402

FRAME_SIZE = 512
HOP_SIZE = 256
N_MELS = 96
_N_BINS = FRAME_SIZE // 2 + 1
_LOG_SCALE = 10000.0


def _slaney_hz2mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    log_step = np.log(6.4) / 27.0
    with np.errstate(divide="ignore"):
        return np.where(f < 1000.0, f / (200.0 / 3.0),
                        15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / log_step)


def _slaney_mel2hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    log_step = np.log(6.4) / 27.0
    return np.where(m < 15.0, m * (200.0 / 3.0),
                    1000.0 * np.exp(log_step * (m - 15.0)))


def mel_filterbank() -> np.ndarray:
    """(96, 257) Slaney-mel triangles over 0-8000 Hz, each with unit area.

    This is Essentia MelBands(warpingFormula="slaneyMel", normalize="unit_tri",
    weighting="linear"), which is what TensorflowInputMusiCNN uses.
    """
    edges = _slaney_mel2hz(np.linspace(_slaney_hz2mel(0.0),
                                       _slaney_hz2mel(SAMPLE_RATE / 2),
                                       N_MELS + 2))
    freqs = np.arange(_N_BINS) * SAMPLE_RATE / FRAME_SIZE
    fb = np.zeros((N_MELS, _N_BINS))
    for i in range(N_MELS):
        left, centre, right = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None) * (2.0 / (right - left))
    return fb.astype(np.float32)


_FILTERBANK = None


def _filterbank() -> np.ndarray:
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = mel_filterbank()
    return _FILTERBANK


# Essentia Windowing(type="hann", normalized=False): symmetric Hann, then the
# frame is rolled by half its length (zero-phase). The roll changes nothing
# about the magnitude spectrum but is kept so intermediate values match.
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_SIZE) / (FRAME_SIZE - 1))).astype(np.float32)


def _frames(audio: np.ndarray) -> np.ndarray:
    """Essentia FrameCutter(startFromZero=True): first frame at 0, last frame
    zero-padded to a full 512."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))
    n_frames = -(-(len(audio) - FRAME_SIZE) // HOP_SIZE) + 1
    padded_len = (n_frames - 1) * HOP_SIZE + FRAME_SIZE
    audio = np.pad(audio, (0, padded_len - len(audio)))
    idx = np.arange(FRAME_SIZE)[None, :] + HOP_SIZE * np.arange(n_frames)[:, None]
    return audio[idx]


def mel_frames(audio: np.ndarray) -> np.ndarray:
    """(n_frames, 96) log-compressed mel bands, identical to Essentia's
    TensorflowInputMusiCNN applied to FrameCutter(512, 256) frames."""
    frames = _frames(audio) * _WINDOW
    frames = np.roll(frames, FRAME_SIZE // 2, axis=1)
    power = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) ** 2
    bands = power @ _filterbank().T
    return np.log10(1.0 + _LOG_SCALE * bands).astype(np.float32)


def patches(mel: np.ndarray) -> np.ndarray:
    """(n_patches, 128, 96) windows of mel frames with hop 62; a trailing
    partial patch is discarded (Essentia lastPatchMode="discard")."""
    size, hop = registry.PATCH_SIZE, registry.PATCH_HOP
    n = len(mel)
    if n < size:
        return np.zeros((0, size, mel.shape[1]), dtype=np.float32)
    count = 1 + (n - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return np.ascontiguousarray(mel[idx], dtype=np.float32)
```

- [ ] **Step 5: Run to verify they pass**

Run: `python3 -m pytest tests/analysis/test_frontend.py -v`
Expected: all PASS, including `test_mel_frames_match_essentia_input_musicnn` on the Mac.

- [ ] **Step 6: Commit**

```bash
git add src/music_recommendations/analysis/frontend.py src/music_recommendations/analysis/registry.py tests/analysis/test_frontend.py
git commit -m "feat(analysis): numpy mel front-end matching TensorflowInputMusiCNN"
```

---

### Task 3: EffNet inference through TensorFlow

**Files:**
- Rewrite: `src/music_recommendations/analysis/embedding.py`
- Modify: `pyproject.toml` (optional-dependency `analysis`)
- Test: `tests/analysis/test_embedding.py`

**Interfaces:**
- Consumes: `frontend.decode`, `frontend.mel_frames`, `frontend.patches`, `registry.EFFNET_INPUT/EFFNET_OUTPUT/BATCH_SIZE/MODELS_DIR/EFFNET_FILE`.
- Produces: `embedding.effnet_frames(mp3_path) -> np.ndarray (n_patches, 1280)` float32; `embedding.embed_patches(p: np.ndarray) -> np.ndarray (n, 1280)`; `embedding.load_audio(path)` kept as an alias of `frontend.decode` for the existing docstring contract.

- [ ] **Step 1: Install TensorFlow locally and change the dependency**

In `pyproject.toml` replace

```toml
analysis = ["essentia-tensorflow"]
```

with

```toml
analysis = ["tensorflow>=2.16"]
```

Then run:

```bash
python3 -m pip install "tensorflow>=2.16"
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

Expected: a version 2.16 or newer prints. (`uv.lock` is regenerated in Task 6 once all dependency edits are in.)

- [ ] **Step 2: Write the failing embedding tests**

`tests/analysis/test_embedding.py`:

```python
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
```

- [ ] **Step 3: Run to verify they fail**

Run: `python3 -m pytest tests/analysis/test_embedding.py -v`
Expected: FAIL (ImportError from essentia-based module, or AttributeError `embed_patches`).

- [ ] **Step 4: Rewrite embedding.py**

Replace the whole of `src/music_recommendations/analysis/embedding.py` with:

```python
"""Audio -> Discogs-EffNet per-patch embeddings (n, 1280), via TensorFlow.

The frozen graph is the same file Essentia's TensorflowPredictEffnetDiscogs
loads; we feed it ourselves because Essentia does not ship on Linux ARM.
The graph was frozen with a fixed batch of 64 patches, so inputs are
zero-padded up to a multiple of 64 and the padding rows are dropped again
(Essentia's lastBatchMode="same").
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import frontend, registry

SAMPLE_RATE = frontend.SAMPLE_RATE

# TensorFlow logs a wall of INFO on import; silence before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_session = None
_input = None
_output = None


def _load():
    global _session, _input, _output
    if _session is not None:
        return
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
    _input = graph.get_tensor_by_name(registry.EFFNET_INPUT + ":0")
    _output = graph.get_tensor_by_name(registry.EFFNET_OUTPUT)
    _session = tf.compat.v1.Session(graph=graph)


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
```

- [ ] **Step 5: Run to verify they pass**

Run: `python3 -m pytest tests/analysis/test_embedding.py -v`
Expected: 3 PASS. If `get_tensor_by_name` raises for `PartitionedCall:1`, list the graph's operations with `[op.name for op in graph.get_operations() if "PartitionedCall" in op.name]` in a scratch script and confirm the name; the strings in the `.pb` file show `PartitionedCall` with outputs `:0` and `:1`, so `:1` is expected to resolve.

- [ ] **Step 6: Commit**

```bash
git add src/music_recommendations/analysis/embedding.py pyproject.toml tests/analysis/test_embedding.py
git commit -m "feat(analysis): run Discogs-EffNet through TensorFlow, no Essentia"
```

---

### Task 4: int8 quantizer

**Files:**
- Create: `src/music_recommendations/analysis/quantize.py`
- Test: `tests/analysis/test_quantize.py`

**Interfaces:**
- Produces: `quantize.to_int8(vec: np.ndarray) -> tuple[bytes, float]` and `quantize.from_int8(data: bytes, scale: float) -> np.ndarray` (float32). Sub-project 2's store uses exactly these two.

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_quantize.py`:

```python
from __future__ import annotations

import numpy as np

from music_recommendations.analysis import quantize


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_round_trip_size_and_similarity():
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(1280).astype(np.float32) * 3.0
    data, scale = quantize.to_int8(vec)
    assert isinstance(data, bytes) and len(data) == 1280
    assert isinstance(scale, float) and scale > 0
    back = quantize.from_int8(data, scale)
    assert back.dtype == np.float32 and back.shape == (1280,)
    assert _cos(vec, back) > 0.999


def test_extremes_survive():
    vec = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    back = quantize.from_int8(*quantize.to_int8(vec))
    assert np.allclose(back, vec, atol=0.05)


def test_zero_vector_does_not_divide_by_zero():
    data, scale = quantize.to_int8(np.zeros(4, np.float32))
    assert np.array_equal(quantize.from_int8(data, scale), np.zeros(4, np.float32))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/analysis/test_quantize.py -v`
Expected: FAIL with `ImportError: cannot import name 'quantize'`

- [ ] **Step 3: Implement**

`src/music_recommendations/analysis/quantize.py`:

```python
"""int8 storage form for embeddings: 1 byte per dimension plus one scale.

Ranking is cosine, so direction is what matters and a per-vector scale
loses nothing that changes an ordering. 1280 floats -> 1280 bytes + 1 float.
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/analysis/test_quantize.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/music_recommendations/analysis/quantize.py tests/analysis/test_quantize.py
git commit -m "feat(analysis): int8 embedding quantizer"
```

---

### Task 5: Remove heads, groove, genre; bump the schema

**Files:**
- Modify: `src/music_recommendations/analysis/__init__.py`
- Modify: `src/music_recommendations/analysis/schema.py`
- Modify: `src/music_recommendations/analysis/registry.py`
- Delete: `src/music_recommendations/analysis/heads.py`, `src/music_recommendations/analysis/groove.py`
- Modify: `contract/features.py`
- Modify: `scripts/fetch_models.py`
- Test: `tests/analysis/test_analyze.py` (new), `tests/test_contract.py`

**Interfaces:**
- Produces: `analyze_track(path) -> {"embedding": np.ndarray(1280,)}`; `FEATURES_VERSION == 3`; `METRICS == {"embedding": "cosine"}`; `contract/features.py FEATURE_KEYS == {"embedding": 1280}`.

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_analyze.py`:

```python
from __future__ import annotations

import importlib.util

import numpy as np

from music_recommendations.analysis import (FEATURES_VERSION, METRICS,
                                            analyze_track, as_json)
from tests.analysis.conftest import needs_effnet


def test_version_and_metrics():
    assert FEATURES_VERSION == 3
    assert METRICS == {"embedding": "cosine"}


def test_no_essentia_anywhere_in_src():
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "src"
    hits = [p for p in src.rglob("*.py")
            if "import essentia" in p.read_text() or "from essentia" in p.read_text()]
    assert hits == []


def test_removed_modules_are_gone():
    for name in ("heads", "groove"):
        assert importlib.util.find_spec(f"music_recommendations.analysis.{name}") is None


@needs_effnet
def test_analyze_track_returns_only_embedding(tone_wav):
    feats = analyze_track(tone_wav)
    assert set(feats) == {"embedding"}
    assert feats["embedding"].shape == (1280,)
    js = as_json(feats)
    assert len(js["embedding"]) == 1280 and isinstance(js["embedding"][0], float)
```

Add to `tests/test_contract.py`:

```python
def test_feature_keys_embedding_only():
    f = _features()
    assert f.FEATURE_KEYS == {"embedding": 1280}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/analysis/test_analyze.py tests/test_contract.py -v`
Expected: `test_version_and_metrics`, `test_no_essentia_anywhere_in_src`, `test_removed_modules_are_gone`, `test_feature_keys_embedding_only` FAIL.

- [ ] **Step 3: Delete the Essentia-only modules**

```bash
git rm src/music_recommendations/analysis/heads.py src/music_recommendations/analysis/groove.py
```

- [ ] **Step 4: Simplify analyze_track**

In `src/music_recommendations/analysis/__init__.py` replace the body of `analyze_track` from the comment `# Imported here, not at module scope` through `return features` with:

```python
    # Imported here, not at module scope: importing TensorFlow costs ~1 s,
    # and a caller that only wants FEATURES_VERSION or METRICS should not pay.
    from . import embedding

    mp3_path = Path(mp3_path)
    if not mp3_path.exists():
        raise FileNotFoundError(mp3_path)

    frames = embedding.effnet_frames(mp3_path)
    if len(frames) == 0:
        raise ValueError(f"{mp3_path}: too short for one EffNet patch")
    return {"embedding": frames.mean(axis=0).astype(np.float32)}
```

Update the docstring's first line to `"""Run EffNet; returns {"embedding": (1280,) float32}."""`. Fix the module docstring's "essentia" mention to say TensorFlow.

- [ ] **Step 5: Bump schema**

In `src/music_recommendations/analysis/schema.py` set:

```python
# 3: Essentia replaced by ffmpeg + numpy mel + TensorFlow (Linux ARM has no
#    Essentia wheels). Same model, same maths, but not bit-identical: cosine
#    with v2 vectors is >0.99, yet a corpus must not mix the two. Genre head
#    and groove dropped.
FEATURES_VERSION = 3
```

and

```python
METRICS = {
    "embedding": "cosine",
}
```

Delete the `groove` and `genre` paragraphs from the METRICS comment.

- [ ] **Step 6: Trim the registry and fetch script**

In `registry.py` delete the `Head` dataclass, the `HEADS` dict with its comment block, and `model_url`. Keep `MODELS_DIR`, `EFFNET_FILE`, `EFFNET_URL`, `EFFNET_OUTPUT`, and the constants from Task 2. Update the module docstring to: `"""Where the EffNet graph lives and the framing parameters it expects."""`

Replace `scripts/fetch_models.py` `main()` with:

```python
def main() -> None:
    registry.MODELS_DIR.mkdir(exist_ok=True)
    fetch(registry.EFFNET_URL, registry.MODELS_DIR / registry.EFFNET_FILE)
    print("models ready")
```

and its docstring first line with `"""Download the Discogs-EffNet graph into models/."""`.

- [ ] **Step 7: Update the contract**

In `contract/features.py` replace the `FEATURE_KEYS` block with:

```python
# analyze_track(mp3_path) -> dict with exactly these keys.
FEATURE_KEYS = {
    "embedding": 1280,   # Discogs-EffNet penultimate layer, patch-mean
}
```

and change the docstring line `It is read-only (see CLAUDE.md).` to `Changing it is cross-cutting (see CLAUDE.md).`

- [ ] **Step 8: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: all PASS. `tests/server/test_store.py` and `test_embed_worker.py` use literal `genre`/`groove` keys in fixture dicts; that is data, not a contract, and they still pass. If any test imports `heads` or `groove`, delete that test (those modules are gone by design).

- [ ] **Step 9: Commit**

```bash
git add -A src/music_recommendations/analysis contract/features.py scripts/fetch_models.py tests/analysis/test_analyze.py tests/test_contract.py
git commit -m "refactor(analysis): embedding only — drop genre head, groove, essentia; FEATURES_VERSION 3"
```

---

### Task 6: End-to-end parity against Essentia on the fixture tracks

**Files:**
- Test: `tests/analysis/test_parity.py`
- Modify: `README.md`, `pyproject.toml` (regenerate `uv.lock` if `uv` is installed)

**Interfaces:**
- Consumes: `analyze_track`, `corpus.download.download_preview(track, dest_dir)`, `corpus.deezer.fresh_preview_url(track_id)`.

- [ ] **Step 1: Write the parity test**

`tests/analysis/test_parity.py`:

```python
"""Old pipeline vs new on real audio. Runs only where essentia is installed
(the Mac); the VM and CI skip it. This is the merge gate for the ARM port."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from music_recommendations.analysis import analyze_track
from tests.analysis.conftest import needs_effnet, needs_essentia

FIXTURE = Path(__file__).resolve().parents[2] / "contract" / "fixture.json"
N_TRACKS = int(os.environ.get("PARITY_TRACKS", "8"))
pytestmark = [needs_essentia, needs_effnet]


def _essentia_embedding(mp3: Path) -> np.ndarray:
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
    from music_recommendations.analysis import registry

    model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(registry.MODELS_DIR / registry.EFFNET_FILE),
        output=registry.EFFNET_OUTPUT,
    )
    frames = model(MonoLoader(filename=str(mp3), sampleRate=16000)())
    return np.asarray(frames).mean(axis=0)


def _download(track: dict, dest: Path) -> Path | None:
    from music_recommendations.corpus import deezer, download

    try:
        return download.download_preview(track, dest)
    except Exception:  # noqa: BLE001 - signed URL expired; refresh once
        url = deezer.fresh_preview_url(track["track_id"])
        if not url:
            return None
        return download.download_preview({**track, "preview_url": url}, dest)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fixture_tracks_cosine_above_099(tmp_path):
    tracks = json.loads(FIXTURE.read_text())["tracks"][:N_TRACKS]
    scores = []
    for track in tracks:
        mp3 = _download(track, tmp_path)
        if mp3 is None:
            continue
        new = analyze_track(mp3)["embedding"]
        old = _essentia_embedding(mp3)
        scores.append((track["title"], _cos(new, old)))
    assert len(scores) >= 3, "network: could not fetch enough previews"
    worst = min(scores, key=lambda s: s[1])
    print("\n".join(f"{c:.4f}  {t}" for t, c in scores))
    assert worst[1] > 0.99, f"worst parity {worst}"
```

- [ ] **Step 2: Run it on the Mac**

Run: `python3 -m pytest tests/analysis/test_parity.py -v -s`
Expected: PASS, printing a cosine per track, all above 0.99. If a track is below 0.99, the likeliest cause is frame alignment (Essentia's predictor may not use `startFromZero=True`). Check by shifting: in a scratch script compute the new embedding after prepending 256 zeros to the decoded audio and compare again. If the shifted version scores higher on every track, change `_frames` in `frontend.py` to prepend `HOP_SIZE` zeros and update the Task 2 count test accordingly. Record the outcome in the commit message.

- [ ] **Step 3: Update README and lock file**

In `README.md` replace line 9-10 (`pip install -e ".[dev]"` and `fetch_models.py` comment) with:

```
    python3 -m pip install -e ".[dev,analysis]"   # analysis extra = tensorflow
    brew install ffmpeg                            # or apt install ffmpeg
    python3 scripts/fetch_models.py                # downloads EffNet into models/
```

If `uv` is installed (`which uv`), run `uv lock`; otherwise leave `uv.lock` and note it in the commit.

- [ ] **Step 4: Full suite, then commit**

Run: `python3 -m pytest -q`
Expected: all PASS (parity test included on the Mac).

```bash
git add tests/analysis/test_parity.py README.md pyproject.toml uv.lock
git commit -m "test(analysis): Essentia parity gate on fixture tracks; README for TF setup"
```

---

### Task 7: Analysis-capable Docker image, smoke-tested on the VM

**Files:**
- Create: `deploy/Dockerfile`
- Create: `deploy/.dockerignore`
- Create: `scripts/analyze_one.py`

**Interfaces:**
- Produces: image `essentia:latest` with ffmpeg, TensorFlow, the project, and the EffNet graph baked in. Sub-project 3 adds api/worker entrypoints on top of this same file.

- [ ] **Step 1: Write a one-file CLI for smoke tests**

`scripts/analyze_one.py`:

```python
"""Analyze one audio file and print the embedding's shape, norm, and timing.
Usage: python3 scripts/analyze_one.py path/to/preview.mp3"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis import analyze_track  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    t0 = time.perf_counter()
    vec = analyze_track(path)["embedding"]
    dt = time.perf_counter() - t0
    print(f"{path.name}: shape={vec.shape} norm={float((vec ** 2).sum() ** 0.5):.3f} "
          f"seconds={dt:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the Dockerfile**

`deploy/.dockerignore`:

```
.git
ios
legacy
notebooks
docs
audio_cache
corpus_snapshot
models
.venv
__pycache__
*.pyc
```

`deploy/Dockerfile`:

```dockerfile
# Base image for every Essentia service. Build from the repo root:
#   docker build -f deploy/Dockerfile -t essentia:latest .
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY contract ./contract
COPY scripts ./scripts
RUN pip install --no-cache-dir -e ".[analysis]"

# Bake the EffNet graph so the container needs no network at start.
RUN python3 scripts/fetch_models.py

ENV TF_CPP_MIN_LOG_LEVEL=2
CMD ["python3", "-c", "import music_recommendations.analysis as a; print('analysis ok, v', a.FEATURES_VERSION)"]
```

- [ ] **Step 3: Build and smoke-test on the VM**

Push the branch, then on the VM:

```bash
git push -u origin analysis-arm
ssh ubuntu@146.235.195.66 '
  set -e
  [ -d ~/essentia ] || git clone https://github.com/Gabeyocum28/Essentia.git ~/essentia
  cd ~/essentia && git fetch && git checkout analysis-arm && git pull
  docker build -f deploy/Dockerfile -t essentia:latest .
  curl -sL -o /tmp/preview.mp3 "$(python3 -c "import json;print(json.load(open(\"contract/fixture.json\"))[\"tracks\"][0][\"preview_url\"])")"
  docker run --rm -v /tmp/preview.mp3:/tmp/preview.mp3 essentia:latest python3 scripts/analyze_one.py /tmp/preview.mp3
'
```

Expected: the last line prints `shape=(1280,) norm=... seconds=...`. If the fixture's signed preview URL has expired (curl returns an HTML error page and ffmpeg fails), fetch a fresh URL: `curl -s https://api.deezer.com/track/2711778 | python3 -c "import json,sys;print(json.load(sys.stdin)['preview'])"` and download that instead. Record the measured seconds per track in the commit message; the spec assumes a few seconds per 30 s preview on 2 cores.

- [ ] **Step 4: Commit and open the PR**

```bash
git add deploy/Dockerfile deploy/.dockerignore scripts/analyze_one.py
git commit -m "build: ARM-capable analysis image; smoke-tested on the Oracle VM"
git push
gh pr create --base main --title "Analysis on ARM: Essentia replaced by ffmpeg + numpy + TensorFlow" --body "Implements sub-project 1 of docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md. Includes the spec and the solo AGENTS.md from branch solo-migration-spec."
```

The PR base is `main`, so it carries the spec commit too.

---

## Self-review

- **Spec coverage.** 4.1 frontend (Tasks 1-2), embedding rewrite (Task 3), quantize (Task 4), heads and groove deleted with contract change (Task 5), dependency swap (Task 3 and 6), verification tests including the parity gate (Tasks 1-6), spec gate "image analyzes a fixture track on the VM" (Task 7). Spec 4.1 says the last patch is zero-padded; measurement shows Essentia discards it (`lastPatchMode="discard"`), and the plan follows the measurement. The spec is corrected in the same commit as Task 2.
- **Placeholders.** None; every step carries its code and command.
- **Type consistency.** `effnet_frames`, `embed_patches`, `mel_frames`, `patches`, `to_int8`, `from_int8`, `decode`, `DecodeError` are named identically in tests and implementation; registry constants `PATCH_SIZE/PATCH_HOP/BATCH_SIZE/EFFNET_INPUT` defined in Task 2 before use in Tasks 2-3.
