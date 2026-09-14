# Task 1 report — Analysis v2 (CLAP, zero-shot feel, rhythm, licence register)

Worktree: `/Users/gabrielyocum/Projects/Essentia/.claude/worktrees/task1-analysis-v2`
Branch: `task1/analysis-v2` (off `main`, dbc5f2d)

## What changed

New modules under `src/music_recommendations/analysis/`:

- `clap.py` — Microsoft CLAP 2023 loaded once behind a lock from
  `models/v2/CLAP_weights_2023.pth` (never the library's own hub download),
  `torch.set_num_threads(2)`. `embed_audio(waveforms, sr=44100)` goes
  through the wrapper's tensor path (`_get_audio_embeddings`), not its file
  loader, so torchaudio/torchcodec file decoding is never used and the
  preview is decoded exactly once. Three 7 s windows per track, evenly
  spaced (the wrapper's own loader crops at *random*, which would make
  analysis non-reproducible), mean-pooled, L2-normalized. All windows of a
  group go through as one batch. `embed_text` likewise normalized.
- `feel_v2.py` — `PROMPT_BANK` (8 contrastive pairs), `FEEL_KEYS`,
  `TEMPERATURE = 25`, `feel_scores()` = softmax over the two poles' cosines,
  positive probability. Text embeddings computed once per process.
- `rhythm.py` — `RHYTHM_KEYS` and `rhythm_features(waveform, sr)`: tempo from
  Beat This! (median inter-beat interval), `beat_strength` = mean sigmoid
  activation at the detected beats, `loudness_lufs` + `loudness_range`
  (pyloudnorm, BS.1770, floored at -70), `key`/`mode`/`key_strength`
  (librosa CQT chroma vs Krumhansl profiles). All imports are lazy so the
  module is readable without torch/librosa installed.
- `v2.py` — `analyze_tracks(paths)` keeping the v1 signature: decode once per
  path (`DecodeError` on failure or a too-short clip), batch CLAP, feel,
  rhythm; returns `{"embedding": (1024,), "feel": (8,), "rhythm": {...},
  "_features_version": 4}` or the exception in that path's slot.

Modified:

- `analysis/__init__.py` — `analyze_tracks`/`analyze_track` now forward to v2
  (a thin forwarder rather than `= v2.analyze_tracks`, so importing the
  package stays free of torch). v1 kept as `analyze_tracks_v1` /
  `analyze_track_v1` for the parity test only. `as_json` now passes the
  `rhythm` dict and `_features_version` through.
- `analysis/schema.py` — `FEATURES_VERSION = 4` with the reason recorded;
  METRICS note updated for the 1024-d normalized CLAP vector.
- `analysis/registry.py` — a v2 block: `V2_DIR`, `CLAP_URL`/`CLAP_WEIGHTS`,
  `BEAT_THIS_URL`/`BEAT_THIS_CKPT`. v1 block untouched.
- `contract/features.py` — `FEATURE_KEYS = {"embedding": 1024, "feel": 8}` and
  the new `RHYTHM_KEYS` tuple with per-field meaning.
- `scripts/fetch_models.py` — fetches v2 weights into `models/v2/` (CLAP from
  HuggingFace, Beat This! `final0` from the authors' share) and pre-caches the
  GPT-2 tokenizer msclap needs at load time, so a container needs no network
  for its first job. Downloads now go to `.part` and are length-checked
  against `Content-Length` before renaming — see "Beat This!" below for why.
  v1 fetching kept.
- `deploy/Dockerfile` — installs torch/torchaudio from the PyTorch CPU index
  (aarch64 cp311 wheels; avoids ~2 GB of CUDA), then `-e ".[analysis]"`, then
  bakes all model files. TensorFlow still present (v1).
- `pyproject.toml` — analysis extra = torch, msclap, librosa, pyloudnorm,
  soxr, `beat_this @ git+…`, **plus tensorflow**, which stays until Task 4
  deletes v1 and is flagged as such in a comment. `torchcodec` is *not*
  needed: we never use msclap's file loader.
- `docs/THIRD_PARTY.md` — every runtime dependency, notable transitive
  package, model weight and system component with licence + link.
- `tests/test_licences.py` — parses pyproject deps (incl. extras, PEP 503
  normalized), demands a register row for each, demands every row state a
  licence, and fails on `NC`/`NonCommercial` anywhere except one explicitly
  named, scheduled-for-removal component (the v1 EffNet model, which *is*
  CC BY-NC-SA and is still shipped this sprint). A fourth test asserts that
  allowlist has exactly one entry, so the cutover cannot forget to empty it.
- Tests: `tests/analysis/test_v2.py` (new, `needs_v2`-gated),
  `tests/test_contract.py` (new widths, FEEL_KEYS, RHYTHM_KEYS),
  `tests/analysis/conftest.py` (`needs_v2`, `needs_clap_weights`),
  `test_analyze.py`/`test_parity.py` retargeted at the v1 functions,
  `test_quantize.py` moved to 1024-d and `needs_v2` for the real-embedding
  round trip.

## Beat This! — resolved, no fallback needed

`File2Beats(checkpoint_path="final0")` failed in the spike because the
package resolves the shortname against a WebDAV share
(`beat_this.inference.CHECKPOINT_URL`) that the spike could not reach
cleanly. Working URL:

    https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt

It is fetched into `models/v2/beat_this-final0.ckpt` and passed as a local
path. One trap worth recording: the first `curl` of it returned HTTP 200 but
a **truncated 58 MB** file (the real one is 81 MB; curl reported
`HTTP/2 stream was not closed cleanly`), which torch then rejected as a
corrupt zip. `fetch_models.fetch` now verifies `Content-Length` and only
renames the `.part` file when it matches, so a truncated model can never be
mistaken for a cached one.

`rhythm.py` still carries the `librosa.beat.beat_track` fallback, used when
the checkpoint is absent or fails to load — a host that skipped
`fetch_models.py` gets a tempo rather than an exception. It is cached
per process, so a missing checkpoint costs one import attempt, not one
per track.

## Measured (2 threads, real 30 s preview, `2711781.mp3`)

| stage | time |
|---|---|
| CLAP load (once per process) | 5.5 s |
| Beat This! load (once) | 0.3 s |
| decode (librosa, 30 s → 44.1 kHz mono) | 0.18 s |
| CLAP embed, 1 track / 3 windows | 0.17 s |
| 16 prompt embeddings (once per process) | 0.35 s |
| feel scores | < 1 ms |
| rhythm (Beat This! + loudness + chroma) | 1.1 s |
| **`analyze_tracks` on 4 previews** | **5.55 s → 1.39 s/track** |

Well inside the plan's < 12 s/track target; `GROUP_SIZE` will not need
lowering. Rhythm, not CLAP, is now the per-track cost.

Sanity checks on the same preview (jazz, "All Blues"): cosine to itself at
half gain 0.991, to a click track 0.097, to a C-major triad loop 0.287;
`"a jazz trumpet solo"` 0.277 vs `"heavy metal guitar"` -0.119. Feel:
acoustic 0.99, vocal 0.005, energy 0.16, valence 0.87. Rhythm: 136.4 BPM,
beat strength 0.97, -27.6 LUFS, LRA 6.0, key D minor at strength 0.73.

## RED/GREEN

- `test_licences.py`: verified RED by adding a fake dependency to pyproject
  (`test_every_declared_dependency_has_a_row` failed) and by flipping
  librosa's licence cell to `CC BY-NC 4.0` (both NC tests failed); GREEN
  after reverting both.
- Beat This! integration: RED first — `_load_beat_tracker()` returned None
  against the truncated checkpoint (tests still passed via the librosa
  fallback, which is exactly why the report had to check the tracker
  directly rather than trust the tempo assertion); GREEN with the complete
  file: 120 BPM click → 120.00, 90 BPM click → 90.91.
- The eight-axis softmax test was written against a synthetic text matrix and
  verified to fail if the positive/negative columns are swapped.

## Test counts

- `PYTHONPATH=$PWD/src python3 -m pytest -q` → **379 passed, 17 skipped**
  (the skips are the Essentia parity gate and the v2 tests, which this
  interpreter cannot import).
- `PYTHONPATH=$PWD/src <v2-venv>/bin/python -m pytest tests/analysis/test_v2.py -q`
  → **15 passed**.

(The two `test_embedding.py` failures seen on first run were this worktree
having no `models/` — fixed by symlinking the main checkout's EffNet graph
and heads in, not by a code change.)

## Concerns

1. **The store has not been touched** (per the brief). `put_track` persists
   `embedding`, `feel` and `features_version`; the new `rhythm` dict and
   `_features_version` key are simply ignored today, so rhythm reaches
   nothing until Task 3 adds `get_many_rhythm` and the `LIVE` filter starts
   demanding version 4. Until then the corpus holds version-3 rows that the
   ranking still reads — Task 3/4 are what make this safe, and the two must
   land before any redeploy.
2. **`feel` changed meaning, not just width.** `server/app.py::_feel_blend`
   and the web `MathPanel` name v1's eleven heads; with eight differently
   scaled axes they will mislabel bars until Task 3. Nothing crashes (the
   code reads the vector positionally) which is the dangerous part.
3. **CLAP is ~700 MB of weights and ~2.5 GB resident**. The API process will
   pay that too once `/search/text` lands (Task 3); on a 2-CPU VM running
   api + worker + TensorFlow, memory is the thing to watch, and the compose
   `mem_limit` values probably need revisiting at the cutover.
4. **`soxr` is LGPL-2.1** (dynamically linked, unmodified, pulled in by Beat
   This!). That is fine for a commercial service but it is the only copyleft
   item in the register, so it is worth a deliberate decision rather than a
   discovery.
5. **The Docker image is not built here.** The torch CPU-index install and
   `pip install -e ".[analysis]"` are written to the plan's measured facts
   (aarch64 cp311 wheels exist) but unverified on this machine; the image
   also now bakes ~800 MB of weights, so build time and image size will jump.
6. The v2 venv needed `pytest` installed to run the gated tests; it is a
   scratch venv, not part of the repo.
