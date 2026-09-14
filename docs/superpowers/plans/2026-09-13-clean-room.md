# Clean-room Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every non-commercial component (the Discogs-EffNet analysis and the Deezer-only catalogue) with MIT/BSD/ISC-licensed equivalents behind a pluggable source interface, re-analyze the corpus, and keep the app working throughout.

**Architecture:** see `docs/superpowers/specs/2026-09-13-clean-room-design.md`. Analysis v2 = Microsoft CLAP (1024-d audio embedding + zero-shot feel axes from contrastive text prompts) + Beat This!/librosa rhythm + pyloudnorm loudness + librosa key. Sources = `corpus/sources/{base,deezer,jamendo}.py`. Ranking = CLAP cosine − feel − tempo terms; `/search/text`; UMAP galaxy. Worker gains a `reanalyze` arm. Cutover removes TensorFlow and the EffNet files.

**Tech Stack:** PyTorch CPU (aarch64 wheels exist for cp311), `msclap` (MIT), `beat_this` (MIT, from GitHub), `librosa`, `pyloudnorm`, `umap-learn`, `torchcodec` (msclap's loader needs it; otherwise decode with librosa and call the wrapper's tensor path).

**Measured on the Mac (2 CPU threads, 30 s preview at 44.1 kHz):** decode 1.5 s (librosa), CLAP load 2.7 s, CLAP audio embed 3.95 s, six text prompts 0.19 s, loudness 0.03 s. Zero-shot cosines on a jazz preview: calm 5.6 > energetic 3.2, acoustic 6.5 > electronic 2.2, sad 4.5 ≈ happy 4.2 (CLAP similarities are unnormalized logits; use softmax over each pair). `beat_this.inference.File2Beats(checkpoint_path="final0")` failed to load the checkpoint by name in the spike — resolve the checkpoint name/URL from the Beat This! README (the pretrained names and download) or pass a local path; fall back to `librosa.beat.beat_track` if it cannot be made to work in an hour.

## Global Constraints

- Every new dependency must appear in `docs/THIRD_PARTY.md` with its licence and link, and `tests/test_licences.py` fails when `pyproject.toml` lists a runtime dependency absent from that file or when any listed licence contains "NC" / "NonCommercial".
- No TensorFlow, no Essentia-derived model file, and no `models/heads` after Task 4. `FEATURES_VERSION = 4`. Ranking uses only version-4 rows.
- Contract: `Track` gains optional `source` and `attribution_url` (both may be absent); everything else unchanged. iOS decodes unknown keys fine; the contract test allows the two optional keys.
- Track ids: existing Deezer ids stay bare; other sources use `"<source>:<id>"`.
- Branch `clean-room` off `main`. `python3 -m pytest -q` and `cd web && npm test && npm run build` green before each commit. Tasks 1 and 2 may run in parallel in worktrees (disjoint files); Task 3 after both merge; Task 4 last.
- The Mac has TensorFlow and Essentia installed; neither may be imported by v2 code. A dedicated venv is fine for local v2 testing (`torch`, `msclap`, `torchcodec`, `librosa`, `pyloudnorm`, `beat_this`); the project venv gets them via `pyproject.toml` `analysis` extra replaced: `analysis = ["torch>=2.2", "msclap>=1.3", "torchcodec", "librosa>=0.10", "pyloudnorm>=0.1", "beat_this @ git+https://github.com/CPJKU/beat_this"]` (pip supports the direct reference; `uv lock` too).

## Facts

- `analysis/__init__.py::analyze_tracks(paths) -> list[dict | Exception]` is what the worker calls; keep the signature. Current keys: `embedding` (1280), `feel` (11). `analysis/frontend.py` (ffmpeg decode + mel) and `embedding.py` (TF EffNet) and `feel.py` (heads) are v1 and get deleted in Task 4; `quantize.py` stays (int8 storage works for 1024-d).
- `store.put_track` stores `embedding` (int8 + scale), `feel` (list), `features_version`; `LIVE` filter = analyzed + not duplicate. `get_many_feel` projects feel.
- `worker.py`: `_tick` → `_claim_group` → `process_jobs` → `analyze_tracks`; crawl arms call `corpus.crawl.from_charts/snowball/deep_cuts` and `corpus.deezer.*`; `_fresh_track(track_id)` uses `deezer.get_track`; `download_preview(url)`.
- `app.py`: `AXIS_FEATURES`, `_feel_blend` (mean |Δ| over feel dims, `_FEEL_ALIGN_CACHE`), `_viz_subset`, `_top8` PCA via `viz.project_top8`, `/search` → `deezer.search`, `/preview/{id}` → `_fresh_preview` → `deezer.fresh_preview_url`.
- Web: `MathPanel` renders `math.feel {seed, rec}` with `feel_keys`; Recommendations has the feel slider (`DEFAULT_FEEL` in `client.ts`); Search screen calls `api.search`.

---

### Task 1: Analysis v2 (worktree)

**Files:** create `src/music_recommendations/analysis/clap.py`, `feel_v2.py` (rename to `feel.py` in Task 4 when v1 is deleted; until then import as `feel_v2`), `rhythm.py`, `v2.py` (the new `analyze_tracks`), `docs/THIRD_PARTY.md`, `tests/test_licences.py`, `tests/analysis/test_v2.py`; modify `pyproject.toml` (analysis extra), `scripts/fetch_models.py` (download CLAP 2023 weights and the Beat This! checkpoint into `models/v2/`; keep EffNet fetching until Task 4), `deploy/Dockerfile` (install the analysis extra; torch CPU wheel; `models/v2` baked; TensorFlow still present until Task 4), `contract/features.py` (`FEATURE_KEYS = {"embedding": 1024, "feel": 8}`, `RHYTHM_KEYS = ("tempo_bpm", "beat_strength", "loudness_lufs", "loudness_range", "key", "mode", "key_strength")`), `analysis/schema.py` (`FEATURES_VERSION = 4`), `analysis/__init__.py` (delegate `analyze_tracks` to `v2.analyze_tracks` when `ANALYSIS_V2=1` env or always once Task 4 lands — implement as: `analyze_tracks = v2.analyze_tracks`, and keep the v1 module importable only for the parity test until Task 4).

- [ ] `clap.py`: `load()` once (lock), `embed_audio(waveforms: list[np.ndarray], sr=44100) -> (n,1024)` via the wrapper's tensor path (decode with librosa/ffmpeg ourselves; do not depend on torchaudio file loading), L2-normalized; `embed_text(prompts) -> (k,1024)` L2-normalized; `torch.set_num_threads(2)`. Weights path from `models/v2/`.
- [ ] `feel_v2.py`: `PROMPT_BANK = [("energy", "energetic, fast, intense music", "calm, slow, gentle music"), ("valence", "happy, joyful, uplifting music", "sad, melancholic music"), ("tension", "tense, dark, anxious music", "relaxed, peaceful music"), ("acoustic", "acoustic instruments, unplugged", "electronic, synthesized, produced"), ("danceable", "danceable, groovy, rhythmic music", "music that is not for dancing"), ("vocal", "a singer with vocals", "instrumental music without vocals"), ("bright", "bright, crisp, high-frequency sound", "dark, warm, muffled sound"), ("density", "dense, busy, layered arrangement", "sparse, minimal arrangement")]`; `FEEL_KEYS` = the eight names; `feel_scores(audio_emb (n,1024)) -> (n,8)` = softmax over `[cos(a,pos), cos(a,neg)] * TEMPERATURE` with `TEMPERATURE = 25` (CLAP logit scale), taking the positive probability. Text embeddings computed once and cached. Test: on two synthetic "audio" vectors equal to the positive and negative prompt embeddings, scores ≈ 1 and ≈ 0.
- [ ] `rhythm.py`: `rhythm_features(waveform, sr) -> dict` with `tempo_bpm` (Beat This! median inter-beat interval; fallback `librosa.beat.beat_track`), `beat_strength` (mean beat activation, or librosa onset-strength mean normalized), `loudness_lufs`, `loudness_range` (pyloudnorm; `-70` floor), `key` (0–11), `mode` (`"major"|"minor"`), `key_strength` (librosa chroma + Krumhansl templates). Tests: a 120-BPM click track → tempo within ±3 BPM; silence → `-70` LUFS and `key_strength 0`; a C-major triad loop → key 0 major.
- [ ] `v2.analyze_tracks(paths)`: decode once per path (librosa, mono, 44.1 kHz; `DecodeError` on failure), batch CLAP, feel, rhythm; returns `{"embedding": (1024,) float32, "feel": (8,), "rhythm": {...}, "_features_version": 4}` or an Exception per path. Test with the fixture preview downloaded in the test (skip when offline) or a synthetic WAV.
- [ ] `THIRD_PARTY.md` + `tests/test_licences.py` (parse `pyproject.toml` dependencies incl. the analysis extra; every distribution name must appear as a row; rows carry licence text; fail on "NC"/"NonCommercial"). Populate rows for everything present.
- [ ] Commit `feat(analysis): v2 — CLAP embedding, zero-shot feel axes, rhythm/loudness/key; licence register`.

### Task 2: Sources (worktree)

**Files:** create `src/music_recommendations/corpus/sources/{__init__,base,deezer,jamendo}.py`, `tests/corpus/test_sources.py`; modify `worker.py` (crawl via `sources.active()`; `_fresh_track` via `sources.track(id)`; `download_preview` via `sources.preview_url`), `server/app.py` (`/search` fans out to active sources; `/preview/{id}` resolves via the id's source; `_playable` passes through `source`/`attribution_url`), `server/store.py` (`_meta` stores `source`, `attribution`; `_contract` includes `source` and `attribution_url` when present), `contract/features.py` (`TRACK_OPTIONAL_FIELDS = {"source", "attribution_url"}`), `tests/test_contract.py`, web `TrackRow` (small "via Jamendo · CC BY-SA" line with the link when present).

- [ ] `base.py`: `class Source(Protocol)`: `name`, `owns(track_id) -> bool`, `search`, `track`, `preview_url`, `candidates(step) -> list[Track]`, `attribution(track) -> dict | None`. `sources.active() -> list[Source]` from `SOURCES` env (default `deezer`); `sources.for_id(track_id)`.
- [ ] `deezer.py`: wraps existing `corpus/deezer.py` and `crawl.py` (charts / snowball / deep cuts rotation moves here from `worker.crawl_step`, which now just calls `source.candidates(step)` round-robin across active sources); attribution `{"source": "deezer", "url": "https://www.deezer.com/track/<id>", "license": None}`.
- [ ] `jamendo.py`: v3 API (`https://api.jamendo.com/v3.0/`, `client_id` from `JAMENDO_CLIENT_ID`): `tracks/?search=`, `tracks/?id=`, `audio` field as preview URL (full CC track; analysis still uses the first 30 s), `candidates(step)` rotates over `tracks/?tags=<tag>&order=popularity_total` for a tag list (jazz, blues, soul, funk, electronic, rock, folk, classical, hiphop, ambient) and `?featured=1`; attribution from `license_ccurl` and `shareurl`; ids `jamendo:<id>`; rate limiting 0.2 s; tests with a stubbed HTTP layer.
- [ ] Contract/store/app/web edits as listed; Deezer ids remain bare. Tests: a jamendo track round-trips `source`/`attribution_url` into `/recommend` and `/search` results; a Deezer track has `source: "deezer"` and no attribution_url; iOS untouched.
- [ ] Commit `feat(sources): pluggable music sources — Deezer (dev) and Jamendo; attribution in Track`.

### Task 3: Ranking, text search, UMAP galaxy, web

**Files:** `server/app.py`, `server/axes.py`, `server/viz.py`, `server/store.py` (`LIVE` gains `features_version: 4`; `get_many_rhythm`), `tests/server/*`, web `client.ts`, `types.ts`, `Recommendations.tsx` (tempo slider beside feel), `Search.tsx` ("by description" toggle → `/search/text`), `MathPanel.tsx` (eight feel bars + BPM/key/loudness rows), `Galaxy.tsx` unchanged (consumes x,y).

- [ ] `LIVE` filter includes `features_version: FEATURES_VERSION`; `corpus_ids`/`base_matrix`/`tracks_since` honour it; test.
- [ ] Feel z-scoring: per-dimension mean/std over the feel matrix (recomputed with the snapshot), `feel_dist = mean |z_seed − z_cand|`; test that a dominant dimension no longer dominates.
- [ ] Tempo term: `store.get_many_rhythm` → `tempo` matrix (n,1) aligned like feel; `tempo_dist` octave-tolerant as in the spec; query param `tempo` (default 0.2); `math` gains `tempo_dist`, `rhythm: {seed, rec}`; tests.
- [ ] `GET /search/text?q=&limit=25`: `clap.embed_text([q])` in the API (lazy load of the text tower only; if loading the full CLAP is the only option, load it lazily and document the ~600 MB), cosine against the corpus matrix, results as Tracks with `score`; test with `embed_text` monkeypatched.
- [ ] UMAP: `viz.project_umap(matrix, seed=0) -> (n,2)` with `umap-learn`; `_top8` stays for tour/extremes; `/viz/map` and `/viz/walk` use UMAP coords cached per subset identity (≤4); falls back to PCA when `n < 50`; tests (shape, determinism with the seed, fallback).
- [ ] Web: types/client params; tempo slider; description toggle; math panel rows; attribution line already from Task 2; tests.
- [ ] Commit `feat(ranking,web): version-4 corpus, z-scored feel, tempo term, text search, UMAP galaxy`.

### Task 4: Cutover

- [ ] `worker.py`: `reanalyze` arm — when `store.stale_count() > 0` (live rows with `features_version < 4`), claim up to `GROUP_SIZE` stale ids (oldest first), re-download via their source, `analyze_tracks`, `put_track`; runs BEFORE crawling each tick; log `reanalyzed N, remaining M`. `store.stale_ids(limit)` / `stale_count()`. Tests.
- [ ] Delete v1: `analysis/frontend.py`, `embedding.py`, `feel.py` (v1), `registry.py` EffNet/heads entries, `tests/analysis/test_parity.py`, `parity_baseline.json`, the `needs_essentia` machinery; rename `feel_v2` → `feel`; `fetch_models.py` fetches only v2 weights; Dockerfile drops TensorFlow; `pyproject` analysis extra = the v2 list; `THIRD_PARTY.md` updated; README rewritten for v2; `.env.example` gains `SOURCES`, `JAMENDO_CLIENT_ID`.
- [ ] Deploy from the branch; watch the re-analysis rate on the VM (target < 12 s/track with 2 threads; if slower, set `GROUP_SIZE=2`); after the fixture seeds are re-analyzed, verify `/recommend` and the galaxy; record before/after timings; open the PR. Do not merge.

## Self-review
Coverage: licences (T1 register + test; T4 removals), sources (T2), analysis (T1), ranking/text/UMAP (T3), cutover (T4). Placeholders: the Beat This! checkpoint-name resolution is explicitly delegated with a fallback. Type consistency: `embedding` 1024, `feel` 8 (`FEEL_KEYS`), `rhythm` dict keys = `RHYTHM_KEYS`; `source`/`attribution_url` names shared by store, contract, app, web.
