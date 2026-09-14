# Clean-room Essentia: sellable stack, pluggable sources

Date: 2026-09-13. Status: approved in conversation ("I don't care even if it means scrapping everything… act like this could be sold some day"; source decision: pluggable, Deezer for development only).

## 1. Goal

Remove every non-commercial dependency from the product and make the music source swappable, so the same app can be sold with a licensed catalogue.

### What is not sellable today (verified at the source on 2026-09-13)

- Discogs-EffNet and all its classifier heads: CC BY-NC-SA 4.0 (essentia.upf.edu/models). Every embedding in the corpus was produced by it.
- Deezer API and previews: "strictly limited for a non-commercial purpose" (developers.deezer.com/termsofuse).

### What replaces them (verified)

| Need | Choice | Licence | Notes |
|---|---|---|---|
| Audio embedding + text/mood | Microsoft CLAP (2023) | MIT, code and weights | 1024-d joint audio–text space; zero-shot prompts; CPU inference |
| Tempo, beats, downbeats | Beat This! | MIT, code and weights | training data mixed; weights MIT |
| Loudness, dynamic range | pyloudnorm | MIT | EBU R128 |
| Key, mode, chroma | librosa | ISC | |
| Instrument / vocal presence | PANNs Cnn14 | MIT | optional second opinion; AudioSet-trained |
| Galaxy layout | umap-learn | BSD-3 | replaces PCA for the map; PCA kept for tour/extremes |
| Runtime | PyTorch (CPU) | BSD | replaces TensorFlow in the image |
| First shippable catalogue | Jamendo API | commercial via licensing@jamendo.com; CC tracks; attribution + backlink required | Deezer stays behind the interface for development |

Rejected: MERT (CC BY-NC), madmom models (CC BY-NC-SA), MTG-Jamendo/Essentia heads (CC BY-NC-SA), LAION-CLAP music checkpoints (weights' terms unstated; training data mixed).

A licence register (`docs/THIRD_PARTY.md`) lists every model and library with its licence and a link, and a test fails if `pyproject.toml` gains a dependency not listed there.

### Non-goals

- Changing the app's screens or the ranking UX. The contract stays; fields are only added.
- Training our own models.
- Full-length audio. Previews (30 s) remain the unit of analysis.

## 2. Architecture

```
 sources/ (deezer | jamendo | local)  →  worker (crawl, analyze v2)  →  Atlas
                                                                          ↓
 web / iOS  →  /api  →  ranking (CLAP cosine − w·feel − tempo term)  ←  matrices
```

Only `analysis/`, the crawler's source layer, and the feel/tempo scoring change. Store, caches, dedupe, deployment, web and iOS stay.

## 3. Sources

`src/music_recommendations/corpus/sources/`: `base.py` defines `Source` with `name`, `search(q, limit) -> list[Track]`, `track(id) -> Track | None`, `preview_url(id) -> str | None`, `candidates(step) -> list[Track]` (the crawl arms), and `attribution(id) -> {"source", "url", "license"} | None`. `deezer.py` wraps the existing `corpus/deezer.py` and `crawl.py`; `jamendo.py` implements the Jamendo v3 API (client id from `JAMENDO_CLIENT_ID`; search, track, `audio` stream URL as the preview, tags/genres as crawl arms, per-track `license_ccurl`). `SOURCES` env selects which sources crawl (default `deezer` in dev).

Track ids: Deezer ids stay bare (existing corpus); other sources are prefixed `jamendo:123`. Track documents gain `source` and `attribution` (url, license). `Track` responses gain two optional fields, `source` and `attribution_url`; the contract test allows them; Swift's decoder ignores unknown keys. The web app shows "via Jamendo · CC BY-SA" with the backlink when present.

## 4. Analysis v2

`analysis/` is rewritten around PyTorch:

- `clap.py`: loads Microsoft CLAP 2023 once; `embed_audio(paths) -> (n, 1024)`; `embed_text(prompts) -> (k, 1024)`; batched.
- `feel.py`: a prompt bank of contrastive pairs producing continuous scores in [0, 1] as `softmax(cos(audio, pos), cos(audio, neg))`: energy (energetic/calm), valence (happy/sad), tension (tense/relaxed), acoustic (acoustic/electronic), danceable (danceable/not), vocal (vocals/instrumental), brightness (bright/dark), density (busy/sparse), plus the CLAP audio embedding's own cosine. The bank lives in one table so prompts can be tuned by ear.
- `rhythm.py`: Beat This! → `tempo_bpm`, `beat_strength` (mean beat activation), `downbeat_ratio`; pyloudnorm → `loudness_lufs`, `loudness_range`; librosa → `key`, `mode`, `key_strength`.
- `analyze_tracks(paths) -> list[dict | Exception]` returns `{"embedding": (1024,), "feel": (8,), "rhythm": {...}}`. `FEATURES_VERSION = 4`. `contract/features.py FEATURE_KEYS = {"embedding": 1024, "feel": 8}` plus `RHYTHM_KEYS`.
- The worker gains a `reanalyze` arm: while any live track has `features_version < 4`, it re-downloads and re-analyzes those first (before crawling); progress is logged. Ranking uses only version-4 rows (`store.LIVE` gains `features_version: 4`), so the visible corpus shrinks at cutover and grows back over the re-analysis window.

Measured on the Mac (see the plan) and to be re-measured on the VM: CLAP embed per 30 s preview, Beat This! per preview, loudness. Target: under 6 s per track on the VM so a 19k-track re-analysis finishes within a day.

## 5. Ranking

- `sounds_like`: `cos(clap) − w_feel·mean|Δfeel| − w_tempo·tempo_dist`, where `tempo_dist = min(|log2(bpm_a/bpm_b)|, |log2(bpm_a/bpm_b) ± 1|)` (octave-tolerant, in [0, 0.5]); `feel` (default 0.3) and `tempo` (default 0.2) query params; sliders in the web app; math panel shows the eight feel bars, BPM, key, loudness for seed and pick.
- `surprise`: unchanged (CLAP cosine, centrality-corrected).
- New additive endpoint `GET /search/text?q=` : CLAP text embedding against the corpus (top 25), so "smoky late-night trumpet" works; the web search box gets a "by description" toggle.
- Galaxy: UMAP (n_neighbors 15, min_dist 0.1, fixed seed) on the snapshot subset, cached like the PCA; tour/extremes keep PCA.

## 6. Deployment

The Docker image installs `torch` (aarch64 CPU wheel), `msclap`, `beat_this`, `librosa`, `pyloudnorm`, and downloads the CLAP and Beat This! weights at build time into `/app/models/`. TensorFlow and the Essentia-derived model files are removed. Memory: CLAP ~600 MB resident in the worker; the API does not load models (text search embeds prompts through a small in-API CLAP text tower, ~150 MB, loaded lazily).

## 7. Cutover

1. Ship analysis v2 + sources behind the same endpoints; bump `FEATURES_VERSION`.
2. Deploy; the worker re-analyzes the existing corpus (Deezer previews, development only) at full speed; the API serves version-4 rows only.
3. Remove `models/heads`, `discogs-effnet` fetching, and every TensorFlow import; delete `feel.py` (v1) and the EffNet front-end; update `THIRD_PARTY.md`.
4. Jamendo source enabled via `SOURCES=deezer,jamendo` in dev; production sale requires the Jamendo licence and removing `deezer` from `SOURCES`.

## 8. Risks

- CLAP's music understanding is weaker than EffNet's on fine-grained style; the text axis and rhythm features compensate. If "sounds like" quality drops noticeably, PANNs Cnn14 embeddings (MIT) can be concatenated.
- Zero-shot prompt scores are relative, not calibrated; z-score each feel dimension over the corpus before distance so no axis dominates.
- Re-analysis window: the visible corpus shrinks to zero at cutover and refills over ~a day; the fixture seeds are re-analyzed first.
- Jamendo's catalogue is independent music; mainstream searches will not resolve there. That is the product reality of a licensed source.
