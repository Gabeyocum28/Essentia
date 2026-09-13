# Speed, Full Features, and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every user-facing request fast on the same hardware, roughly double the worker's throughput, bring the iOS-only SOUND mode to the web app, let iOS use the seed subset, and give the web app a real visual design.

**Architecture:** Server: derived insights caches (subset, PCA, pairwise, MST) are computed from a *viz snapshot* of the corpus matrix that refreshes at most every `VIZ_REFRESH_S` seconds or when the corpus grew by 5%, instead of on every new track; PCA uses the 1280×1280 covariance eigendecomposition instead of a thin SVD (7× faster, same components); track metadata is served from an in-process cache filled in batches; responses are gzipped. Worker: patches from up to three tracks are packed into one fixed 64-patch inference batch, previews are downloaded in parallel with inference, TensorFlow threads are pinned. Web: a same-origin audio proxy feeds Web Audio decoding for a browser mel spectrogram, self-similarity matrix and band-solo playback; a design token layer and component restyle. iOS: passes the seed and rec ids to the global viz endpoints.

**Tech Stack:** FastAPI, numpy, pymongo, TensorFlow; React + TypeScript + Vite, Web Audio API, Web Workers; Swift (config-level edits only).

**Baseline (measured 2026-09-13, warm caches, 7.5k tracks, crawl running):** `/recommend` 1.0–2.3 s; `/viz/map` 9–11 s (2.5 MB); `/viz/tour` 7.5–8.3 s; `/viz/extremes` 9.1 s; `/viz/walk` 12.4 s; `/viz/hubs` 3.4 s; `/viz/mst` 2.7 s; `/viz/histogram` 0.3 s; `/search` 0.7 s; `/seed` (analyzed) 0.15 s. Server breakdown: thin SVD 5.5 s vs covariance+eigh 0.8 s; pairwise 1.4 s; MST 0.4 s; walk 0.7 s; `get_many_tracks` for 7.5k ids 0.39 s; `corpus_ids` cold 1.2 s. Worker: a 64-patch batch takes 2.9–3.5 s on 2 cores; a 30 s preview yields 28 patches, so 56% of each batch is zero padding; the graph input is fixed at (64, 128, 96). Web bundle 300 KB.

## Global Constraints

- No new resources: same VM, same container limits, same Atlas tier.
- Public contract unchanged; endpoints may gain optional params. iOS must keep working without rebuild until Task 3.
- Numeric results of recommendations are unchanged. PCA coordinates may differ by float noise and sign convention must be preserved (largest-magnitude coordinate per component positive).
- Every derived-cache change is covered by a test that proves no recompute inside the refresh window and a recompute after it.
- Web: `npm test && npm run build` green; Python: `python3 -m pytest -q` green; run before each commit. Branch `speed-features-polish` off `main`.
- Design: dark, one system; no UI framework; Google Fonts (Inter) allowed; keep every existing behaviour and test id.
- Web app must work at 400px width.

## Facts (do not re-derive)

- `app.py`: `_corpus_matrix`/`_build` maintain `_MATRIX_CACHE["embedding"]` (rows appended as the corpus grows; the matrix object is replaced on growth). `_viz_subset(seed_id, extra_ids, corpus=None)` keys its LRU on `id(matrix_all)`; `_top8`/`_mst` key on `id(subset)`; `viz.pairwise_cosine` single-entry identity cache; all guarded by `_VIZ_CACHE_LOCK`; `_UNIT_CACHE` holds `(matrix, unit)` float32. `recommend` calls `store.get_track` once per result (10 sequential Atlas round trips). `/viz/map` calls `get_many_tracks(subset_ids)` for every point (0.4 s) and returns full track dicts for all points (2.1 MB JSON). No gzip middleware. `viz.project_top8`: row-normalize, mean-center, thin SVD, top-8, sign fix, variance fractions over the full spectrum.
- `worker.py`: `_tick` claims one job (`dequeue_job(timeout=5)`), `process_job(track_id)` downloads (`download_preview(url)` → temp mp3), `analyze_track(mp3)`, `store.put_track`. `analysis/__init__.py::analyze_track` → `embedding.effnet_frames(path)` → `frontend.decode` → `mel_frames` → `patches` → `embed_patches` (pads to `registry.BATCH_SIZE=64`, runs one `session.run` per 64-chunk). TF threading defaults (0/0) on a 2-core container.
- iOS `APIClient.swift`: `vizHubs()`, `vizTour()`, `vizMST()`, `vizExtremes(pc:limit:)` build URLs via a `get(path, query:)` helper; callers in `Features/Insights/*.swift` (InsightsView loads tour/mst/hubs/extremes lazily and knows the seed track id and `map.recs`).
- Web: `styles.css` 176 lines of tokens + component classes; screens `Search/Seed/Recommendations/Insights`; insights views draw on canvases; player is one `<audio>` on `/api/preview/{id}` (302 to Deezer, cross-origin, so samples cannot be read). iOS SOUND mode: mel spectrogram (96 bands, FFT 2048, hop 1024, dB floor −80, inferno-like colormap stops `(0,0,0.02)→(0.20,0.03,0.35)→(0.55,0.10,0.42)→(0.90,0.35,0.15)→(0.99,0.75,0.20)→(0.99,0.98,0.80)`, 70 dB range anchored at the track's peak); self-similarity matrix (pool to ≤256 columns, L2-normalize, cosine Gram, same colormap, tap to seek); band solo (keep only one frequency band audible, driven from a strip beside the spectrogram and from WhySimilar bars).

## File structure

- Server: `server/app.py` (snapshot policy, meta cache, gzip, batch fetches, audio proxy), `server/viz.py` (`project_top8` via covariance), `server/store.py` (no change expected).
- Worker: `analysis/embedding.py` (`embed_patch_groups`, TF threads), `worker.py` (grouped claim, parallel downloads).
- iOS: `Networking/APIClient.swift`, `Features/Insights/InsightsView.swift` (call sites only).
- Web: `src/audio/{decode.ts, fft.ts, mel.ts, ssm.ts, bandSolo.ts, spectrogram.worker.ts}`, `src/insights/Sound.tsx`, `SpectrogramView.tsx`, `SelfSimilarity.tsx`, `BandStrip.tsx`; design: `src/styles.css` (tokens + components), `src/components/{TopBar,Card,Skeleton}.tsx`, small edits in every screen.
- Tests beside each module; `tests/server/test_speed.py` for snapshot/PCA/meta/gzip.

---

### Task 1: Server speed

**Files:** modify `src/music_recommendations/server/app.py`, `src/music_recommendations/server/viz.py`; create `tests/server/test_speed.py`; modify `tests/server/conftest.py` (clear the new caches).

**Interfaces produced:** `VIZ_REFRESH_S` (env, default 300), `VIZ_GROWTH_PCT` (default 5); `app._viz_snapshot() -> (ids, matrix)`; `app._tracks_cached(ids) -> list[dict|None]`; `viz.project_top8` same signature; `GET /viz/map?...&points=compact` returning `points: {ids, x, y}` without `tracks`; `GZipMiddleware`.

- [ ] **Step 1: PCA via covariance (test first)**

`tests/server/test_speed.py`:

```python
import numpy as np
from music_recommendations.server import viz


def _svd_reference(matrix):
    m = np.asarray(matrix, dtype=float)
    unit = m / np.linalg.norm(m, axis=1, keepdims=True)
    c = unit - unit.mean(axis=0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    coords = u[:, :8] * s[:8]
    for col in range(8):
        if coords[np.argmax(np.abs(coords[:, col])), col] < 0:
            coords[:, col] = -coords[:, col]
    return coords, (s[:8] ** 2) / float(np.sum(s ** 2))


def test_project_top8_matches_the_svd_reference():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((300, 40)).astype(np.float32)
    coords, var = viz.project_top8(m)
    ref, ref_var = _svd_reference(m)
    assert np.allclose(var, ref_var, atol=1e-6)
    for col in range(8):
        assert np.allclose(coords[:, col], ref[:, col], atol=1e-4)   # same sign convention


def test_project_top8_small_and_degenerate():
    coords, var = viz.project_top8(np.ones((2, 3), np.float32))
    assert coords.shape == (2, 8) and var.shape == (8,)
    coords, var = viz.project_top8(np.zeros((1, 5), np.float32))
    assert coords.shape == (1, 8)
```

Implement `viz.project_top8` as: row-normalize (float32→float64 for the 1280² step is fine), center, `C = Xc.T @ Xc` (d×d), `w, V = np.linalg.eigh(C)` ascending → take the top-8 eigenpairs in descending order, `coords = Xc @ V[:, :k]`, sign-fix as before, `variance = w_top / w.sum()` (eigenvalues of `X^T X` equal `s²`, so this is the same fraction over the full spectrum). Handle `d < 8` padding and `n < 2` exactly as before. Keep the docstring's numerical-identity note but say "same components as the thin SVD, computed via the 1280×1280 covariance (7× faster at 8k rows)".

- [ ] **Step 2: Viz snapshot policy (test first)**

Append to `test_speed.py` (uses `fake_mongo`, `client`, `TRACK`, `store` as `test_viz.py` does):

```python
def test_viz_snapshot_is_reused_inside_the_window(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    for tid, v in {"a": [1, 0], "b": [0.9, 0.1], "c": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    ids1, m1 = app_mod._viz_snapshot()
    store.put_track({**TRACK, "track_id": "d"}, {"embedding": [0.5, 0.5]})
    ids2, m2 = app_mod._viz_snapshot()
    assert m2 is m1 and ids2 == ids1                     # one new track, inside the window: no rebuild


def test_viz_snapshot_refreshes_after_the_window_or_growth(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    for tid in ("a", "b"):
        store.put_track({**TRACK, "track_id": tid}, {"embedding": [1.0, 0.0]})
    ids1, m1 = app_mod._viz_snapshot()
    monkeypatch.setattr(app_mod, "VIZ_REFRESH_S", 0.0)
    store.put_track({**TRACK, "track_id": "c"}, {"embedding": [0.0, 1.0]})
    ids2, m2 = app_mod._viz_snapshot()
    assert "c" in ids2 and m2 is not m1
    monkeypatch.setattr(app_mod, "VIZ_REFRESH_S", 3600.0)
    for i in range(10):                                   # +>5% growth forces a refresh too
        store.put_track({**TRACK, "track_id": f"g{i}"}, {"embedding": [1.0, 1.0]})
    ids3, _ = app_mod._viz_snapshot()
    assert len(ids3) == 13
```

Implement in `app.py`: module globals `VIZ_REFRESH_S = float(os.environ.get("VIZ_REFRESH_S", "300"))`, `VIZ_GROWTH_PCT = float(os.environ.get("VIZ_GROWTH_PCT", "5"))`, `_VIZ_SNAPSHOT: tuple[float, list[str], np.ndarray] | None`. `_viz_snapshot()` (under `_VIZ_CACHE_LOCK`): get the live `(ids, matrix)` from `_viz_embedding_corpus()`; if a snapshot exists and `time.monotonic() - stamp < VIZ_REFRESH_S` and `len(ids) < len(snap_ids) * (1 + VIZ_GROWTH_PCT/100)`, return the snapshot; else store `(now, ids, matrix)` (the live matrix object itself, no copy) and purge subset/top8/mst caches. Then `_viz_subset` uses `_viz_snapshot()` instead of `_viz_embedding_corpus()` when no `corpus` is passed, and `/viz/map` passes `_viz_snapshot()` as its `corpus` for the projection while still computing recs on the live matrix. Seeds analyzed after the snapshot (a fresh `/seed`) must still appear: if `seed_id` is not in the snapshot ids, force a refresh once. `conftest.clear_matrix_cache` clears `_VIZ_SNAPSHOT`.

- [ ] **Step 3: Track metadata cache and batch fetches (test first)**

```python
def test_tracks_cached_batches_misses_and_reuses_hits(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    for tid in ("a", "b", "c"):
        store.put_track_meta({**TRACK, "track_id": tid})
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks", lambda ids: (calls.append(list(ids)), real(ids))[1])
    got = app_mod._tracks_cached(["a", "b"])
    assert [t["track_id"] for t in got] == ["a", "b"] and calls == [["a", "b"]]
    got = app_mod._tracks_cached(["b", "c", "zzz"])
    assert calls[-1] == ["c", "zzz"] and got[2] is None


def test_recommend_fetches_tracks_in_one_batch(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    # seed + 3 corpus tracks
    ...  # mirror an existing recommend test's setup from test_app.py
    calls = []
    real = store.get_many_tracks
    monkeypatch.setattr(store, "get_many_tracks", lambda ids: (calls.append(1), real(ids))[1])
    monkeypatch.setattr(store, "get_track", lambda tid: (_ for _ in ()).throw(AssertionError("per-track fetch")))
    r = client.get("/recommend?track_id=<seed>&axis=sounds_like&limit=3")
    assert r.status_code == 200 and len(calls) == 1
```

Implement `_TRACK_META: OrderedDict[str, dict]` capped at 50000 entries with a lock; `_tracks_cached(ids)` returns cached hits and batch-fetches misses (one `get_many_tracks` call for all misses, caching non-None results; `None` for unknown ids is not cached). Replace the per-result `store.get_track` in `recommend` (both axis branches) and in `/viz/walk`, `/viz/hubs`, `/viz/extremes`, `/viz/map`'s seed lookup with `_tracks_cached`. `/viz/map`'s point tracks use `_tracks_cached(subset_ids)`. `_playable` unchanged. `put_track`/`put_track_meta` paths in `/seed` should also insert into the cache (a small `_remember_track(track)`).

- [ ] **Step 4: Gzip and compact points**

`app.add_middleware(GZipMiddleware, minimum_size=1024)` (from `fastapi.middleware.gzip`). `/viz/map` gains `points: Literal["full", "compact"] = "full"`; compact omits `points.tracks`. Tests: `client.get("/viz/map?...", headers={"accept-encoding": "gzip"})` has `content-encoding: gzip`; compact response has no `tracks` key and the full one still does.

- [ ] **Step 5: Suite, commit**

`python3 -m pytest -q` green. Commit `perf(server): viz snapshot policy, covariance PCA, track metadata cache, gzip, compact map points`.

---

### Task 2: Worker throughput

**Files:** modify `src/music_recommendations/analysis/embedding.py`, `src/music_recommendations/analysis/__init__.py`, `src/music_recommendations/worker.py`; tests in `tests/analysis/test_embedding.py`, `tests/server/test_worker.py`.

**Interfaces produced:** `embedding.embed_patch_groups(groups: list[np.ndarray]) -> list[np.ndarray]` (each `(n_i,128,96)` → `(n_i,1280)`, packed into 64-patch batches across groups, results identical to per-group `embed_patches`); `analysis.analyze_tracks(paths: list[Path]) -> list[dict | Exception]`; worker `GROUP_SIZE` (default 3), `DOWNLOAD_THREADS` (default 3).

- [ ] **Step 1: Packing (test first)**

```python
@needs_effnet
def test_embed_patch_groups_matches_per_group_results():
    rng = np.random.default_rng(1)
    groups = [rng.random((28, 128, 96), np.float32), rng.random((30, 128, 96), np.float32), rng.random((70, 128, 96), np.float32)]
    packed = embedding.embed_patch_groups(groups)
    for g, out in zip(groups, packed):
        assert out.shape == (len(g), 1280)
        assert np.allclose(out, embedding.embed_patches(g), atol=1e-4)


def test_embed_patch_groups_batches_are_packed(monkeypatch):
    calls = []
    monkeypatch.setattr(embedding, "_run_batch", lambda chunk: (calls.append(len(chunk)), np.zeros((len(chunk), 1280), np.float32))[1])
    groups = [np.zeros((28, 128, 96), np.float32), np.zeros((28, 128, 96), np.float32), np.zeros((10, 128, 96), np.float32)]
    embedding.embed_patch_groups(groups)
    assert calls == [64, 2]        # 66 patches → one full batch + a remainder (padded inside _run_batch)
```

Implement: factor the `session.run` on a 64-chunk (with zero padding) into `_run_batch(chunk) -> np.ndarray` (returns only the real rows); `embed_patches(p)` uses it; `embed_patch_groups(groups)` concatenates all patches, runs `_run_batch` over 64-slices, then splits results back per group. At `_load()`, set `tf.config.threading.set_intra_op_parallelism_threads(2)` and `set_inter_op_parallelism_threads(1)` before the session is created (guard with try/except in case the runtime is already initialized). `analyze_tracks(paths)`: decode + mel + patches per path (catch per-path exceptions and keep them in the result list), one `embed_patch_groups` call, mean per group → feature dicts.

- [ ] **Step 2: Grouped worker loop (test first)**

```python
def test_tick_processes_a_group_of_embed_jobs_with_one_analysis_call(fake_mongo, monkeypatch):
    for tid in ("1", "2", "3"):
        store.put_track_meta({**TRACK, "track_id": tid}); store.enqueue_embed(tid)
    monkeypatch.setattr(worker, "download_preview", lambda url: _fake_mp3(tmp))   # see the file's existing helpers
    calls = []
    monkeypatch.setattr(worker, "analyze_tracks", lambda paths: (calls.append(len(paths)), [dict(FEATURES) for _ in paths])[1])
    monkeypatch.setattr(worker.deezer, "get_track", lambda t: {**TRACK, "track_id": t})
    worker._tick()
    assert calls == [3] and store.corpus_size() == 3 and store.queued_count() == 0


def test_group_isolates_a_failed_download(fake_mongo, monkeypatch):
    ...  # one of three downloads raises → that job is failed, the other two are stored
```

Implement in `worker.py`: `_claim_group()` claims up to `GROUP_SIZE` embed jobs (first with `dequeue_job(timeout=5)`, then non-blocking `timeout=0` claims, embed kind only; an attribution job stops the group and is processed alone as today); `process_jobs(track_ids)`: resolve metadata and download previews in a `ThreadPoolExecutor(DOWNLOAD_THREADS)`; jobs whose download fails get `fail_job`; the rest go through `analyze_tracks` in one call; per-track exceptions → `fail_job`; successes → `put_track` + `clear_embed_marker`; log one line per track plus a group line `"[worker] group of N: X.Xs (Y tracks/min)"`. Keep `process_job(track_id)` as a thin wrapper (`process_jobs([track_id])`) so existing tests hold. Temp files always unlinked.

- [ ] **Step 3: Suite, commit**

`python3 -m pytest -q` green (the effnet-gated test runs on the Mac). Commit `perf(worker): pack patches across tracks into fixed batches; parallel downloads; pinned TF threads`.

---

### Task 3: iOS passes the seed subset

**Files:** modify `ios/Hackathon/Hackathon/Networking/APIClient.swift`, `ios/Hackathon/Hackathon/Features/Insights/InsightsView.swift` (and any other caller of the four methods).

- [ ] Add optional parameters `seed: String? = nil, recs: [String] = []` to `vizHubs`, `vizTour`, `vizMST`, `vizExtremes`; append `track_id` and `recs` (comma-joined) query items when present. In `InsightsView` pass the seed track id and `map.recs.map(\.track_id)` at each call site. Update the comment on those methods. No Xcode build is available here: keep the edits minimal and syntactically conservative (mirror the existing `URLQueryItem` style exactly). Commit `feat(ios): insights views ask for the seed's subset`.

---

### Task 4: SOUND mode in the web app

**Files:** server `app.py` (`GET /preview/{id}/audio`); web `src/audio/{decode.ts, fft.ts, mel.ts, ssm.ts, bandSolo.ts}`, `src/audio/spectrogram.worker.ts`, `src/insights/{Sound.tsx, SpectrogramView.tsx, SelfSimilarity.tsx, BandStrip.tsx}`, `src/insights/WhySimilar.tsx` (bar click solos the band), `src/screens/Insights.tsx` (SOUND tab), `src/player/usePlayer.tsx` (expose the audio element / a `MediaElementSource` hook), tests for `fft.ts`, `mel.ts`, `ssm.ts`, `bandSolo.ts`.

- [ ] **Step 1: Audio proxy (test first)** — `GET /preview/{track_id}/audio` streams the Deezer mp3 bytes through the API with `content-type: audio/mpeg`, `cache-control: private, max-age=600`, using `urllib`/`httpx` streaming in chunks; 404 when no preview. Test with a monkeypatched fetch. Document in the route docstring that this is used only by the web SOUND mode (bytes are needed for analysis) and playback still uses the redirect.
- [ ] **Step 2: DSP helpers with tests** — `fft.ts`: radix-2 real FFT (2048) returning magnitudes; test: a 440 Hz tone at 44.1 kHz peaks at bin round(440·2048/44100)=20. `mel.ts`: 96 Slaney mel filters over 0–sr/2 for FFT 2048; `melSpectrogram(samples, sr, {fft: 2048, hop: 1024, bands: 96}) -> {frames: Float32Array[], db floor −80, peak}`; test: silence → all −80; tone → peak band contains 440 Hz. `ssm.ts`: pool frames to ≤256 columns, L2-normalize, cosine Gram; test: diagonal ≈ 1, symmetric. `colormap.ts`: the six inferno-like stops; test: t=0 dark, t=1 light. `bandSolo.ts`: given an `AudioContext`, a `MediaElementAudioSourceNode` and `[lo, hi]` Hz, builds a two-stage highpass+lowpass (BiquadFilter, Q≈0.7071, 2 cascaded each for 24 dB/oct) with a bypass gain, `setBand(lo,hi)`, `clear()`; test with a stub AudioContext that records node connections.
- [ ] **Step 3: Worker + decoding** — `decode.ts`: fetch `/api/preview/{id}/audio` → `AudioContext.decodeAudioData` → mono Float32Array + sampleRate (cache per track id in memory, 5 entries). `spectrogram.worker.ts` computes mel frames + SSM off the main thread (Vite `new Worker(new URL(..., import.meta.url), {type: "module"})`).
- [ ] **Step 4: Views** — `Sound.tsx` (SOUND tab): loads audio for the selected rec (or the seed), shows a loading skeleton, then `SpectrogramView` (canvas, time → right, low frequency at bottom, colormap anchored at the track's peak with a 70 dB range, playhead synced to the player when this track is playing, click to seek via the player), the `BandStrip` beside it (drag to choose a band, shows "240 – 1.2k Hz", toggles solo), and `SelfSimilarity` (square canvas, click to seek). `WhySimilar` bars: clicking a bar solos that band (calls the shared band-solo controller) and highlights it; a "Solo off" button. The player must route its `<audio>` through the Web Audio graph only once (create the `MediaElementSource` lazily on first solo; keep direct playback otherwise; note Safari's autoplay rules: create the context on a user gesture).
- [ ] **Step 5: Verify, commit** — `npm test && npm run build`; commit `feat(web): SOUND mode — spectrogram, self-similarity, band solo`.

---

### Task 5: Visual design

**Files:** `web/src/styles.css` (rewrite around tokens), `web/index.html` (Inter font link, theme-color), `web/src/components/{TopBar.tsx, Card.tsx, Skeleton.tsx}`, edits to every screen and insights view for class names/structure only (no behaviour changes; all tests keep passing).

- [ ] **Step 1: Tokens and base** — palette: background `#0b0d12`, surface `#141824`, surface-2 `#1b2030`, border `rgba(255,255,255,.08)`, text `#eef1f6`, muted `#98a2b3`, accent `#5b9cff` (hover `#7db3ff`), seed `#ffd60a`, cyan `#64d2ff`, danger `#ff5c5c`; radii 8/12/16; spacing scale 4/8/12/16/24/32; shadows `0 1px 0 rgba(255,255,255,.04) inset, 0 8px 24px rgba(0,0,0,.35)`; font Inter with the system stack; `.mono` → `ui-monospace`. Focus rings visible (`:focus-visible`). Reduced-motion respected.
- [ ] **Step 2: Components** — `TopBar`: brand "Essentia" with a small gradient mark, back link, current section; `Card`: surface + border + padding; `Skeleton`: shimmering placeholders used in Search results, Seed header, Recommendations list, Insights while loading. Buttons: primary (accent gradient), secondary (surface-2), chip (pill, `aria-pressed` styled), icon buttons for play. Track rows: 56px artwork with soft shadow, title/artist hierarchy, score as a small mono pill on the right, hover lift. Now-playing bar: blurred artwork backdrop (`filter: blur(24px)` on a background copy with an overlay), progress as a thin accent bar, play/pause icon button. Insights: segmented control as pills in a surface track; chips row; canvases in a Card with 16px radius; captions in muted mono; rec strip with 48px artwork and a yellow ring for the seed; MathPanel as a two-column key/value grid.
- [ ] **Step 3: Layout** — max width 960px centered; search page hero (input + button in one rounded control); results as a responsive list; seed page with a large artwork (200px) and two big axis buttons; recommendations header shows the seed compactly with "See the math" as a secondary button; insights fills the viewport height for the canvas with a sticky control row. 400px: everything stacks; rec strip scrolls horizontally.
- [ ] **Step 4: Verify** — `npm test && npm run build`; in the browser (dev server) check every screen at desktop and 400px widths and note what was checked. Commit `style(web): design system — tokens, top bar, cards, skeletons, restyled screens`.

---

### Task 6: Deploy and measure

- [ ] Push, PR against main, deploy from the branch on the VM (`git checkout speed-features-polish && docker compose -f deploy/docker-compose.yml up -d --build`), then rerun the baseline timing script (two passes, warm) and record a before/after table in the report; verify the worker log's group line shows tracks/min and compare with the ~12/min baseline; check container memory; other sites 200. Do not merge.

## Self-review

- Coverage: speed (Tasks 1–2, measured in 6), all features (Task 4 web SOUND mode; Task 3 iOS subset), prettier (Task 5), no extra resources (constraints). Each cache change has a window/growth test. The audio proxy is the one new bandwidth cost (480 KB per SOUND-mode track), used only on demand.
- Placeholders: Task 1 Step 3's second test and Task 2 Step 2's second test are sketched with `...` — the implementer completes them following the neighbouring tests in the same files; everything else is concrete.
- Type consistency: `_viz_snapshot`, `_tracks_cached`, `embed_patch_groups`, `_run_batch`, `analyze_tracks`, `process_jobs` are named identically across tasks and tests.
