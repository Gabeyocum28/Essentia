# Essentia on Oracle + Atlas: design

Date: 2026-09-11. Status: approved in conversation, awaiting written review.

## 1. Goal

Run every part of Essentia on one Oracle VM with all data in MongoDB Atlas,
and ship a web app alongside the iOS app. Nothing runs on the developer's
Mac after this: not analysis, not the crawler, not a tunnel.

Target host: `ubuntu@146.235.195.66`, VM.Standard.A1.Flex (aarch64), 2 OCPU,
12 GB RAM, 193 GB disk, Ubuntu 24.04, Docker preinstalled.
Domain: `essentia.gabeyocum.com` (A record to the VM, set by the user).
Data: the user's existing Atlas 512 MB cluster, shared with another app.

### Non-goals

- Accounts, auth, persistence of user state. The app stays anonymous.
- Atlas Vector Search. Ranking stays in-process numpy; the corpus fits in RAM.
- Bit-identical embeddings with the old Mac pipeline. "Close enough" is the
  accepted bar (cosine similarity > 0.99 on the fixture tracks).
- Keeping Redis or the file snapshot mode. Both are deleted.
- Feature parity on day one for every /viz screen in the web app. They are
  ported in priority order after the core flow works.

## 2. Why this is hard: Essentia has no Linux ARM wheels

Essentia publishes macOS ARM and Linux x86 wheels only, and its TensorFlow
bridge needs the TensorFlow C library, which Google also does not publish for
Linux ARM. TensorFlow's Python wheel does exist for Linux aarch64. So the VM
runs the EffNet model through TensorFlow directly and Essentia's contribution
(audio decoding plus a mel-spectrogram front-end) is reimplemented in numpy.

## 3. Architecture

```
 iOS app ─┐                      ┌─ api     (FastAPI, uvicorn)      ─┐
          ├─ https://essentia… ─►│  caddy   (TLS, static web/, /api)│──► Atlas
 web app ─┘                      └─ worker  (embed + crawl loop)    ─┘
                                        │
                                        └─► Deezer (search, previews)
```

One Docker Compose stack on the VM, three services, one Atlas database.
Both clients speak the existing HTTP contract; the iOS app changes only its
base URL.

## 4. Sub-project 1: analysis on ARM

Folder: `src/music_recommendations/analysis/`, `tests/analysis/`.

### 4.1 Components

- `frontend.py` (new). `decode(mp3_path) -> np.ndarray` shells out to
  ffmpeg (`-ac 1 -ar 16000 -f f32le`) and returns float32 mono at 16 kHz.
  `mel_frames(audio) -> np.ndarray` produces the (n, 96) input EffNet
  expects, matching Essentia's `TensorflowInputMusiCNN`: 512-sample frames,
  256 hop, Hann window, 96 mel bands over 0–8000 Hz computed from the power
  spectrum, then `log10(1 + 10000 * x)`. Patches of 128 frames with hop 62,
  as Essentia's `TensorflowPredictEffnetDiscogs` uses; a trailing partial
  patch is discarded, and the last batch is zero-padded to the graph's fixed
  batch of 64 with the padding rows dropped from the output.
- `embedding.py` (rewritten). Loads `discogs-effnet-bs64-1.pb` with
  `tf.compat.v1` graph import once per process, feeds
  `serving_default_melspectrogram`, reads `PartitionedCall:1`, and returns
  (n_patches, 1280). `effnet_frames()` keeps its name and shape so
  `analyze_track()` is unchanged.
- `quantize.py` (new). `to_int8(vec) -> (bytes, float scale)` with
  `scale = max(abs(vec)) / 127`; `from_int8(bytes, scale) -> np.ndarray`.
  Ranking uses cosine, so a per-vector scale loses nothing that matters.
- `heads.py` and the genre head are deleted. `contract/features.py`
  `FEATURE_KEYS` becomes `{"embedding": 1280}`. Nothing in the API reads
  the genre vector; it cost 1.6 KB per track.
- `groove.py` is deleted along with any remaining Essentia import. The
  `essentia-tensorflow` optional dependency is replaced by `tensorflow` and
  `numpy`; ffmpeg is a system package in the Docker image.

### 4.2 Verification

- `tests/analysis/test_frontend.py`: synthetic signals (silence, a 440 Hz
  tone) produce the expected mel band peaks and frame counts.
- `tests/analysis/test_parity.py`: skipped unless `essentia` is importable
  (so it runs on the Mac once, not in CI or on the VM). For each fixture
  track with a cached preview, compares the new pipeline's mean embedding
  with Essentia's and asserts cosine similarity > 0.99. This is the gate
  for merging sub-project 1.
- `tests/analysis/test_quantize.py`: round-trip cosine > 0.999 on random
  and real vectors.

## 5. Sub-project 2: Atlas storage layer

Folder: `src/music_recommendations/server/`, `tests/server/`.

### 5.1 Collections (database `essentia`)

`tracks`, one document per analyzed or pending track:

```
{ _id: "3135556",                  // Deezer track id, string
  title, artist, album, artwork_url,
  embedding: BinData(1280 bytes),  // int8, absent until analyzed
  scale: 0.0123,                   // float, absent until analyzed
  features_version: 3,
  analyzed_at: ISODate | absent }
```

`jobs`, the replacement for the Redis queues:

```
{ _id: "embed:3135556" | "attr:<seed>:<rec>",
  kind: "embed" | "attr",
  track_id, seed_id, rec_id,
  state: "queued" | "running" | "done" | "failed",
  claimed_at: ISODate | null, attempts: int, error: string | null,
  created_at: ISODate }
```

`cache`, TTL-indexed on `expires_at`:

```
{ _id: "preview:3135556" | "attr:<seed>:<rec>", value: any, expires_at }
```

Indexes: `tracks.analyzed_at` (for incremental matrix refresh),
`jobs.state + jobs.created_at` (claim order), `cache.expires_at` TTL.

Size budget: about 0.6 KB metadata + 1.3 KB embedding per track. 100k
tracks is roughly 190 MB plus indexes, inside the shared 512 MB cluster.

### 5.2 Store API

`store.py` keeps every existing function name and signature so `app.py`
changes are limited to removing snapshot branches:
`put_track`, `put_track_meta`, `get_track`, `get_many_tracks`,
`get_features`, `get_many_features`, `corpus_size`, `corpus_ids`,
`enqueue_embed`, `dequeue_job`, `clear_embed_marker`, `get_attribution`,
`put_attribution`, `enqueue_attribution`, `clear_attribution_marker`,
`get_cached_preview`, `put_cached_preview`.

Two additions:

- `base_matrix() -> (ids, float32 matrix)` loads every analyzed track's
  embedding, dequantized, in one query projecting only `embedding` and
  `scale`. Called once at API startup.
- `tracks_since(ts) -> (ids, rows)` returns tracks analyzed after `ts`, so
  the API's matrix cache tops itself up every 30 s instead of re-reading
  the corpus. The existing `corpus_size()` change-detection is replaced by
  this timestamp watermark.

`dequeue_job` becomes an atomic `find_one_and_update` from `queued` to
`running` setting `claimed_at`; a sweeper in the worker returns jobs stuck
in `running` for over 10 minutes to `queued` with `attempts + 1`, and marks
them `failed` after 3 attempts. `/seed` polls the job document instead of
Redis.

Connection: `MONGODB_URI` env var, read lazily, one `MongoClient` per
process. `snapshot.py` and all `REDIS_URL` handling are deleted.

### 5.3 Testing

`tests/server/` swaps the current FakeRedis for `mongomock`. The existing
tests for `/search`, `/seed`, `/recommend`, and `/viz/*` keep their
assertions; only fixtures change. A new test covers the job state machine
(claim, retry, fail) and the matrix watermark refresh.

## 6. Sub-project 3: VM deployment

Folder: `deploy/`, plus `ios/Hackathon/Hackathon/Networking/AppConfig.swift`.

- `deploy/Dockerfile`: `python:3.11-slim` on arm64, installs ffmpeg,
  the project, and `tensorflow` (aarch64 wheel). Runs
  `scripts/fetch_models.py` at build time so the image carries the EffNet
  graph. One image, two commands (api, worker).
- `deploy/compose.yml`: services `api` (uvicorn, 2 workers, port 8000
  internal), `worker` (`python -m music_recommendations.worker`), `caddy`
  (ports 80/443, volume for certificates). `env_file: .env` supplies
  `MONGODB_URI` and `PUBLIC_BASE_URL=https://essentia.gabeyocum.com`.
  `restart: unless-stopped` on all three.
- `deploy/Caddyfile`: `essentia.gabeyocum.com` serves `/api/*` to `api:8000`
  with the prefix stripped, everything else from the built web app with
  SPA fallback to `index.html`.
- `deploy/bootstrap.sh`: run once on the VM as `ubuntu`. Fixes the firewall
  (moves the 80/443 ACCEPT rules above the REJECT rule and persists them),
  clones the repo to `~/essentia`, prompts for the Atlas URI into `.env`,
  and runs `docker compose up -d --build`. The Oracle security list must
  allow 80 and 443; that is a console step the script prints a reminder for.
- Worker (`src/music_recommendations/worker.py`, new): one loop that
  claims jobs (embed first, then attr), downloads a fresh Deezer preview,
  runs `analyze_track`, and writes the int8 vector. When the queue is empty
  it runs one crawl step: the existing corpus crawler logic from
  `corpus/` walks Deezer jazz playlists and artists and enqueues unseen
  tracks, rate-limited to stay under Deezer's public API limits. Crawl
  progress lives in a `crawl` document in `cache` (no TTL) so a restart
  resumes.
- iOS: `AppConfig.swift` default base URL becomes
  `https://essentia.gabeyocum.com/api`. The ATS exception for the old IP is
  removed since the domain has a real certificate.
- `scripts/push_tracks.py`, `embed_worker.py`, `embed_worker_run.sh`,
  `export_snapshot.py`, `seed_tracks.py` are deleted. `README.md` and the
  design spec's hosting sections are updated to describe the VM.

Cutover: the old Redis on the retired VM is unreachable, so there is no
data migration. The corpus starts from the 30 fixture tracks (seeded by the
worker on first boot if `tracks` is empty) and grows by crawling.

## 7. Sub-project 4: web app

Folder: `web/`. React 18 + TypeScript, Vite, no UI framework. `deploy/Dockerfile`
gains a `node` build stage that produces `web/dist`, copied into a `caddy`
image; that is the `caddy` service in compose and the only thing serving the
web app.

Screens, mirroring iOS:

1. Search: query box, results list from `/search`, tap to seed.
2. Seed: calls `/seed`, shows progress until `status: "ready"`.
3. Recommendations: axis toggle ("More sounds like this" / "Nothing like
   this"), results from `/recommend`, an `<audio>` player fed by
   `/preview/{id}`.
4. Insights: the galaxy map (`/viz/map`) and topology (`/viz/mst`) first,
   drawn on `<canvas>`; walk, histogram, hubs, tour, attribute, extremes
   follow in that order in later plans.

A small `api.ts` client mirrors `APIClient.swift`. Vitest for the client and
the ranking-presentation helpers; no browser end-to-end suite.

## 8. Error handling

- Deezer preview 403 (expired signature): the worker re-fetches via
  `/track/{id}` before download, as the old worker did.
- Analysis failure: job marked `failed` with the error string; `/seed`
  returns 502 with `{"error": "analysis failed"}`; the track document stays
  without an embedding and is skipped by `base_matrix`.
- Atlas unreachable at API startup: the API serves `/search` and `/axes`
  and returns 503 on `/seed` and `/recommend` until a background retry
  connects. It never crashes on boot.
- Atlas storage nearing quota: the crawler stops enqueuing when
  `tracks.countDocuments()` exceeds `CORPUS_CAP` (env, default 100000).

## 9. Order of work and gates

1. Analysis on ARM. Gate: parity test passes on the Mac; the Docker image
   analyzes a fixture track on the VM.
2. Atlas store. Gate: pytest green with mongomock; `/recommend` returns the
   same ranking for a fixture seed as before the swap.
3. Deployment. Gate: `https://essentia.gabeyocum.com/api/axes` responds;
   the iOS app searches, seeds, and recommends against it; the worker has
   embedded the fixture set.
4. Web app. Gate: the four screens work in a browser against the live API.

Each step gets its own implementation plan under `docs/superpowers/plans/`.
