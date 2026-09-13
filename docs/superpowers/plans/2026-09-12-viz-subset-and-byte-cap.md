# Viz Subset and Byte Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the corpus grow until the Atlas database holds 450 MB, while the insights endpoints keep working by computing over a bounded, seed-anchored subset of tracks.

**Architecture:** The crawler stops when `dataSize + indexSize` of the database reaches `CORPUS_BYTES_CAP` (default 450 MB), read from `dbStats`; the track cap becomes a distant backstop. Recommendations and the histogram keep using the full corpus (one matrix-vector product). Every other insights endpoint runs on a "viz subset": the `VIZ_MAX` (default 8000) tracks most similar to a seed, plus the seed and, for `/viz/map`, that axis's recommendations. `/viz/tour`, `/viz/mst`, `/viz/hubs`, `/viz/extremes` gain an optional `track_id` query parameter selecting that subset; without it they use the first `VIZ_MAX` rows of the corpus matrix (a stable fallback for old clients). Subsets are cached per seed (small LRU) and the existing identity-pinned caches (top-8 PCA, MST, pairwise) key on the subset matrix, so nothing else changes. Memory is trimmed: the unit-normalized full matrix and the pairwise matrix become float32.

**Tech Stack:** FastAPI, numpy, pymongo, mongomock; React client.

**Spec:** user decision 2026-09-12 ("200 MB cap, rework insights to a subset"); memory numbers below.

## Global Constraints

- Public HTTP contract unchanged; the four global viz endpoints only gain an optional `track_id`.
- Env: `CORPUS_BYTES_CAP` (bytes, default `471859200` = 450 MB), `VIZ_MAX` (default `8000`), `CORPUS_CAP` (default `300000`, backstop).
- Memory budget at 100k tracks inside the 4 GB API container: base matrix float32 512 MB; unit matrix float32 512 MB; up to 2 cached subsets × (8000×1280 float32 = 41 MB); pairwise float32 per subset 256 MB, at most 2 held. Total under 1.7 GB plus TensorFlow.
- Branch `viz-subset` off `main`. `python3 -m pytest -q` green before each commit; web: `npm test && npm run build`.

## Facts

- `app.py`: `_viz_embedding_corpus()` returns `(ids, matrix)` for the whole corpus via `_corpus_matrix`; `_viz_embedding_corpus_min2()` wraps it for tour/mst/extremes; `_top8`/`_mst` caches hold `(matrix, …)` and compare by identity; `viz.pairwise_cosine` caches one `(matrix, similarity)` pair and computes in float64; `_UNIT_CACHE[feature_key] = (id(matrix), unit)` with `unit` float64; `_similarity(feature_key, matrix, seed_vec, metric)` gives seed-vs-all cosine using that cache; `viz_hubs` uses the full pairwise; `viz_walk` uses `viz.shortest_walk` over a k-NN graph from pairwise; `viz_map` calls the recommend machinery for the recs then projects the whole matrix.
- `worker.crawl_step` guards on `store.corpus_size() >= CORPUS_CAP or store.queued_count() >= MAX_QUEUED`.
- mongomock does not implement `db.command("dbStats")` (raises `NotImplementedError`).
- Web client: `api.vizHubs()`, `vizTour()`, `vizMst()`, `vizExtremes(pc, limit)` take no seed; callers are `Proof.tsx`, `Tour.tsx`, `Topology.tsx`, `Extremes.tsx`, all rendered under `Insights` which knows the seed `id`.

---

### Task 1: Server subset, float32 caches, byte cap

**Files:**
- Modify: `src/music_recommendations/server/app.py`, `src/music_recommendations/server/viz.py`, `src/music_recommendations/server/store.py`, `src/music_recommendations/worker.py`, `deploy/.env.example`
- Test: `tests/server/test_viz.py`, `tests/server/test_store.py`, `tests/server/test_worker.py`

**Interfaces:**
- Produces: `store.data_size_bytes() -> int`; `app.VIZ_MAX`; `app._viz_subset(seed_id: str | None, extra_ids: list[str] = ()) -> (ids, matrix)`; query param `track_id: str | None = None` on `/viz/tour`, `/viz/mst`, `/viz/hubs`, `/viz/extremes`; worker `CORPUS_BYTES_CAP`.

- [ ] **Step 1: Store byte size (test first)**

Append to `tests/server/test_store.py`:

```python
def test_data_size_bytes_falls_back_when_dbstats_is_unsupported(fake_mongo):
    store.put_track(TRACK, FEATURES)
    assert store.data_size_bytes() >= 1


def test_data_size_bytes_uses_dbstats_when_available(fake_mongo, monkeypatch):
    monkeypatch.setattr(fake_mongo, "command", lambda name: {"dataSize": 1000, "indexSize": 24})
    assert store.data_size_bytes() == 1024
```

Implement in `store.py`:

```python
def data_size_bytes() -> int:
    """Data plus index bytes for the database (what Atlas counts against the
    free tier). mongomock has no dbStats; estimate from document counts."""
    try:
        stats = db().command("dbStats")
        return int(stats.get("dataSize", 0)) + int(stats.get("indexSize", 0))
    except (NotImplementedError, TypeError):
        n = db().tracks.count_documents({})
        return n * 2048
```

- [ ] **Step 2: Worker byte cap (test first)**

In `tests/server/test_worker.py` add:

```python
def test_crawl_step_stops_at_the_byte_cap(fake_mongo, monkeypatch):
    monkeypatch.setattr(worker, "CORPUS_BYTES_CAP", 10)
    monkeypatch.setattr(worker.store, "data_size_bytes", lambda: 11)
    monkeypatch.setattr(worker.crawl, "from_charts", lambda *a, **k: _tracks(["1"]))
    assert worker.crawl_step() == 0
```

`worker.py`: `CORPUS_BYTES_CAP = int(os.environ.get("CORPUS_BYTES_CAP", str(200 * 1024 * 1024)))`; `CORPUS_CAP` default becomes `"200000"`; in `crawl_step` the guard becomes

```python
    size = store.data_size_bytes()
    if size >= CORPUS_BYTES_CAP or store.corpus_size() >= CORPUS_CAP \
            or store.queued_count() >= MAX_QUEUED:
        if size >= CORPUS_BYTES_CAP:
            print(f"[worker] byte cap reached: {size/1e6:.1f} MB of {CORPUS_BYTES_CAP/1e6:.0f} MB", flush=True)
        return 0
```

and the per-step log line appends `f"  db {size/1e6:.1f} MB"`.

`deploy/.env.example`: replace the `CORPUS_CAP` block with:

```
# Stop crawling when the Atlas database (data + indexes) reaches this many
# bytes. 200 MB leaves the rest of the free 512 MB cluster for other data.
CORPUS_BYTES_CAP=209715200
# Backstop track count; the byte cap normally wins.
CORPUS_CAP=200000
# The insights endpoints compute over at most this many tracks nearest the
# seed (dense n×n similarity: 8000 rows = 256 MB float32).
VIZ_MAX=8000
```

- [ ] **Step 3: float32 caches**

`viz.normalized_rows`: `np.asarray(matrix, dtype=np.float32)`; `pairwise_cosine` unchanged otherwise (result float32). `app._similarity`: build `unit` as float32 (`rank_mod.normalize(np.asarray(matrix, dtype=np.float32))`), and the seed vector likewise. Check `rank_mod.normalize` keeps dtype; if it upcasts, cast the result. Existing tests assert numeric closeness with tolerances; if any asserts exact float64 equality, loosen to `np.allclose(..., atol=1e-5)` and say so.

- [ ] **Step 4: The subset (tests first)**

Append to `tests/server/test_viz.py` (use the file's existing helpers for seeding a fake corpus with `fake_mongo` and its `client`; adapt names):

```python
def test_viz_subset_is_seed_nearest_and_bounded(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 3)
    # five tracks on a line; seed "c" in the middle
    vecs = {"a": [1, 0], "b": [0.9, 0.1], "c": [0.7, 0.3], "d": [0.5, 0.5], "e": [0, 1]}
    for tid, v in vecs.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    ids, matrix = app_mod._viz_subset("c")
    assert "c" in ids and len(ids) == 3
    assert set(ids) == {"b", "c", "d"}            # the two nearest plus the seed
    assert matrix.shape == (3, 2)
    assert ids == sorted(ids)                      # stable order
    ids2, _ = app_mod._viz_subset("c", extra_ids=["e"])
    assert set(ids2) == {"b", "c", "d", "e"}       # extras appended even when far


def test_viz_subset_without_seed_is_the_first_rows(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid in ("a", "b", "c"):
        store.put_track({**TRACK, "track_id": tid}, {"embedding": [1.0, 0.0]})
    ids, matrix = app_mod._viz_subset(None)
    assert ids == ["a", "b"] and matrix.shape == (2, 2)


def test_tour_and_mst_share_the_subset_for_a_seed(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 3)
    for tid, v in {"a": [1, 0], "b": [0.9, 0.1], "c": [0.7, 0.3], "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    tour = client.get("/viz/tour?track_id=a").json()
    mst = client.get("/viz/mst?track_id=a").json()
    assert tour["ids"] == mst["ids"] and len(tour["ids"]) == 3 and "d" not in tour["ids"]
    hubs = client.get("/viz/hubs?track_id=a").json()
    assert {h["track_id"] for h in hubs["all_counts"]} == set(tour["ids"])


def test_map_subset_includes_surprise_recs(fake_mongo, monkeypatch):
    from music_recommendations.server import app as app_mod
    monkeypatch.setattr(app_mod, "VIZ_MAX", 2)
    for tid, v in {"a": [1, 0], "b": [0.99, 0.01], "c": [0.98, 0.02], "d": [0, 1]}.items():
        store.put_track({**TRACK, "track_id": tid}, {"embedding": v})
    body = client.get("/viz/map?track_id=a&axis=surprise&limit=1&correction=off").json()
    rec_ids = {r["track_id"] for r in body["recs"]}
    assert rec_ids <= set(body["points"]["ids"])   # far recs are still drawn
```

Implement in `app.py`:

```python
VIZ_MAX = int(os.environ.get("VIZ_MAX", "8000"))
_SUBSET_CACHE: "OrderedDict[tuple[int, str | None, tuple[str, ...]], tuple[list[str], np.ndarray]]" = OrderedDict()
_SUBSET_KEEP = 2


def _viz_subset(seed_id: str | None, extra_ids: "list[str] | tuple[str, ...]" = ()) -> tuple[list[str], np.ndarray]:
    """At most VIZ_MAX rows of the corpus for the insights endpoints.

    With a seed: the seed plus its VIZ_MAX-1 nearest tracks by cosine over the
    FULL matrix (one matrix-vector product), plus any extra ids the caller
    needs drawn (an axis's recs, which for `surprise` are far away). Without
    a seed: the first VIZ_MAX rows, for clients that predate the parameter.
    Row order is the full matrix's order, so tour/mst/hubs agree on indices.
    The dense n×n work downstream happens on this copy, never on the corpus.
    """
    ids_all, matrix_all = _viz_embedding_corpus()
    extra = tuple(t for t in extra_ids if t in set(ids_all))
    key = (id(matrix_all), seed_id, extra)
    cached = _SUBSET_CACHE.get(key)
    if cached is not None and len(cached[0]) <= len(ids_all):
        _SUBSET_CACHE.move_to_end(key)
        return cached
    n = len(ids_all)
    if seed_id is None or seed_id not in ids_all:
        rows = np.arange(min(n, VIZ_MAX))
    else:
        seed_row = ids_all.index(seed_id)
        sims = _similarity("embedding", matrix_all, matrix_all[seed_row], "cosine")
        keep = min(n, VIZ_MAX)
        rows = np.argpartition(-sims, keep - 1)[:keep] if keep < n else np.arange(n)
        rows = set(rows.tolist()) | {seed_row}
        rows = np.array(sorted(rows))
    index = {t: i for i, t in enumerate(ids_all)}
    wanted = set(rows.tolist()) | {index[t] for t in extra}
    rows = np.array(sorted(wanted))
    subset = (list(np.array(ids_all, dtype=object)[rows]), np.ascontiguousarray(matrix_all[rows]))
    _SUBSET_CACHE[key] = subset
    while len(_SUBSET_CACHE) > _SUBSET_KEEP:
        _SUBSET_CACHE.popitem(last=False)
    return subset
```

Then route changes:
- `viz_tour`, `viz_mst`, `viz_extremes`: add `track_id: str | None = None`; replace `_viz_embedding_corpus_min2()` with `_viz_subset_min2(track_id)` (same 404 rule, wrapping `_viz_subset`).
- `viz_hubs`: add `track_id: str | None = None`; use `_viz_subset(track_id)`.
- `viz_walk`: use `_viz_subset(from_, extra_ids=[to])` so both endpoints are present; the walk is over the subset graph.
- `viz_map`: compute recs as today (full corpus), then `ids, matrix = _viz_subset(track_id, extra_ids=[r["track_id"] for r in recs])` for the points and projection.
- `viz_histogram`: unchanged (seed vs full corpus, O(n)).
- The `clear_matrix_cache` fixture in `tests/server/conftest.py` must also clear `app._SUBSET_CACHE`.
- The `_top8`/`_mst` caches currently store one entry under `"embedding"`; change them to dicts keyed by `id(matrix)` holding `(matrix, …)` and trimmed to 2 entries so two seeds alternate without thrashing (the subset LRU keeps those matrices alive).

Also `pairwise_cosine` keeps its single-entry cache (it is identity-pinned; a second seed recomputes 0.3 s at 8000 rows). Fine.

- [ ] **Step 5: Suite, commit**

`python3 -m pytest -q` green. Commit: `feat(server,worker): seed-anchored viz subset, float32 caches, and an Atlas byte cap for the crawler`.

---

### Task 2: Web client passes the seed; redeploy

**Files:**
- Modify: `web/src/api/client.ts`, `web/src/api/client.test.ts`, `web/src/insights/Proof.tsx`, `Tour.tsx`, `Topology.tsx`, `Extremes.tsx`, `web/src/screens/Insights.tsx` (pass `id` down), README web/deploy notes.

- [ ] **Step 1: Client**

`vizHubs(track_id?: string)`, `vizTour(track_id?: string)`, `vizMst(track_id?: string)`, `vizExtremes(pc, limit = 4, track_id?: string)` add the query param when given (via the existing `q()` helper, which drops undefined). Tests: each builds the URL with `track_id` when provided and without it otherwise. Pass the seed id from `Insights` into `Tour`, `Topology`, `Proof`, `Extremes` as a `seedId` prop and forward it. Adjust existing component tests' mocks (`toHaveBeenCalledWith` now includes the id).

- [ ] **Step 2: Verify, commit, redeploy**

`npm test && npm run build`; `python3 -m pytest -q`. Commit `feat(web): insights views ask for the seed's subset`. Push branch, open a PR against main titled "Viz subset and 200 MB byte cap". On the VM: `cd ~/stacks/essentia && git fetch && git checkout viz-subset && git pull --ff-only`, then edit nothing in `.env` (the new variables have defaults), `docker compose -f deploy/docker-compose.yml up -d --build`, verify `/api/axes`, `/api/viz/tour?track_id=2711778` returns ≤ VIZ_MAX ids, the worker log shows `db N MB` on its next crawl line, other sites 200. Never print `.env`.

## Self-review

- Coverage: byte cap (Task 1 Step 2), subset for every dense endpoint (Step 4), float32 memory (Step 3), client passes the seed (Task 2), env documented. Old clients (iOS) keep working via the no-seed fallback.
- Placeholders: none. Type consistency: `_viz_subset(seed_id, extra_ids)` used by map/walk/hubs/tour/mst/extremes; `data_size_bytes` used by worker; `VIZ_MAX` monkeypatched in tests as a module attribute (read at call time, not import time — implement it as a module global read inside the function).
