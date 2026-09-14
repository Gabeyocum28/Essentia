# Task 2 report — pluggable music sources (Deezer dev, Jamendo shippable)

Worktree: `/Users/gabrielyocum/Projects/Essentia/.claude/worktrees/agent-a4f462144a03a81b2`
Branch: `agent-a4f462144a03a81b2` (see commit below)

## What changed, per file

### New — `src/music_recommendations/corpus/sources/`
- **`base.py`** — `Source` Protocol (`name`, `owns`, `search`, `track`,
  `preview_url`, `candidates(step)`, `candidate_label()`, `attribution`),
  `split_id()` (bare id → deezer; `"<name>:<id>"` → that name), and a
  `BaseSource` mixin carrying id qualification, the crawl label and
  `enabled()`/`disabled_reason()`.
  `candidate_label()` was added to the Protocol so the worker can keep its
  existing `[worker] crawl <label>: ...` line without knowing any source's
  internals.
- **`__init__.py`** — the registry. `active()` reads `SOURCES` (comma list,
  default `deezer`), raises `ValueError` on an unknown name, drops a
  configured-but-unusable source with a one-time log line, and raises
  `RuntimeError` if that source was the only one. `for_id()` resolves an id
  to its source whether or not it is switched on (a corpus id must stay
  playable after a `SOURCES` change) and returns `None` for an unknown
  namespace. Instances are cached per name (sources hold rate-limit and
  label state). `reset()` for tests.
- **`deezer.py`** — `DeezerSource`, wrapping `server/deezer.py` (search /
  get_track / fresh_preview_url) and `corpus/crawl.py`. The chart /
  snowball / deep-cuts rotation and `_grow_roots` moved here verbatim from
  `worker.crawl_step`; `step // 3` still indexes within each arm. Ids stay
  bare. `attribution()` returns `None` — Deezer is a development source and
  a "via Deezer" credit on every row is noise the licence does not ask for.
- **`jamendo.py`** — `JamendoSource` against v3. One HTTP choke point,
  `_get(path, **params)`, which adds `client_id`/`format=json`, holds a
  0.2 s delay under a lock, and returns `{}` rather than raising. Mapping
  `name/artist_name/album_name/image|album_image/audio`, ids
  `jamendo:<id>`, attribution `{"source","url"(shareurl),"license"(license_ccurl)}`
  — omitted when there is no `shareurl`, since a credit with no link back is
  not attribution. `candidates(step)` rotates the ten tags by
  `popularity_total` then `featured=1`. `enabled()` is false without
  `JAMENDO_CLIENT_ID`.

### Modified
- **`contract/features.py`** — `TRACK_OPTIONAL_FIELDS = frozenset({"source",
  "attribution_url"})`, documented as absent-not-null.
- **`server/store.py`** — `_source_of()` (explicit `source` wins, else read
  off the id, else `deezer`); `_meta()` stores `source` always and
  `attribution` when present; `_contract()` publishes `source` and
  `attribution_url` (from `attribution.url`) only when present, never the
  whole attribution object; `_TRACK_PROJECTION` replaces the two inline
  projections so both Track reads carry the new fields.
- **`server/app.py`** — dropped the direct `deezer` import for
  `corpus.sources`. New `_public_track()` narrows a source's Track to the
  contract shape plus the two optional keys (search results never pass
  through the store, so this is where that narrowing happens);
  `_remember_track` now builds its cache entry through it, so the metadata
  cache no longer strips the credit. `/search` fans out over `active()` in
  order, tolerates a single source failing, and falls back to the fixture
  only when every source failed. `_fresh_preview` and therefore
  `/preview/{id}` and `/preview/{id}/audio` route via `sources.for_id`.
  `/seed`'s cold lookup goes through the new `_from_source`.
- **`worker.py`** — `_fresh_track` resolves via `sources.for_id(...).track`;
  `crawl_step` round-robins `active()` and calls
  `candidates(step // len(sources))`, keeping the cap / queue-depth /
  byte-cap guards, the cursor, the backoff and the log line verbatim;
  `_grow_roots` and the `crawl` import moved out to the Deezer source;
  `main()` prints the active sources (unguarded, so a bad `SOURCES` stops
  the process instead of crawling nothing forever).
- **`web/src/api/types.ts`** — `Track` gains `source?: string;
  attribution_url?: string | null;`.
- **`web/src/components/Attribution.tsx`** (new) — `licenceLabel()` derives
  e.g. `CC BY-SA 3.0` from the deed path and returns `null` when the URL is
  not a deed (guessing a licence would be a false claim); `<Attribution>`
  renders a muted `via Jamendo · CC BY-SA 3.0` backlink, or nothing when
  there is no `attribution_url`.
- **`web/src/components/TrackRow.tsx`** — the credit is a sibling of the
  select button inside a new `.track-row-body` wrapper (an `<a>` inside a
  `<button>` is invalid HTML).
- **`web/src/styles.css`** — `.track-row-body`, `.track-row-attribution`.

### Tests
- **New** `tests/corpus/test_sources.py` (23): registry/ordering/unknown-name,
  instance reuse, `for_id` including a switched-off source and an unknown
  namespace, Jamendo disabled-without-key (and refusing when it is the only
  source), Jamendo field mapping / missing audio / missing shareurl /
  album-image fallback / tag-then-featured rotation / `_get` returning `{}`
  on a dead network, Deezer tagging, Deezer no-attribution, the three-arm
  rotation with `step // 3`, and root growth.
- **New** `web/src/components/TrackRow.test.tsx` (4).
- `tests/test_contract.py` — asserts `TRACK_OPTIONAL_FIELDS` and that it is
  disjoint from `TRACK_FIELDS`.
- `tests/server/test_store.py` — attribution round-trip, licence kept on the
  row but not published, bare-id default, source read off a namespaced id,
  `get_many_tracks` carrying the optional keys. Existing exact-equality
  assertions updated for the new `source` key.
- `tests/server/test_app.py` — a `_contract_shape()` helper replaces the four
  exact-set assertions (exact required set, plus a subset of the optional
  fields — internal fields like `dedupe_key` still cannot leak); new tests
  for `/search` fan-out order, the attribution backlink in search results,
  one source being down, `/preview` routing to the owning source, a 404 for
  an unknown namespace, and a Jamendo track's credit surviving into
  `/recommend`. `app_module.deezer` → a direct `server.deezer` import.
- `tests/server/test_worker.py` — new round-robin, label-logging,
  no-active-sources and `_fresh_track`-routing tests; `worker.deezer` /
  `worker.crawl` / `worker._grow_roots` re-pointed at the modules that own
  them now.

## RED / GREEN

RED: after the source package and the store/app/worker rewiring, 15 failures
— 6 exact-Track-shape assertions that had not been told about the new
`source` key, 1 search stub with the wrong arity, 6 worker tests whose
patch targets had moved, plus the 2 pre-existing model-file failures.
GREEN after updating those expectations and adding the new suites.

## Counts

- Python: `397 passed, 15 skipped, 2 failed`.
  The two failures are **pre-existing and environmental**:
  `tests/analysis/test_embedding.py::test_embed_patch_groups_*` need
  `models/discogs-effnet-bs64-1.pb`, and `models/` is gitignored so it is
  not present in this worktree. Nothing in this change touches `analysis/`.
  (Baseline before this change: the same 2 failures.)
- Web: `26 files, 153 tests passed`; `npm run build` clean.

## Concerns

1. **`contract/` was edited.** The worktree's `AGENTS.md` (solo-project
   version) allows it as a cross-cutting change made in one PR; the task
   brief asked for `TRACK_OPTIONAL_FIELDS` explicitly. The iOS client was
   NOT updated — it decodes `Track` and will simply ignore two new optional
   keys, but it will therefore show no attribution. If Jamendo is ever
   switched on for the iPhone app, that is a licence problem and iOS needs
   the same change as `TrackRow`.
2. **The brief and the task spec disagreed on Deezer attribution.** The
   brief said Deezer should emit `https://www.deezer.com/track/<id>`; the
   task requirements said the web must show nothing for Deezer and a Deezer
   track must have no `attribution_url`. I followed the task requirements:
   `DeezerSource.attribution()` returns `None`.
3. **Jamendo has never been run against the real API** — there is no
   `JAMENDO_CLIENT_ID` here, so every Jamendo test stubs `_get`. The field
   names and endpoints come from the brief, not from a live response; the
   first real crawl may need mapping corrections.
4. **Jamendo audio is a full track, not a 30 s preview.** `download_preview`
   in the worker fetches the whole file into a temp mp3 before analysis
   reads its first 30 s — for a 6-minute CC track that is ~6x the bytes and
   the download time of a Deezer preview. Worth a range request or a
   duration cap before a large Jamendo crawl.
5. **Preview caching TTL** is still the Deezer-shaped 10 minutes for every
   source, even though Jamendo URLs do not expire. Harmless, just wasteful.
6. **No deploy/env wiring** — `SOURCES` and `JAMENDO_CLIENT_ID` are not in
   `deploy/`; left alone deliberately (out of scope, and the VM is off
   limits for this task).
