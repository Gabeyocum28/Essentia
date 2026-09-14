# Task 4 report — cutover: re-analysis arm, v1 deleted, TensorFlow out of the image

Branch: `clean-room` (main checkout), one commit off `042d7f0`. Not pushed, no
PR, nothing deployed — the third brief bullet (deploy/PR) is deliberately not
done here.

## 1. The re-analysis arm

### `src/music_recommendations/worker.py`

- **`reanalyze_step() -> int`** — takes `store.stale_ids(GROUP_SIZE)` (live
  rows below `FEATURES_VERSION`, oldest `analyzed_at` first), downloads each
  preview in parallel on `DOWNLOAD_THREADS`, runs the group through one
  `analyze_tracks` call, and `store.put_track`s each result. Logs
  `[worker] reanalyzed N, remaining M (Xs for K)` in a `finally`, so the line
  appears even when the group blew up. Never raises: a store error on
  `stale_ids` is caught and returns 0.
- **`_reanalyze_one(track_id)`** (pool thread) reads metadata from
  `store.get_track(id)` and fetches only the AUDIO via
  `sources.for_id(id).preview_url(id)`. Deliberately not `source.track(id)`:
  a re-analysis is not a re-crawl, and the title/artist/attribution written
  at crawl time must survive it. `put_track` re-`$set`s only the fields
  `_meta` produces, and `attribution` is absent from that dict when the
  contract-shaped track has none, so the stored attribution is untouched
  (covered by `test_reanalyze_keeps_the_metadata_it_already_had`).
- **Tick order** (`_tick`): `requeue_stale` → `reanalyze_step()` →
  claim/process embed+attribution jobs → crawl. One group per tick, not a
  drain loop: with 19k stale rows a drain loop would stall every `/seed` for
  a day, and `/seed` only waits 20 s. Crawling is skipped on any tick that
  re-analyzed something (the rotation cursor lives in Atlas, so nothing is
  lost). The dedupe guard in `_enqueue_new` is untouched.

### How a re-analysis failure is tracked

`store.fail_reanalysis(track_id, error)` sets **`reanalysis_failed_at`**
(server `$currentDate`) and `reanalysis_error` on the TRACK document, and
`store._STALE` now excludes rows that have `reanalysis_failed_at`. That is the
whole mechanism:

- the queue is oldest-first, so without it one dead preview sits at the head
  forever and starves everything behind it;
- the row keeps its old vectors and metadata — still a playable seed by id,
  just out of `LIVE`, `stale_ids` and `stale_count`;
- `put_track` `$unset`s both fields on any successful store, so a row that
  becomes analyzable again (or the next version bump) is not hidden forever;
- `store.reanalysis_failed_count()` exists for ops visibility.

Failure paths that hit it: no metadata in the store, no source owns the id, the
source has no preview URL any more, the download raised, `analyze_tracks` put
an exception in that path's slot, or `put_track` itself failed.

## 2. v1 deleted

Removed: `analysis/frontend.py`, `analysis/embedding.py`, the eleven-head
`analysis/feel.py`, `registry.py`'s EffNet/heads block (and its `NamedTuple`
import), `analyze_tracks_v1`/`analyze_track_v1` and `_warn_missing_heads` from
`analysis/__init__.py`, `scripts/feel_backfill.py` (the arm IS the backfill),
`fetch_models.fetch_v1()`, `tests/analysis/test_parity.py`,
`parity_baseline.json`, `test_embedding.py`, `test_frontend.py`,
`test_feel.py`, `tests/server/test_feel_backfill.py`, and the
`needs_essentia`/`needs_effnet`/`run_essentia` machinery in
`tests/analysis/conftest.py`.

`analysis/feel_v2.py` → `analysis/feel.py` (git mv), with every importer
updated: `analysis/v2.py`, `server/app.py`, `server/store.py` docstring,
`contract/features.py`, `tests/test_contract.py`, `tests/server/test_viz.py`,
`tests/analysis/test_v2.py`, `web/src/insights/MathPanel.test.tsx`.

`tests/analysis/test_analyze.py` is rewritten as the removal guard: the v1
modules must not be importable, `feel_v2` must not resolve, `feel` must not
have `feel_vectors`, no `import tensorflow`/`import essentia` anywhere in
`src/`, and importing the package must not pull torch or librosa into
`sys.modules`.

**`process_attribution` was ported, not deleted.** It used `embedding.py`'s
EffNet. It now decodes with `v2.decode` (44.1 kHz float) and embeds with
`clap.embed_audio([samples])`; `_write_wav` is gone (no temp wav, no 16 kHz
resample), `_embed_waveform(samples)` lost its module argument. The band edges
in `viz.py` are unchanged on purpose — the phone's band-solo filter is their
exact complement, so moving them is a lockstep client change; the stale "EffNet
is 16 kHz" comment now says that instead.

`grep -rn "tensorflow\|essentia\b\|effnet\|frontend\.\|feel_v2" src scripts
tests deploy pyproject.toml docs README.md` comes back with only: the project's
own name (`essentia` the database, the containers, the domain), the historical
paragraph in `analysis/CLAUDE.md`, and the new guard tests that name what must
stay gone.

## 3. Packaging and deploy

- **`pyproject.toml`** — `tensorflow` dropped; the analysis extra is exactly
  `torch, msclap, librosa, pyloudnorm, soxr, beat_this`. Description no longer
  says "Essentia analysis". **`uv.lock` regenerated** (`uv lock`) — zero
  `tensorflow` lines left.
- **`deploy/Dockerfile`** — apt line is now `ffmpeg git libsndfile1 curl`
  (**finding 1**: `git` is required to `pip install beat_this @ git+…`);
  `TF_CPP_MIN_LOG_LEVEL` gone; `python3 scripts/fetch_models.py` bakes only
  `models/v2`.
- **`deploy/docker-compose.yml`** — `TF_CPP_MIN_LOG_LEVEL` removed from both
  services; api `6g → 5g`, worker `3g → 4g`, with the reason (only the worker
  loads the audio tower) written next to each.
- **`deploy/.env.example`** — `SOURCES=deezer` and an empty
  `JAMENDO_CLIENT_ID=` with comments (Deezer dev-only, Jamendo shippable,
  monthly quota), a re-analysis paragraph, and GROUP_SIZE re-explained for
  CLAP. No Atlas URI anywhere; a test asserts `mongodb+srv://` never appears.
- **`tests/deploy/test_compose.py`** — new cases for the mem limits, the `git`
  apt entry, the absence of TensorFlow, and the two new env keys.
- **`docs/THIRD_PARTY.md`** — EffNet and TensorFlow rows deleted; Beat This!
  weights row now says code MIT / weights CC BY 4.0 with the upstream link and
  the attribution; ffmpeg row corrected to the **GPL** Debian build with a
  section explaining why it does not propagate (separate process, source
  available, unmodified); a recorded **LGPL decision** section for
  soxr/libsndfile/ffmpeg (dynamically linked, unmodified, never patched, never
  static, notices + source offer shipped). Jamendo row added.
- **`tests/test_licences.py`** — `NC_ALLOWED` is now `{}`; the "only tolerated
  NC component" test became `test_nothing_non_commercial_is_tolerated_any_more`
  asserting both the empty set and zero NC rows; the extra test now asserts
  **equality** with the v2 list, plus a new test that `tensorflow` appears in
  neither `pyproject.toml`, `uv.lock`, nor the Dockerfile.
- **`README.md`** — Setup split (dev extra vs analysis extra), new "Analysis"
  section (CLAP / feel / rhythm / FEATURES_VERSION and the re-analysis arm /
  the API never analyzes), new "Sources and licensing" section (SOURCES,
  Deezer dev-only, Jamendo + attribution, the register), feel-slider section
  rewritten for eight z-scored axes plus the tempo slider, parity-test
  instructions replaced.

## 4. Review findings

| # | Status |
|---|---|
| 1 | Done — `git` (and `libsndfile1`) in the apt line; TF gone. |
| 2 | Done — `/seed` NEVER analyzes. `_ANALYZE_SEM`, `_fetch_preview_audio`, `_download_preview`, `_to_plain` and the `analyze_track`/`frontend` imports are deleted; an unknown seed always goes through `_seed_via_worker`. No 502 path remains. Tests assert `app_module` has no analyzer attribute and that `/seed` opens no socket. |
| 3 | Done — `offset=(step // arms) * PER_PAGE` on both arms (label carries `+N`), envelope `headers.status`/`error_message` logged when not success, transport failures logged by path (never the URL — it carries the client id), `audioformat=mp32` pinned in `_get` as `AUDIO_FORMAT` with a note beside `FEATURES_VERSION` in `analysis/schema.py`. |
| 4 | Done — both sides. `worker.download_preview` and `app._open_upstream` send `Range: bytes=0-1048575`, and both cap the bytes they read (the proxy counts as it streams and drops a `Content-Length` larger than the cap so the `<audio>` element is not left waiting). |
| 5 | Done, **unbuilt** — see §5. |
| 6 | Done — see the THIRD_PARTY bullets above. |
| 7 | Done — `tests/analysis/test_v2.py` no longer hardcodes a scratchpad path; a module fixture downloads `contract/fixture.json`'s first track and skips on any failure. |
| 8 | Done — `rhythm.py` says 81 MB. |
| 9 | Done — a rhythm failure now keeps `embedding` + `feel` and simply omits `rhythm` (the store already treats it as optional and the tempo term as no-penalty), with a once-per-process warning. |
| 10 | **Not measured** — see §6. |
| 11 | Done — the note beside `SLEEP` says the free quota is monthly, so pacing cannot keep a runaway crawl inside it. |
| 12 | Done — `_RHYTHM_ALIGN_CACHE` holds `(matrix, tempo float32 column, present)` only; the dict list is dropped (`del rows`) as soon as the column is built. The math panel's `rec` rhythm is filled in by a new `_fill_rec_rhythm(recs)` — ONE `get_many_rhythm` query for the ≤ limit rows actually displayed, after the page is chosen. |

## 5. iOS (unbuilt)

`Track.swift` gains `source: String?` and `attributionURL: URL?` with
snake_case `CodingKeys`. They are `var … = nil`, not `let`: a `let` with a
default is not decoded by the synthesized initializer, and without a default
the memberwise `Track(trackID:…)` calls in `Models/VizMap.swift`,
`VizT1.swift`, `VizT2.swift` and two test files would stop compiling.

`Shared/Components.swift` gains `licenceLabel(_ url: URL?) -> String?` (parses
`/licenses/<code>/<version>/` out of the deed URL — mirrors
`web/src/components/Attribution.tsx`) and an `AttributionLine` view rendering
`via Jamendo · CC BY-SA 3.0` as a `Link` with `.buttonStyle(.plain)` so it
opens the backlink without firing the row's own tap target. It renders nothing
when `attributionURL` is nil (the Deezer case). Added to both `TrackRow` and
`TrackHeader`. `TrackDecodingTests.swift` gains four cases.

**This is not compiled.** There is no Xcode toolchain in this environment, so
the Swift is deliberately conservative (no new files, no project-file changes,
no new APIs beyond `Link`/`String.prefix`). It needs a build before the phone
ships, and per finding 5 Jamendo must not be switched on for the phone until
it does.

## 6. Temperature (finding 10) — skipped, with the reason

Not measured. The measurement needs CLAP: `msclap`, `torch` and `librosa` are
all absent from this interpreter (`importlib.util.find_spec` → None for each),
there is no second interpreter or virtualenv in the checkout or beside it, and
`models/v2/` does not exist locally — only the old EffNet/heads files do. Every
v2 test skips here for exactly that reason. So no per-axis distribution over
the fixture previews could be produced, and `TEMPERATURE` stays at 25.0.

This is the one measurement the brief allowed to be skipped if no previews were
reachable; the reachable half (the previews) was fine, the model was not. It
should be run on the VM once the image is built — the check is: analyze ~200
previews, and if more than 30% of any axis's values fall outside [0.05, 0.95],
drop `TEMPERATURE` to 10 (a FEATURES_VERSION bump, since it changes every
number).

## 7. Tests

`PYTHONPATH=$PWD/src python3 -m pytest -q` → **455 passed, 16 skipped**
(baseline 467/17 — the drop is the deleted v1 suites; ~40 new tests were
added). `cd web && npm test -- --run` → **171 passed**; `npm run build` clean.

New coverage: 11 worker tests for the arm (invisible-until-run, one group per
tick, metadata survives, the log line, dead preview marked, undecodable marked,
unknown source marked, mark cleared by a later success, ordering vs the crawl,
store blip), 4 store tests (`fail_reanalysis`, still-playable, clearing,
counting), 3 jamendo tests (paging, `audioformat`, the 200-with-an-error
envelope), 3 `/seed` tests, 2 preview-cap tests on each side, 5 deploy tests,
1 licence test, 4 Swift tests, and the rewritten `test_analyze.py` guards.

## 8. Left undone / concerns

1. **iOS is unbuilt** (§5).
2. **Temperature not measured** (§6).
3. **`models/msd-musicnn-1.pb` + `.json` are still tracked in git** — MSD
   MusiCNN from essentia.upf.edu, i.e. another CC BY-NC-SA model. Nothing
   loads them (only frozen `legacy/` names them), they are not in the image and
   not in the register, so no shipped artifact contains them. Left alone
   because `legacy/` is frozen and this was outside the brief, but they are the
   last non-commercial bytes in the repository and deleting them (with the
   legacy scripts' knowledge that they must be re-fetched) would finish the
   job.
4. **`/seed` can no longer answer 502.** A permanently bad preview now returns
   200 `{"status": "unanalyzed"}` and the worker fails the embed job. That is
   finding 2's explicit ruling, but it does change the contract's §8 behaviour;
   `server/CLAUDE.md` was updated to say so, and nothing in `contract/` named
   the 502.
5. **Crawling pauses while a backfill is running.** Intentional (two cores,
   and new tracks cannot be shown while old ones are invisible), but at cutover
   it means no new discovery for the length of the backfill.
6. **The re-analysis rate is unmeasured.** The brief's < 12 s/track target and
   the `GROUP_SIZE=2` fallback are a deploy-step observation; nothing here can
   time a real CLAP pass.
7. **`scripts/atlas_check.py` still writes its probe row at version 3** —
   unchanged from Task 3, and now doubly safe: the probe cannot join the corpus
   AND the arm will try to re-analyze it once. It is a one-row no-op that ends
   in `reanalysis_failed_at` if the id is not real.
8. **`uv.lock` was regenerated**, so dependency pins moved for unrelated
   packages too. It resolved offline-clean and the suite is green, but it is a
   bigger diff than the pyproject change alone.
