# Task 4 — fix round 1

Branch `clean-room`, one commit on top of `d4c824a` (Task 3's fix round).
Not pushed, not deployed, not merged.

## Critical

**1. A group-wide analysis failure marks nothing, and trips a breaker.**
`analyze_tracks` *raising* (rather than filling each slot) means the MODEL is
broken — a missing checkpoint, an OOM, a bad image — so charging it to the
three rows that happened to be in the group would retire the whole corpus
three rows at a time. `reanalyze_step` now returns 0 from that branch without
touching a row, and `_group_failed` counts it: at
`REANALYZE_MAX_GROUP_FAILURES = 3` consecutive ones the arm sets
`_halted_until = now + REANALYZE_HALT_S (600 s)` and says so **once**, not per
tick. A group that actually runs resets the counter (so two blips a day apart
never add up to a halt), and the halt lifts by itself. The periodic
"reanalyzed N, remaining M" line is suppressed on a group-wide failure — the
failure line is the report.

**2. A `put_track` failure marks nothing.** The analysis succeeded and the
store did not; that is Atlas, not the track. It logs
`could not store (…); leaving the row alone` and skips. Note it also does not
record an attempt: a store that cannot take the `put_track` cannot take the
attempt write either.

**3. Only classifiable failures can retire a row, and only after three.**
- `worker.UnfetchableTrack` is raised by `_reanalyze_one` for the three
  "verdict about this row" cases (no metadata, no source owns the id, the
  source has no preview URL). A **download** failure deliberately does not
  raise it — a CDN 500 or a socket timeout is evidence about the world.
- `worker._classifiable(exc)` is `UnfetchableTrack` or `analysis.v2.DecodeError`
  (imported lazily by name, so a host without the analysis extra still runs
  the worker).
- `store.record_reanalysis_failure(track_id, error, classifiable)` replaces
  `fail_reanalysis`. It ALWAYS `$inc`s `reanalysis_attempts` and stamps
  `reanalysis_attempted_at`; it sets `reanalysis_failed_at` only when
  `classifiable and attempts >= REANALYSIS_MAX_ATTEMPTS (3)`. `_STALE` still
  excludes only `reanalysis_failed_at`, so rows with attempts < 3 remain
  stale work. `put_track` `$unset`s all five fields.
- `reanalysis_attempted_at` is also a **sort key**, which is what stops a
  retried row from keeping the head of an oldest-first queue: `_STALE_SORT` is
  now `reanalyze_priority DESC, reanalysis_attempted_at ASC, analyzed_at ASC`
  (prioritized rows, then never-tried, then longest-since-tried, then
  oldest). The compound index is `features_version ASC` + those three, same
  directions, so the sort still comes off the index.

**4. Stale rows read as unanalyzed to the API.**
- `/seed` is ready only when `_is_current(features)`. A row at an older
  version is **not** re-crawled or re-queued as a cold embed — it already has
  its metadata and is already in the arm's queue — it is pushed to the front
  with `store.prioritize_reanalysis` (the `reanalyze_priority` timestamp the
  sort's first key honours) and waited on exactly like a cold seed.
- `_await_features` is version-aware too; without that it would answer
  "ready" the instant it was asked about the row it had just rejected.
- `_require_current(track_id, features)` raises **409**
  `{"status": "unanalyzed", "track_id", "detail"}` and prioritizes the row;
  it guards `/recommend`, `/viz/map` (before its 404, so the client is told
  "queued for re-analysis" rather than "not in corpus") and `/viz/histogram`.
- A width mismatch that slips past the version check (a same-version row of
  the wrong width) is caught in `_similarity`, the single funnel every
  seed-vs-corpus comparison goes through, with the same 409.

## Important

**5. Embed jobs come before the backfill.** `_reanalyze_budget()` returns 1
when `store.queued_count() > 0`, else `GROUP_SIZE`; `_tick` passes it to
`reanalyze_step(limit)`. A cold /seed blocks a client for 20 s and must not
spend it behind a full re-analysis group — but the backfill still creeps
forward, so a busy queue cannot starve it either. A `queued_count` error logs
and falls back to the full group.

**6. `v2.decode(duration=ANALYSIS_SECONDS=30.0)`.** Not an optimization: a
Deezer preview is 30 s and Jamendo hands back the whole track (the Range cap
stops at ~1 MB ≈ 85 s at 96 kbps), so without it one corpus held head-only
embeddings beside head-plus-middle ones. Commented as a FEATURES_VERSION-bump
constant. Test asserts 45 s in → 30 s out (skipped here: no analysis extra).

**7. `beat_this` pinned** to `@b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c`
(HEAD of `main`, fetched from the GitHub API). `uv lock` re-run; the lock now
records `?rev=b95c8ab…`.

**8. Dockerfile.** `--extra-index-url` instead of `--index-url` (the CPU index
mirrors torch and little else, so making it the only index breaks resolution
of everything it does not carry); `torch==2.10.0+cpu` and
`torchaudio==2.10.0+cpu` pinned via an `ARG TORCH_VERSION` — I verified both
have `cp311 manylinux_2_28_aarch64` wheels on that index; `build-essential`
added so an sdist fallback can compile on arm64.

**9. Memory consistent with Task 3's `TEXT_SEARCH` gate.** Task 3 took the
gate rather than a text-only load (msclap builds the audio encoder and loads
the full state dict before the caption encoder exists), so with the flag ON
the API holds the WHOLE model. api `mem_limit` is therefore **6g** again, with
a comment saying exactly that (3g would do with the flag off; 6g is what stops
the first text query OOM-killing the API). `.env.example`, README and
`server/CLAUDE.md` all say the same thing; the compose test now asserts
`["6", "4"]`.

## Minor

**10. THIRD_PARTY.md** — `transformers` promoted to a first-class row (it is
how the text tower runs GPT-2's tokenizer) and removed from the transitive
table; the LGPL/GPL source offer now runs from **us**: "on request we provide
the corresponding source for these packages, in the versions the distributed
image contains", with the upstream URLs described as where we get it, not as
the discharge of the obligation. `beat_this`'s pin is named in its row.

**11. Logging** — the arm's periodic line is now
`reanalyzed N, remaining M (gave up on K; …s for L)`, so a backfill that is
"finishing" because it retired half the corpus is visible in the same line.
`_fill_rec_rhythm` logs when `get_many_rhythm` returns a different number of
rows than it was asked for instead of silently blanking the panel.

**12. README** says the iOS credit line is **unbuilt/unverified** (no Xcode
toolchain in this environment). `scripts/atlas_check.py` writes its probe at
`FEATURES_VERSION` and `FEATURE_KEYS["embedding"]` width — the old version-3
probe never exercised the path a real write takes, and since the arm landed it
briefly parked a fake row at the head of the backfill queue.

## 13. Background UMAP layouts are serialized (from the Task 3 re-review)

`_UMAP_PENDING` only deduped a burst against ONE subset identity; two distinct
subsets (a second seed, a grown corpus, `/viz/walk` beside `/viz/map`) each
started their own 2–15 s single-threaded numba job on a 2-core VM. A
module-level `_UMAP_WORK_LOCK` is now held for the whole layout inside the
background thread, so at most one computes at a time. Nothing a caller can see
changes: the request never waits for a layout either way — it is answered with
PCA and upgraded when the layout lands — so a queued subset simply stays on
PCA a few seconds longer.

Test `test_two_subsets_lay_out_one_at_a_time` records entry/exit around a
stubbed `project_umap` that sleeps 150 ms, asserts **two distinct matrix
identities** actually ran (otherwise `_UMAP_PENDING` alone would make the test
vacuous) and that the concurrency count never exceeded 1. Verified it fails
with the lock removed and passes with it.

## Tests

`PYTHONPATH=$PWD/src python3 -m pytest -q` → **483 passed, 17 skipped**
(baseline on `d4c824a`: 465/16). `cd web && npm test -- --run` → **174
passed** (27 files); `npm run build` clean.

New: 13 worker tests (group-wide failure marks nothing, breaker halts + says
it once + self-heals, one good group resets it, put_track failure marks
nothing, DecodeError counts to 3, unclassifiable never retires, dead preview
3-then-out, unknown source, failed row goes to the back, prioritized row jumps
the queue, the gave-up count in the log, embed jobs before the backfill); 6
store tests (three-strikes, unclassifiable never retires, backoff sort,
priority sort, still-playable, every field cleared); 5 app tests (stale seed
not ready, stale seed prioritized and not embed-queued, version-aware wait,
`/recommend` and `/viz/map` 409); 1 viz test (serialized layouts); 1 analysis
test (the 30 s window). A new autouse `clear_worker_state` fixture resets the
breaker between tests — without it a test that deliberately breaks the model
left the arm halted for the next one.

## Concerns

1. The 30 s decode window and the `beat_this` pin both change stored numbers
   in principle; neither is a FEATURES_VERSION bump here because nothing has
   been analyzed at version 4 in production yet. If anything HAS been, it must
   be re-analyzed.
2. `torch==2.10.0+cpu` / `torchaudio==2.10.0+cpu` are verified to EXIST as
   cp311 aarch64 wheels, not verified to import or run — no torch in this
   environment, and the image is built on the VM in the next step.
3. The analysis tests (including the new 30 s one) still skip here: no
   msclap/torch/librosa, so `ANALYSIS_SECONDS` is unexercised locally.
4. `record_reanalysis_failure` is two writes (find_one_and_update, then the
   give-up stamp) rather than one atomic update; two workers racing the same
   row could double-count attempts. There is one worker.
5. A permanently 404ing preview URL is unclassifiable, so it is retried
   forever — the attempted-at sort keeps it at the back, so it costs one
   download per full lap of the queue, not a stall.
