# Task 4 — fix round 2

Branch `clean-room`, one commit on top of `bc0266c`. Not pushed, not
deployed, not merged.

## D1 (critical) — a priority stamp that outlived its attempt froze the backfill

The loop: `/seed` on a v3 row stamps `reanalyze_priority`; it was the first
DESCENDING sort key, so the arm took that row every tick; its preview 404s,
which is *unclassifiable*, so the row is never retired; and nothing cleared
the stamp, because only a successful `put_track` did. With
`_reanalyze_budget()` returning 1 while the embed queue is busy, the backfill
made **zero** progress, for ever.

Two fixes, as instructed:

1. **The attempt spends the stamp.** `record_reanalysis_failure` now
   `$unset`s `reanalyze_priority` in the same update that `$inc`s the
   counters and stamps `reanalysis_attempted_at`. `put_track` already cleared
   it on success, so the stamp now buys exactly one attempt either way.
2. **Stamps expire.** `PRIORITY_TTL_S = 600`. `stale_ids` is now **two
   queries** rather than one three-key sort — a sort key cannot express "only
   if it is still fresh", and that is the whole point: first
   `{**_STALE, reanalyze_priority: {$gte: now - TTL}}` sorted by
   `reanalyze_priority DESC`, then, if the limit is not filled, the backlog
   (`priority absent OR older than the cutoff`) sorted by
   `reanalysis_attempted_at ASC, analyzed_at ASC`. The predicates are
   complementary, so no id can come back twice. A sparse
   `reanalyze_priority DESC` index backs the first query (the field is absent
   on almost every row, and a stamp lives ten minutes).

Tests: `test_a_priority_is_spent_by_the_attempt_it_buys` (prioritised row
fails unclassifiably → it is not first on the next `stale_ids`, the stamp is
gone, the row is still stale work), `test_a_forgotten_priority_expires`,
`test_a_fresh_priority_still_wins_over_an_expired_one`, and at the worker
level `test_a_seeded_row_whose_preview_is_dead_does_not_stall_the_backfill`,
which reproduces the exact loop end to end.

## D2 (important) — blips no longer add up to a verdict

`reanalysis_attempts` counted every failure and the retirement check read it,
so two network blips plus one `DecodeError` retired a perfectly good
recording. There are now two counters:

- `reanalysis_attempts` — every failure. This is the backoff signal, and it
  is what `reanalysis_attempted_at` accompanies.
- `reanalysis_classifiable_attempts` — only failures that are a verdict about
  the track (`UnfetchableTrack`, `DecodeError`). **Only this one retires**,
  at `REANALYSIS_MAX_ATTEMPTS = 3`.

`record_reanalysis_failure` returns the classifiable count (what the worker
logs a give-up against, now worded "giving up after N failures of its own"),
and `put_track` `$unset`s the new field with the rest. Test
`test_blips_do_not_add_up_to_a_verdict`: blip, blip, decode-error → attempts
3, verdicts 1, not retired; two more decode errors → retired.

## D3 (important) — flat 409 bodies, and clients that understand them

**Server.** `HTTPException(409, {...})` nests the payload under `detail`, so
the web client (which reads `.detail` as a string) rendered
"409: [object Object]". Replaced with a dedicated `UnanalyzedSeed` exception
and an `@app.exception_handler` returning
`JSONResponse(409, {"status": "unanalyzed", "track_id": …, "reason": …})` —
flat. `_unanalyzed()` now builds that exception; every raise site
(`_require_current` on `/recommend`, `/viz/map`, `/viz/histogram`, and the
width guard in `_similarity`) is unchanged. A test asserts the body is
exactly those three keys with no `detail`.

**Web.** `request()` reads `body.detail ?? body.reason`, so an `ApiError`
carries the reason string either way; new `UNANALYZED` constant and
`isUnanalyzed(err)` helper. `Recommendations` gains a `reanalyzing` status: a
409 shows *"We're re-analyzing this track with the new audio model. This
usually takes a few seconds."* over the skeleton list and retries after
`REANALYZE_RETRY_MS = 3000`, up to `REANALYZE_MAX_RETRIES = 5`, then falls
through to the ordinary error box with its Try again button. The retry timer
is cleared on unmount and on any fresh run (a slider move resets the attempt
count). `Insights` maps 409 onto its existing `unanalyzed` state, whose copy
now mentions re-analysis. Tests: client parses the flat body and
`isUnanalyzed` discriminates; the screen shows the message, retries and
renders on success, gives up after exactly `MAX_RETRIES + 1` calls, and still
reports an ordinary 500 immediately; Insights' 409 is not a failure.

**iOS (unbuilt).** `APIError.reanalyzing` with that same sentence as its
`errorDescription`, thrown from the one status check in `APIClient.get/post`
when the code is 409. A test in `APIClientTests.swift` asserts a 409 throws
it and that the description mentions re-analysing. Still not compiled — no
Xcode toolchain here.

## Tests

`PYTHONPATH=$PWD/src python3 -m pytest -q` → **488 passed, 17 skipped**
(baseline `bc0266c`: 483/17). `cd web && npm test -- --run` → **181 passed**
(27 files, baseline 174); `npm run build` clean.

New: 5 store tests (priority spent, priority expires, fresh beats expired,
two counters, plus the widened cleared-fields assertion), 1 worker test (the
dead-preview stall), 2 app tests (flat body shape on `/recommend` and
`/viz/map`), 2 web client tests, 4 `Recommendations` tests, 1 `Insights`
test, 1 Swift test.

## Concerns

1. `stale_ids` is two round trips per tick now instead of one; both are
   indexed and bounded by `limit`, but it is twice the query count on the
   arm's hot path.
2. The priority cutoff compares a client-side `_now()` against a
   server-stamped `$currentDate`, so a badly skewed API clock could expire
   stamps early or late — the same known trade-off store.py's header already
   records for cache expiry.
3. The iOS 409 mapping is written but uncompiled and unseen on a device.
