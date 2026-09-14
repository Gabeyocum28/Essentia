"""MongoDB Atlas read/write. Database `essentia`, three collections:

  tracks  _id=track_id, title, artist, album, artwork_url,
          embedding (int8 bytes), scale (float), features_version, analyzed_at
          -- the last four are absent until the track is analyzed --
          feel (8 floats, analysis/feel.FEEL_KEYS) and rhythm (the seven
          contract RHYTHM_KEYS) -- both absent on rows analyzed before they
          existed, and both optional to the ranking.
  jobs    _id="embed:{id}" | "attr:{seed}:{rec}", kind, state, claimed_at,
          attempts, error, created_at  (+ track_id or seed_id/rec_id)
  cache   _id="preview:{id}" | "attr:{seed}:{rec}", value, expires_at (TTL)

Signed preview URLs are never stored on a track (they expire in minutes);
GET /preview re-signs and caches them in `cache`.

Every function here is the same name app.py, corpus/ingest.py and the
worker called before this was MongoDB. Only the backend changed.

Timestamps are naive UTC throughout (not timezone-aware): mongomock strips
tzinfo off datetimes on round-trip, so comparing an aware `_now()` against a
value read back from mongomock raises. Using naive UTC everywhere avoids the
mismatch in both the mongomock-backed tests and the real Atlas driver.

The timestamps that drive watermarks and stale detection -- `analyzed_at`,
`claimed_at`, `created_at` -- are stamped by the MongoDB server itself via
`$currentDate`, not by the calling process's clock. The API and the embed
worker are different hosts; if their clocks drifted, a client-stamped
`analyzed_at` could sort before a watermark that was really taken later,
silently hiding a row from `tracks_since`/`corpus_ids`. A single server
clock removes that failure mode. `_now()` remains in use for read-side
comparisons (e.g. cache expiry, the stale-job cutoff) where both sides of
the comparison are evaluated in this process.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pymongo
from pymongo import ReturnDocument

from music_recommendations.analysis.quantize import from_int8, to_int8
from music_recommendations.analysis.schema import FEATURES_VERSION
from contract.features import RHYTHM_KEYS
from music_recommendations.server.dedupe import dedupe_key

TRACK_FIELDS = ("title", "artist", "album", "artwork_url")
VERSION_KEY = "_features_version"   # what corpus/ingest.py expects to read back

_client: pymongo.MongoClient | None = None
_indexes_ready = False
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.utcnow()


def db():
    """Lazily connect using MONGODB_URI / MONGODB_DB; one client per process."""
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is not set")
        _client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client[os.environ.get("MONGODB_DB", "essentia")]


def reset() -> None:
    """Forget the client and caches (tests, and after a fork)."""
    global _client, _indexes_ready, _ids_cache
    if _client is not None:
        _client.close()
    _client = None
    _indexes_ready = False
    _ids_cache = None


def ensure_indexes() -> None:
    """Idempotent; called once per process before the first write."""
    global _indexes_ready
    if _indexes_ready:
        return
    d = db()
    d.tracks.create_index("analyzed_at")
    # The crawler asks "which of these 100 keys do we already have?" once per
    # batch (existing_keys), and the backfill groups the whole collection on
    # it; neither is affordable as a scan at corpus scale.
    d.tracks.create_index("dedupe_key")
    # The re-analysis arm asks "how many rows are below the current version,
    # and which are the oldest" every tick (stale_count / stale_ids, _STALE).
    # On the bare `analyzed_at` index that is a scan of the whole corpus for
    # a count that is usually zero once the backfill has drained; leading on
    # features_version (an equality-style range) makes the count cheap, and
    # the remaining keys are _STALE_SORT in the SAME directions, which is
    # what lets the arm's three-key sort come off the index rather than out
    # of a blocking in-memory sort of the whole stale set.
    d.tracks.create_index([("features_version", pymongo.ASCENDING),
                           *_STALE_SORT])
    # The priority queue stale_ids checks first: a tiny index (almost no row
    # carries the field, and a stamp lives PRIORITY_TTL_S) that turns "is
    # anyone waiting?" -- asked every tick -- into a bounded lookup instead
    # of a scan for a field that is usually absent everywhere.
    d.tracks.create_index([("reanalyze_priority", pymongo.DESCENDING)],
                          sparse=True)
    d.jobs.create_index([("state", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)])
    d.cache.create_index("expires_at", expireAfterSeconds=0)
    _indexes_ready = True


# ---- tracks ----

def _contract(doc: dict) -> dict:
    """The document as a Track: the contract fields, plus the two optional
    ones when this row has them (contract.features.TRACK_OPTIONAL_FIELDS).

    Only `attribution.url` is published, not the whole attribution object:
    the licence deed and the source name are already implied by the link and
    the `source` field, and the contract fixes what a Track may carry.
    Absent, never null, when there is nothing to say.
    """
    out = {"track_id": doc["_id"], **{k: doc.get(k) for k in TRACK_FIELDS},
           "preview_url": ""}
    if doc.get("source"):
        out["source"] = doc["source"]
    url = (doc.get("attribution") or {}).get("url")
    if url:
        out["attribution_url"] = url
    return out


def _source_of(track: dict) -> str:
    """Which catalogue a track came from.

    An explicit `source` wins; otherwise it is read off the id, where a
    namespaced `"jamendo:42"` names its own source and a bare `"42"` is
    Deezer (corpus/sources/base.py explains why Deezer ids stay bare).
    """
    name = track.get("source")
    if name:
        return str(name)
    prefix, sep, _local = str(track.get("track_id") or "").partition(":")
    return prefix if sep and prefix else "deezer"


def _meta(track: dict) -> dict:
    """The contract fields plus the dedupe key derived from them, the
    source, and the attribution the source handed over.

    `dedupe_key` is stored, not computed on read, so the crawler's
    "have we got this recording already?" question is one indexed `$in`
    instead of a scan. It is NOT a contract field: _contract() projects
    TRACK_FIELDS only, so it can never reach a response.

    `attribution` is stored whole and published narrow (see _contract): if
    the clients ever need the licence deed as well as the backlink, the row
    already has it and no re-crawl is needed.
    """
    out = {**{k: track.get(k) for k in TRACK_FIELDS},
           "dedupe_key": dedupe_key(track.get("title"), track.get("artist")),
           "source": _source_of(track)}
    attribution = track.get("attribution")
    if attribution:
        out["attribution"] = attribution
    return out


# What every Track read projects: the contract fields plus the two optional
# ones _contract may publish.
_TRACK_PROJECTION = {**{k: 1 for k in TRACK_FIELDS}, "source": 1, "attribution": 1}


# A track that is analyzed and has not been retired as a duplicate of
# another recording. Every CORPUS ENUMERATION (corpus_ids, base_matrix,
# tracks_since) filters on this, which is what removes retired rows from
# ranking, the viz snapshot and the crawl watermark in one place.
#
# Deliberately NOT applied to get_features/get_many_features/get_track: a
# retired id is still a legitimate SEED (a user can search it up on Deezer
# and press play), and refusing to look it up would 404 a playable track.
# `features_version` is part of it since v4: the CLAP embedding shares no
# space at all with the v3 embedding it replaced (analysis/schema.py), so a
# corpus that mixes them ranks nonsense against nonsense and nothing in the
# numbers says so. Filtering here means one predicate retires every stale row
# from ranking, the viz snapshot and the crawl watermark at once, and the
# worker's re-analysis arm (stale_ids) is what brings them back.
LIVE = {"analyzed_at": {"$exists": True}, "duplicate_of": {"$exists": False},
        "features_version": FEATURES_VERSION}


def _live_since(stamp: datetime) -> dict:
    """LIVE, plus "analyzed at or after `stamp`" -- the incremental form the
    watermark readers use. Spelled once so a change to LIVE cannot be
    forgotten in corpus_ids' fast path or tracks_since."""
    return {**LIVE, "analyzed_at": {"$gte": stamp}}


# The feel vector is eleven probabilities in [0, 1], stored as plain BSON
# doubles rather than quantized like the embedding: 11 numbers is ~90 bytes a
# row, and rounding them to four decimals keeps the document small while
# staying far finer than the heads themselves are calibrated.
FEEL_DP = 4


def _feel(features: dict) -> list[float] | None:
    """The feel vector as short floats, or None if this analysis has none."""
    value = features.get("feel")
    if value is None:
        return None
    return [round(float(v), FEEL_DP)
            for v in np.asarray(value, dtype=np.float32).ravel()]


# The rhythm dict is stored whole and read back whole: seven named,
# human-readable numbers (contract/features.py RHYTHM_KEYS) totalling well
# under 200 bytes a row. Unknown keys are dropped rather than stored, so a
# future extractor cannot quietly widen every document.
def _rhythm(features: dict) -> dict | None:
    """The rhythm dict narrowed to RHYTHM_KEYS, or None if there is none."""
    value = features.get("rhythm")
    if not isinstance(value, dict):
        return None
    out = {k: value[k] for k in RHYTHM_KEYS if k in value}
    return out or None


def put_track(track: dict, features: dict) -> None:
    """Upsert contract fields plus the analyzed embedding (and feel/rhythm).

    `feel` is optional on the way in: the ranking treats a missing vector as
    "no penalty" rather than hiding the track. `rhythm` is optional the same
    way and for the same reason -- a track whose beat tracker failed simply
    takes no tempo penalty.

    `features_version` comes off the analysis dict's `_features_version`
    (VERSION_KEY) and is what LIVE filters on: a caller that omits it writes
    version 0, which is a row no corpus enumeration will ever serve. That is
    deliberate -- a writer that does not say which stack produced a vector
    has not earned a place in the ranking.
    """
    ensure_indexes()
    data, scale = to_int8(np.asarray(features["embedding"], dtype=np.float32))
    fields = {**_meta(track), "embedding": data, "scale": float(scale),
              "features_version": int(features.get(VERSION_KEY, 0))}
    feel = _feel(features)
    if feel is not None:
        fields["feel"] = feel
    rhythm = _rhythm(features)
    if rhythm is not None:
        fields["rhythm"] = rhythm
    db().tracks.update_one(
        {"_id": track["track_id"]},
        # A successful analysis clears any earlier re-analysis give-up mark:
        # whatever was broken about this row's audio evidently is not any
        # more, and leaving the mark would hide the row from a future
        # version bump's backfill.
        {"$set": fields, "$currentDate": {"analyzed_at": True},
         "$unset": {"reanalysis_failed_at": "", "reanalysis_error": "",
                    "reanalysis_attempts": "",
                    "reanalysis_classifiable_attempts": "",
                    "reanalysis_attempted_at": "", "reanalyze_priority": ""}},
        upsert=True,
    )


def put_track_meta(track: dict) -> None:
    """Upsert contract fields only; leaves any embedding in place."""
    ensure_indexes()
    db().tracks.update_one({"_id": track["track_id"]}, {"$set": _meta(track)}, upsert=True)


def get_track(track_id: str) -> dict | None:
    doc = db().tracks.find_one({"_id": track_id}, _TRACK_PROJECTION)
    return _contract(doc) if doc else None


def get_many_tracks(track_ids: list[str]) -> list[dict | None]:
    if not track_ids:
        return []
    found = {d["_id"]: _contract(d) for d in
             db().tracks.find({"_id": {"$in": track_ids}}, _TRACK_PROJECTION)}
    return [found.get(t) for t in track_ids]


# What every feature read projects. `feel` is included but never required:
# _features() omits the key entirely when the row has not been scored, which
# is exactly what app.py's _rows_for tests for.
_FEATURE_FIELDS = {"embedding": 1, "scale": 1, "features_version": 1,
                   "feel": 1, "rhythm": 1}


def _features(doc: dict | None) -> dict | None:
    if not doc or doc.get("embedding") is None:
        return None
    out = {"embedding": from_int8(doc["embedding"], doc["scale"]).tolist(),
           VERSION_KEY: doc.get("features_version", 0)}
    if doc.get("feel") is not None:
        out["feel"] = [float(v) for v in doc["feel"]]
    if doc.get("rhythm") is not None:
        out["rhythm"] = dict(doc["rhythm"])
    return out


def get_features(track_id: str) -> dict | None:
    return _features(db().tracks.find_one({"_id": track_id}, _FEATURE_FIELDS))


def get_many_features(track_ids: list[str]) -> list[dict | None]:
    """Features per requested id, in order; None for an unanalyzed id.

    No LIVE filter on purpose: an id asked for by name (a seed, an
    attribution pair, a crawl candidate) is answered even if it was retired
    as a duplicate. Only the corpus enumerations hide retired rows.
    """
    if not track_ids:
        return []
    found = {d["_id"]: _features(d) for d in
             db().tracks.find({"_id": {"$in": track_ids}}, _FEATURE_FIELDS)}
    return [found.get(t) for t in track_ids]


def get_many_feel(track_ids: list[str]) -> list[list[float] | None]:
    """The feel vector per requested id, in order; None where the row has none.

    A projection of `feel` alone, deliberately NOT get_many_features: the
    ranking's feel matrix wants eleven floats a row, and the generic read
    hands back a 1280-int8 embedding per candidate that has to be
    dequantized to float32 before it can be discarded. Same no-LIVE-filter
    reasoning as get_many_features -- an id asked for by name is answered.
    """
    if not track_ids:
        return []
    found = {}
    for doc in db().tracks.find({"_id": {"$in": track_ids}}, {"feel": 1}):
        vector = doc.get("feel")
        if vector is not None:
            found[doc["_id"]] = [float(v) for v in vector]
    return [found.get(t) for t in track_ids]


def get_many_rhythm(track_ids: list[str]) -> list[dict | None]:
    """The rhythm dict per requested id, in order; None where there is none.

    A projection of `rhythm` alone, for the same reason get_many_feel
    exists: the ranking's tempo term wants one float a row and the generic
    read would dequantize a 1024-int8 embedding per candidate to get it.
    No LIVE filter, like every other by-name read.
    """
    if not track_ids:
        return []
    found = {}
    for doc in db().tracks.find({"_id": {"$in": track_ids}}, {"rhythm": 1}):
        value = doc.get("rhythm")
        if value is not None:
            found[doc["_id"]] = dict(value)
    return [found.get(t) for t in track_ids]


def get_analyzed_at(track_id: str) -> datetime | None:
    doc = db().tracks.find_one({"_id": track_id}, {"analyzed_at": 1})
    return doc.get("analyzed_at") if doc else None


def tracks_since(stamp: datetime) -> list[tuple[str, np.ndarray]]:
    """(id, float32 vector) for every track analyzed at or after `stamp`.

    Inclusive, not strict: MongoDB stores datetimes at millisecond
    resolution and the API and the embed worker are separate processes, so
    two tracks analyzed within the same millisecond (by the same process or
    different ones) can legitimately tie on `analyzed_at`. A strict `$gt`
    would then silently drop whichever of the tied rows a caller already
    has the watermark for. Using `$gte` means the boundary row (the one
    exactly at `stamp`) can come back again; callers that page through this
    by re-using the last-seen `analyzed_at` as the next `stamp` must dedupe
    by track id.

    Retired duplicates (`duplicate_of` set) and rows from a superseded
    feature version are excluded, like every other corpus enumeration -- a
    caller paging this to build a matrix must not re-introduce a row
    corpus_ids() no longer lists.
    """
    cursor = db().tracks.find(_live_since(stamp),
                              {"embedding": 1, "scale": 1}).sort("analyzed_at", 1)
    return [(d["_id"], from_int8(d["embedding"], d["scale"])) for d in cursor]


# corpus_ids() is called on every /recommend. Instead of shipping every id
# each time, remember the ids seen so far and the newest analyzed_at, and ask
# only for tracks newer than that. Tracks are only ever added.
_ids_cache: tuple[datetime, list[str]] | None = None


def corpus_ids() -> list[str]:
    """All analyzed, non-retired track ids, sorted (see LIVE)."""
    global _ids_cache
    with _lock:
        if _ids_cache is None:
            # analyzed_at, not embedding: only put_track ever sets either
            # field (together), so the two filters are equivalent -- but
            # analyzed_at has an index (ensure_indexes) and embedding does
            # not.
            docs = list(db().tracks.find(LIVE, {"analyzed_at": 1}))
            newest = max((d["analyzed_at"] for d in docs), default=datetime(1970, 1, 1))
            _ids_cache = (newest, sorted(d["_id"] for d in docs))
            return list(_ids_cache[1])
        newest, ids = _ids_cache
        # $gte, not $gt: millisecond-resolution timestamps mean a track
        # analyzed in another process within the same millisecond as the
        # watermark would otherwise be missed. Re-fetching the watermark
        # row itself is harmless -- the set union below dedupes it.
        fresh = list(db().tracks.find(_live_since(newest), {"analyzed_at": 1}))
        if fresh:
            newest = max(d["analyzed_at"] for d in fresh)
            ids = sorted(set(ids) | {d["_id"] for d in fresh})
            _ids_cache = (newest, ids)
        return list(ids)


def corpus_size() -> int:
    return len(corpus_ids())


# Rows that ARE analyzed and not retired, but under a superseded feature
# version: exactly what LIVE now excludes. They are invisible to ranking
# until something re-analyzes them, so the worker's re-analysis arm needs
# to see both how many are left and which to take next.
#
# `reanalysis_failed_at` is what takes a row OUT of that queue for good, and
# it is set only after REANALYSIS_MAX_ATTEMPTS classifiable failures (see
# record_reanalysis_failure). A row that has failed once or twice is still
# stale work: retiring a track on one bad afternoon at the source would
# quietly delete a third of the corpus, and nothing in the numbers would say
# so. Marking the track document rather than the jobs collection is
# deliberate: re-analysis is not a queued job, it is a property of the row.
_STALE = {"analyzed_at": {"$exists": True}, "duplicate_of": {"$exists": False},
          "features_version": {"$lt": FEATURES_VERSION},
          "reanalysis_failed_at": {"$exists": False}}

# How many CLASSIFIABLE failures a row gets before it leaves the queue.
# Counted separately from `reanalysis_attempts` (which counts every attempt,
# and is what the backoff sort reads) because otherwise two network blips
# plus one real DecodeError would retire a perfectly good recording: three
# failures is not three verdicts.
REANALYSIS_MAX_ATTEMPTS = 3

# How long a /seed priority stamp outranks the backlog. It buys ONE attempt
# (record_reanalysis_failure and put_track both clear it), so this is only
# the backstop for a stamp nothing ever came back for -- a crash between the
# stamp and the attempt, or a client that asked and went away. Without it a
# forgotten stamp is permanent: `reanalyze_priority` is the first, DESCENDING
# sort key, so the row would be handed to the arm every tick for ever.
PRIORITY_TTL_S = 600

# The arm works two queues, in order.
#
# FIRST, rows someone is actively waiting on: a /seed on a superseded track
# stamps `reanalyze_priority`, and the most recent stamp goes first. This is a
# separate QUERY rather than a leading sort key because a sort key cannot
# express "and only if the stamp is still fresh" -- and an expired or
# forgotten stamp that still sorted first would hand the arm the same row
# every tick for ever (PRIORITY_TTL_S above).
_PRIORITY_SORT = [("reanalyze_priority", pymongo.DESCENDING)]
#
# THEN the backlog:
#   reanalysis_attempted_at  ASCENDING, absent (never tried) first, then the
#                        longest-ago attempt. This is the backoff: a row that
#                        just failed goes behind every row that has not been
#                        tried, so one bad track cannot hold the head of the
#                        queue and starve the other 19,000.
#   analyzed_at          ASCENDING, oldest first -- so an interrupted backfill
#                        makes forward progress (a re-analyzed row gets a
#                        fresh stamp and sorts to the back).
_STALE_SORT = [("reanalysis_attempted_at", pymongo.ASCENDING),
               ("analyzed_at", pymongo.ASCENDING)]


def stale_count() -> int:
    """How many live-analyzed rows are below the current feature version."""
    return db().tracks.count_documents(_STALE)


def _priority_cutoff() -> datetime:
    return _now() - timedelta(seconds=PRIORITY_TTL_S)


def stale_ids(limit: int = 100) -> list[str]:
    """The next `limit` stale ids: freshly prioritized rows first (newest
    stamp first), then the backlog in _STALE_SORT order.

    Two queries rather than one sort, so an expired stamp is genuinely
    ignored instead of merely sorting oddly. The two predicates are
    complementary, so no id can come back from both.
    """
    if limit <= 0:
        return []
    limit = int(limit)
    cutoff = _priority_cutoff()
    tracks = db().tracks
    ids = [d["_id"] for d in
           tracks.find({**_STALE, "reanalyze_priority": {"$gte": cutoff}},
                       {"_id": 1}).sort(_PRIORITY_SORT).limit(limit)]
    if len(ids) >= limit:
        return ids
    backlog = {**_STALE, "$or": [{"reanalyze_priority": {"$exists": False}},
                                 {"reanalyze_priority": {"$lt": cutoff}}]}
    ids += [d["_id"] for d in
            tracks.find(backlog, {"_id": 1})
            .sort(_STALE_SORT).limit(limit - len(ids))]
    return ids


def prioritize_reanalysis(track_id: str) -> None:
    """Put this row at the FRONT of the re-analysis queue.

    /seed on a row the previous stack analyzed: the client is blocked on
    this one track, and behind it may be the whole corpus. Stamping
    `reanalyze_priority` is what the queue's first sort key honours. Later
    stamps beat earlier ones, which is right -- the most recent person
    waiting is served first.

    Does nothing to a row that is not stale; the arm only ever reads _STALE.
    """
    db().tracks.update_one({"_id": track_id},
                           {"$currentDate": {"reanalyze_priority": True}})


def record_reanalysis_failure(track_id: str, error: str,
                              classifiable: bool) -> int:
    """One failed re-analysis attempt against this row; returns the count.

    EVERY failure bumps `reanalysis_attempts` and stamps
    `reanalysis_attempted_at`, because that timestamp is the queue's backoff
    key -- without it a row that fails keeps the head of an oldest-first
    queue and nothing behind it is ever reached. Every failure also SPENDS
    any priority stamp: the attempt is what the stamp bought, and a stamp
    that outlives its attempt hands the arm the same row every tick for ever.

    Only a CLASSIFIABLE failure bumps `reanalysis_classifiable_attempts`, and
    only that counter can retire the row, at REANALYSIS_MAX_ATTEMPTS.
    Classifiable means the failure was a verdict about this track -- its
    audio will not decode, no source owns its id, the source no longer offers
    a preview. An unclassifiable failure (the CDN 500ing, a socket timeout, a
    broken model) is evidence about the WORLD; charging it to the track is
    how a bad afternoon retires a corpus, and counting the two in ONE counter
    means two blips plus one real DecodeError retires a good recording.

    A retired row keeps its old vectors and stays readable by id (a seed the
    user can still play); it is simply out of LIVE, out of `stale_ids` and
    out of `stale_count`. put_track clears every one of these fields, so a
    row that becomes analyzable again rejoins the queue on its own.

    Returns the CLASSIFIABLE count -- what a caller logs a give-up against.
    """
    increments = {"reanalysis_attempts": 1}
    if classifiable:
        increments["reanalysis_classifiable_attempts"] = 1
    doc = db().tracks.find_one_and_update(
        {"_id": track_id},
        {"$set": {"reanalysis_error": str(error)[:500]},
         "$inc": increments,
         "$currentDate": {"reanalysis_attempted_at": True},
         "$unset": {"reanalyze_priority": ""}},
        return_document=ReturnDocument.AFTER,
    )
    verdicts = int((doc or {}).get("reanalysis_classifiable_attempts", 0))
    if classifiable and verdicts >= REANALYSIS_MAX_ATTEMPTS:
        db().tracks.update_one(
            {"_id": track_id},
            {"$currentDate": {"reanalysis_failed_at": True}},
        )
    return verdicts


def reanalysis_failed_count() -> int:
    """How many rows the re-analysis arm has given up on (ops visibility)."""
    return db().tracks.count_documents({"reanalysis_failed_at": {"$exists": True}})


def existing_keys(keys: list[str]) -> set[str]:
    """Which of these dedupe keys the corpus already holds, in one query.

    Retired duplicates are ignored: their key belongs to the primary that
    replaced them, so counting them again would be double-counting -- and
    if the primary itself were ever removed, the key should be free to
    return.
    """
    keys = [k for k in dict.fromkeys(keys) if k]
    if not keys:
        return set()
    docs = db().tracks.find(
        {"dedupe_key": {"$in": keys}, "duplicate_of": {"$exists": False}},
        {"dedupe_key": 1},
    )
    return {d["dedupe_key"] for d in docs if d.get("dedupe_key")}


def mark_duplicate(track_id: str, primary_id: str) -> None:
    """Retire `track_id` as another edition of `primary_id`.

    The document stays (its features still answer an explicit seed); it just
    leaves every corpus enumeration. The in-process id cache is dropped so
    this takes effect without a restart -- it only ever grows otherwise.
    """
    global _ids_cache
    db().tracks.update_one({"_id": track_id},
                           {"$set": {"duplicate_of": primary_id}})
    with _lock:
        _ids_cache = None


def base_matrix() -> tuple[list[str], np.ndarray]:
    """Every live embedding as one float32 matrix, ids in row order.

    "Live" is analyzed and not retired as a duplicate (see LIVE), so a
    backfilled duplicate stops being a rankable row everywhere at once.

    Preallocated rather than stacked: a corpus of tens of thousands of
    tracks means a stack of per-row arrays (and the list holding them)
    doubles peak memory versus filling one array in place.
    """
    # analyzed_at, not embedding: only put_track ever sets either field
    # (together), so the two filters are equivalent -- but analyzed_at has
    # an index (ensure_indexes) and embedding does not.
    n = db().tracks.count_documents(LIVE)
    if n == 0:
        return [], np.empty((0, 1), dtype=np.float32)

    cursor = db().tracks.find(LIVE, {"embedding": 1, "scale": 1}).sort("_id", 1)
    ids: list[str] = []
    matrix: np.ndarray | None = None
    i = 0
    for d in cursor:
        vec = from_int8(d["embedding"], d["scale"])
        if matrix is None:
            matrix = np.empty((n, vec.shape[0]), dtype=np.float32)
        if i >= n:
            # A track landed mid-read, past what count_documents saw:
            # dropping it (rather than growing the array) keeps this a
            # single allocation; the next call picks it up.
            break
        matrix[i] = vec
        ids.append(d["_id"])
        i += 1
    if matrix is None:
        return [], np.empty((0, 1), dtype=np.float32)
    if i < n:
        # A track vanished mid-read (or the count raced a write the other
        # way): truncate to the rows actually filled.
        matrix = matrix[:i]
    return ids, matrix


# ---- jobs (internal; not part of the HTTP contract) ----
#
# One document per pending unit of work. `state` moves queued -> running ->
# (deleted on success | failed). A queued or running document is the dedup
# guard a set of pending ids used to be; requeue_stale() is the TTL.

_POLL_S = 0.5


def _attr_pair(seed_id: str, rec_id: str) -> str:
    """Job/cache key half: colon-joined, matching the cache key format
    (`attr:{seed}:{rec}`). dequeue_job's returned payload is pipe-joined
    instead -- worker.py splits it on "|" -- so it is built
    separately there, not through this helper."""
    return f"{seed_id}:{rec_id}"


def _enqueue(job_id: str, fields: dict) -> bool:
    ensure_indexes()
    existing = db().jobs.find_one({"_id": job_id}, {"state": 1})
    if existing and existing["state"] in ("queued", "running"):
        return False
    db().jobs.update_one(
        {"_id": job_id},
        {"$set": {**fields, "state": "queued", "claimed_at": None,
                  "attempts": 0, "error": None},
         "$currentDate": {"created_at": True}},
        upsert=True,
    )
    return True


def enqueue_embed(track_id: str) -> bool:
    """Queue a track for analysis. False if already queued or running."""
    return _enqueue(f"embed:{track_id}", {"kind": "embed", "track_id": track_id})


def enqueue_attribution(seed_id: str, rec_id: str) -> bool:
    return _enqueue(f"attr:{_attr_pair(seed_id, rec_id)}",
                    {"kind": "attr", "seed_id": seed_id, "rec_id": rec_id})


def _claim(kind: str) -> dict | None:
    return db().jobs.find_one_and_update(
        {"kind": kind, "state": "queued"},
        {"$set": {"state": "running"}, "$inc": {"attempts": 1},
         "$currentDate": {"claimed_at": True}},
        sort=[("created_at", pymongo.ASCENDING)],
        return_document=ReturnDocument.AFTER,
    )


def dequeue_job(timeout: int = 5) -> tuple[str, str] | None:
    """Next job as (kind, payload), embed jobs first; polls up to `timeout` s."""
    deadline = time.monotonic() + timeout
    while True:
        job = _claim("embed")
        if job:
            return "embed", job["track_id"]
        job = _claim("attr")
        if job:
            # Pipe-joined, not _attr_pair's colon: worker.py
            # splits this payload on "|".
            return "attribution", f"{job['seed_id']}|{job['rec_id']}"
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_S)


def dequeue_embed(timeout: int = 5) -> str | None:
    deadline = time.monotonic() + timeout
    while True:
        job = _claim("embed")
        if job:
            return job["track_id"]
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_S)


def clear_embed_marker(track_id: str) -> None:
    """The job finished (or the caller gave up): drop it so it can be requeued."""
    db().jobs.delete_one({"_id": f"embed:{track_id}"})


def clear_attribution_marker(seed_id: str, rec_id: str) -> None:
    db().jobs.delete_one({"_id": f"attr:{_attr_pair(seed_id, rec_id)}"})


def fail_job(job_id: str, error: str) -> None:
    db().jobs.update_one({"_id": job_id},
                         {"$set": {"state": "failed", "error": error[:500]}})


def requeue_stale(max_age_s: int = 600, max_attempts: int = 3) -> int:
    """Return jobs stuck in `running` to `queued`, or fail them past the cap.

    `attempts` is incremented by `_claim()` on every dequeue, so the cap
    counts total claims (including the one currently stuck), not retries.
    """
    cutoff = _now() - timedelta(seconds=max_age_s)
    stale = list(db().jobs.find({"state": "running", "claimed_at": {"$lt": cutoff}}))
    for job in stale:
        if job["attempts"] >= max_attempts:
            fail_job(job["_id"], f"gave up after {job['attempts']} attempts")
        else:
            db().jobs.update_one({"_id": job["_id"]},
                                 {"$set": {"state": "queued", "claimed_at": None}})
    return len(stale)


# ---- cache: attribution results and signed preview URLs ----

_PREVIEW_TTL_S = 600


def _cache_get(key: str):
    doc = db().cache.find_one({"_id": key})
    if not doc:
        return None
    if doc.get("expires_at") is not None and doc["expires_at"] <= _now():
        return None
    return doc["value"]


def _cache_put(key: str, value, ttl: int | None) -> None:
    ensure_indexes()
    expires = _now() + timedelta(seconds=ttl) if ttl else None
    db().cache.replace_one({"_id": key}, {"_id": key, "value": value, "expires_at": expires},
                           upsert=True)


def get_attribution(seed_id: str, rec_id: str) -> dict | None:
    return _cache_get(f"attr:{seed_id}:{rec_id}")


def put_attribution(seed_id: str, rec_id: str, payload: dict, ttl: int | None = None) -> None:
    """A ready result is kept forever; a failure gets a TTL so it is retried."""
    _cache_put(f"attr:{seed_id}:{rec_id}", payload, ttl)


def get_cached_preview(track_id: str) -> str | None:
    return _cache_get(f"preview:{track_id}")


def put_cached_preview(track_id: str, url: str, ttl: int = _PREVIEW_TTL_S) -> None:
    _cache_put(f"preview:{track_id}", url, ttl)


# ---- durable state (crawl cursor etc.): cache documents that never expire ----

def get_state(key: str) -> dict | None:
    return _cache_get(f"state:{key}")


def put_state(key: str, value: dict) -> None:
    _cache_put(f"state:{key}", value, None)


def queued_count() -> int:
    """How many jobs are waiting (not running, not failed)."""
    return db().jobs.count_documents({"state": "queued"})


_SYSTEM_DBS = ("admin", "local", "config")


def _db_stats_bytes(database) -> int:
    stats = database.command("dbStats")
    return int(stats.get("dataSize", 0)) + int(stats.get("indexSize", 0))


def data_size_bytes() -> int:
    """Data plus index bytes for the WHOLE CLUSTER (what Atlas counts against
    the free tier's 512 MB), not just our own database.

    Atlas bills the cluster, so a leftover database beside `essentia` --
    a scratch copy, an older crawl -- counts against the same cap while
    being invisible to a single-database dbStats. The crawler's byte cap is
    only a real brake if it sees what Atlas sees.

    Three levels of fallback, because this must never be the thing that
    stops a crawl: the cluster sum, then our own database, then a
    2 KB-per-track estimate (mongomock, which implements neither
    list_database_names' stats nor dbStats).
    """
    try:
        client = db().client
        names = [n for n in client.list_database_names() if n not in _SYSTEM_DBS]
        if not names:
            # mongomock lists nothing until a write lands, and a cluster that
            # reports no user databases is a broken read, not an empty disk.
            # Either way, 0 would silently disable the crawler's byte cap.
            raise NotImplementedError("no user databases listed")
        return sum(_db_stats_bytes(client[name]) for name in names)
    except (pymongo.errors.PyMongoError, NotImplementedError,
            TypeError, AttributeError) as exc:
        print(f"data_size_bytes: cluster-wide dbStats unavailable ({exc!r}); "
              f"falling back to this database")
    try:
        return _db_stats_bytes(db())
    except (pymongo.errors.PyMongoError, NotImplementedError, TypeError) as exc:
        print(f"data_size_bytes: dbStats unavailable ({exc!r}); "
              f"estimating from document count")
    return db().tracks.count_documents({}) * 2048


def failed_ids(track_ids: list[str]) -> set[str]:
    """Which of these track ids have a permanently failed embed job, in one
    query -- so a crawl never re-enqueues a track that already failed."""
    if not track_ids:
        return set()
    docs = db().jobs.find(
        {"_id": {"$in": [f"embed:{i}" for i in track_ids]}, "state": "failed"},
        {"_id": 1},
    )
    return {d["_id"].split(":", 1)[1] for d in docs}
