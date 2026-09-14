"""FastAPI app. Routes mirror contract/contract.md exactly. Mock-first: serve
contract/fixture.json until the corpus lands.

Routes:
  GET  /search       -- query the switched-on sources (or fixture fallback)
  GET  /search/text  -- CLAP text-to-audio search over the corpus (not contract)
  POST /seed         -- mark a track as the seed for recommendations
  GET  /axes         -- list available recommendation axes
  GET  /recommend    -- ranked, scored tracks for a seed + axis
  GET  /preview/{id} -- 302 to a freshly signed Deezer preview (not contract)
  GET  /preview/{id}/audio -- same-origin mp3 stream for the web SOUND mode
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from collections import OrderedDict
from contextvars import ContextVar
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from starlette.requests import Request
from typing import Literal, NamedTuple

from contract.features import AXES
from music_recommendations.analysis import analyze_track, frontend
from music_recommendations.analysis.feel_v2 import FEEL_KEYS
from music_recommendations.analysis.schema import METRICS
from music_recommendations.corpus import sources
from music_recommendations.server import dedupe, store, viz
from music_recommendations.server.axes import AXIS_FEATURES, BLENDED_AXES


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """A read-only API process (no put_track/enqueue_* call of its own) would
    otherwise never run ensure_indexes(), leaving Atlas without the
    analyzed_at/jobs/cache indexes until some writer happened to start
    first. _safe() so an unreachable Atlas at boot doesn't crash the API --
    it falls back to serving the fixture, same as every other store call."""
    _safe(store.ensure_indexes)
    yield


app = FastAPI(title="Essencia", lifespan=_lifespan)


# ---- stable preview URLs ----
#
# A Deezer preview URL is not data, it is a 15-minute lease: every one of the
# 90k stored with the corpus was dead within minutes of being written, and a
# sampled 3000 were 100% expired. Persisting it was the mistake. So the stored
# string never reaches a client -- tracks go out pointing at THIS server, at a
# URL that never expires, and GET /preview re-signs at the moment of play.
#
# Contract-safe: contract.md fixes the Track shape and says preview_url is
# "https://...". It does not say whose host.

_public_base: ContextVar[str] = ContextVar("public_base", default="")


class _CaptureBaseURL:
    """Record this request's own origin so _playable can build absolute URLs.

    Plain ASGI rather than @app.middleware("http"): BaseHTTPMiddleware runs the
    endpoint in a child task, and a ContextVar set here would land in a context
    the endpoint may not share. This wrapper runs in the caller's own task.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            _public_base.set(str(Request(scope).base_url).rstrip("/"))
        await self.app(scope, receive, send)


app.add_middleware(_CaptureBaseURL)

# The viz payloads are long lists of near-identical JSON numbers and repeated
# track dicts; gzip takes /viz/map's 2.1 MB to a small fraction of it, which
# on a phone over cell is most of the wall clock. Added AFTER _CaptureBaseURL
# so it sits outermost: compression is the last thing to happen on the way
# out, and the base-URL ContextVar is still set in the endpoint's own task
# (both are plain ASGI wrappers, so neither hops tasks). minimum_size keeps
# it off the small responses -- /preview's 302 above all -- where the header
# and CPU cost more than the bytes saved.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)


def _base() -> str:
    """Origin to hand clients. PUBLIC_BASE_URL wins: behind a tunnel or a proxy
    the request's own host header is the internal one, not the reachable one."""
    return os.environ.get("PUBLIC_BASE_URL", "").rstrip("/") or _public_base.get()


def _playable(track: dict | None) -> dict | None:
    """Give a track this server's stable preview URL.

    Keyed on track_id alone, NOT on the track already having a preview_url:
    the store never round-trips one (a 15-minute signature is not worth
    persisting), so requiring one meant every corpus track went out with no
    way to play it. Callers that synthesize placeholder rows for unknown ids
    pass None here and fall through to their own literal, which keeps
    preview_url null rather than promising audio that would 404.
    """
    if not track or not track.get("track_id"):
        return track
    base = _base()
    if not base:
        return track
    return {**track, "preview_url": f"{base}/preview/{track['track_id']}"}


def _public_track(track: dict) -> dict:
    """A source's Track narrowed to what the contract publishes.

    Sources hand back the contract fields plus `source` and the full
    `attribution` object; only the backlink out of that object is a contract
    field (`attribution_url`), and a source may carry extra keys of its own.
    Search results do not go through the store, so this is where that
    narrowing happens for them.
    """
    out = {"track_id": track.get("track_id"),
           **{k: track.get(k) for k in store.TRACK_FIELDS},
           "preview_url": track.get("preview_url")}
    if track.get("source"):
        out["source"] = track["source"]
    url = (track.get("attribution") or {}).get("url") or track.get("attribution_url")
    if url:
        out["attribution_url"] = url
    return out


def _from_source(track_id: str) -> dict | None:
    """This track straight from the catalogue that owns its id, or None.

    Used when the store has never seen the id -- a track the user searched
    up that the crawler has not reached. A source failure is None, not a
    500: /seed has a fixture fallback behind this.
    """
    source = sources.for_id(track_id)
    if source is None:
        return None
    try:
        found = source.track(track_id)
    except Exception:
        return None
    return _public_track(found) if found else None


def _fresh_preview(track_id: str) -> str | None:
    """A currently-playable URL for this track, from cache or from its source.

    The id says which catalogue to ask (sources.for_id): Deezer ids are bare
    digits and have to be re-signed every few minutes; a `jamendo:...` id
    resolves to a stable CC-licensed audio URL. An id from a source this
    build does not know gets None, which the callers turn into a 404.
    """
    cached = _safe(store.get_cached_preview, track_id)
    if cached:
        return cached
    source = sources.for_id(track_id)
    if source is None:
        return None
    url = source.preview_url(track_id)
    if url:
        _safe(store.put_cached_preview, track_id, url)
    return url


@app.get("/preview/{track_id}")
def preview(track_id: str) -> RedirectResponse:
    """302 to a freshly signed Deezer preview URL.

    A redirect, not a proxy: the audio still streams from Deezer's CDN
    straight to the phone, so this server carries one small request per play
    rather than 480 KB of mp3 -- which is what makes it viable on a free host.

    no-store because the target dies in ~15 minutes; a cached 302 would send
    a client to a URL that 403s long after this response looked fine.
    """
    url = _fresh_preview(track_id)
    if url is None:
        raise HTTPException(404, f"no preview available for track {track_id}")
    return RedirectResponse(url, status_code=302,
                            headers={"Cache-Control": "no-store"})


_AUDIO_CHUNK = 64 * 1024


def _open_upstream(url: str):
    """Open the upstream preview URL. Its own function so tests can stub it."""
    return urllib.request.urlopen(url, timeout=10)


@app.get("/preview/{track_id}/audio")
def preview_audio(track_id: str) -> StreamingResponse:
    """Stream the Deezer preview mp3 through this server, same-origin.

    Used ONLY by the web SOUND mode, which needs the raw bytes: a browser
    cannot read samples out of a cross-origin mp3 (decodeAudioData wants the
    bytes, and Deezer's CDN sends no CORS header), so the spectrogram and
    self-similarity views must fetch the audio from our own origin. Ordinary
    playback still uses the 302 above, which keeps 480 KB per play off this
    host; this route costs that much only when someone opens SOUND mode.

    Cached privately for 10 minutes -- comfortably inside the ~15-minute
    life of the signed upstream URL, and long enough that flipping between
    recs doesn't refetch.
    """
    url = _fresh_preview(track_id)
    if url is None:
        raise HTTPException(404, f"no preview available for track {track_id}")
    try:
        upstream = _open_upstream(url)
    except Exception as exc:  # network flake, 403 on an expired signature, ...
        # Logged, not echoed: the exception text can carry the signed CDN URL.
        print(f"preview audio upstream failed for {track_id}: {exc!r}", flush=True)
        raise HTTPException(502, "upstream preview fetch failed") from exc

    def chunks():
        try:
            while True:
                chunk = upstream.read(_AUDIO_CHUNK)
                if not chunk:
                    return
                yield chunk
        finally:
            upstream.close()

    # An mp3 is already compressed, so gzipping it costs CPU and buys nothing
    # -- and worse, GZipMiddleware would strip the Content-Length and force
    # the <audio> element into a non-seekable stream. `identity` tells
    # Starlette's gzip middleware to pass the body through untouched.
    headers = {
        "Cache-Control": "private, max-age=600",
        "Content-Encoding": "identity",
    }
    length = upstream.headers.get("Content-Length") if hasattr(upstream, "headers") else None
    if length:
        # Both of these are what let the element seek inside the preview
        # instead of treating it as an open-ended stream.
        headers["Content-Length"] = str(length)
        headers["Accept-Ranges"] = "bytes"

    return StreamingResponse(
        chunks(),
        media_type="audio/mpeg",
        headers=headers,
    )

_FIXTURE_PATH = Path(__file__).parents[3] / "contract" / "fixture.json"


def _fixture_tracks() -> list[dict]:
    return json.loads(_FIXTURE_PATH.read_text())["tracks"]


def _fixture_track(track_id: str) -> dict | None:
    return next(
        (t for t in _fixture_tracks() if t["track_id"] == track_id), None
    )


def _download_preview(url: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    path = Path(name)
    with urllib.request.urlopen(url, timeout=10) as resp:
        path.write_bytes(resp.read())
    return path


def _safe(fn, *args, default=None):
    """store call, but a down store means mock-first fallback, not a 500."""
    try:
        return fn(*args)
    except Exception:
        return default


# ---- track metadata cache ----
#
# Title/artist/album/artwork never change once a track is crawled, but every
# endpoint that returns tracks was paying an Atlas round trip for them:
# /recommend did ten sequential get_track calls (1-2 s of the request), and
# /viz/map fetched all VIZ_MAX point dicts every time (0.39 s at 7.5k).
# Metadata is small and immutable, so it is held in process, keyed by id,
# and every read goes through _tracks_cached -- which collapses whatever it
# does not hold into ONE get_many_tracks call.
#
# Cached dicts are treated as read-only by callers (_playable already builds
# a new dict), so entries are never copied on the way out.
_TRACK_META: "OrderedDict[str, dict]" = OrderedDict()
# ~50k tracks of six short strings: tens of MB, an order of magnitude under
# the embedding matrix already in this process. Module-level so tests can
# shrink it.
_TRACK_META_MAX = 50000
# Its own lock, not _VIZ_CACHE_LOCK: this is read on the /recommend path,
# which must not queue behind an Insights request's cache bookkeeping.
_TRACK_META_LOCK = threading.Lock()


def _remember_track(track: dict | None) -> dict | None:
    """Put a track dict in the metadata cache (LRU-trimmed); give back what
    was cached.

    Normalized to the shape store.get_track returns, so a track that arrived
    from /search or Deezer (carrying a live 15-minute preview signature and
    whatever else Deezer sent) cannot leak that expiring URL to a later
    reader: preview_url is "" here exactly as it is out of the store, and
    _playable re-points it at this server.
    """
    if not track or not track.get("track_id"):
        return None
    track_id = track["track_id"]
    entry = _public_track(track)
    entry["preview_url"] = ""
    with _TRACK_META_LOCK:
        _TRACK_META[track_id] = entry
        _TRACK_META.move_to_end(track_id)
        while len(_TRACK_META) > _TRACK_META_MAX:
            _TRACK_META.popitem(last=False)
    return entry


def _tracks_cached(track_ids: "list[str] | tuple[str, ...]") -> list[dict | None]:
    """Metadata for these ids, in order; one bulk fetch for whatever is missing.

    None for an id the store does not know. A miss is NOT cached: an id can
    be absent because the worker has not written its metadata yet, and a
    cached None would outlive that by the life of the process.

    A store failure does not raise: the bulk read goes through _safe(), so an
    Atlas blip degrades to None for every uncached id, which the viz callers
    turn into id-only track placeholders (_unknown_track). The screen keeps
    its geometry and loses only the titles, matching the module-wide fallback
    described in server/CLAUDE.md rather than failing the whole request.
    """
    track_ids = list(track_ids)
    found: dict[str, dict] = {}
    misses: list[str] = []
    with _TRACK_META_LOCK:
        for track_id in track_ids:
            hit = _TRACK_META.get(track_id)
            if hit is not None:
                _TRACK_META.move_to_end(track_id)
                found[track_id] = hit
            elif track_id not in misses:
                misses.append(track_id)

    if misses:
        fetched = _safe(store.get_many_tracks, misses) or [None] * len(misses)
        for track_id, track in zip(misses, fetched):
            if track:
                found[track_id] = _remember_track(track)
    return [found.get(track_id) for track_id in track_ids]


class SeedRequest(BaseModel):
    track_id: str


@app.get("/axes")
def get_axes() -> dict:
    return {"axes": AXES}


@app.get("/search")
def search(q: str) -> dict:
    """Every switched-on source, concatenated in SOURCES order.

    Per-source failures are tolerated -- one catalogue being down must not
    empty a search that another could answer -- but a search where every
    source failed falls back to the fixture, which is what the single-source
    (Deezer-only) case has always done.
    """
    results: list[dict] = []
    failures = 0
    try:
        active = sources.active()
    except Exception:
        active = []
    for source in active:
        try:
            results.extend(_playable(_public_track(t)) for t in source.search(q))
        except Exception:
            failures += 1
    if results or (active and failures < len(active)):
        return {"results": results}
    needle = q.lower()
    hits = [
        t for t in _fixture_tracks()
        if needle in t["title"].lower()
        or needle in t["artist"].lower()
        or needle in t["album"].lower()
    ]
    return {"results": hits}


# ---- GET /search/text — not part of contract/contract.md (like /viz/*) ----

# How many results the text search returns by default. Larger than /search's
# page because the corpus is the whole catalogue rather than one query's
# page, and a phrase like "late night piano" is a region, not a track.
TEXT_LIMIT_DEFAULT = 25
TEXT_LIMIT_MAX = 50


@app.get("/search/text")
def search_text(q: str,
                limit: int = Query(TEXT_LIMIT_DEFAULT, ge=1,
                                   le=TEXT_LIMIT_MAX)) -> dict:
    """Search the CORPUS by description: cosine between the typed phrase and
    every analyzed track, in CLAP's shared audio/text space.

    This is the one thing the clean-room stack gives us that the v1 one
    could not: EffNet has no text tower, so "hazy late-night trumpet" was
    not a question the old corpus could be asked at all.

    MEMORY: the first call LOADS CLAP INTO THE API PROCESS -- ~700 MB of
    weights and roughly **2.5 GB resident** once torch's allocator is warm.
    That is why the import is inside the handler rather than at module
    scope: an API container that never serves a text search never pays it,
    and the box does not have room for a copy in every process by accident.
    On a 2-CPU VM running the API beside the embed worker, this is the
    number to size `mem_limit` against (see deploy/.env.example).

    503, not 500, when CLAP is unavailable (not installed, or the weights
    were never fetched): the rest of the API is fine, and a client should
    hide the toggle rather than report the service down.
    """
    query = q.strip()
    if not query:
        raise HTTPException(400, "q must not be empty")

    try:
        from music_recommendations.analysis import clap

        vector = np.asarray(clap.embed_text([query]), dtype=np.float32).ravel()
    except (ImportError, FileNotFoundError) as exc:
        raise HTTPException(
            503, f"text search unavailable: CLAP could not be loaded ({exc})"
        ) from exc

    corpus = tuple(_safe(store.corpus_ids, default=[]))
    ids, matrix, _ = (_corpus_matrix(corpus, "embedding", "cosine",
                                     want_correction=False)
                      if corpus else ([], np.empty((0, 1)), None))
    if not ids:
        return {"results": []}
    if matrix.shape[1] != vector.shape[0]:
        # A corpus of version-3 (or synthetic) vectors cannot be compared
        # with a CLAP text vector at all. Saying so is the only honest
        # answer; numpy would otherwise raise a shape error as a 500.
        raise HTTPException(
            503, "text search unavailable: the corpus is not in CLAP space "
                 f"({matrix.shape[1]}-d rows against a {vector.shape[0]}-d "
                 "text vector)"
        )

    similarity = _similarity("embedding", matrix, vector, "cosine")
    order = [int(i) for i in np.argsort(similarity)[::-1][:limit]]
    _tracks_cached([ids[i] for i in order])   # one metadata read for the page
    return {"results": [{**_rec_track(ids[i]),
                         "score": round(float(similarity[i]), 4)}
                        for i in order]}


# TensorFlow inference is CPU- and memory-heavy; endpoints run on FastAPI's
# threadpool, so an unbounded burst of /seed calls would run that many
# analyses at once on a 4 GB container. Cap concurrent analyses instead of
# concurrent requests -- requests beyond the cap just wait their turn.
_ANALYZE_SEM = threading.Semaphore(2)


@app.post("/seed")
def seed(req: SeedRequest) -> dict:
    ready = {"track_id": req.track_id, "status": "ready"}
    if _safe(store.get_features, req.track_id) is not None:
        return ready

    track = _safe(store.get_track, req.track_id)
    if track is None:
        track = _from_source(req.track_id)
    if track is None:
        track = _fixture_track(req.track_id)
    if track is None:
        raise HTTPException(404, f"track {req.track_id} not found")

    mp3 = _fetch_preview_audio(req.track_id, track)
    if mp3 is None:
        # Deezer preview fetch failed -- flake, timeout, 404, or a signature
        # that could not be renewed. Don't 500 on a transient failure; hand
        # the job to the worker.
        return _seed_via_worker(req.track_id, track)

    try:
        with _ANALYZE_SEM:
            features = _to_plain(analyze_track(mp3))
        _safe(store.put_track, track, features)
        _remember_track(track)
    except (NotImplementedError, ImportError):
        # Analysis can't run on this host (no aarch64 essentia wheels on the
        # ARM VM). Hand the job to the worker and wait.
        return _seed_via_worker(req.track_id, track)
    except (frontend.DecodeError, ValueError) as exc:
        # The preview itself is bad (undecodable, too short). Not transient,
        # so do not queue it; tell the client (spec §8).
        raise HTTPException(502, "analysis failed") from exc
    finally:
        mp3.unlink(missing_ok=True)
    return ready


def _fetch_preview_audio(track_id: str, track: dict) -> "Path | None":
    """The preview mp3 for analysis, or None if no signature could be had.

    Tries the URL already in hand, then re-signs once. A track that came from
    /search carries a live signature and downloads first try; one read back
    from the store carries a dead one, so the 403 is expected rather than a
    flake. corpus/download.py takes the same try-then-re-sign shape for the
    same reason. urllib raises OSError subclasses (URLError, socket.timeout).
    """
    # Lazily: re-signing costs a Deezer call, so it must not happen when the
    # URL already in hand works.
    for source in (lambda: track.get("preview_url"),
                   lambda: _fresh_preview(track_id)):
        url = source()
        if not url:
            continue
        try:
            return _download_preview(url)
        except OSError:
            continue
    return None


# How long /seed waits for the embed worker before failing loudly. Module
# constants so tests can shrink them instead of sleeping 20 real seconds.
_EMBED_WAIT_S = 20.0
_EMBED_POLL_S = 0.5


def _seed_via_worker(track_id: str, track: dict) -> dict:
    _safe(store.put_track_meta, track)
    _remember_track(track)
    queued = _safe(store.enqueue_embed, track_id)
    if queued is None:
        # The store is down: there is no queue to hand to and no features to
        # await. Mock-first as before -- "ready", fixture-fallback recs.
        return {"track_id": track_id, "status": "ready"}
    status = "ready" if _await_features(track_id) else "unanalyzed"
    return {"track_id": track_id, "status": status}


def _await_features(track_id: str) -> bool:
    """Poll until the worker writes features:{id}, or the wait window closes."""
    deadline = time.monotonic() + _EMBED_WAIT_S
    while True:
        if _safe(store.get_features, track_id) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_EMBED_POLL_S)


class _CorpusMatrix(NamedTuple):
    corpus: tuple[str, ...]        # the corpus:ids state this was built from
    ids: list[str]                 # the rows actually present, in row order
    matrix: np.ndarray
    correction: np.ndarray | None  # centrality, computed only if an axis wants it
    # Every id this key has been LOOKED UP for, whether or not it turned out
    # to have the feature. Sparse keys need it: `feel` is absent on every row
    # scripts/feel_backfill.py has not reached, so those ids never enter
    # `ids` -- and a `fresh` computed from `ids` alone re-queried all of them
    # on every single request, which is the whole cost the cache exists to
    # avoid. An id enters this set once and stays; a row that gains its
    # vector later is picked up at the next process start (or the next
    # corpus rebuild), which is the same cadence a backfill runs on.
    attempted: frozenset[str] = frozenset()


# One assembled matrix per feature key, extended as the corpus grows.
# Every seed ranks against the same vectors, so reading and JSON-parsing each
# track's features per request was pure repetition -- 25 s at 7.8k tracks and
# linear from there. Parsing is the cost, not the round trip, so a cache that
# rebuilt whenever corpus:ids changed bought nothing while a crawl was running:
# it changes every few seconds. Only genuinely new ids are parsed and appended.
# Held in process: the corpus is numpy-sized, not database-sized.
_MATRIX_CACHE: dict[str, _CorpusMatrix] = {}
# Endpoints are sync, so FastAPI runs them on a threadpool: without this two
# concurrent requests would each build the matrix, and the first one to finish
# would be overwritten by the second.
_MATRIX_LOCK = threading.Lock()


def _rows_for(track_ids: list[str], feature_key: str) -> tuple[list[str], list[np.ndarray]]:
    """Fetch and vectorize a set of tracks, skipping any without this feature."""
    ids, rows = [], []
    if feature_key == "feel":
        # Eleven floats per row, so the generic read below was fetching a
        # 1280-byte int8 embedding and dequantizing it to float32 for every
        # candidate purely to throw it away. store.get_many_feel projects
        # `feel` alone and does no dequantization at all.
        for track_id, vector in zip(track_ids, store.get_many_feel(track_ids)):
            if vector:
                ids.append(track_id)
                rows.append(np.atleast_1d(np.asarray(vector, dtype=float)))
        return ids, rows
    for track_id, features in zip(track_ids, store.get_many_features(track_ids)):
        if features and feature_key in features:
            ids.append(track_id)
            rows.append(_vector(features, feature_key))
    return ids, rows


def _cold_matrix(corpus: tuple[str, ...],
                 feature_key: str) -> tuple[list[str], np.ndarray]:
    """Every row, for a cache that has nothing yet.

    The store hands back the whole embedding matrix in one query instead of
    one fetch per track. Anything in `corpus` that the matrix does not yet
    contain (analyzed between the two reads) is appended the usual way.
    """
    if feature_key == "embedding":
        ids, matrix = store.base_matrix()
        # Hoisted: as a comprehension condition this rebuilt the whole set
        # once per candidate.
        known_rows = set(ids)
        extra = [t for t in corpus if t not in known_rows]
        if extra:
            more_ids, rows = _rows_for(extra, feature_key)
            if rows:
                ids = ids + more_ids
                matrix = np.vstack([matrix, np.stack(rows).astype(matrix.dtype)])
        return ids, matrix
    ids, rows = _rows_for(list(corpus), feature_key)
    return ids, (np.stack(rows) if rows else np.empty((0, 1)))


def _corpus_matrix(corpus: tuple[str, ...], feature_key: str, metric: str,
                   want_correction: bool) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """The corpus as one matrix, parsing only what it has not seen before."""
    with _MATRIX_LOCK:
        return _build(corpus, feature_key, metric, want_correction)


def _build(corpus: tuple[str, ...], feature_key: str, metric: str,
           want_correction: bool) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    cached = _MATRIX_CACHE.get(feature_key)
    known = set(cached.ids) if cached else set()
    attempted = cached.attempted if cached else frozenset()
    # One set for both the subset test and the membership checks below;
    # `known.issubset(corpus)` would build its own copy of a 90k-id tuple.
    corpus_set = set(corpus)

    # Tracks only ever get added, so the common case is a short tail of new ids.
    # A track disappearing means someone cleared the store: drop it all and rebuild.
    # The shrink test reads `ids` (the rows actually held), not `attempted`:
    # an id that was looked up and had no vector was never a row, so its
    # absence from the corpus is not evidence the store was cleared.
    if cached is not None and not known.issubset(corpus_set):
        cached, known, attempted = None, set(), frozenset()

    # Iterated over the tuple, not the set: row order must follow corpus order.
    # Against `attempted`, not `known`: an id with no vector for this key is
    # not a row and never will be one, so testing `known` re-fetched every
    # unscored track on every request.
    fresh = [i for i in corpus if i not in attempted]
    if cached is None:
        ids, matrix = _cold_matrix(corpus, feature_key)
        cached = _CorpusMatrix(corpus, ids, matrix, None, frozenset(corpus))
    elif fresh:
        ids, rows = _rows_for(fresh, feature_key)
        seen = attempted | frozenset(fresh)
        if rows:
            cached = _CorpusMatrix(
                corpus, cached.ids + ids,
                # _vector() returns float64; without the cast the whole corpus
                # matrix (float32 from store.base_matrix) would be upcast on
                # the first growth step and double in memory.
                np.vstack([cached.matrix,
                           np.stack(rows).astype(cached.matrix.dtype, copy=False)]),
                None,  # the corpus moved, so any cached centrality is stale
                seen,
            )
        else:
            cached = cached._replace(corpus=corpus, attempted=seen)
    _MATRIX_CACHE[feature_key] = cached

    if want_correction and cached.correction is None and len(cached.ids):
        from music_recommendations.server import rank as rank_mod

        # A property of the corpus rather than of the seed, so it is cached
        # beside it. It therefore includes the seed's own row, where the
        # pre-cache code excluded it -- one row in thousands, and it makes the
        # correction the same for every seed instead of subtly seed-dependent.
        cached = cached._replace(
            correction=rank_mod.centrality(cached.matrix, metric)
        )
        _MATRIX_CACHE[feature_key] = cached

    return cached.ids, cached.matrix, (cached.correction if want_correction else None)


# Normalizing the corpus is the dominant per-request cost on a cosine axis:
# rank.scores() re-divides the whole matrix every call, ~900 MB of allocation
# at 90k tracks. The unit rows only change when rows are appended, and
# _corpus_matrix hands back a NEW ndarray when that happens, so keying on the
# matrix object itself invalidates this cache for free.
#
# The matrix is held IN the tuple, not reduced to id(matrix): a bare id lets
# the array it named be freed, and a later array allocated at the same address
# would then be served a unit matrix belonging to a different (possibly
# differently-sized) corpus. Same reason viz._PAIRWISE_CACHE holds its matrix.
_UNIT_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _similarity(feature_key: str, matrix: np.ndarray, seed_vec: np.ndarray,
                metric: str) -> np.ndarray:
    """Seed against every row, reusing a cached unit matrix where cosine allows."""
    from music_recommendations.server import rank as rank_mod

    if metric != "cosine" or not len(matrix):
        return rank_mod.scores(seed_vec, matrix, metric)

    cached = _UNIT_CACHE.get(feature_key)
    if cached is not None and cached[0] is matrix:
        unit = cached[1]
    else:
        unit = rank_mod.normalize(np.asarray(matrix, dtype=np.float32))
        _UNIT_CACHE[feature_key] = (matrix, unit)
    return unit @ rank_mod.normalize(np.asarray(seed_vec, dtype=np.float32))


# ---- feel: the second half of "sounds like this" ----
#
# Embedding cosine ranks by STYLE. It cannot tell a hushed solo take from a
# full-band blast of the same idiom, because both sit in the same corner of
# CLAP space. The feel vector (eight zero-shot axes, analysis/feel_v2.py)
# carries exactly what the cosine drops: energy, mood, texture. The blend is
#
#     blended = cos(embedding) - feel * feel_dist - tempo * tempo_dist
#
# a straight subtraction rather than a percentile blend (_blended) because
# every term is already on the same scale: a cosine in [-1, 1] against two
# distances that are O(1) by construction. Both weights at 0 reproduce the
# embedding-only order EXACTLY, which is what makes the sliders safe to ship.
#
# feel_dist is a mean absolute difference of Z-SCORES, not of the raw
# probabilities. The v2 axes are zero-shot softmaxes over a contrastive pair
# and they are nothing like equally spread: `acoustic` saturates near 0 or 1
# on almost every track while `density` lives inside a band a few hundredths
# wide. On raw values the widest axis simply decides the ranking and the
# narrow ones are rounding error. Dividing each dimension by its own standard
# deviation over the corpus asks "how unusual is this difference FOR THIS
# AXIS", which is the question the slider is supposed to be weighting.

FEEL_DEFAULT = 0.3
FEEL_MAX = 3.0

# Standard deviations are measured, so a corpus where an axis is constant
# (one track, a synthetic fixture) would divide by zero and turn every
# distance into an inf or a nan. The floor makes such an axis contribute a
# difference of ~0 instead, which is the truth: an axis with no spread
# distinguishes nothing.
_FEEL_STD_FLOOR = 1e-6


class _FeelRows(NamedTuple):
    """The feel vectors of one ranking's rows, lined up with its matrix.

    `rows` and `seed` are the RAW probabilities, because those are what the
    math panel draws as bars; only `distance` is z-scored.
    """
    distance: np.ndarray   # (n,) mean |z(rec) - z(seed)|, 0.0 where missing
    rows: np.ndarray       # (n, 8) aligned vectors, zeros where missing
    present: np.ndarray    # (n,) bool -- whether that row has a feel vector
    seed: np.ndarray       # (8,) the seed's own vector


# The alignment between the embedding matrix's rows and the feel matrix's is
# pure bookkeeping over two id lists, but it is O(corpus) and would otherwise
# be rebuilt on every request. Both matrices are held IN the tuple and
# re-checked with `is` -- the same discipline as _UNIT_CACHE: a bare id() lets
# the array it named be freed and a later array at the same address would be
# served someone else's alignment. Either matrix growing (a crawl, a backfill)
# hands back a new object and invalidates this for free.
# (matrix, feel_matrix, rows, present, mean, std)
_FEEL_ALIGN_CACHE: tuple[np.ndarray, ...] | None = None
_FEEL_ALIGN_LOCK = threading.Lock()
# Rows still waiting on scripts/feel_backfill.py are a deploy-time fact, not a
# per-request one: say it once per process rather than on every /recommend.
_FEEL_MISSING_LOGGED = False


def _feel_alignment(ids: list[str], matrix: np.ndarray, feel_ids: list[str],
                    feel_matrix: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rows, present, mean, std) for `ids`, drawn from the feel matrix.

    `mean`/`std` are per-dimension over the WHOLE feel matrix (every scored
    row in the corpus, not just this request's), which is what makes the
    z-score a statement about the corpus rather than about the candidate
    list. They are computed here, once, because this is the one place that
    already runs exactly when the feel matrix changes identity.
    """
    global _FEEL_ALIGN_CACHE, _FEEL_MISSING_LOGGED

    with _FEEL_ALIGN_LOCK:
        cached = _FEEL_ALIGN_CACHE
        if (cached is not None and cached[0] is matrix
                and cached[1] is feel_matrix):
            return cached[2], cached[3], cached[4], cached[5]

    at = {track_id: row for row, track_id in enumerate(feel_ids)}
    take = np.array([at.get(track_id, -1) for track_id in ids], dtype=np.int64)
    present = take >= 0
    width = feel_matrix.shape[1] if feel_matrix.ndim == 2 else 0
    rows = np.zeros((len(ids), width), dtype=np.float32)
    if width and present.any():
        rows[present] = np.asarray(feel_matrix, dtype=np.float32)[take[present]]

    missing = int(len(ids) - present.sum())
    if missing and not _FEEL_MISSING_LOGGED:
        _FEEL_MISSING_LOGGED = True
        print(f"feel: {missing} of {len(ids)} corpus rows have no feel vector "
              f"and rank unpenalized — run scripts/feel_backfill.py")

    scored = np.asarray(feel_matrix, dtype=np.float32)
    if scored.ndim == 2 and scored.shape[0]:
        mean = scored.mean(axis=0)
        std = np.maximum(scored.std(axis=0), _FEEL_STD_FLOOR)
    else:
        mean = np.zeros(width, dtype=np.float32)
        std = np.ones(width, dtype=np.float32)

    with _FEEL_ALIGN_LOCK:
        _FEEL_ALIGN_CACHE = (matrix, feel_matrix, rows, present, mean, std)
    return rows, present, mean, std


def _feel_rows(corpus: tuple[str, ...], ids: list[str], matrix: np.ndarray,
               seed_features: dict) -> _FeelRows | None:
    """The feel penalty for every row of `matrix`, or None if it cannot apply.

    None when the seed itself has no feel vector, or nothing in the corpus
    does: a partial backfill must never hide tracks, so the absence of the
    number means "no penalty", not "rank last".
    """
    seed = seed_features.get("feel")
    if seed is None or not ids:
        return None
    # euclidean is nominal here -- no correction is asked for, so the metric
    # only ever reaches rank.centrality, which this never calls. The distance
    # below is a mean ABSOLUTE difference, chosen over L2 so one wildly
    # different dimension cannot dominate the other seven.
    feel_ids, feel_matrix, _ = _corpus_matrix(corpus, "feel", "euclidean",
                                              want_correction=False)
    if not feel_ids:
        return None
    rows, present, mean, std = _feel_alignment(ids, matrix, feel_ids, feel_matrix)
    seed_vec = np.asarray(seed, dtype=np.float32).ravel()
    if rows.shape[1] != seed_vec.shape[0]:
        return None
    distance = np.abs((rows - mean) / std - (seed_vec - mean) / std).mean(axis=1)
    distance[~present] = 0.0
    return _FeelRows(distance.astype(np.float32), rows, present, seed_vec)


# ---- tempo: the number a listener can actually name ----
#
# Two tracks can sit in the same corner of CLAP space and the same corner of
# feel space and still be a ballad and a double-time burner. Tempo is the one
# dimension of that the embedding reliably throws away (it is trained on 7 s
# windows with a contrastive objective, not a beat tracker), and it is also
# the dimension a listener can name, so it gets its own term and its own
# slider.
#
# The distance is octave-folded: 90 and 180 BPM are the same groove counted
# differently, and every beat tracker in existence disagrees with every other
# about which one to report. Working in log2 of the ratio makes "double" and
# "half" both exactly 1.0 away, so folding is one min() over three offsets
# rather than a table of special cases.

TEMPO_DEFAULT = 0.2
TEMPO_MAX = 3.0

# Past half an octave there is nothing left to say: the two tracks are simply
# at different tempi, and letting the penalty grow without bound would make
# one 60-BPM outlier beat the entire cosine ranking. Clipping keeps the term
# comparable in size with the cosine it is subtracted from.
TEMPO_MAX_DIST = 0.5


class _RhythmRows(NamedTuple):
    """The rhythm of one ranking's rows, lined up with its matrix."""
    distance: np.ndarray        # (n,) octave-folded tempo distance, 0 if absent
    rows: list[dict | None]     # (n,) the stored rhythm dict, or None
    present: np.ndarray         # (n,) bool -- whether that row has a tempo
    seed: dict                  # the seed's own rhythm dict


# Keyed and re-checked on the embedding matrix's identity, exactly like
# _FEEL_ALIGN_CACHE (and for the same reason: a bare id() can be reused by a
# later array). One Mongo read of `rhythm` for the whole corpus per matrix
# identity, not per request.
_RHYTHM_ALIGN_CACHE: "tuple[np.ndarray, np.ndarray, list[dict | None], np.ndarray] | None" = None
_RHYTHM_ALIGN_LOCK = threading.Lock()


def _rhythm_alignment(ids: list[str], matrix: np.ndarray
                      ) -> tuple[np.ndarray, list[dict | None], np.ndarray]:
    """(tempo, rows, present) for `ids`, in matrix row order."""
    global _RHYTHM_ALIGN_CACHE

    with _RHYTHM_ALIGN_LOCK:
        cached = _RHYTHM_ALIGN_CACHE
        if cached is not None and cached[0] is matrix:
            return cached[1], cached[2], cached[3]

    rows = _safe(store.get_many_rhythm, list(ids), default=None)
    if rows is None or len(rows) != len(ids):
        rows = [None] * len(ids)
    tempo = np.array(
        [float(r.get("tempo_bpm") or 0.0) if isinstance(r, dict) else 0.0
         for r in rows],
        dtype=np.float32,
    )
    # A stored 0.0 means "no beat could be found" (contract/features.py), so
    # it is an absence, not a tempo of zero.
    present = tempo > 0.0

    with _RHYTHM_ALIGN_LOCK:
        _RHYTHM_ALIGN_CACHE = (matrix, tempo, rows, present)
    return tempo, rows, present


def _tempo_distance(seed_bpm: float, tempo: np.ndarray,
                    present: np.ndarray) -> np.ndarray:
    """min(|d|, |d-1|, |d+1|) for d = log2(seed / candidate), clipped.

    0 wherever the candidate has no tempo: a missing number must never be a
    penalty, or a partial backfill would quietly hide half the corpus.
    """
    distance = np.zeros(tempo.shape, dtype=np.float32)
    if seed_bpm <= 0.0 or not present.any():
        return distance
    ratio = np.log2(seed_bpm / np.where(present, tempo, 1.0))
    folded = np.minimum(np.abs(ratio),
                        np.minimum(np.abs(ratio - 1.0), np.abs(ratio + 1.0)))
    distance = np.clip(folded, 0.0, TEMPO_MAX_DIST).astype(np.float32)
    distance[~present] = 0.0
    return distance


def _rhythm_rows(corpus: tuple[str, ...], ids: list[str], matrix: np.ndarray,
                 seed_features: dict) -> _RhythmRows | None:
    """The tempo penalty for every row of `matrix`, or None if it cannot apply.

    None when the seed has no rhythm at all -- same rule as the feel term:
    absence means "no penalty", never "rank last".
    """
    seed = seed_features.get("rhythm")
    if not isinstance(seed, dict) or not ids:
        return None
    tempo, rows, present = _rhythm_alignment(ids, matrix)
    distance = _tempo_distance(float(seed.get("tempo_bpm") or 0.0),
                               tempo, present)
    return _RhythmRows(distance, rows, present, seed)


def _tempo_blend(axis: str, weight: float, similarity: np.ndarray,
                 rhythm: _RhythmRows | None) -> np.ndarray:
    """`similarity` minus the tempo penalty, on sounds_like only.

    surprise is untouched for the same reason the feel term leaves it alone:
    "nothing like this" is already a request to leave the neighbourhood.
    """
    if axis != "sounds_like" or rhythm is None or not weight:
        return similarity
    return similarity - weight * rhythm.distance


# What the math panel shows for a row with no rhythm on either side. Spelled
# once so the keys are always present in the response, which is what lets the
# client test `math.rhythm != null` instead of probing for the key.
_NO_RHYTHM_MATH = {"tempo_dist": None, "rhythm": None}


def _rhythm_math(rhythm: _RhythmRows | None, index: int) -> dict:
    """The math-panel half: both rhythm dicts and the distance between them."""
    if rhythm is None or not rhythm.present[index]:
        return dict(_NO_RHYTHM_MATH)
    return {
        "tempo_dist": round(float(rhythm.distance[index]), 4),
        "rhythm": {"seed": dict(rhythm.seed),
                   "rec": dict(rhythm.rows[index] or {})},
    }


def _feel_blend(axis: str, weight: float, similarity: np.ndarray,
                feel: _FeelRows | None) -> np.ndarray:
    """`similarity` minus the feel penalty, on sounds_like only.

    surprise is deliberately untouched: "nothing like this" is already a
    request to leave the seed's neighbourhood, and penalizing a different
    feel there would pull the answers back towards it.
    """
    if axis != "sounds_like" or feel is None or not weight:
        return similarity
    return similarity - weight * feel.distance


def _feel_math(feel: _FeelRows | None, index: int) -> dict:
    """The math-panel half: the two vectors and the distance between them."""
    if feel is None or not feel.present[index]:
        return {"feel_dist": None, "feel": None}
    return {
        "feel_dist": round(float(feel.distance[index]), 4),
        "feel": {"seed": [round(float(v), 4) for v in feel.seed],
                 "rec": [round(float(v), 4) for v in feel.rows[index]]},
    }


def _percentile(values: np.ndarray) -> np.ndarray:
    """Each score as "better than X% of the corpus", 0-100.

    Blending raw scores across feature keys is meaningless: cosine and
    euclidean-derived similarities occupy different ranges, so a weighted sum
    is decided by whichever happens to have the wider spread. Percentiles are
    scale-free, and they are also the only version of the number a listener
    can read -- 0.835 says nothing, 99.9 says a great deal.
    """
    if len(values) < 2:
        return np.zeros(len(values))
    return 100.0 * values.argsort().argsort() / (len(values) - 1)


def _blended(corpus: tuple[str, ...], seed_features: dict,
             weights: dict[str, float]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Weighted mean of per-key percentiles; also returns each key's own."""
    from music_recommendations.server import rank as rank_mod

    base_ids: list[str] = []
    fused = np.zeros(0)
    parts: dict[str, np.ndarray] = {}

    for feature_key, weight in weights.items():
        metric = METRICS.get(feature_key, "cosine")
        ids, matrix, _ = _corpus_matrix(corpus, feature_key, metric,
                                        want_correction=False)
        if not ids or feature_key not in seed_features:
            continue
        ranked = _percentile(
            rank_mod.scores(_vector(seed_features, feature_key), matrix, metric)
        )
        if not base_ids:
            base_ids, fused = ids, np.zeros(len(ids))
        elif ids != base_ids:
            # A track missing one feature is not in that key's rows; line the
            # columns back up rather than adding mismatched positions.
            at = {t: i for i, t in enumerate(ids)}
            ranked = np.array([ranked[at[t]] if t in at else 0.0 for t in base_ids])
        parts[feature_key] = ranked
        fused = fused + weight * ranked

    return base_ids, fused, parts


# Second net, after the crawler's key guard: whatever duplicates are already
# stored must not reach a result list twice.
#
# Two candidates are the same recording when their dedupe keys match, or --
# for a re-release whose title says nothing -- when their embeddings are
# within this cosine. 0.995 was measured: 8.8% of a 20k corpus has a
# near-identical partner at that threshold, and 320 of 323 sampled pairs
# above it were the same recording under another Deezer id (the rest were
# genuine covers, which the key check keeps because it includes the artist).
_NEAR_DUPLICATE_COSINE = 0.995

# Every candidate the collapse drops needs another to take its place, so the
# ranked scan looks this many times `limit` deep. One extra pass is enough
# for a corpus where duplicates are ~10% of rows, and it keeps the whole
# thing bounded: at most 2*limit+1 candidates, `limit` kept keys, `limit`
# kept rows -- O(limit^2) dot products and ONE bulk metadata read.
_COLLAPSE_WIDEN = 1


def _scan_width(limit: int) -> int:
    """How many ranked candidates a result loop may look at for `limit` rows."""
    return limit * (1 + _COLLAPSE_WIDEN) + 1


def _dedupe_key(track_id: str) -> str | None:
    """A track's dedupe key from cached metadata; None if we have no metadata.

    Read from the metadata cache rather than the stored `dedupe_key` field:
    the cache is already warm for exactly these ids, so this costs nothing,
    and the field never has to be projected into a contract response.
    """
    meta = _tracks_cached([track_id])[0]
    if not meta:
        return None
    return dedupe.dedupe_key(meta.get("title"), meta.get("artist"))


def _unit_rows(feature_key: str, matrix: np.ndarray) -> np.ndarray | None:
    """The unit matrix _similarity just cached for this matrix, if any."""
    cached = _UNIT_CACHE.get(feature_key)
    return cached[1] if cached is not None and cached[0] is matrix else None


class _Collapse:
    """Rejects a result candidate that repeats the seed or an earlier result.

    Holds at most `limit` keys and `limit` unit rows -- the state is the
    result list itself, never the corpus.
    """

    def __init__(self, seed_id: str, matrix: np.ndarray | None = None,
                 unit: np.ndarray | None = None):
        self._matrix = matrix
        self._unit = unit
        self._seed_id = seed_id
        # Resolved on first accept(), not here: the caller warms the whole
        # candidate window -- the seed included -- in ONE bulk metadata
        # read, and reading the seed's key in the constructor would cost a
        # round trip of its own ahead of it.
        self._keys: set[str] | None = None
        self._rows: list[np.ndarray] = []

    def _row(self, index: int | None) -> np.ndarray | None:
        if index is None or self._matrix is None:
            return None
        if self._unit is not None:
            return self._unit[index]
        # No cached unit matrix (a non-cosine axis, or a matrix _similarity
        # did not normalize): normalize the ONE row, not the corpus.
        vec = np.asarray(self._matrix[index], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else None

    def accept(self, track_id: str, index: int | None = None) -> bool:
        """True if this candidate is a new recording; records it if so."""
        if self._keys is None:
            seed_key = _dedupe_key(self._seed_id)
            self._keys = {seed_key} if seed_key else set()
        key = _dedupe_key(track_id)
        if key and key in self._keys:
            return False
        row = self._row(index)
        if row is not None and any(float(row @ prev) > _NEAR_DUPLICATE_COSINE
                                   for prev in self._rows):
            return False
        if key:
            self._keys.add(key)
        if row is not None:
            self._rows.append(row)
        return True


def _take(order: np.ndarray, ids: list[str], seed_id: str,
          limit: int, collapse: "_Collapse | None" = None) -> list[int]:
    """The first `limit` ranked rows that are not the seed (nor a duplicate).

    Also warms the metadata cache for exactly those rows: building the
    result list first and fetching after turns what used to be `limit`
    sequential store.get_track round trips (1-2 s of every /recommend) into
    one bulk read, or none at all once the ids are cached.

    With a `collapse`, the whole candidate window is warmed UP FRONT
    instead: the collapse test reads each candidate's title and artist, and
    fetching those one at a time would undo the single round trip this
    function exists for.
    """
    if collapse is not None:
        order = [int(idx) for idx in order[:_scan_width(limit)]]
        _tracks_cached([seed_id, *(ids[idx] for idx in order)])

    chosen: list[int] = []
    for idx in order:
        idx = int(idx)
        if ids[idx] == seed_id:
            continue
        if collapse is not None and not collapse.accept(ids[idx], idx):
            continue
        chosen.append(idx)
        if len(chosen) == limit:
            break
    if collapse is None:
        _tracks_cached([ids[idx] for idx in chosen])
    return chosen


def _rec_track(track_id: str) -> dict:
    """A result's track dict, from the metadata cache _take just warmed."""
    return _playable(_tracks_cached([track_id])[0]) or {"track_id": track_id}


@app.get("/recommend")
def recommend(track_id: str, axis: str,
              limit: int = Query(10, ge=1, le=50),
              feel: float = Query(FEEL_DEFAULT, ge=0, le=FEEL_MAX),
              tempo: float = Query(TEMPO_DEFAULT, ge=0, le=TEMPO_MAX)) -> dict:
    """`feel` weights the eight-dimension feel penalty on sounds_like and
    `tempo` the octave-folded tempo penalty. Both are ignored on every other
    axis, and both at 0 give the embedding-only order.

    `score` is the number the list was RANKED by, so on sounds_like it is
    the blended value

        cos - feel * feel_dist - tempo * tempo_dist

    and can therefore sit below the raw cosine, and below zero. With both
    weights at 0 (and on every other axis) it is exactly the raw
    similarity."""
    if axis not in AXIS_FEATURES and axis not in BLENDED_AXES:
        raise HTTPException(400, f"unknown axis {axis!r}")

    seed_features = _safe(store.get_features, track_id)
    corpus = tuple(_safe(store.corpus_ids, default=[]))
    ranked_against = [i for i in corpus if i != track_id]

    if seed_features is None or not ranked_against:
        results = _fixture_fallback(track_id, limit)
    elif axis in BLENDED_AXES:
        ids, fused, parts = _blended(corpus, seed_features, BLENDED_AXES[axis])
        # No matrix to compare rows in: a blended axis scores in no single
        # vector space, so duplicates are collapsed on the key alone.
        chosen = _take(np.argsort(fused)[::-1], ids, track_id, limit,
                       _Collapse(track_id))
        results = [
            {**_rec_track(ids[idx]), "score": round(float(fused[idx]) / 100.0, 4)}
            for idx in chosen
        ]
    else:
        feature_key, direction = AXIS_FEATURES[axis]
        from music_recommendations.server import rank as rank_mod

        # The right metric depends on how the vector was built, so analysis
        # declares it per feature key rather than the server assuming cosine.
        metric = METRICS.get(feature_key, "cosine")
        ids, matrix, correction = _corpus_matrix(corpus, feature_key, metric,
                                                 want_correction=direction == -1)
        seed_vec = _vector(seed_features, feature_key)

        # The seed is a row in the cached matrix like any other, so ask for one
        # extra and drop it — cheaper than rebuilding the matrix per seed.
        # Wider still (_scan_width) so the duplicates _take collapses have
        # replacements to draw on; rank() sorts the whole column regardless
        # of `limit`, so asking for more is free.
        similarity = _similarity(feature_key, matrix, seed_vec, metric)
        # The blend is what is RANKED and what is REPORTED: a result whose
        # score was not the number it was sorted by would read as an
        # out-of-order list in the client.
        blended = _feel_blend(
            axis, feel,
            similarity,
            _feel_rows(corpus, ids, matrix, seed_features) if feel else None,
        )
        blended = _tempo_blend(
            axis, tempo, blended,
            _rhythm_rows(corpus, ids, matrix, seed_features) if tempo else None,
        )
        order = rank_mod.rank(
            seed_vec, matrix, direction=direction, limit=_scan_width(limit),
            metric=metric, correction=correction, similarity=blended,
        )
        chosen = _take(order, ids, track_id, limit,
                       _Collapse(track_id, matrix, _unit_rows(feature_key, matrix)))
        results = [
            {**_rec_track(ids[idx]), "score": float(blended[idx])}
            for idx in chosen
        ]

    return {"seed_track_id": track_id, "axis": axis, "results": results}


# ---- /viz/map — demo/debug, NOT part of contract/contract.md (like GET /) ----

# The top-8 PC projection of the embedding matrix, kept beside the matrix it
# was computed from: recomputed only when _corpus_matrix hands back a new
# object (corpus grew or was rebuilt), served from memory otherwise. One SVD
# serves /viz/map + /viz/walk (columns 0:2) and /viz/tour + /viz/extremes
# (all 8 columns). Keyed by id(matrix), not a single fixed slot: the
# seed-anchored subset means two different seeds' matrices can both be live
# at once (a caller alternating between two seeds), so a single-entry cache
# thrashed on every other request. The matrix is stored in the VALUE and
# re-checked with `is`, so a reused id is a miss rather than a wrong answer,
# and entries whose subset matrix is no longer cached are purged by
# _viz_subset (see _purge_viz_caches) instead of pinning it in memory.
_TOP8_CACHE: "OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

# The MST edge list of the embedding matrix, keyed the same way. Not part of
# contract/contract.md — see /viz/mst below.
_MST_CACHE: "OrderedDict[int, tuple[np.ndarray, list[tuple[int, int, float]]]]" = OrderedDict()

# /viz/hubs' two per-row arrays (neighbour counts, mean centrality), keyed by
# (id(matrix), k) because the count depends on k while the centrality does
# not — one entry per (subset, k) is simpler than splitting them, and both
# are one float/int per row (~64 KB at VIZ_MAX) against the subset matrix's
# ~40 MB. Same identity discipline as the two above: the matrix is stored in
# the value and re-checked with `is`, so a reused id is a miss.
_HUBS_CACHE: "OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
_CACHE_KEEP = 4

# One lock for all four viz caches (_SUBSET_CACHE, _TOP8_CACHE, _MST_CACHE,
# _HUBS_CACHE).
# Endpoints run on FastAPI's threadpool, and every one of these caches is a
# read-modify-write (lookup + move_to_end, insert + evict, purge): without
# this, two concurrent /viz calls could interleave a move_to_end with a
# popitem and drop the entry that was just promoted, or leave an OrderedDict
# mid-mutation. Deliberately NOT _MATRIX_LOCK: _viz_subset calls
# _corpus_matrix, which takes that lock, so sharing one would self-deadlock.
_VIZ_CACHE_LOCK = threading.Lock()


def _top8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = id(matrix)
    with _VIZ_CACHE_LOCK:
        cached = _TOP8_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _TOP8_CACHE.move_to_end(key)
            return cached[1], cached[2]
    # The SVD runs outside the lock: it is the expensive part, and two
    # threads racing on the same matrix would only compute it twice.
    coords8, variance = viz.project_top8(matrix)
    with _VIZ_CACHE_LOCK:
        _TOP8_CACHE[key] = (matrix, coords8, variance)
        while len(_TOP8_CACHE) > _CACHE_KEEP:
            _TOP8_CACHE.popitem(last=False)
    return coords8, variance


# The UMAP layout of a subset matrix, keyed exactly like _TOP8_CACHE: UMAP
# is seconds of work at VIZ_MAX rows (single-threaded, because a fixed
# random_state disables its parallelism), so recomputing it per request is
# not an option. Same identity discipline -- matrix in the value, re-checked
# with `is` -- and purged with the others when its subset is superseded.
_UMAP_CACHE: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = OrderedDict()


def _projection(matrix: np.ndarray) -> np.ndarray:
    """The (n, 2) galaxy coordinates: UMAP, cached per subset identity.

    /viz/map and /viz/walk only. /viz/tour and /viz/extremes stay on _top8
    because they are ABOUT the principal components -- a UMAP axis has no
    variance fraction to report and no "most extreme on PC3" to rank by.

    Under viz.UMAP_MIN_ROWS rows project_umap falls back to the same top-2
    PCA columns this used to return, so small corpora are unchanged.
    """
    key = id(matrix)
    with _VIZ_CACHE_LOCK:
        cached = _UMAP_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _UMAP_CACHE.move_to_end(key)
            return cached[1]
    # Outside the lock, like _top8: it is the expensive part, and two threads
    # racing on one matrix would only compute it twice.
    xy = viz.project_umap(matrix)
    with _VIZ_CACHE_LOCK:
        _UMAP_CACHE[key] = (matrix, xy)
        while len(_UMAP_CACHE) > _CACHE_KEEP:
            _UMAP_CACHE.popitem(last=False)
    return xy


def _mst(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    key = id(matrix)
    with _VIZ_CACHE_LOCK:
        cached = _MST_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _MST_CACHE.move_to_end(key)
            return cached[1]
    edges = viz.minimum_spanning_tree(matrix)
    with _VIZ_CACHE_LOCK:
        _MST_CACHE[key] = (matrix, edges)
        while len(_MST_CACHE) > _CACHE_KEEP:
            _MST_CACHE.popitem(last=False)
    return edges


def _compute_hub_arrays(matrix: np.ndarray,
                        neighbor_k: int) -> tuple[np.ndarray, np.ndarray]:
    """(neighbour counts, mean centrality) per row of the subset matrix.

    Deliberately no `similarity.copy()`: the pairwise matrix is n^2 float32
    (256 MB at VIZ_MAX), so copying it to blank the diagonal doubled the peak
    for one number per row. Take the top k+1 per row instead — the self index
    is always among them, being the row's maximum — and drop the one self
    entry per row, which is exactly the old top-k-excluding-self set.
    """
    rows = len(matrix)
    similarity = viz.pairwise_cosine(matrix)
    # kth=neighbor_k is in range: neighbor_k <= rows - 1 by construction.
    neighbors = np.argpartition(-similarity, neighbor_k, axis=1)[:, :neighbor_k + 1]
    # Exactly one self index per row (argpartition returns distinct indices).
    keep = neighbors != np.arange(rows)[:, None]
    neighbors = neighbors[keep].reshape(rows, neighbor_k)
    counts = np.bincount(neighbors.ravel(), minlength=rows)
    centrality = (similarity.sum(axis=1) - np.diag(similarity)) / (rows - 1)
    return counts, centrality


def _hub_arrays(matrix: np.ndarray,
                neighbor_k: int) -> tuple[np.ndarray, np.ndarray]:
    """_compute_hub_arrays, memoized per (subset matrix, k).

    The arrays are deterministic in the subset and k, and the Insights screen
    asks for the same subset repeatedly (the hub list, then the same list with
    a different limit), so an uncached /viz/hubs paid the O(n^2) argpartition
    every time — the measured 2.4 s warm.
    """
    key = (id(matrix), neighbor_k)
    with _VIZ_CACHE_LOCK:
        cached = _HUBS_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _HUBS_CACHE.move_to_end(key)
            return cached[1], cached[2]
    # Outside the lock, like _top8: the argpartition is the expensive part and
    # a race only costs a duplicate computation.
    counts, centrality = _compute_hub_arrays(matrix, neighbor_k)
    with _VIZ_CACHE_LOCK:
        _HUBS_CACHE[key] = (matrix, counts, centrality)
        while len(_HUBS_CACHE) > _CACHE_KEEP:
            _HUBS_CACHE.popitem(last=False)
    return counts, centrality


@app.get("/viz/map")
def viz_map(track_id: str, axis: str,
            limit: int = Query(10, ge=1, le=50),
            correction: Literal["on", "off"] = "on",
            points: Literal["full", "compact"] = "full",
            feel: float = Query(FEEL_DEFAULT, ge=0, le=FEEL_MAX),
            tempo: float = Query(TEMPO_DEFAULT, ge=0, le=TEMPO_MAX)) -> dict:
    """Everything the wow screen needs in one payload: the whole corpus as 2D
    points, the seed, and the recs with the actual numbers behind each score.

    `points=compact` drops points.tracks — the ids/x/y are all a client that
    only draws the galaxy needs, and the per-point track dicts are ~2 MB of
    the 2.1 MB response at VIZ_MAX rows. Default stays "full" so existing
    clients are untouched.

    The rec list here is drawn from the viz SNAPSHOT, so it can lag
    /recommend by up to VIZ_REFRESH_S: a track analyzed inside the refresh
    window has no row to be positioned at, and every client reads rec.x and
    rec.y unconditionally, so it is skipped and the next-best candidate
    takes its place (the list is still `limit` long). It appears at the next
    refresh. /recommend itself is unaffected — it always ranks and returns
    over the live matrix.
    """
    blended_weights = BLENDED_AXES.get(axis)
    if blended_weights is None and axis not in AXIS_FEATURES:
        raise HTTPException(400, f"unknown axis {axis!r}")

    seed_features = _safe(store.get_features, track_id)
    corpus = tuple(_safe(store.corpus_ids, default=[]))
    if seed_features is None or not corpus:
        raise HTTPException(404, f"track {track_id} not analyzed")

    # The map is always embedding space, whatever the axis scores with.
    emb_ids, emb_matrix, _ = _corpus_matrix(corpus, "embedding", "cosine",
                                            want_correction=False)
    if track_id not in emb_ids:
        raise HTTPException(404, f"track {track_id} not in corpus")

    # Ranked on the live matrix, drawn on the SNAPSHOT: the ranking must be
    # the one /recommend just served, while the projection must reuse the
    # subset/PCA the other Insights endpoints already hold. Taken before the
    # ranking so a candidate the snapshot predates can be FILTERED OUT of
    # the list (see the docstring) rather than forcing a refresh -- one
    # newly crawled track landing in someone's top ten would otherwise
    # rebuild the snapshot and cold-start every derived cache.
    snapshot_ids, snapshot_matrix = _viz_snapshot(require=track_id,
                                                  live=(emb_ids, emb_matrix))
    snapshot_id_set = set(snapshot_ids)
    # Every row the snapshot is missing is a candidate the loop below may
    # skip, so ask the ranking for that many more: the list stays `limit`
    # long whenever there are older candidates left to fill it. rank() sorts
    # the whole column regardless of `limit`, so a wider ask is free.
    skippable = max(0, len(emb_ids) - len(snapshot_id_set))

    from music_recommendations.server import rank as rank_mod

    if blended_weights is not None:
        # A blended axis scores in no single vector space, so there is no one
        # metric to report. Rank on the same fused percentile /recommend uses:
        # Insights has to explain the list the user actually saw, and a second
        # ranking here would show different neighbours than the rec list did.
        direction, use_correction = 1, False
        seed_vec = _vector(seed_features, "embedding")
        ids, fused, parts = _blended(corpus, seed_features, blended_weights)
        order = np.argsort(fused)[::-1]
        # The panel still gets real arithmetic: embedding cosine is the number
        # the map's own axes are drawn from, and `parts` carries each key's
        # percentile so the blend can be shown as the sum it is.
        emb_at = {tid: i for i, tid in enumerate(emb_ids)}
        metric, matrix, correction = "cosine", emb_matrix, None
        feel_rows = rhythm_rows = None
    else:
        feature_key, direction = AXIS_FEATURES[axis]
        metric = METRICS.get(feature_key, "cosine")
        use_correction = direction == -1 and correction == "on"
        ids, matrix, correction = _corpus_matrix(corpus, feature_key, metric,
                                                 want_correction=use_correction)
        seed_vec = _vector(seed_features, feature_key)
        similarity = _similarity(feature_key, matrix, seed_vec, metric)
        # Computed even at feel=0, unlike /recommend: the math panel shows the
        # per-dimension comparison whether or not it is currently weighted,
        # and that is the whole point of a slider you can turn back down.
        # But only where it can ever apply: _feel_blend and _feel_math both
        # ignore it off sounds_like, so building the whole feel matrix for a
        # `surprise` map was pure cost.
        feel_rows = (_feel_rows(corpus, ids, matrix, seed_features)
                     if axis == "sounds_like" else None)
        rhythm_rows = (_rhythm_rows(corpus, ids, matrix, seed_features)
                       if axis == "sounds_like" else None)
        similarity = _feel_blend(axis, feel, similarity, feel_rows)
        similarity = _tempo_blend(axis, tempo, similarity, rhythm_rows)
        order = rank_mod.rank(seed_vec, matrix, direction=direction,
                              limit=_scan_width(limit) + skippable, metric=metric,
                              correction=correction, similarity=similarity)

    emb_id_set = set(emb_ids)
    # Same two nets as /recommend (see _Collapse): Insights explains the list
    # the user saw, so it must collapse the same duplicates. A blended axis
    # has no single vector space, so it gets the key check only.
    collapse = _Collapse(track_id, *((None, None) if blended_weights is not None
                                     else (matrix, _unit_rows(feature_key, matrix))))
    # The scan is bounded (widened by the rows the snapshot is missing, and
    # again by the duplicates the collapse may drop), and its metadata is
    # read in one go: _Collapse reads a title and artist per candidate.
    order = [int(idx) for idx in order[:_scan_width(limit) + skippable]]
    _tracks_cached([track_id, *(ids[idx] for idx in order)])
    recs = []
    for idx in order:
        rec_id = ids[idx]
        if rec_id == track_id or rec_id not in emb_id_set:
            continue
        # Before the limit truncation, so the list is still `limit` long and
        # every entry has a row to be positioned at.
        if rec_id not in snapshot_id_set:
            continue
        if not collapse.accept(rec_id, None if blended_weights is not None else idx):
            continue
        if blended_weights is not None:
            row = emb_at.get(rec_id)
            if row is None:
                continue
            score = round(float(fused[idx]) / 100.0, 4)
            math = viz.score_math(seed_vec, matrix[row], metric, None)
            math["metric"] = "blend"
            math["parts"] = {
                key: round(float(values[idx]), 1)
                for key, values in parts.items()
            }
            math.update(_feel_math(None, idx))
            math.update(_rhythm_math(None, idx))
        else:
            score = float(similarity[idx])
            math = viz.score_math(
                seed_vec, matrix[idx], metric,
                float(correction[idx]) if correction is not None else None,
            )
            # feel_rows/rhythm_rows are already None off sounds_like.
            math.update(_feel_math(feel_rows, idx))
            math.update(_rhythm_math(rhythm_rows, idx))
        recs.append({"track_id": rec_id, "score": score, "math": math})
        if len(recs) == limit:
            break

    # The points/projection are seed-anchored, not the whole corpus: recs
    # (e.g. `surprise`'s far neighbours) are drawn even when they fall
    # outside the nearest-VIZ_MAX ring, via extra_ids.
    subset_ids, subset_matrix = _viz_subset(
        track_id, extra_ids=[rec["track_id"] for rec in recs],
        corpus=(snapshot_ids, snapshot_matrix),
    )
    xy = _projection(subset_matrix)
    position = {tid: i for i, tid in enumerate(subset_ids)}

    # One bulk metadata read for everything this response names, before any
    # of it is assembled: in "full" that is the whole subset (the points),
    # in "compact" only the seed and the recs.
    named = (list(subset_ids) if points == "full"
             else [track_id, *(rec["track_id"] for rec in recs)])
    named_meta = _tracks_cached(named)

    recs = [
        {**_viz_track(rec["track_id"]), **rec,
         "x": float(xy[position[rec["track_id"]], 0]),
         "y": float(xy[position[rec["track_id"]], 1])}
        for rec in recs
    ]

    seed_track = _playable(_tracks_cached([track_id])[0]) or {"track_id": track_id}
    seed_pos = position[track_id]
    seed = {
        **seed_track,
        "x": float(xy[seed_pos, 0]),
        "y": float(xy[seed_pos, 1]),
    }
    axis_info = {"id": axis, "metric": metric, "direction": direction}
    if blended_weights is not None:
        axis_info["metric"] = "blend"
        axis_info["weights"] = dict(blended_weights)
    if direction == -1:
        axis_info["correction"] = "on" if use_correction else "off"

    point_data = {
        "ids": subset_ids,
        "x": [round(float(v), 4) for v in xy[:, 0]],
        "y": [round(float(v), 4) for v in xy[:, 1]],
    }
    if points == "full":
        # Straight off the one bulk read above rather than _viz_track per
        # row: at VIZ_MAX that would be 8000 cache lookups to the same end.
        point_data["tracks"] = [
            _playable(track) or _unknown_track(point_id)
            for point_id, track in zip(subset_ids, named_meta)
        ]

    return {
        "points": point_data,
        "seed": seed,
        "recs": recs,
        "axis": axis_info,
        # Sent rather than hardcoded in the client: the dimension order is
        # analysis/feel_v2.PROMPT_BANK's insertion order, and a client with
        # its own copy of the list would mislabel every bar the day an axis
        # moves.
        "feel_keys": list(FEEL_KEYS),
    }


# Insights endpoints (tour/mst/hubs/walk/map) do dense n x n work over
# whatever matrix they're handed; VIZ_MAX bounds that matrix's row count
# regardless of how large the analyzed corpus grows. Read at call time
# (not cached at import) so tests can monkeypatch it per-test.
VIZ_MAX = int(os.environ.get("VIZ_MAX", "8000"))

# Seed id + extra ids -> (matrix_all, ids, subset_matrix) for that seed's
# subset. The full matrix the subset was sliced from is stored alongside it
# (mirroring _top8/_mst) and re-checked with `is` on every lookup: the
# subset itself is a COPY, not a view, so it does not by itself keep
# id(matrix_all) from being reused by a later, differently-sized matrix once
# _MATRIX_CACHE replaces the original during a crawl. A key whose stored
# matrix no longer matches is evicted and treated as a miss rather than
# risking rows from a stale corpus, and every miss also purges the entries
# belonging to superseded full matrices (_purge_viz_caches) so a stale key
# cannot pin one in memory. The cap bounds what is left.
_SUBSET_CACHE: "OrderedDict[tuple[int, str | None, tuple[str, ...]], tuple[np.ndarray, list[str], np.ndarray]]" = OrderedDict()
# Four, not two: one Insights screen asks for the seed's subset with the recs
# of the corrected map, the recs of the raw map, and (from /viz/tour, /viz/mst,
# /viz/hubs, /viz/extremes) the recs it was handed — distinct `extra` tuples,
# so a 2-entry cap thrashed within a single screen load.
_SUBSET_KEEP = 4


def _purge_viz_caches(matrix_all: np.ndarray) -> None:
    """Drop every viz cache entry that belongs to a superseded full matrix.

    Caller must hold _VIZ_CACHE_LOCK. A _SUBSET_CACHE value holds a strong
    reference to the full matrix it was sliced from, so one stale entry pins
    a whole ~450 MB corpus matrix that _MATRIX_CACHE has already replaced.
    LRU eviction alone does not do this: the stale entry can stay inside the
    keep window indefinitely if it is never looked up again. _TOP8_CACHE,
    _MST_CACHE, _HUBS_CACHE and _UMAP_CACHE pin subset matrices the same
    way, so they are purged down to whatever subsets are still cached.
    """
    for key in [k for k, v in _SUBSET_CACHE.items() if v[0] is not matrix_all]:
        _SUBSET_CACHE.pop(key, None)
    live = [value[2] for value in _SUBSET_CACHE.values()]
    for cache in (_TOP8_CACHE, _MST_CACHE, _HUBS_CACHE, _UMAP_CACHE):
        for key in [k for k, v in cache.items()
                    if not any(v[0] is subset for subset in live)]:
            cache.pop(key, None)


# (matrix, row norms) for whatever matrix _seed_cosine last saw. One float
# per row, so this is ~30 KB at 8k tracks against the matrix's ~40 MB.
_ROW_NORMS: "tuple[np.ndarray, np.ndarray] | None" = None


def _seed_cosine(matrix_all: np.ndarray, seed_row: int) -> np.ndarray:
    """One seed row against every row, cosine, without a unit matrix.

    Deliberately NOT _similarity: that fills _UNIT_CACHE, which is a single
    slot holding the LIVE matrix for /recommend. Calling it here on the
    snapshot matrix evicted the live one, so /recommend and every subset
    miss took turns re-normalizing a whole corpus (1.2 GB of allocation
    each at 230k rows) to serve the other. Dividing the raw matrix-vector
    product by the row norms is the same number, and the norms are one
    float per row, cached beside the matrix they belong to.
    """
    global _ROW_NORMS

    seed_vec = np.asarray(matrix_all[seed_row], dtype=np.float32)
    seed_norm = float(np.linalg.norm(seed_vec))
    unit_seed = seed_vec / (seed_norm or 1.0)

    with _VIZ_CACHE_LOCK:
        cached = _ROW_NORMS
    if cached is not None and cached[0] is matrix_all:
        norms = cached[1]
    else:
        norms = np.linalg.norm(matrix_all, axis=1)
        with _VIZ_CACHE_LOCK:
            _ROW_NORMS = (matrix_all, norms)

    return (matrix_all @ unit_seed) / np.where(norms == 0.0, 1.0, norms)


def _viz_subset(seed_id: str | None,
                extra_ids: "list[str] | tuple[str, ...]" = (),
                corpus: "tuple[list[str], np.ndarray] | None" = None,
                ) -> tuple[list[str], np.ndarray]:
    """VIZ_MAX rows plus any extra ids, of the corpus, for the insights endpoints.

    With a seed: the seed plus its VIZ_MAX-1 nearest tracks by cosine over the
    FULL matrix (one matrix-vector product), plus any extra ids the caller
    needs drawn (an axis's recs, which for `surprise` are far away). Without
    a seed: the first VIZ_MAX rows, for clients that predate the parameter.
    Row order is the full matrix's order, so tour/mst/hubs agree on indices.
    The dense n×n work downstream happens on this copy, never on the corpus.

    `corpus` is the (ids, matrix) pair a caller already holds (/viz/map has
    just taken the snapshot), passed in so this does not re-derive it.
    Otherwise the pair comes from _viz_snapshot, NOT the live matrix: the
    cache key below is id(matrix_all), so a live matrix would be a new
    object -- and a cold subset, PCA and MST -- on every newly crawled
    track. `seed_id` is passed as the snapshot's `require`, so a track
    analyzed inside the window still gets its own screen.
    """
    ids_all, matrix_all = (corpus if corpus is not None
                           else _viz_snapshot(require=seed_id))
    # Hoisted out of the generator: as an inline condition this rebuilt the
    # whole id set once per extra id.
    known = set(ids_all)
    extra = tuple(t for t in extra_ids if t in known)
    key = (id(matrix_all), seed_id, extra)
    with _VIZ_CACHE_LOCK:
        cached = _SUBSET_CACHE.get(key)
        if cached is not None:
            if cached[0] is matrix_all:
                _SUBSET_CACHE.move_to_end(key)
                return cached[1], cached[2]
            _SUBSET_CACHE.pop(key, None)

    n = len(ids_all)
    if seed_id is None or seed_id not in ids_all:
        rows = set(range(min(n, VIZ_MAX)))
    else:
        seed_row = ids_all.index(seed_id)
        sims = _seed_cosine(matrix_all, seed_row)
        keep = min(n, VIZ_MAX)
        nearest = np.argpartition(-sims, keep - 1)[:keep] if keep < n else np.arange(n)
        rows = set(nearest.tolist()) | {seed_row}

    index = {t: i for i, t in enumerate(ids_all)}
    rows |= {index[t] for t in extra}
    rows = np.array(sorted(rows))

    ids, subset_matrix = (list(np.array(ids_all, dtype=object)[rows]),
                          np.ascontiguousarray(matrix_all[rows]))
    with _VIZ_CACHE_LOCK:
        _SUBSET_CACHE[key] = (matrix_all, ids, subset_matrix)
        # Any miss is the moment to notice a superseded corpus matrix: purge
        # before the LRU trim so eviction spends its budget on live entries.
        _purge_viz_caches(matrix_all)
        while len(_SUBSET_CACHE) > _SUBSET_KEEP:
            _SUBSET_CACHE.popitem(last=False)
    return ids, subset_matrix


def _viz_subset_min2(seed_id: str | None,
                     extra_ids: "list[str] | tuple[str, ...]" = (),
                     ) -> tuple[list[str], np.ndarray]:
    """Like _viz_subset, but the T2 endpoints (/viz/tour, /viz/mst,
    /viz/extremes) are pinned to a single 404 message regardless of whether
    the corpus is empty or has exactly one track -- SVD/MST need at least
    two rows either way."""
    try:
        ids, matrix = _viz_subset(seed_id, extra_ids)
    except HTTPException:
        ids, matrix = [], np.empty((0, 1))
    if len(ids) < 2:
        raise HTTPException(404, "needs at least two tracks")
    return ids, matrix


# ---- the viz snapshot ----
#
# Every derived insights cache (_SUBSET_CACHE, _TOP8_CACHE, _MST_CACHE,
# viz._PAIRWISE_CACHE) keys on the identity of the matrix it was computed
# from, and _corpus_matrix hands back a NEW matrix object the moment the
# crawler analyzes one more track. With a crawl running that is every few
# seconds, so those caches were cold on essentially every Insights request:
# the PCA, the pairwise similarity and the MST were all recomputed for the
# sake of one extra row in eight thousand.
#
# So the insights endpoints read a SNAPSHOT instead of the live matrix. It
# holds the live matrix object itself (no copy -- the snapshot only pins a
# matrix _MATRIX_CACHE would otherwise have dropped) and refreshes when the
# window closes, when the corpus has grown by VIZ_GROWTH_PCT, or when a
# caller needs a track the snapshot predates (`require`: a track the user
# just seeded must appear on its own Insights screen).
#
# What this trades away: a track analyzed inside the window is not drawn
# until the next refresh. That is the intended trade -- a point appearing up
# to VIZ_REFRESH_S late is invisible, a 9 s map is not.
VIZ_REFRESH_S = float(os.environ.get("VIZ_REFRESH_S", "300"))
VIZ_GROWTH_PCT = float(os.environ.get("VIZ_GROWTH_PCT", "5"))

# (taken at, ids, matrix)
_VIZ_SNAPSHOT: "tuple[float, list[str], np.ndarray] | None" = None


def _viz_snapshot(require: "str | list[str] | tuple[str, ...] | None" = None,
                  live: "tuple[list[str], np.ndarray] | None" = None,
                  ) -> tuple[list[str], np.ndarray]:
    """The corpus matrix the insights endpoints project, stable for a window.

    `require` is the track ids the caller must be able to find -- its seed.
    If the snapshot predates any of them (and the live corpus does have them), it refreshes
    once; the refreshed snapshot is the live matrix, so it then holds all of
    them. Without this a track analyzed inside the window would rank as a
    rec but have no row to draw at, and every client reads rec.x/rec.y
    unconditionally. An id that is in neither is NOT a reason to rebuild --
    otherwise a bogus ?track_id= would refresh on every request.

    `live` is the current (ids, matrix) pair when the caller already holds
    it (/viz/map ranks on it), so the growth check does not repeat that
    caller's corpus_ids read.
    """
    global _VIZ_SNAPSHOT, _ROW_NORMS, _FEEL_ALIGN_CACHE, _RHYTHM_ALIGN_CACHE

    # Read live first: this is also the call that appends newly analyzed rows
    # to _MATRIX_CACHE, and it is the cheap part (only new ids are parsed).
    ids, matrix = live if live is not None else _viz_embedding_corpus()
    wanted = [require] if isinstance(require, str) else list(require or ())

    with _VIZ_CACHE_LOCK:
        snapshot = _VIZ_SNAPSHOT
        if snapshot is not None:
            stamp, snap_ids, snap_matrix = snapshot
            fresh = time.monotonic() - stamp < VIZ_REFRESH_S
            # A corpus that SHRANK means the store was cleared and rebuilt:
            # the snapshot's ids no longer describe the live corpus, so it
            # has to be retaken rather than merely "not grown enough".
            small_growth = (
                len(snap_ids) <= len(ids) < len(snap_ids) * (1.0 + VIZ_GROWTH_PCT / 100.0)
            )
            held = set(snap_ids) if wanted else set()
            live_ids = set(ids) if wanted else set()
            missing = any(t in live_ids and t not in held for t in wanted)
            if fresh and small_growth and not missing:
                return snap_ids, snap_matrix
        # Two threads can miss together and both store a snapshot; benign,
        # since both store the same live (ids, matrix) pair and the loser's
        # write is identical to the winner's. The lock is here for the
        # OrderedDict mutations in _purge_viz_caches, not for exclusivity.
        _VIZ_SNAPSHOT = (time.monotonic(), ids, matrix)
        # The derived caches belong to the superseded matrix; drop them now
        # rather than letting them pin it until the LRU happens to evict.
        # _ROW_NORMS, _FEEL_ALIGN_CACHE and _RHYTHM_ALIGN_CACHE hold strong
        # references to the old matrix too (the feel alignment holds the
        # embedding matrix AND the feel matrix, so a stale entry pins both).
        _ROW_NORMS = None
        _FEEL_ALIGN_CACHE = None
        _RHYTHM_ALIGN_CACHE = None
        _purge_viz_caches(matrix)
    return ids, matrix


def _viz_embedding_corpus() -> tuple[list[str], np.ndarray]:
    corpus = tuple(_safe(store.corpus_ids, default=[]))
    if not corpus:
        raise HTTPException(404, "analyzed corpus is empty")
    ids, matrix, _ = _corpus_matrix(
        corpus, "embedding", "cosine", want_correction=False
    )
    if not ids:
        raise HTTPException(404, "analyzed corpus is empty")
    return ids, matrix


_MAX_RECS = 50


def _rec_ids(recs: str | None) -> tuple[str, ...]:
    """Parse the optional `recs` query param: comma-separated track ids the
    caller wants guaranteed a row in the subset, so rec highlighting still
    works on `surprise`, whose recs sit outside the seed's nearest ring.

    Capped at _MAX_RECS so a hostile query cannot widen the subset without
    bound; ids not in the corpus are dropped by _viz_subset itself.
    """
    if not recs:
        return ()
    return tuple(part for part in
                 (chunk.strip() for chunk in recs.split(",")) if part)[:_MAX_RECS]


def _unknown_track(track_id: str) -> dict:
    """The placeholder row for an id the store has no metadata for -- an id
    in the embedding matrix whose track document has not landed yet."""
    return {
        "track_id": track_id,
        "title": track_id,
        "artist": "Unknown artist",
        "album": "",
        "artwork_url": None,
        "preview_url": None,
    }


def _viz_track(track_id: str) -> dict:
    """One track's display dict, from the metadata cache. Callers that need
    many warm it first with a single _tracks_cached(ids) call, so this is a
    memory hit rather than a round trip per row."""
    return _playable(_tracks_cached([track_id])[0]) or _unknown_track(track_id)


@app.get("/viz/walk")
def viz_walk(from_: str = Query(alias="from"), to: str = Query(),
             k: int = Query(8, ge=1, le=50)) -> dict:
    ids, matrix = _viz_subset(from_, extra_ids=[to])
    position = {track_id: index for index, track_id in enumerate(ids)}
    for endpoint in (from_, to):
        if endpoint not in position:
            raise HTTPException(404, f"track {endpoint} not in corpus")

    try:
        path, geodesic, ambient = viz.shortest_walk(
            matrix, position[from_], position[to], k=k
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    xy = _projection(matrix)
    _tracks_cached([ids[index] for index in path])   # one read for the walk
    steps = [
        {
            **_viz_track(ids[index]),
            "x": float(xy[index, 0]),
            "y": float(xy[index, 1]),
        }
        for index in path
    ]
    return {
        "path": steps,
        "geodesic": geodesic,
        "ambient": ambient,
        "detour": geodesic / ambient if ambient > 0 else 1.0,
        "k": min(k, len(ids) - 1),
    }


@app.get("/viz/histogram")
def viz_histogram(track_id: str) -> dict:
    seed_features = _safe(store.get_features, track_id)
    if not seed_features or "embedding" not in seed_features:
        raise HTTPException(404, f"track {track_id} not analyzed")
    ids, matrix = _viz_embedding_corpus()
    if track_id not in ids:
        raise HTTPException(404, f"track {track_id} not analyzed")

    # Through _similarity, not rank.scores directly: this is the same full
    # embedding matrix /recommend just ranked against, so the cached float32
    # unit matrix is already in hand and re-normalizing it here would be the
    # single most expensive thing the endpoint does.
    similarities = _similarity(
        "embedding", matrix, _vector(seed_features, "embedding"), "cosine"
    )
    values = np.delete(similarities, ids.index(track_id))
    counts, edges = np.histogram(values, bins=60, range=(-1.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    rec_scores = np.sort(values)[::-1][:10]
    threshold = float(rec_scores[-1]) if len(rec_scores) else 0.0
    percentile = float(100.0 * np.mean(values <= threshold)) if len(values) else 0.0
    dimension = int(matrix.shape[1])
    return {
        "bins": [float(value) for value in centers],
        "counts": [int(value) for value in counts],
        "rec_scores": [float(value) for value in rec_scores],
        "percentile": percentile,
        "null": {"mean": 0.0, "sd": float(1.0 / np.sqrt(dimension))},
        "corpus": {
            "mean": float(values.mean()) if len(values) else 0.0,
            "sd": float(values.std()) if len(values) else 0.0,
        },
    }


@app.get("/viz/hubs")
def viz_hubs(track_id: str | None = None,
             k: int = Query(8, ge=1, le=50),
             limit: int = Query(5, ge=1, le=20),
             recs: str | None = None) -> dict:
    ids, matrix = _viz_subset(track_id, _rec_ids(recs))
    if len(ids) < 2:
        raise HTTPException(404, "hubness needs at least two tracks")
    neighbor_k = min(k, len(ids) - 1)
    counts, centrality = _hub_arrays(matrix, neighbor_k)

    index = np.arange(len(ids))
    hub_order = np.lexsort((index, -counts))
    central_order = np.lexsort((index, -centrality))
    isolated_order = np.lexsort((index, centrality))

    # Only the rows this response names, in one read: the subset is up to
    # VIZ_MAX rows but at most 3 x limit of them are shown.
    _tracks_cached([ids[i] for order in (hub_order, central_order, isolated_order)
                    for i in order[:limit]])

    def rows(order: np.ndarray, field: str, values: np.ndarray) -> list[dict]:
        return [
            {**_viz_track(ids[i]), field: float(values[i])}
            for i in order[:limit]
        ]

    return {
        "hubs": [
            {**_viz_track(ids[i]), "count": int(counts[i])}
            for i in hub_order[:limit]
        ],
        "central": rows(central_order, "centrality", centrality),
        "isolated": rows(isolated_order, "centrality", centrality),
        "expected_k": neighbor_k,
        "all_counts": [
            {"track_id": track_id, "count": int(count)}
            for track_id, count in zip(ids, counts)
        ],
    }


@app.get("/viz/tour")
def viz_tour(track_id: str | None = None, recs: str | None = None) -> dict:
    """Per-track top-8 PC coordinates + variance explained (T2.1).

    Non-contract debug/demo endpoint, like /viz/map and /viz/hubs. Columns
    0-1 of coords8 come from the same SVD, sign convention and matrix-pinned
    cache (_TOP8_CACHE) as /viz/map's x/y, but they are NOT the same numbers
    unless the subsets match: both are seed-anchored, and /viz/map's subset
    also contains that axis's recs. Pass the same ids in `recs` to line the
    two up.
    """
    ids, matrix = _viz_subset_min2(track_id, _rec_ids(recs))
    coords8, variance = _top8(matrix)
    coords8_b64 = base64.b64encode(
        np.asarray(coords8, dtype="<f4").tobytes()
    ).decode("ascii")
    return {
        "ids": ids,
        "coords8": coords8_b64,
        "variance": [float(v) for v in variance],
    }


@app.get("/viz/mst")
def viz_mst(track_id: str | None = None, recs: str | None = None) -> dict:
    """The n-1 MST edges over cosine distance — Prim in numpy (T2.2).

    Non-contract debug/demo endpoint. The H0 barcode's death times are
    exactly these edge weights.
    """
    ids, matrix = _viz_subset_min2(track_id, _rec_ids(recs))
    edges = _mst(matrix)
    return {"ids": ids, "edges": [[i, j, d] for i, j, d in edges]}


@app.get("/viz/extremes")
def viz_extremes(track_id: str | None = None,
                 pc: int = Query(1, ge=1, le=8),
                 limit: int = Query(4, ge=1, le=10),
                 recs: str | None = None) -> dict:
    """Top/bottom tracks along one principal component (T2.4).

    Non-contract debug/demo endpoint. Reuses /viz/tour's top-8 PC cache —
    for the same (track_id, recs) pair, which is the same subset.
    """
    ids, matrix = _viz_subset_min2(track_id, _rec_ids(recs))
    coords8, variance = _top8(matrix)
    col = pc - 1
    values = coords8[:, col]

    k = min(limit, len(ids))
    low_order = np.argsort(values, kind="stable")          # most negative first
    high_order = np.argsort(-values, kind="stable")         # most positive first
    _tracks_cached([ids[i] for order in (low_order, high_order)
                    for i in order[:k]])

    return {
        "pc": pc,
        "variance_pct": float(variance[col] * 100.0),
        "low": [_viz_track(ids[i]) for i in low_order[:k]],
        "high": [_viz_track(ids[i]) for i in high_order[:k]],
    }


@app.get("/viz/attribute")
def viz_attribute(seed: str, rec: str) -> dict:
    """Which frequency bands carry this pair's similarity (T2.6).

    Non-contract debug/demo endpoint, and the only /viz route that can't
    answer from the matrix alone: the counterfactual has to go back through
    the real model, which lives on the Mac worker. So this route is a
    mailbox — it serves the cached answer, or queues the pair and says
    "pending" while the worker band-stops, re-embeds, and writes the result.
    """
    if seed == rec:
        raise HTTPException(400, "seed and rec must differ")

    ids, _ = _viz_embedding_corpus()
    present = set(ids)
    for track_id in (seed, rec):
        if track_id not in present:
            raise HTTPException(404, f"track {track_id} not analyzed")

    cached = _safe(store.get_attribution, seed, rec)
    if cached:
        return cached

    _safe(store.enqueue_attribution, seed, rec)
    return {"status": "pending"}


def _fixture_fallback(track_id: str, limit: int) -> list[dict]:
    tracks = [t for t in _fixture_tracks() if t["track_id"] != track_id][:limit]
    return [
        {**_playable(t), "score": round(0.95 - 0.05 * i, 4)}
        for i, t in enumerate(tracks)
    ]


def _vector(features: dict, key: str) -> np.ndarray:
    value = features[key]
    return np.atleast_1d(np.asarray(value, dtype=float))


def _to_plain(features: dict) -> dict:
    """np arrays/scalars -> JSON-serializable lists/floats."""
    return {
        k: v.tolist() if hasattr(v, "tolist") else v for k, v in features.items()
    }


# ---- the web app: web/dist built into the image, served at / ----
# Declared last so every API route above wins; unknown GET paths without a
# file extension fall back to index.html for client-side routing.
#
# WEB_DIST is read lazily (at request time, not import time) so tests can
# monkeypatch the env var per-test without reloading this module.


def _web_dist() -> Path:
    return Path(os.environ.get(
        "WEB_DIST", str(Path(__file__).resolve().parents[3] / "web" / "dist")
    ))


def _spa_index() -> FileResponse:
    index = _web_dist() / "index.html"
    if not index.exists():
        raise HTTPException(404, "web app not built")
    return FileResponse(index)


@app.get("/", include_in_schema=False)
def spa_root() -> FileResponse:
    return _spa_index()


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str) -> FileResponse:
    if "." in path.rsplit("/", 1)[-1]:
        # Containment check: Path("/x") / "/etc/passwd" == "/etc/passwd" (an
        # absolute right operand discards the left), and ".." segments are
        # only normalized by well-behaved clients/proxies, not by us. Resolve
        # both sides and require the result to still live under the root.
        root = _web_dist().resolve()
        file = (root / path.lstrip("/")).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise HTTPException(404)
        return FileResponse(file)
    return _spa_index()
