"""FastAPI app. Routes mirror contract/contract.md exactly. Mock-first: serve
contract/fixture.json until the corpus lands.

Routes:
  GET  /search      -- query Deezer (or fixture fallback) for tracks
  POST /seed         -- mark a track as the seed for recommendations
  GET  /axes         -- list available recommendation axes
  GET  /recommend    -- ranked, scored tracks for a seed + axis
  GET  /preview/{id} -- 302 to a freshly signed Deezer preview (not contract)
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
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from starlette.requests import Request
from typing import Literal, NamedTuple

from contract.features import AXES
from music_recommendations.analysis import analyze_track, frontend
from music_recommendations.analysis.schema import METRICS
from music_recommendations.server import deezer, store, viz
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


def _fresh_preview(track_id: str) -> str | None:
    """A signed, currently-valid preview URL, from cache or from Deezer."""
    cached = _safe(store.get_cached_preview, track_id)
    if cached:
        return cached
    url = deezer.fresh_preview_url(track_id)
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


class SeedRequest(BaseModel):
    track_id: str


@app.get("/axes")
def get_axes() -> dict:
    return {"axes": AXES}


@app.get("/search")
def search(q: str) -> dict:
    try:
        return {"results": [_playable(t) for t in deezer.search(q)]}
    except Exception:
        needle = q.lower()
        hits = [
            t for t in _fixture_tracks()
            if needle in t["title"].lower()
            or needle in t["artist"].lower()
            or needle in t["album"].lower()
        ]
        return {"results": hits}


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
        try:
            track = deezer.get_track(req.track_id)
        except Exception:
            track = None
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
    # One set for both the subset test and the membership checks below;
    # `known.issubset(corpus)` would build its own copy of a 90k-id tuple.
    corpus_set = set(corpus)

    # Tracks only ever get added, so the common case is a short tail of new ids.
    # A track disappearing means someone cleared the store: drop it all and rebuild.
    if cached is not None and not known.issubset(corpus_set):
        cached, known = None, set()

    # Iterated over the tuple, not the set: row order must follow corpus order.
    fresh = [i for i in corpus if i not in known]
    if cached is None:
        ids, matrix = _cold_matrix(corpus, feature_key)
        cached = _CorpusMatrix(corpus, ids, matrix, None)
    elif fresh:
        ids, rows = _rows_for(fresh, feature_key)
        if rows:
            cached = _CorpusMatrix(
                corpus, cached.ids + ids, np.vstack([cached.matrix, np.stack(rows)]),
                None,  # the corpus moved, so any cached centrality is stale
            )
        else:
            cached = cached._replace(corpus=corpus)
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


@app.get("/recommend")
def recommend(track_id: str, axis: str,
              limit: int = Query(10, ge=1, le=50)) -> dict:
    if axis not in AXIS_FEATURES and axis not in BLENDED_AXES:
        raise HTTPException(400, f"unknown axis {axis!r}")

    seed_features = _safe(store.get_features, track_id)
    corpus = tuple(_safe(store.corpus_ids, default=[]))
    ranked_against = [i for i in corpus if i != track_id]

    if seed_features is None or not ranked_against:
        results = _fixture_fallback(track_id, limit)
    elif axis in BLENDED_AXES:
        ids, fused, parts = _blended(corpus, seed_features, BLENDED_AXES[axis])
        results = []
        for idx in np.argsort(fused)[::-1]:
            if ids[idx] == track_id:
                continue
            track = store.get_track(ids[idx])
            results.append({**_playable(track), "score":
                            round(float(fused[idx]) / 100.0, 4)})
            if len(results) == limit:
                break
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
        similarity = _similarity(feature_key, matrix, seed_vec, metric)
        order = rank_mod.rank(
            seed_vec, matrix, direction=direction, limit=limit + 1,
            metric=metric, correction=correction, similarity=similarity,
        )
        results = []
        for idx in order:
            if ids[idx] == track_id:
                continue
            track = store.get_track(ids[idx])
            results.append({**_playable(track), "score": float(similarity[idx])})
            if len(results) == limit:
                break

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
_CACHE_KEEP = 4

# One lock for all three viz caches (_SUBSET_CACHE, _TOP8_CACHE, _MST_CACHE).
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


def _projection(matrix: np.ndarray) -> np.ndarray:
    """First two PC columns — numerically identical to viz.project_2d's
    output, since /viz/map's xy must not change when this cache was added."""
    coords8, _ = _top8(matrix)
    return coords8[:, :2]


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


@app.get("/viz/map")
def viz_map(track_id: str, axis: str,
            limit: int = Query(10, ge=1, le=50),
            correction: Literal["on", "off"] = "on") -> dict:
    """Everything the wow screen needs in one payload: the whole corpus as 2D
    points, the seed, and the recs with the actual numbers behind each score."""
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
    else:
        feature_key, direction = AXIS_FEATURES[axis]
        metric = METRICS.get(feature_key, "cosine")
        use_correction = direction == -1 and correction == "on"
        ids, matrix, correction = _corpus_matrix(corpus, feature_key, metric,
                                                 want_correction=use_correction)
        seed_vec = _vector(seed_features, feature_key)
        similarity = _similarity(feature_key, matrix, seed_vec, metric)
        order = rank_mod.rank(seed_vec, matrix, direction=direction,
                              limit=limit + 1, metric=metric,
                              correction=correction, similarity=similarity)

    emb_id_set = set(emb_ids)
    recs = []
    for idx in order:
        rec_id = ids[idx]
        if rec_id == track_id or rec_id not in emb_id_set:
            continue
        track = _safe(store.get_track, rec_id)
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
        else:
            score = float(similarity[idx])
            math = viz.score_math(
                seed_vec, matrix[idx], metric,
                float(correction[idx]) if correction is not None else None,
            )
        recs.append({
            **(_playable(track) or {"track_id": rec_id}),
            "score": score,
            "math": math,
        })
        if len(recs) == limit:
            break

    # The points/projection are seed-anchored, not the whole corpus: recs
    # (e.g. `surprise`'s far neighbours) are drawn even when they fall
    # outside the nearest-VIZ_MAX ring, via extra_ids.
    subset_ids, subset_matrix = _viz_subset(
        track_id, extra_ids=[rec["track_id"] for rec in recs],
        corpus=(emb_ids, emb_matrix),
    )
    xy = _projection(subset_matrix)
    position = {tid: i for i, tid in enumerate(subset_ids)}
    for rec in recs:
        pos = position[rec["track_id"]]
        rec["x"] = float(xy[pos, 0])
        rec["y"] = float(xy[pos, 1])

    seed_track = _playable(_safe(store.get_track, track_id)) or {"track_id": track_id}
    seed_pos = position[track_id]
    seed = {
        **seed_track,
        "x": float(xy[seed_pos, 0]),
        "y": float(xy[seed_pos, 1]),
    }
    point_tracks = []
    for point_id, track in zip(subset_ids, _safe(store.get_many_tracks, subset_ids,
                                                  default=[])):
        point_tracks.append(_playable(track) or {
            "track_id": point_id,
            "title": point_id,
            "artist": "Unknown artist",
            "album": "",
            "artwork_url": None,
            "preview_url": None,
        })
    axis_info = {"id": axis, "metric": metric, "direction": direction}
    if blended_weights is not None:
        axis_info["metric"] = "blend"
        axis_info["weights"] = dict(blended_weights)
    if direction == -1:
        axis_info["correction"] = "on" if use_correction else "off"

    return {
        "points": {
            "ids": subset_ids,
            "x": [round(float(v), 4) for v in xy[:, 0]],
            "y": [round(float(v), 4) for v in xy[:, 1]],
            "tracks": point_tracks,
        },
        "seed": seed,
        "recs": recs,
        "axis": axis_info,
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
    keep window indefinitely if it is never looked up again. _TOP8_CACHE and
    _MST_CACHE pin subset matrices the same way, so they are purged down to
    whatever subsets are still cached.
    """
    for key in [k for k, v in _SUBSET_CACHE.items() if v[0] is not matrix_all]:
        _SUBSET_CACHE.pop(key, None)
    live = [value[2] for value in _SUBSET_CACHE.values()]
    for cache in (_TOP8_CACHE, _MST_CACHE):
        for key in [k for k, v in cache.items()
                    if not any(v[0] is subset for subset in live)]:
            cache.pop(key, None)


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
    just built it), passed in so this does not re-derive the same pair.
    """
    ids_all, matrix_all = corpus if corpus is not None else _viz_embedding_corpus()
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
        sims = _similarity("embedding", matrix_all, matrix_all[seed_row], "cosine")
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


def _viz_track(track_id: str) -> dict:
    return _playable(_safe(store.get_track, track_id)) or {
        "track_id": track_id,
        "title": track_id,
        "artist": "Unknown artist",
        "album": "",
        "artwork_url": None,
        "preview_url": None,
    }


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
    similarity = viz.pairwise_cosine(matrix)
    without_self = similarity.copy()
    np.fill_diagonal(without_self, -np.inf)
    neighbors = np.argpartition(
        -without_self, neighbor_k - 1, axis=1
    )[:, :neighbor_k]
    counts = np.bincount(neighbors.ravel(), minlength=len(ids))
    centrality = (similarity.sum(axis=1) - np.diag(similarity)) / (len(ids) - 1)

    index = np.arange(len(ids))
    hub_order = np.lexsort((index, -counts))
    central_order = np.lexsort((index, -centrality))
    isolated_order = np.lexsort((index, centrality))

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
