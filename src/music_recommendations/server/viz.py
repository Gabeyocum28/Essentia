"""Math for GET /viz/map — the demo/debug galaxy of embedding space.

NOT part of contract/contract.md (like GET /). The map is always a 2D
projection of the EMBEDDING matrix regardless of axis: the picture answers
"where does the music lie", the axis only changes which scores are attached
to the recs. /viz/map and /viz/walk draw it with UMAP (project_umap, which
keeps neighbourhoods rather than variance); /viz/tour and /viz/extremes stay
on PCA, because those two are ABOUT the principal components.

PCA over eigen-libraries: the corpus is numpy-sized, the projection is two
principal components of an (n, 1024) matrix, and we already hold that matrix
in process (app._MATRIX_CACHE). Rows are L2-normalized first so the picture
matches the cosine geometry the ranking actually uses — otherwise loudness
becomes the first principal component.
"""
from __future__ import annotations

import heapq
import threading
from collections import OrderedDict

import numpy as np


_PAIRWISE_LOCK = threading.RLock()
# id(matrix) -> (matrix, similarity), same identity-keyed shape as
# _GRAPH_CACHE below (the matrix is in the value so a recycled id() is a miss,
# not a wrong answer) and for the same reason: with a seed-anchored subset a
# caller alternating between two seeds thrashed the single slot this used to
# be, recomputing a dense n×n every other request.
#
# Both caches here are bounded by VIZ_MAX on the subset they hold, and
# neither is reachable from app's _purge_viz_caches — app clears them only
# through clear_geometry_cache().
_PAIRWISE_CACHE: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_PAIRWISE_KEEP = 2
# (id(matrix), k) -> (matrix, adjacency). The matrix is held in the VALUE ON
# PURPOSE: keying by bare id(matrix) let the old array be garbage-collected
# after a corpus growth, and a later array reusing the same address would
# silently serve a graph whose node indices belong to the old, smaller
# corpus — so the stored matrix is re-checked with `is` and a reused id is a
# miss rather than a wrong answer.
#
# An LRU dict rather than the single slot this used to be: the seed-anchored
# subset means two seeds' matrices can both be live at once (a caller
# alternating between two walks), and a one-entry cache rebuilt the whole
# k-NN graph on every other request — the measured 1.8 s warm /viz/walk.
# Bounded at _GRAPH_KEEP, matching app._CACHE_KEEP, because each entry pins
# its subset matrix.
_GRAPH_CACHE: "OrderedDict[tuple[int, int], tuple[np.ndarray, list[dict[int, float]]]]" = OrderedDict()
_GRAPH_KEEP = 4


def clear_geometry_cache() -> None:
    with _PAIRWISE_LOCK:
        _PAIRWISE_CACHE.clear()
        _GRAPH_CACHE.clear()


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Dense cosine matrix cached by matrix identity; safe through ~10k rows.

    float32 throughout (normalized_rows casts): at 10k rows that is 400 MB
    rather than 800 MB. It is still an n^2 allocation, so this intentionally
    remains a small-corpus visualization primitive — bounded by VIZ_MAX on
    the seed-anchored subset — rather than a ranking dependency.
    """
    key = id(matrix)
    with _PAIRWISE_LOCK:
        cached = _PAIRWISE_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _PAIRWISE_CACHE.move_to_end(key)
            return cached[1]
        unit = normalized_rows(matrix)
        similarity = np.clip(unit @ unit.T, -1.0, 1.0)
        _PAIRWISE_CACHE[key] = (matrix, similarity)
        _PAIRWISE_CACHE.move_to_end(key)
        while len(_PAIRWISE_CACHE) > _PAIRWISE_KEEP:
            _PAIRWISE_CACHE.popitem(last=False)
        return similarity


def _build_knn_graph(matrix: np.ndarray, k: int) -> list[dict[int, float]]:
    """Symmetrized k-NN adjacency (cosine distance) for every row."""
    n = len(matrix)
    distance = 1.0 - pairwise_cosine(matrix)
    adjacency: list[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        candidates = np.argpartition(distance[i], k)[:k + 1]
        neighbors = [int(j) for j in candidates if j != i]
        neighbors.sort(key=lambda j: (distance[i, j], j))
        for j in neighbors[:k]:
            weight = float(distance[i, j])
            adjacency[i][j] = min(adjacency[i].get(j, weight), weight)
            adjacency[j][i] = min(adjacency[j].get(i, weight), weight)
    return adjacency


def _knn_graph(matrix: np.ndarray, k: int) -> list[dict[int, float]]:
    """_build_knn_graph, memoized per (matrix identity, k)."""
    key = (id(matrix), k)
    with _PAIRWISE_LOCK:
        cached = _GRAPH_CACHE.get(key)
        if cached is not None and cached[0] is matrix:
            _GRAPH_CACHE.move_to_end(key)
            return cached[1]
        adjacency = _build_knn_graph(matrix, k)
        _GRAPH_CACHE[key] = (matrix, adjacency)
        while len(_GRAPH_CACHE) > _GRAPH_KEEP:
            _GRAPH_CACHE.popitem(last=False)
        return adjacency


def shortest_walk(matrix: np.ndarray, start: int, end: int,
                  k: int = 8) -> tuple[list[int], float, float]:
    """Dijkstra on the symmetrized k-NN graph of normalized embeddings.

    Both the k-NN edges and their weights use cosine distance (1 - cosine),
    matching the numbers shown in the walkthrough. The graph is cached by
    matrix identity and k because embedding matrices are already cache-owned
    by the server for the lifetime of one corpus state.

    The matrix is used as handed in, never re-cast: it arrives as the float32
    seed subset, and an `asarray(..., dtype=float)` rebinding both doubled it
    and produced a NEW object, which missed pairwise_cosine's identity cache
    and recomputed the n^2 similarity for the walk, the MST and the hubs
    separately.
    """
    n = len(matrix)
    if not (0 <= start < n and 0 <= end < n):
        raise IndexError("walk endpoint outside matrix")
    if start == end:
        return [start], 0.0, 0.0
    k = min(max(int(k), 1), n - 1)

    adjacency = _knn_graph(matrix, k)

    distances = [float("inf")] * n
    previous = [-1] * n
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == end:
            break
        for neighbor, weight in adjacency[node].items():
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    if not np.isfinite(distances[end]):
        raise ValueError("k-NN graph is disconnected")
    path = []
    node = end
    while node != -1:
        path.append(node)
        if node == start:
            break
        node = previous[node]
    path.reverse()
    similarity = pairwise_cosine(matrix)
    return path, float(distances[end]), float(1.0 - similarity[start, end])


def project_2d(matrix: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, 2): first two principal components of the row-normalized,
    mean-centered matrix. Deterministic (sign-fixed) so points don't jump
    between requests as the corpus grows."""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    centered = unit - unit.mean(axis=0)

    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2))

    # SVD of the thin side: components are V rows, coordinates U * S.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    xy = u[:, :2] * s[:2]
    if xy.shape[1] < 2:  # degenerate d == 1
        xy = np.hstack([xy, np.zeros((xy.shape[0], 1))])

    # Fix the sign convention: make each component's largest-magnitude
    # coordinate positive, so the layout is stable across recomputes.
    for col in range(2):
        peak = np.argmax(np.abs(xy[:, col]))
        if xy[peak, col] < 0:
            xy[:, col] = -xy[:, col]
    return xy


# Rows per pass of the covariance accumulation. Bounds the float64 working
# copy (2048 x 1024 x 8 B = 16 MB) while the normalized matrix itself stays
# float32; the seam does not change the answer, only the summation order.
_PCA_BLOCK = 2048


def project_top8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, d) -> (coords8 (n, 8) float64, variance fractions (8,) float64).

    Same row-normalization, mean-centering, and per-component sign-fixing
    rule as project_2d, extended to all 8 components — columns 0 and 1 agree
    with project_2d's output to float noise (same components as the thin
    SVD, computed via the 1024×1024 covariance: 7× faster at 8k rows),
    so one decomposition serves /viz/map, /viz/walk, and /viz/tour.

    Covariance rather than a thin SVD because d is fixed at 1024 while n
    grows: `Xc.T @ Xc` is (d, d) whatever n is, and its eigenvectors ARE the
    SVD's right singular vectors with eigenvalues s^2, so `Xc @ V[:, :k]`
    reproduces `U[:, :k] * s[:k]` to float noise. Accumulated in float64 in
    row blocks — a float32 gram loses too much on 8k nearly-parallel rows,
    and a float64 copy of the whole matrix is 82 MB.

    variance[i] is s_i^2 over the FULL spectrum (all d eigenvalues, the
    zeros included), not just the top 8, per T2.1's talking point ("Top-8
    PCs hold 37.4% of variance").

    d < 8 corpora (including the fixture-sized ones in tests) pad the
    unused columns with zeros rather than erroring.
    """
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    if n < 2:
        return np.zeros((n, 8)), np.zeros(8)
    d = int(matrix.shape[1])

    unit = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    unit = unit / np.where(norms == 0.0, np.float32(1.0), norms)
    # float64 mean: the centering is what the covariance sees, so the one
    # quantity every block shares is worth carrying at full precision.
    mean = unit.mean(axis=0, dtype=np.float64)

    def blocks():
        for start in range(0, n, _PCA_BLOCK):
            stop = min(start + _PCA_BLOCK, n)
            yield start, stop, unit[start:stop].astype(np.float64) - mean

    covariance = np.zeros((d, d))
    for _, _, block in blocks():
        covariance += block.T @ block

    # eigh returns ascending eigenvalues; reverse for descending components.
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values[::-1], 0.0, None)
    vectors = vectors[:, ::-1]

    k = min(8, n, d)
    coords = np.empty((n, k))
    for start, stop, block in blocks():
        coords[start:stop] = block @ vectors[:, :k]

    # Fix the sign convention per component: make each component's
    # largest-magnitude coordinate positive (same rule as project_2d).
    # eigh's sign is arbitrary, exactly as the SVD's was.
    for col in range(k):
        peak = np.argmax(np.abs(coords[:, col]))
        if coords[peak, col] < 0:
            coords[:, col] = -coords[:, col]

    if k < 8:
        coords = np.hstack([coords, np.zeros((n, 8 - k))])

    total_variance = float(values.sum())
    variance = np.zeros(8)
    if total_variance > 0.0:
        variance[:k] = values[:k] / total_variance
    return coords, variance


# Below this many rows UMAP has nothing to work with: it builds a k-NN graph
# on n_neighbors=15 and then optimizes a fuzzy simplicial set over it, so at
# 20 or 30 points the "neighbourhood" is most of the corpus and the layout is
# noise with a confident shape. PCA is the honest answer there -- it is also
# what every fixture-sized test corpus gets.
UMAP_MIN_ROWS = 50
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def project_umap(matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    """(n, d) -> (n, 2) UMAP coordinates on cosine distance.

    PCA answers "which two directions carry the most variance", which in a
    1024-d contrastive embedding is a nearly uniform ball: the galaxy came
    out as one blob because that is genuinely what the top two components
    look like. UMAP answers a different question -- keep near neighbours
    near -- and that is the one the picture is being asked.

    `random_state=seed` makes the layout reproducible, which matters because
    the client redraws the same subset across requests and points jumping
    between renders reads as a bug. It also forces UMAP single-threaded (it
    warns about exactly this); at VIZ_MAX rows that is the cost of a stable
    picture.

    Under UMAP_MIN_ROWS rows it falls back to the top-2 PCA columns rather
    than erroring: small corpora (every test fixture, a cold deploy) still
    get a map, just the old one.

    umap is imported inside the function: it pulls numba and llvmlite and
    costs seconds of JIT on first import, and nothing that merely imports
    this module for its other primitives should pay that.
    """
    matrix = np.asarray(matrix)
    n = int(matrix.shape[0])
    if n < UMAP_MIN_ROWS:
        return np.asarray(project_top8(matrix)[0][:, :2], dtype=float)

    import umap

    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric="cosine", random_state=int(seed))
    # Row-normalized first, like every other projection here: cosine is the
    # geometry the ranking uses, and the metric argument alone would leave
    # the k-NN graph built on unnormalized rows.
    return np.asarray(reducer.fit_transform(normalized_rows(matrix)),
                      dtype=float)


def minimum_spanning_tree(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    """Prim's algorithm over the dense cosine-distance matrix.

    Returns exactly n-1 (i, j, d) edges, i < j, sorted ascending by d.
    Deterministic tie-breaking: when multiple unvisited nodes tie for the
    cheapest edge, the lowest index wins. Reuses pairwise_cosine's cache --
    the matrix is passed through untouched, so /viz/walk, /viz/mst and
    /viz/hubs on one subset share a single similarity matrix -- and so stays
    cheap through the ~10k-row scale that primitive already documents; not
    engineered past it.
    """
    n = len(matrix)
    if n < 2:
        return []

    distance = 1.0 - pairwise_cosine(matrix)

    in_tree = np.zeros(n, dtype=bool)
    best_dist = distance[0].copy()
    best_from = np.zeros(n, dtype=int)
    in_tree[0] = True
    best_dist[0] = np.inf

    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        masked = np.where(in_tree, np.inf, best_dist)
        min_val = float(masked.min())
        j = int(np.flatnonzero(masked == min_val)[0])
        i = int(best_from[j])
        a, b = (i, j) if i < j else (j, i)
        edges.append((a, b, min_val))

        in_tree[j] = True
        candidate = distance[j]
        better = (~in_tree) & (candidate < best_dist)
        best_dist = np.where(better, candidate, best_dist)
        best_from = np.where(better, j, best_from)

    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    return edges


def score_math(seed_vec: np.ndarray, rec_vec: np.ndarray, metric: str,
               centrality: float | None) -> dict:
    """The numbers behind one recommendation's score, for the math panel.

    cosine:    score == dot / (seed_norm * rec_norm)
    euclidean: score == 1 / (1 + distance)
    """
    seed_vec = np.asarray(seed_vec, dtype=float)
    rec_vec = np.asarray(rec_vec, dtype=float)
    return {
        "metric": metric,
        "dot": float(seed_vec @ rec_vec),
        "seed_norm": float(np.linalg.norm(seed_vec)),
        "rec_norm": float(np.linalg.norm(rec_vec)),
        "distance": (
            float(np.linalg.norm(seed_vec - rec_vec))
            if metric == "euclidean" else None
        ),
        "centrality": centrality,
    }


# ---- occlusion attribution (T2.6): which frequencies carry a similarity ----

ATTRIBUTION_BANDS = 10
# These edges are a CONTRACT, not a tuning choice: the phone's band-solo
# player filters with the exact complement of this mask, so changing them
# silently desynchronizes what the user hears from what the bars claim. They
# were chosen when the model ran at 16 kHz (nothing above its 8 kHz Nyquist
# was worth probing); CLAP runs at 44.1 kHz and could see higher, but the
# audible payoff above 8 kHz on a 30 s lossy preview is small and the cost of
# moving them is a client change in lockstep.
ATTRIBUTION_LO_HZ = 60.0
ATTRIBUTION_HI_HZ = 7800.0


def band_edges(count: int = ATTRIBUTION_BANDS,
               lo_hz: float = ATTRIBUTION_LO_HZ,
               hi_hz: float = ATTRIBUTION_HI_HZ) -> list[tuple[float, float]]:
    """Log-spaced band boundaries — equal ratios, so each band is the same
    number of octaves and the bass isn't lumped into one bar."""
    ratio = hi_hz / lo_hz
    edges = [lo_hz * ratio ** (k / count) for k in range(count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(count)]


def _periodic_hann(size: int) -> np.ndarray:
    """Periodic (not symmetric) Hann: at 50% overlap the shifted copies sum
    to a constant, which is the COLA condition overlap-add reconstruction
    depends on."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)


def _stop_gain(freqs: np.ndarray, lo_hz: float, hi_hz: float,
               taper_bins: int) -> np.ndarray:
    """1 outside [lo, hi], 0 inside, raised-cosine across `taper_bins` at each
    edge — a brick wall is a sinc in time and rings audibly (Gibbs).

    Deliberately NOT width-adaptive: this mask has to stay the exact
    complement of the band-solo mask on the phone, which tapers a fixed
    number of bins. A band too narrow for its taper is a resolution problem,
    and the caller fixes it by analyzing with a longer window (see the
    worker's 8192-point call), not by quietly using a different filter here.
    """
    gain = np.ones_like(freqs)
    inside = np.nonzero((freqs >= lo_hz) & (freqs <= hi_hz))[0]
    if inside.size == 0:
        return gain
    gain[inside] = 0.0
    first, last = int(inside[0]), int(inside[-1])
    for step in range(1, taper_bins + 1):
        ramp = 0.5 * (1 - np.cos(np.pi * step / (taper_bins + 1)))
        below, above = first - step, last + step
        if below >= 0:
            gain[below] = min(gain[below], ramp)
        if above < gain.size:
            gain[above] = min(gain[above], ramp)
    return gain


def band_stop(audio: np.ndarray, sample_rate: float, lo_hz: float,
              hi_hz: float, fft_size: int = 2048, hop: int = 1024,
              taper_bins: int = 4) -> np.ndarray:
    """Remove [lo_hz, hi_hz] from a mono waveform, keeping everything else.

    The counterfactual the attribution measures: STFT, zero the bins in the
    band (tapered), inverse-FFT each frame with its ORIGINAL phase, and
    overlap-add. Phase never leaves the pipeline, so no Griffin-Lim is
    needed and an all-pass mask reconstructs the input exactly.
    """
    audio = np.asarray(audio, dtype=float).ravel()
    window = _periodic_hann(fft_size)
    gain = _stop_gain(np.fft.rfftfreq(fft_size, 1.0 / sample_rate),
                      lo_hz, hi_hz, taper_bins)

    # Pad by a full frame at each end (and out to a whole number of hops) so
    # every real sample sits under complete window coverage. Without this the
    # first and last samples are covered by a vanishing window, and dividing
    # the overlap-add by that envelope leaks unfiltered audio into the
    # counterfactual — which is exactly the band we claim to have deleted.
    tail = (-(audio.size + fft_size)) % hop
    padded = np.concatenate([
        np.zeros(fft_size), audio, np.zeros(fft_size + tail),
    ])

    output = np.zeros_like(padded)
    envelope = np.zeros_like(padded)
    for start in range(0, padded.size - fft_size + 1, hop):
        stop = start + fft_size
        spectrum = np.fft.rfft(padded[start:stop] * window)
        output[start:stop] += np.fft.irfft(spectrum * gain, n=fft_size)
        envelope[start:stop] += window

    resynthesized = np.divide(output, envelope,
                              out=np.zeros_like(output), where=envelope > 1e-8)
    return resynthesized[fft_size:fft_size + audio.size]
