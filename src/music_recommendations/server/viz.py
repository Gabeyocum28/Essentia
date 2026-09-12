"""Math for GET /viz/map — the demo/debug galaxy of embedding space.

NOT part of contract/contract.md (like GET /). The map is always a 2D PCA of
the EMBEDDING matrix regardless of axis: the picture answers "where does the
music lie", the axis only changes which scores are attached to the recs.

PCA over eigen-libraries: the corpus is numpy-sized, the projection is two
principal components of an (n, 1280) matrix, and we already hold that matrix
in process (app._MATRIX_CACHE). Rows are L2-normalized first so the picture
matches the cosine geometry the ranking actually uses — otherwise loudness
becomes the first principal component.
"""
from __future__ import annotations

import heapq
import threading

import numpy as np


_PAIRWISE_LOCK = threading.RLock()
_PAIRWISE_CACHE: tuple[np.ndarray, np.ndarray] | None = None
# (matrix, {k: adjacency}) — the matrix is held in the tuple ON PURPOSE:
# keying by bare id(matrix) let the old array be garbage-collected after a
# corpus growth, and a later array reusing the same address would silently
# serve a graph whose node indices belong to the old, smaller corpus.
_GRAPH_CACHE: tuple[np.ndarray, dict[int, list[dict[int, float]]]] | None = None


def clear_geometry_cache() -> None:
    global _PAIRWISE_CACHE, _GRAPH_CACHE
    with _PAIRWISE_LOCK:
        _PAIRWISE_CACHE = None
        _GRAPH_CACHE = None


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Dense cosine matrix cached by matrix identity; safe through ~10k rows.

    At 20k rows float64 approaches 3.2 GB, so this intentionally remains a
    small-corpus visualization primitive rather than a ranking dependency.
    """
    global _PAIRWISE_CACHE
    with _PAIRWISE_LOCK:
        if _PAIRWISE_CACHE is not None and _PAIRWISE_CACHE[0] is matrix:
            return _PAIRWISE_CACHE[1]
        unit = normalized_rows(matrix)
        similarity = np.clip(unit @ unit.T, -1.0, 1.0)
        _PAIRWISE_CACHE = (matrix, similarity)
        return similarity


def shortest_walk(matrix: np.ndarray, start: int, end: int,
                  k: int = 8) -> tuple[list[int], float, float]:
    """Dijkstra on the symmetrized k-NN graph of normalized embeddings.

    Both the k-NN edges and their weights use cosine distance (1 - cosine),
    matching the numbers shown in the walkthrough. The graph is cached by
    matrix identity and k because embedding matrices are already cache-owned
    by the server for the lifetime of one corpus state.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    if not (0 <= start < n and 0 <= end < n):
        raise IndexError("walk endpoint outside matrix")
    if start == end:
        return [start], 0.0, 0.0
    k = min(max(int(k), 1), n - 1)

    global _GRAPH_CACHE
    with _PAIRWISE_LOCK:
        if _GRAPH_CACHE is None or _GRAPH_CACHE[0] is not matrix:
            _GRAPH_CACHE = (matrix, {})
        graphs = _GRAPH_CACHE[1]
        adjacency = graphs.get(k)
        if adjacency is None:
            similarity = pairwise_cosine(matrix)
            distance = 1.0 - similarity
            adjacency = [dict() for _ in range(n)]
            for i in range(n):
                candidates = np.argpartition(distance[i], k)[:k + 1]
                neighbors = [int(j) for j in candidates if j != i]
                neighbors.sort(key=lambda j: (distance[i, j], j))
                for j in neighbors[:k]:
                    weight = float(distance[i, j])
                    adjacency[i][j] = min(adjacency[i].get(j, weight), weight)
                    adjacency[j][i] = min(adjacency[j].get(i, weight), weight)
            graphs[k] = adjacency

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


def project_top8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, d) -> (coords8 (n, 8) float64, variance fractions (8,) float64).

    Same row-normalization, mean-centering, and per-component sign-fixing
    rule as project_2d, extended to all 8 components — columns 0 and 1 are
    numerically identical to project_2d's output (same SVD, same sign
    convention), so one SVD serves /viz/map, /viz/walk, and /viz/tour.

    variance[i] is s_i^2 over the FULL spectrum of the thin SVD (all
    min(n, d) singular values), not just the top 8, per T2.1's talking
    point ("Top-8 PCs hold 37.4% of variance").

    d < 8 corpora (including the fixture-sized ones in tests) pad the
    unused columns with zeros rather than erroring.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if n < 2:
        return np.zeros((n, 8)), np.zeros(8)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    centered = unit - unit.mean(axis=0)

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total_variance = float(np.sum(s ** 2))
    k = min(8, u.shape[1])
    coords = u[:, :k] * s[:k]

    # Fix the sign convention per component: make each component's
    # largest-magnitude coordinate positive (same rule as project_2d).
    for col in range(k):
        peak = np.argmax(np.abs(coords[:, col]))
        if coords[peak, col] < 0:
            coords[:, col] = -coords[:, col]

    if k < 8:
        coords = np.hstack([coords, np.zeros((n, 8 - k))])

    variance = np.zeros(8)
    if total_variance > 0.0:
        variance[:k] = (s[:k] ** 2) / total_variance
    return coords, variance


def minimum_spanning_tree(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    """Prim's algorithm over the dense cosine-distance matrix.

    Returns exactly n-1 (i, j, d) edges, i < j, sorted ascending by d.
    Deterministic tie-breaking: when multiple unvisited nodes tie for the
    cheapest edge, the lowest index wins. Reuses pairwise_cosine's cache,
    so this stays cheap through the ~10k-row scale that primitive already
    documents; not engineered past it.
    """
    matrix = np.asarray(matrix, dtype=float)
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
# EffNet consumes 16 kHz mono, so the model literally cannot see anything
# above 8 kHz: bands stop just under that Nyquist rather than at the 12 kHz
# a 44.1 kHz display spectrogram could show.
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
