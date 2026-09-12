"""rank.py: normalize + matmul + argsort, per spec §2.2."""
import numpy as np
import pytest

from music_recommendations.server import rank


def test_scores_is_cosine_similarity():
    seed = np.array([1.0, 0.0])
    matrix = np.array([
        [1.0, 0.0],    # identical -> 1.0
        [0.0, 1.0],    # orthogonal -> 0.0
        [-1.0, 0.0],   # opposite -> -1.0
        [2.0, 0.0],    # same direction, bigger magnitude -> still 1.0
    ])
    s = rank.scores(seed, matrix)
    assert s == pytest.approx([1.0, 0.0, -1.0, 1.0])


def test_scores_zero_vector_yields_zero_not_nan():
    seed = np.array([1.0, 0.0])
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]])
    s = rank.scores(seed, matrix)
    assert not np.isnan(s).any()
    assert s[0] == 0.0


def test_rank_returns_nearest_first():
    seed = np.array([1.0, 0.0])
    matrix = np.array([
        [0.0, 1.0],     # orthogonal
        [1.0, 0.1],     # very close
        [-1.0, 0.0],    # opposite
        [1.0, 1.0],     # 45 degrees
    ])
    idx = rank.rank(seed, matrix, limit=4)
    assert list(idx) == [1, 3, 0, 2]


def test_rank_direction_negative_returns_most_distant_first():
    seed = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    idx = rank.rank(seed, matrix, direction=-1, limit=3)
    assert list(idx) == [2, 1, 0]


def test_rank_respects_limit():
    seed = np.array([1.0, 0.0])
    matrix = np.random.default_rng(0).normal(size=(20, 2))
    idx = rank.rank(seed, matrix, limit=5)
    assert len(idx) == 5


# ---- per-axis metric ----
#
# groove is four all-positive, hand-normalized numbers, so every vector points
# into roughly the same direction and only the MAGNITUDE distinguishes a slow
# sparse track from a fast dense one. Cosine is scale-invariant, so it is blind
# to exactly the thing that matters here.
#
# These are real groove vectors from the corpus. Carlita's "Manhattan" (dense
# electronic) and John Denver's "Take Me Home, Country Roads" score 0.996 under
# cosine -- near-identical -- because Denver's vector is roughly half of
# Carlita's in every dimension. Euclidean puts them at 0.62.

DENSE_ELECTRONIC = np.array([0.609, 0.862, 0.852, 0.476])   # Carlita - Manhattan
SPARSE_COUNTRY = np.array([0.304, 0.551, 0.467, 0.273])     # Denver - Country Roads
ALSO_SPARSE = np.array([0.322, 0.573, 0.441, 0.298])        # a near neighbour of it


def test_cosine_cannot_separate_groove_vectors():
    """Why the metric matters: cosine calls two unrelated grooves identical."""
    s = rank.scores(SPARSE_COUNTRY, np.vstack([DENSE_ELECTRONIC]), metric="cosine")
    assert s[0] > 0.99, "cosine is blind to magnitude, which is groove's signal"


def test_euclidean_separates_groove_vectors():
    matrix = np.vstack([DENSE_ELECTRONIC, ALSO_SPARSE])
    s = rank.scores(SPARSE_COUNTRY, matrix, metric="euclidean")
    assert s[1] > s[0], "the other sparse track should outrank the dense one"
    assert rank.rank(SPARSE_COUNTRY, matrix, metric="euclidean", limit=1)[0] == 1


def test_cosine_would_get_that_ranking_wrong():
    """The same two candidates, ranked by cosine, put the wrong one first."""
    matrix = np.vstack([DENSE_ELECTRONIC, ALSO_SPARSE])
    cos = rank.scores(SPARSE_COUNTRY, matrix, metric="cosine")
    euc = rank.scores(SPARSE_COUNTRY, matrix, metric="euclidean")
    assert abs(cos[0] - cos[1]) < 0.01, "cosine cannot tell them apart"
    assert euc[1] - euc[0] > 0.2, "euclidean clearly can"


def test_euclidean_scores_are_similarities_not_distances():
    """Higher must mean closer, so both metrics sort the same direction."""
    matrix = np.vstack([SPARSE_COUNTRY, DENSE_ELECTRONIC])
    s = rank.scores(SPARSE_COUNTRY, matrix, metric="euclidean")
    assert s[0] == 1.0                      # distance 0 -> similarity 1
    assert 0.0 < s[1] < 1.0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError):
        rank.scores(SPARSE_COUNTRY, np.vstack([DENSE_ELECTRONIC]), metric="manhattan")


def test_every_axis_feature_has_a_declared_metric():
    """app.py must not hardcode cosine; analysis declares it per feature key."""
    from music_recommendations.analysis.schema import METRICS
    from music_recommendations.server.axes import AXIS_FEATURES

    for _axis, (feature_key, _direction) in AXIS_FEATURES.items():
        assert feature_key in METRICS, f"{feature_key} has no declared metric"
    assert METRICS["embedding"] == "cosine"


# ---- surprise concentration ----

def test_centrality_matches_the_naive_definition():
    """The O(n*d) identity must give the same answer as the O(n^2) form."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 16))
    fast = rank.centrality(matrix, "cosine")
    slow = np.array([rank.scores(row, matrix, "cosine").mean() for row in matrix])
    assert np.allclose(fast, slow)


def test_correction_stops_one_outlier_winning_every_seed():
    """A track far from everything should not be every seed's 'surprise'."""
    rng = np.random.default_rng(1)
    cluster = rng.normal(size=(30, 8)) * 0.1 + np.array([1.0] + [0.0] * 7)
    outlier = np.array([[-5.0] * 8])
    matrix = np.vstack([cluster, outlier])
    outlier_idx = len(matrix) - 1

    raw = [rank.rank(matrix[i], matrix, direction=-1, limit=1)[0]
           for i in range(len(cluster))]
    assert raw.count(outlier_idx) == len(cluster), "expected the outlier to dominate"

    correction = rank.centrality(matrix, "cosine")
    fixed = [rank.rank(matrix[i], matrix, direction=-1, limit=1,
                       correction=correction)[0] for i in range(len(cluster))]
    assert fixed.count(outlier_idx) < len(cluster), "correction changed nothing"


def test_correction_is_ignored_when_not_supplied():
    """Similar-axis ranking must be untouched by the surprise fix."""
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 8))
    seed = matrix[0]
    assert list(rank.rank(seed, matrix, limit=5)) == list(
        rank.rank(seed, matrix, limit=5, correction=None)
    )
