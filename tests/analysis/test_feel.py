"""feel.py: the eleven heads, the table that says which class is the positive one."""
from __future__ import annotations

import json

import numpy as np
import pytest

from music_recommendations.analysis import registry
from music_recommendations.analysis.feel import FEEL_DIM, FEEL_KEYS

HAVE_HEADS = all(head.graph.exists() for head in registry.HEADS.values())
needs_heads = pytest.mark.skipif(
    not HAVE_HEADS, reason="run scripts/fetch_models.py first"
)
HAVE_HEAD_METADATA = all(head.metadata.exists() for head in registry.HEADS.values())


# ---- the table: no TensorFlow, no model files ----

def test_eleven_keys_in_the_documented_order():
    assert FEEL_KEYS == [
        "danceable", "happy", "sad", "aggressive", "relaxed", "party",
        "acoustic", "electronic", "bright", "tonal", "instrumental",
    ]
    assert FEEL_DIM == 11


def test_every_positive_index_is_a_softmax_column():
    for key, head in registry.HEADS.items():
        assert head.positive in (0, 1), key


def test_head_urls_point_at_the_family_directory():
    graph, metadata = registry.HEADS["tonal"].urls
    assert graph == ("https://essentia.upf.edu/models/classification-heads/"
                     "tonal_atonal/tonal_atonal-discogs-effnet-1.pb")
    assert metadata.endswith(".json")


@pytest.mark.skipif(not HAVE_HEAD_METADATA,
                    reason="run scripts/fetch_models.py first")
def test_each_positive_index_names_the_class_we_think_it_does():
    """The one mistake this table can make is silent: swap an index and the
    corpus is ranked on "not_danceable" with nothing in the output to say so.
    The heads ship their own class order, so check ours against theirs.
    """
    for key, head in registry.HEADS.items():
        classes = json.loads(head.metadata.read_text())["classes"]
        assert len(classes) == 2, key
        assert classes[head.positive] == key, (key, classes, head.positive)


# ---- the graphs ----

@needs_heads
def test_feel_vectors_shape_and_range():
    from music_recommendations.analysis.feel import feel_vectors

    rows = np.random.default_rng(0).normal(size=(4, 1280)).astype(np.float32)
    out = feel_vectors(rows)
    assert out.shape == (4, FEEL_DIM)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@needs_heads
def test_different_embeddings_give_different_feel():
    from music_recommendations.analysis.feel import feel_vectors

    rng = np.random.default_rng(7)
    out = feel_vectors(rng.normal(size=(2, 1280)).astype(np.float32))
    assert not np.allclose(out[0], out[1])


@needs_heads
def test_the_same_embedding_twice_gives_the_same_vector():
    from music_recommendations.analysis.feel import feel_vectors

    row = np.random.default_rng(3).normal(size=(1280,)).astype(np.float32)
    out = feel_vectors(np.stack([row, row]))
    assert np.allclose(out[0], out[1])


@needs_heads
def test_feel_vector_is_the_one_row_case():
    from music_recommendations.analysis.feel import feel_vector, feel_vectors

    row = np.random.default_rng(11).normal(size=(1280,)).astype(np.float32)
    assert np.allclose(feel_vector(row), feel_vectors(row[None, :])[0])


def test_no_rows_needs_no_model():
    """An empty batch must not load eleven graphs (or fail where there are
    none): analyze_tracks can legitimately have nothing to score."""
    from music_recommendations.analysis.feel import feel_vectors

    out = feel_vectors(np.zeros((0, 1280), dtype=np.float32))
    assert out.shape == (0, FEEL_DIM)
