"""The package surface: the version, the metrics, and what is NOT here.

The behaviour of `analyze_tracks` itself lives in test_v2.py, which needs the
analysis extra. These run everywhere, and the removal guards below are the
point of the clean-room work: the non-commercial stack must stay deleted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_version_and_metrics():
    from music_recommendations.analysis import FEATURES_VERSION, METRICS

    assert FEATURES_VERSION == 4
    assert METRICS == {"embedding": "cosine"}


def test_importing_the_package_costs_nothing_heavy():
    """A caller that only wants FEATURES_VERSION must not load torch.

    This is why analyze_tracks is a forwarder into v2 rather than a rebind:
    the store asks "is this row stale?" on every write, and paying a torch
    import for that would put ~700 MB into the API process.
    """
    import sys

    import music_recommendations.analysis  # noqa: F401

    assert "torch" not in sys.modules
    assert "librosa" not in sys.modules


def test_removed_modules_are_gone():
    """The v1 pipeline and its ancestors. `frontend`, `embedding` and the
    eleven-head `feel` ran Discogs-EffNet, which is CC BY-NC-SA -- the whole
    reason the v2 stack exists. Re-adding any of them re-acquires the licence
    problem this project was done to remove."""
    for name in ("heads", "groove", "frontend", "embedding"):
        assert importlib.util.find_spec(
            f"music_recommendations.analysis.{name}") is None


def test_feel_is_the_zero_shot_module_not_the_old_heads():
    """`feel_v2` was renamed onto `feel` at the cutover; anything still
    importing `feel_v2` is reading a module that no longer exists."""
    from music_recommendations.analysis import feel

    assert len(feel.FEEL_KEYS) == 8
    assert not hasattr(feel, "feel_vectors")
    assert importlib.util.find_spec(
        "music_recommendations.analysis.feel_v2") is None


def test_no_tensorflow_or_essentia_anywhere_in_src():
    """Both were removed at the cutover: TensorFlow ran the EffNet graph, and
    Essentia was the generation before that. Neither may come back -- they
    are ~2 GB of image between them and the model one of them loads is not
    sellable."""
    banned = ("import tensorflow", "from tensorflow",
              "import essentia", "from essentia")
    hits = [str(p) for p in SRC.rglob("*.py")
            if any(token in p.read_text() for token in banned)]
    assert hits == []
