from __future__ import annotations

import importlib.util

import numpy as np

from music_recommendations.analysis import (FEATURES_VERSION, METRICS,
                                            analyze_track, analyze_tracks,
                                            as_json)
from tests.analysis.conftest import needs_effnet


def test_version_and_metrics():
    assert FEATURES_VERSION == 3
    assert METRICS == {"embedding": "cosine"}


def test_no_essentia_anywhere_in_src():
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "src"
    hits = [p for p in src.rglob("*.py")
            if "import essentia" in p.read_text() or "from essentia" in p.read_text()]
    assert hits == []


def test_removed_modules_are_gone():
    for name in ("heads", "groove"):
        assert importlib.util.find_spec(f"music_recommendations.analysis.{name}") is None


@needs_effnet
def test_analyze_track_returns_only_embedding(tone_wav):
    feats = analyze_track(tone_wav)
    assert set(feats) == {"embedding"}
    assert feats["embedding"].shape == (1280,)
    js = as_json(feats)
    assert len(js["embedding"]) == 1280 and isinstance(js["embedding"][0], float)


@needs_effnet
def test_analyze_tracks_keeps_a_bad_path_from_sinking_the_group(tone_wav, tmp_path):
    """One unreadable file in a group must not cost the others their
    analysis: its slot holds the exception, the rest hold features, and the
    features are the same ones analyze_track would have produced alone."""
    out = analyze_tracks([tone_wav, tmp_path / "missing.mp3", tone_wav])

    assert isinstance(out[1], Exception)
    for feats in (out[0], out[2]):
        assert set(feats) == {"embedding"}
        assert feats["embedding"].shape == (1280,)
    assert np.allclose(out[0]["embedding"], analyze_track(tone_wav)["embedding"],
                       atol=1e-4)
