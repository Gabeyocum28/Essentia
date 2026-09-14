"""feel_backfill.py: what it scores, what it leaves alone, and that it is safe twice."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from music_recommendations.analysis.schema import FEATURES_VERSION
from music_recommendations.server import store

SCRIPT = Path(__file__).parents[2] / "scripts" / "feel_backfill.py"
spec = importlib.util.spec_from_file_location("feel_backfill", SCRIPT)
feel_backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feel_backfill)

DIM = 11


def fake_feel_vectors(embeddings: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for the eleven heads: every dimension is the
    first embedding component, squashed into [0, 1]. Monkeypatched in so the
    test never loads TensorFlow."""
    rows = np.asarray(embeddings, dtype=np.float32)
    value = 1.0 / (1.0 + np.exp(-rows[:, 0]))
    return np.repeat(value[:, None], DIM, axis=1).astype(np.float32)


@pytest.fixture(autouse=True)
def no_tensorflow(monkeypatch):
    monkeypatch.setattr(feel_backfill, "feel_vectors", fake_feel_vectors)


def _put(track_id: str, first: float, **features) -> None:
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = first
    store.put_track({"track_id": track_id, "title": track_id,
                     "artist": "Miles Davis", "album": "Kind of Blue",
                     "artwork_url": "u"},
                    {"embedding": vec, **features, "_features_version": FEATURES_VERSION})


def _feel(track_id: str) -> list[float] | None:
    return store.db().tracks.find_one({"_id": track_id}, {"feel": 1}).get("feel")


def test_scores_every_row_that_has_none(fake_mongo, capsys):
    _put("a", 1.0)
    _put("b", -1.0)

    assert feel_backfill.run([]) == 0

    for track_id in ("a", "b"):
        vector = _feel(track_id)
        assert len(vector) == DIM
        assert all(0.0 <= v <= 1.0 for v in vector)
    assert _feel("a") != _feel("b")
    assert "scored 2 in" in capsys.readouterr().out


def test_prints_the_count_and_the_rate(fake_mongo, capsys):
    _put("a", 1.0)
    feel_backfill.run([])
    out = capsys.readouterr().out
    assert out.startswith("scored 1 in ")
    assert "tracks/s)" in out


def test_skips_rows_that_already_have_feel(fake_mongo, capsys):
    _put("a", 1.0, feel=[0.5] * DIM)
    _put("b", -1.0)

    feel_backfill.run([])

    assert _feel("a") == [0.5] * DIM     # untouched
    assert _feel("b") is not None
    assert "scored 1 in" in capsys.readouterr().out


def test_all_rescores_rows_that_already_have_feel(fake_mongo, capsys):
    _put("a", 1.0, feel=[0.5] * DIM)

    feel_backfill.run(["--all"])

    assert _feel("a") != [0.5] * DIM
    assert "scored 1 in" in capsys.readouterr().out


def test_a_second_run_has_nothing_left_to_do(fake_mongo, capsys):
    _put("a", 1.0)
    feel_backfill.run([])
    capsys.readouterr()

    assert feel_backfill.run([]) == 0
    assert "scored 0 in" in capsys.readouterr().out


def test_chunking_covers_every_row(fake_mongo, capsys):
    for i in range(7):
        _put(f"t{i}", float(i) / 7.0)

    feel_backfill.run(["--chunk", "2"])

    assert all(_feel(f"t{i}") is not None for i in range(7))
    assert "scored 7 in" in capsys.readouterr().out


def test_an_unanalyzed_row_is_not_scored(fake_mongo, capsys):
    store.put_track_meta({"track_id": "meta", "title": "x", "artist": "y",
                          "album": "z", "artwork_url": "u"})

    feel_backfill.run([])

    assert _feel("meta") is None
    assert "scored 0 in" in capsys.readouterr().out


def test_the_backfilled_vector_reaches_get_features(fake_mongo):
    _put("a", 1.0)
    feel_backfill.run([])

    features = store.get_features("a")
    assert len(features["feel"]) == DIM
