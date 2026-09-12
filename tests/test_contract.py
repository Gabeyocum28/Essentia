import importlib.util
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "contract"


def _features():
    spec = importlib.util.spec_from_file_location("features", CONTRACT / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_axes_ids_match_spec():
    f = _features()
    # Diverged from the spec's original five: groove was dropped (four rhythm
    # numbers could not carry a recommendation on their own), then best_match
    # was cut so the two remaining buttons read as opposites.
    assert [a["id"] for a in f.AXES] == ["sounds_like", "surprise"]


def test_axis_labels_are_the_two_opposites():
    f = _features()
    assert [a["label"] for a in f.AXES] == [
        "More sounds like this", "Nothing like this",
    ]


def test_track_fields():
    f = _features()
    assert f.TRACK_FIELDS == {
        "track_id", "title", "artist", "album", "artwork_url", "preview_url"
    }


def test_feature_keys_embedding_only():
    f = _features()
    assert f.FEATURE_KEYS == {"embedding": 1280}


def test_fixture_thirty_contract_tracks():
    f = _features()
    data = json.loads((CONTRACT / "fixture.json").read_text())
    assert len(data["tracks"]) == 30
    for t in data["tracks"]:
        assert set(t) == f.TRACK_FIELDS
        assert t["preview_url"].startswith("http")
        assert isinstance(t["track_id"], str)
