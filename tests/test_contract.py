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


def test_track_optional_fields():
    """Two keys a Track MAY carry -- and only these two. They are how a
    Creative Commons source's credit reaches the clients; anything else a
    source knows stays server-side."""
    f = _features()
    assert f.TRACK_OPTIONAL_FIELDS == {"source", "attribution_url"}
    assert not (f.TRACK_OPTIONAL_FIELDS & f.TRACK_FIELDS)


def test_feature_keys_are_the_embedding_and_the_feel_vector():
    f = _features()
    assert f.FEATURE_KEYS == {"embedding": 1024, "feel": 8}


def test_feel_dimension_matches_the_prompt_bank():
    """The contract states the width; analysis owns the names. If an axis is
    added or dropped without the contract moving, this is where it shows."""
    from music_recommendations.analysis.feel_v2 import FEEL_KEYS, PROMPT_BANK
    f = _features()
    assert len(FEEL_KEYS) == f.FEATURE_KEYS["feel"]
    assert FEEL_KEYS == [
        "energy", "valence", "tension", "acoustic",
        "danceable", "vocal", "bright", "density",
    ]
    # Every axis is a contrastive pair: a single prompt would score every
    # track high, because CLAP similarities sit in a narrow positive band.
    assert [name for name, _p, _n in PROMPT_BANK] == FEEL_KEYS
    for _name, pos, neg in PROMPT_BANK:
        assert pos and neg and pos != neg


def test_rhythm_keys_match_the_analysis_module():
    """The seven named numbers the app renders and the ranking reads."""
    from music_recommendations.analysis.rhythm import RHYTHM_KEYS
    f = _features()
    assert f.RHYTHM_KEYS == RHYTHM_KEYS
    assert f.RHYTHM_KEYS == (
        "tempo_bpm", "beat_strength", "loudness_lufs", "loudness_range",
        "key", "mode", "key_strength",
    )


def test_fixture_thirty_contract_tracks():
    f = _features()
    data = json.loads((CONTRACT / "fixture.json").read_text())
    assert len(data["tracks"]) == 30
    for t in data["tracks"]:
        # The optional keys are allowed but the fixture is Deezer, which
        # needs neither.
        assert set(t) == f.TRACK_FIELDS
        assert t["preview_url"].startswith("http")
        assert isinstance(t["track_id"], str)
