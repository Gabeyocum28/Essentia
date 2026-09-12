import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "atlas_check.py"
spec = importlib.util.spec_from_file_location("atlas_check", SCRIPT)
atlas_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas_check)


def test_check_round_trips_a_fixture_track_and_cleans_up(fake_mongo, capsys):
    assert atlas_check.run() == 0
    out = capsys.readouterr().out
    assert "tracks:" in out and "round-trip ok" in out
    assert fake_mongo.tracks.find_one({"_id": "atlas-check"}) is None
    assert fake_mongo.jobs.find_one({"_id": "embed:atlas-check"}) is None
