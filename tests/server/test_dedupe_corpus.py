"""The backfill: what it marks, what it leaves alone, and that it is safe twice."""
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from music_recommendations.analysis.quantize import to_int8
from music_recommendations.analysis.schema import FEATURES_VERSION
from music_recommendations.server import store
from music_recommendations.server.dedupe import dedupe_key

SCRIPT = Path(__file__).parents[2] / "scripts" / "dedupe_corpus.py"
spec = importlib.util.spec_from_file_location("dedupe_corpus", SCRIPT)
dedupe_corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedupe_corpus)

DIM = 8


def _vec(**components: float) -> np.ndarray:
    """A DIM-vector from {axis index: weight}, e.g. _vec(**{"1": 0.9})."""
    vec = np.zeros(DIM, dtype=np.float32)
    for axis, weight in components.items():
        vec[int(axis)] = weight
    return vec


E0 = _vec(**{"0": 1.0})
E1 = _vec(**{"1": 1.0})
NEAR_E1 = _vec(**{"1": 1.0, "2": 0.03})          # cosine ~0.9995 with E1
FAR_E1 = _vec(**{"1": 0.9, "3": float(np.sqrt(0.19))})  # cosine ~0.9 with E1

# (id, title, artist, day-of-analysis, embedding). Oldest first.
CORPUS = [
    ("a1", "So What", "Miles Davis", 1, E0),
    ("a2", "So What - 2001 Remaster", "Miles Davis", 2, E0),   # key dup of a1
    ("a3", "So What (Live)", "Miles Davis", 3, E0),            # key dup of a1
    ("b1", "Blue in Green", "Miles Davis", 4, E1),
    ("b2", "Blue in Green (Alternate)", "Miles Davis", 5, NEAR_E1),  # embed dup of b1
    ("cover", "So What", "John Coltrane", 6, E1),              # a cover: keep
    ("c1", "Freddie Freeloader", "Miles Davis", 7, FAR_E1),    # merely similar: keep
]

MARKED = {"a2": "a1", "a3": "a1", "b2": "b1"}
KEPT = {"a1", "b1", "cover", "c1"}


@pytest.fixture
def corpus(fake_mongo):
    """The seven tracks above, written as put_track would write them."""
    for track_id, title, artist, day, vector in CORPUS:
        data, scale = to_int8(vector)
        fake_mongo.tracks.insert_one({
            "_id": track_id, "title": title, "artist": artist,
            "analyzed_at": datetime(2026, 9, day), "embedding": data,
            "scale": float(scale), "dedupe_key": dedupe_key(title, artist),
            "features_version": FEATURES_VERSION,
        })
    return fake_mongo


def _duplicates(db) -> dict[str, str]:
    return {d["_id"]: d["duplicate_of"]
            for d in db.tracks.find({"duplicate_of": {"$exists": True}})}


def test_dry_run_reports_the_counts_and_writes_nothing(corpus, capsys):
    assert dedupe_corpus.run([]) == 0

    out = capsys.readouterr().out
    assert "live analyzed tracks: 7" in out
    assert "pass 1 (key):        2" in out
    assert "pass 2 (embedding):  1" in out
    assert "total to mark:       3" in out
    assert "would mark 3 of 7 (42.9%)" in out
    assert "So What" in out and "Miles Davis" in out  # the top group
    assert _duplicates(corpus) == {}


def test_apply_marks_the_duplicates_and_keeps_the_earliest(corpus, capsys):
    assert dedupe_corpus.run(["--apply"]) == 0

    assert _duplicates(corpus) == MARKED
    out = capsys.readouterr().out
    assert "marked 3 of 7" in out
    assert "restart" in out  # the in-memory matrix reminder


def test_a_second_apply_marks_nothing(corpus, capsys):
    dedupe_corpus.run(["--apply"])
    capsys.readouterr()

    assert dedupe_corpus.run(["--apply"]) == 0
    out = capsys.readouterr().out
    assert "live analyzed tracks: 4" in out
    assert "marked 0 of 4" in out
    assert _duplicates(corpus) == MARKED


def test_marked_tracks_leave_the_corpus(corpus):
    assert set(store.corpus_ids()) == KEPT | set(MARKED)  # also primes the cache

    dedupe_corpus.run(["--apply"])

    assert set(store.corpus_ids()) == KEPT
