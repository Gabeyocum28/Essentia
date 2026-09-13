"""Shared fakes: an in-memory MongoDB (mongomock) standing in for Atlas."""
import mongomock
import pytest

from music_recommendations.server import store


@pytest.fixture(autouse=True)
def clear_matrix_cache():
    """/recommend caches the corpus matrix in module state; tests must not share it."""
    from music_recommendations.server import app, viz

    app._MATRIX_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    app._SUBSET_CACHE.clear()
    app._TRACK_META.clear()
    app._VIZ_SNAPSHOT = None
    viz.clear_geometry_cache()
    yield
    app._MATRIX_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    app._SUBSET_CACHE.clear()
    app._TRACK_META.clear()
    app._VIZ_SNAPSHOT = None
    viz.clear_geometry_cache()


@pytest.fixture
def fake_mongo(monkeypatch):
    """A fresh in-memory database wired into store.db() for one test."""
    database = mongomock.MongoClient().get_database("essentia_test")
    store.reset()
    monkeypatch.setattr(store, "db", lambda: database)
    yield database
    store.reset()
