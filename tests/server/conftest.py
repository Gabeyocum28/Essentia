"""Shared fakes: an in-memory MongoDB (mongomock) standing in for Atlas."""
import mongomock
import pytest

from music_recommendations.server import store


@pytest.fixture(autouse=True)
def clear_matrix_cache():
    """/recommend caches the corpus matrix in module state; tests must not share it."""
    from music_recommendations.server import app, viz

    app._MATRIX_CACHE.clear()
    app._UNIT_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    app._HUBS_CACHE.clear()
    app._SUBSET_CACHE.clear()
    app._TRACK_META.clear()
    app._VIZ_SNAPSHOT = None
    app._ROW_NORMS = None
    app._FEEL_ALIGN_CACHE = None
    app._RHYTHM_ALIGN_CACHE = None
    # Grow-only in production (a corpus id is only ever added); a test that
    # empties the store and re-uses an id would otherwise be served the
    # previous test's tempo.
    app._TEMPO_BY_ID.clear()
    app._UMAP_CACHE.clear()
    viz.clear_geometry_cache()
    yield
    app._MATRIX_CACHE.clear()
    app._UNIT_CACHE.clear()
    app._TOP8_CACHE.clear()
    app._MST_CACHE.clear()
    app._HUBS_CACHE.clear()
    app._SUBSET_CACHE.clear()
    app._TRACK_META.clear()
    app._VIZ_SNAPSHOT = None
    app._ROW_NORMS = None
    app._FEEL_ALIGN_CACHE = None
    app._RHYTHM_ALIGN_CACHE = None
    # Grow-only in production (a corpus id is only ever added); a test that
    # empties the store and re-uses an id would otherwise be served the
    # previous test's tempo.
    app._TEMPO_BY_ID.clear()
    app._UMAP_CACHE.clear()
    viz.clear_geometry_cache()


@pytest.fixture(autouse=True)
def clear_worker_state():
    """The re-analysis circuit breaker is module state and survives a test.

    A test that deliberately breaks the model leaves the arm HALTED, and the
    next test's perfectly good group silently does nothing. Reset both ends.
    """
    from music_recommendations import worker

    worker._group_failures = 0
    worker._halted_until = 0.0
    yield
    worker._group_failures = 0
    worker._halted_until = 0.0


@pytest.fixture
def fake_mongo(monkeypatch):
    """A fresh in-memory database wired into store.db() for one test."""
    database = mongomock.MongoClient().get_database("essentia_test")
    store.reset()
    monkeypatch.setattr(store, "db", lambda: database)
    yield database
    store.reset()
