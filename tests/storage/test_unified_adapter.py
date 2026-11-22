"""Unit tests for UnifiedSearchEngine storage adapter."""
from pathlib import Path
import pytest
from videntra.config.paths import resolve_keyframe_faiss_path, resolve_metadata_db_path
from videntra.storage.adapter import UnifiedSearchEngine


def test_unified_search_engine_auto_detection():
    # If production DB exists, it detects sqlite_faiss
    db_p = resolve_metadata_db_path("data/processed")
    faiss_p = resolve_keyframe_faiss_path("data/processed")

    if db_p.exists() and faiss_p.exists():
        engine = UnifiedSearchEngine(db_path=db_p, faiss_path=faiss_p)
        assert engine.backend == "sqlite_faiss"
        assert len(engine.indexed_videos) > 0
    else:
        engine = UnifiedSearchEngine(db_path="nonexistent.db", faiss_path="nonexistent.bin")
        assert engine.backend == "uninitialized"


def test_unified_search_engine_uninitialized():
    engine = UnifiedSearchEngine(db_path="fake.db", faiss_path="fake.bin")
    res = engine.search("A test query")
    assert res == []
