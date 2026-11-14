"""Unit tests for Multi-Clause Query Ensemble in visual retrieval."""
import pytest
import numpy as np
from videntra.retrieval.visual import ensemble_clause_scores, VisualRetrievalEngine
from videntra.indexing.faiss_store import VisualIndexStore
from videntra.indexing.ids import IDMappingStore
from videntra.encoders.siglip2 import MockSigLIP2Encoder


def test_ensemble_single_clause():
    s_full = 0.45
    combined = ensemble_clause_scores(full_score=s_full, clause_scores=[])
    assert abs(combined - s_full) < 1e-6


def test_ensemble_multi_clause_no_bonus():
    s_full = 0.50
    # One clause matches high, one low, but only 1 hits tau=0.22
    clause_scores = [0.60, 0.10]
    # Expected: 0.70 * 0.50 + 0.30 * 0.60 = 0.35 + 0.18 = 0.53
    combined = ensemble_clause_scores(full_score=s_full, clause_scores=clause_scores, tau_clause=0.22)
    assert abs(combined - 0.53) < 1e-4


def test_ensemble_multi_clause_with_coverage_bonus():
    s_full = 0.50
    # Both clauses exceed tau=0.22
    clause_scores = [0.40, 0.30]
    # Base: 0.70 * 0.50 + 0.30 * 0.40 = 0.35 + 0.12 = 0.47
    # Bonus: 0.05 * (2 / 2) = 0.05
    # Total: 0.52
    combined = ensemble_clause_scores(full_score=s_full, clause_scores=clause_scores, tau_clause=0.22)
    assert abs(combined - 0.52) < 1e-4


def test_visual_engine_search_with_ensemble():
    encoder = MockSigLIP2Encoder()
    index_store = VisualIndexStore(dimension=768, prefer_faiss=False)
    id_mapping = IDMappingStore()

    rng = np.random.RandomState(42)
    # Register 3 mock frames
    for i in range(3):
        mid = id_mapping.register(
            video_id=f"video_{i}",
            timeline_revision="rev1",
            ordinal=i * 10,
            sampled_frame_id=f"frame_{i}",
            timestamp_us=i * 1_000_000,
            timestamp_seconds=float(i),
        )
        vec = rng.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        index_store.add(vec, [mid])

    engine = VisualRetrievalEngine(encoder=encoder, index_store=index_store, id_mapping=id_mapping)

    # Search with multi-clause query
    results = engine.search("A woman walking in the park and talking on the phone", top_k_moments=3)
    assert len(results) > 0
    assert results[0].ranking_score > 0.0
