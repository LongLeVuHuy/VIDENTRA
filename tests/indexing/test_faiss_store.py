"""Unit tests for FAISS and NumPy exact index storage with reference parity."""
from pathlib import Path
import pytest
import numpy as np

from videntra.indexing.faiss_store import VisualIndexStore, NumpyFlatIndex


def test_numpy_flat_index_exact_ranking():
    idx = NumpyFlatIndex(dimension=4)
    # Unit vectors
    v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)

    idx.add(np.vstack([v0, v1, v2]), ids=[100, 101, 102])
    assert len(idx) == 3

    # Query with v0: closest is v0 (sim=1.0), then v2 (sim~0.7071), then v1 (sim=0)
    scores, ids = idx.search(v0, top_k=3)
    assert ids[0][0] == 100
    assert np.isclose(scores[0][0], 1.0, atol=1e-4)
    assert ids[0][1] == 102
    assert np.isclose(scores[0][1], 0.7071, atol=1e-3)
    assert ids[0][2] == 101


def test_visual_index_store_faiss_parity(tmp_path):
    store = VisualIndexStore(dimension=8, prefer_faiss=True)
    rng = np.random.RandomState(42)

    # Generate 50 random unit vectors
    raw = rng.randn(50, 8).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs = raw / norms

    ids = list(range(1000, 1050))
    store.add(vecs, ids=ids)
    assert len(store) == 50

    # Query with 5 random vectors
    q_raw = rng.randn(5, 8).astype(np.float32)
    q_vecs = q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True)

    scores, hit_ids = store.search(q_vecs, top_k=5)
    assert scores.shape == (5, 5)
    assert hit_ids.shape == (5, 5)

    # Parity verification
    assert store.verify_parity(q_vecs, top_k=5) is True

    # Persistence
    save_dir = tmp_path / "idx_store"
    store.save(save_dir)
    assert (save_dir / "index_numpy.npz").exists()

    loaded = VisualIndexStore.load(save_dir, prefer_faiss=True)
    assert len(loaded) == 50
    l_scores, l_ids = loaded.search(q_vecs, top_k=5)
    assert np.allclose(scores, l_scores, atol=1e-4)
    assert np.array_equal(hit_ids, l_ids)
