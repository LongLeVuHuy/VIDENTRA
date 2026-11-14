"""Tests for sharded index and compaction."""
import numpy as np
import pytest

from videntra.indexing.faiss_store import VisualIndexStore
from videntra.indexing.ids import IDMappingStore
from videntra.indexing.shards import ShardedVisualIndex
from videntra.indexing.compaction import compact_index


def test_sharded_visual_index_merging():
    dim = 8
    shard1 = VisualIndexStore(dimension=dim)
    shard2 = VisualIndexStore(dimension=dim)

    # Shard 1 vectors
    vec1 = np.ones((2, dim), dtype=np.float32)
    vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    shard1.add(vec1, [100, 101])

    # Shard 2 vectors (slightly perturbed)
    vec2 = np.ones((2, dim), dtype=np.float32) * 0.9
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    shard2.add(vec2, [200, 201])

    sharded = ShardedVisualIndex([shard1, shard2])
    assert len(sharded) == 4

    q = np.ones((1, dim), dtype=np.float32)
    q = q / np.linalg.norm(q)

    scores, ids = sharded.search(q, top_k=3)
    assert scores.shape == (1, 3)
    assert ids.shape == (1, 3)
    # The merged top hits should come from both shards
    hit_ids = list(ids[0])
    assert len(set(hit_ids)) == 3


def test_compact_index_removes_tombstones():
    dim = 4
    store = VisualIndexStore(dimension=dim)
    mapping = IDMappingStore()

    vecs = np.eye(dim, dtype=np.float32)
    ids = []
    for i in range(dim):
        int_id = mapping.register(
            video_id=f"vid_{i}",
            timeline_revision="rev1",
            ordinal=i,
            sampled_frame_id=f"samp_{i}",
            timestamp_us=i * 1000000,
            timestamp_seconds=float(i),
        )
        ids.append(int_id)

    store.add(vecs, ids)
    assert len(store) == dim
    assert len(mapping) == dim

    # Tombstone the first 2 IDs
    tombstones = [ids[0], ids[1]]
    new_store, new_mapping, removed = compact_index(store, mapping, tombstones)

    assert removed == 2
    assert len(new_store) == dim - 2
    assert len(new_mapping) == dim - 2

    # Querying the new store should not contain the tombstoned IDs
    q = np.eye(1, dim, dtype=np.float32)
    scores, res_ids = new_store.search(q, top_k=dim)
    for tid in tombstones:
        assert tid not in res_ids[0]
