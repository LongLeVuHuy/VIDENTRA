"""Index compaction removing tombstoned vectors and reclaiming memory."""
from typing import Sequence
import numpy as np

from videntra.indexing.faiss_store import VisualIndexStore
from videntra.indexing.ids import IDMappingStore


def compact_index(
    index_store: VisualIndexStore,
    id_mapping: IDMappingStore,
    tombstoned_ids: Sequence[int] = (),
) -> tuple[VisualIndexStore, IDMappingStore, int]:
    """Rebuild a clean, contiguous index excluding tombstoned vectors.

    Returns:
        (new_index_store, new_id_mapping, removed_count)
    """
    tombstone_set = set(tombstoned_ids)
    all_int_ids = [c.int64_id for c in id_mapping._by_int]

    surviving_ids = [i for i in all_int_ids if i not in tombstone_set]
    removed_count = len(all_int_ids) - len(surviving_ids)

    # Reconstruct vectors from reference numpy index
    np_index = index_store.numpy_index
    dim = np_index.dimension
    new_index_store = VisualIndexStore(dimension=dim)
    new_id_mapping = IDMappingStore()

    if surviving_ids and len(np_index.ids) > 0:
        # Extract surviving vectors
        id_to_idx = {int(vid): idx for idx, vid in enumerate(np_index.ids)}
        surviving_vectors = []
        final_ids = []
        for sid in surviving_ids:
            if sid in id_to_idx:
                idx = id_to_idx[sid]
                vec = np_index.vectors[idx]
                coord = id_mapping.get_by_int_id(sid)
                if coord is not None:
                    surviving_vectors.append(vec)
                    final_ids.append(sid)
                    new_id_mapping.register(
                        video_id=coord.video_id,
                        timeline_revision=coord.timeline_revision,
                        ordinal=coord.ordinal,
                        sampled_frame_id=coord.sampled_frame_id,
                        timestamp_us=coord.timestamp_us,
                        timestamp_seconds=coord.timestamp_seconds,
                    )

        if surviving_vectors:
            mat = np.stack(surviving_vectors, axis=0)
            new_index_store.add(mat, final_ids)

    return new_index_store, new_id_mapping, removed_count
