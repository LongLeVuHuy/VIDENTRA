"""Sharded visual vector index supporting multi-shard scaling and unified search."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from videntra.indexing.faiss_store import VisualIndexStore


class ShardedVisualIndex:
    """Manages multiple vector index shards and merges search results."""

    def __init__(self, shards: Sequence[VisualIndexStore] | None = None):
        self.shards: list[VisualIndexStore] = list(shards) if shards else []

    def add_shard(self, shard: VisualIndexStore) -> None:
        self.shards.append(shard)

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search across all shards and return globally top-k scores and int_ids."""
        if not self.shards or len(self) == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        all_batch_scores = []
        all_batch_ids = []

        for q_idx in range(len(query_vectors)):
            single_q = query_vectors[q_idx : q_idx + 1]
            candidate_pairs = []

            for shard in self.shards:
                if len(shard) == 0:
                    continue
                k_search = min(top_k, len(shard))
                scores, ids = shard.search(single_q, top_k=k_search)
                for sc, vid in zip(scores[0], ids[0]):
                    candidate_pairs.append((float(sc), int(vid)))

            # Sort descending by score
            candidate_pairs.sort(key=lambda p: p[0], reverse=True)
            top_pairs = candidate_pairs[:top_k]

            if top_pairs:
                out_scores = [p[0] for p in top_pairs]
                out_ids = [p[1] for p in top_pairs]
            else:
                out_scores = []
                out_ids = []

            all_batch_scores.append(out_scores)
            all_batch_ids.append(out_ids)

        # Pad to dense rectangular array
        max_len = max((len(row) for row in all_batch_scores), default=0)
        final_scores = np.zeros((len(query_vectors), max_len), dtype=np.float32)
        final_ids = np.full((len(query_vectors), max_len), -1, dtype=np.int64)

        for i in range(len(query_vectors)):
            l = len(all_batch_scores[i])
            if l > 0:
                final_scores[i, :l] = all_batch_scores[i]
                final_ids[i, :l] = all_batch_ids[i]

        return final_scores, final_ids
