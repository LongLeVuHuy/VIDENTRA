"""Visual index storage using FAISS CPU Flat Inner Product with NumPy exact reference parity."""
from pathlib import Path
import numpy as np


class NumpyFlatIndex:
    """Exact cosine/inner-product index implemented in NumPy for fallback and parity checking."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int] | None = None) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {vectors.shape[1]}")

        vectors = vectors.astype(np.float32)
        n = vectors.shape[0]

        if ids is None:
            start_id = len(self.ids)
            assigned_ids = np.arange(start_id, start_id + n, dtype=np.int64)
        else:
            assigned_ids = np.asarray(ids, dtype=np.int64)
            if len(assigned_ids) != n:
                raise ValueError(f"IDs count {len(assigned_ids)} != vectors count {n}")

        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, assigned_ids])

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vectors) == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        query_vectors = query_vectors.astype(np.float32)

        k = min(top_k, len(self.vectors))
        # Inner product on normalized vectors = cosine similarity
        sims = np.dot(query_vectors, self.vectors.T)  # [Q, N]

        # Top-k via argpartition
        top_k_indices_local = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        # Sort each query's top-k in descending order
        batch_scores = []
        batch_ids = []
        for q_idx in range(len(query_vectors)):
            local_idx = top_k_indices_local[q_idx]
            q_scores = sims[q_idx, local_idx]
            order = np.argsort(-q_scores)
            sorted_local = local_idx[order]
            batch_scores.append(q_scores[order])
            batch_ids.append(self.ids[sorted_local])

        return np.array(batch_scores, dtype=np.float32), np.array(batch_ids, dtype=np.int64)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(p), vectors=self.vectors, ids=self.ids, dimension=self.dimension)

    @classmethod
    def load(cls, path: Path | str) -> "NumpyFlatIndex":
        p = Path(path)
        data = np.load(str(p))
        dim = int(data["dimension"])
        idx = cls(dimension=dim)
        idx.vectors = data["vectors"]
        idx.ids = data["ids"]
        return idx


class VisualIndexStore:
    """Visual vector index store wrapping FAISS FlatIP with NumPy reference parity."""

    def __init__(self, dimension: int = 768, prefer_faiss: bool = True):
        self.dimension = dimension
        self.prefer_faiss = prefer_faiss
        self.backend = "numpy"
        self._faiss_index = None
        self._faiss_id_map = None
        self._numpy_index = NumpyFlatIndex(dimension=dimension)

        if prefer_faiss:
            try:
                import faiss
                self._faiss = faiss
                self._faiss_base = faiss.IndexFlatIP(dimension)
                self._faiss_index = faiss.IndexIDMap2(self._faiss_base)
                self.backend = "faiss_cpu"
            except Exception:
                self.backend = "numpy"

    def __len__(self) -> int:
        if self.backend == "faiss_cpu" and self._faiss_index is not None:
            return self._faiss_index.ntotal
        return len(self._numpy_index)

    @property
    def numpy_index(self) -> NumpyFlatIndex:
        return self._numpy_index

    def add(self, vectors: np.ndarray, ids: list[int] | np.ndarray | None = None) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = vectors.astype(np.float32)
        n = vectors.shape[0]

        if ids is None:
            current_len = len(self)
            ids_arr = np.arange(current_len, current_len + n, dtype=np.int64)
        else:
            ids_arr = np.asarray(ids, dtype=np.int64)

        # Always update numpy mirror for reference parity
        self._numpy_index.add(vectors, ids_arr)

        if self.backend == "faiss_cpu" and self._faiss_index is not None:
            self._faiss_index.add_with_ids(vectors, ids_arr)

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        query_vectors = query_vectors.astype(np.float32)
        k = min(top_k, len(self))

        if self.backend == "faiss_cpu" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vectors, k)
            return scores.astype(np.float32), indices.astype(np.int64)

        return self._numpy_index.search(query_vectors, k)

    def verify_parity(self, query_vectors: np.ndarray, top_k: int = 5, tolerance: float = 1e-4) -> bool:
        """Verify that FAISS search results match exact NumPy reference."""
        if self.backend != "faiss_cpu" or self._faiss_index is None:
            return True
        k = min(top_k, len(self))
        if k == 0:
            return True

        faiss_scores, faiss_ids = self._faiss_index.search(query_vectors.astype(np.float32), k)
        ref_scores, ref_ids = self._numpy_index.search(query_vectors, k)

        scores_close = np.allclose(faiss_scores, ref_scores, atol=tolerance)
        if not scores_close:
            return False

        # If scores match within tolerance, check ID consistency.
        # Boundary tie: when multiple items share the exact same score at rank k-1,
        # FAISS and NumPy may pick different elements for the last slot.
        # We only require ID agreement for scores strictly higher than the boundary tie score.
        for q_idx in range(len(query_vectors)):
            f_sc = np.round(faiss_scores[q_idx], decimals=4)
            r_sc = np.round(ref_scores[q_idx], decimals=4)
            boundary_score = f_sc[-1]
            strictly_higher_scores = np.unique(f_sc[f_sc > boundary_score])
            for sc in strictly_higher_scores:
                mask_f = np.isclose(f_sc, sc, atol=tolerance)
                mask_r = np.isclose(r_sc, sc, atol=tolerance)
                if set(faiss_ids[q_idx][mask_f]) != set(ref_ids[q_idx][mask_r]):
                    return False

        return True

    def save(self, directory: Path | str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        # Always save numpy representation
        self._numpy_index.save(p / "index_numpy.npz")
        # Save FAISS if active
        if self.backend == "faiss_cpu" and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(p / "index.faiss"))

    @classmethod
    def load(cls, directory: Path | str, prefer_faiss: bool = True) -> "VisualIndexStore":
        p = Path(directory)
        np_path = p / "index_numpy.npz"
        faiss_path = p / "index.faiss"

        if not np_path.exists() and not faiss_path.exists():
            raise FileNotFoundError(f"No index found in {directory}")

        if np_path.exists():
            np_idx = NumpyFlatIndex.load(np_path)
            store = cls(dimension=np_idx.dimension, prefer_faiss=prefer_faiss)
            store._numpy_index = np_idx
            if store.backend == "faiss_cpu" and faiss_path.exists():
                store._faiss_index = store._faiss.read_index(str(faiss_path))
            elif store.backend == "faiss_cpu":
                store._faiss_index.add_with_ids(np_idx.vectors, np_idx.ids)
            return store

        # FAISS only
        import faiss
        faiss_idx = faiss.read_index(str(faiss_path))
        store = cls(dimension=faiss_idx.d, prefer_faiss=True)
        store._faiss_index = faiss_idx
        return store
