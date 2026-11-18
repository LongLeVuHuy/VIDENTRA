"""VideoPrism-LvT-B (768d) clip/sub-clip retrieval, CSLS reranking, and temporal curve helpers."""
from __future__ import annotations

from collections.abc import Iterator, Mapping as AbcMapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import faiss
import numpy as np

VIDEOPRISM_DIM = 768


class _VideoScoreMap(AbcMapping[str, float]):
    """Zero-copy read-only mapping from video_id to NumPy vector entry."""

    __slots__ = ("_vid_to_idx", "_video_ids", "_values")

    def __init__(self, vid_to_idx: dict[str, int], video_ids: list[str], values: np.ndarray) -> None:
        self._vid_to_idx = vid_to_idx
        self._video_ids = video_ids
        self._values = values

    def __getitem__(self, key: str) -> float:
        idx = self._vid_to_idx[key]
        return float(self._values[idx])

    def get(self, key: str, default: float | None = None) -> float | None:  # type: ignore[override]
        idx = self._vid_to_idx.get(key)
        if idx is None:
            return default
        return float(self._values[idx])

    def __contains__(self, key: object) -> bool:
        return key in self._vid_to_idx

    def __iter__(self) -> Iterator[str]:
        return iter(self._video_ids)

    def __len__(self) -> int:
        return len(self._video_ids)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 1e-8, mat / np.maximum(norms, 1e-8), 0.0).astype(np.float32)


def _normalize_vec(vec: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] != VIDEOPRISM_DIM or not np.isfinite(v).all():
        return None
    norm = float(np.linalg.norm(v))
    if norm <= 1e-8:
        return None
    return (v / norm).astype(np.float32)


def _reconstruct_faiss_vectors(index: faiss.Index, expected_count: int) -> np.ndarray | None:
    if index is None or int(index.ntotal) <= 0:
        return None
    n = int(index.ntotal)
    if expected_count > 0 and n != expected_count:
        return None
    if hasattr(index, "reconstruct_n"):
        vecs = np.asarray(index.reconstruct_n(0, n), dtype=np.float32)
    else:
        vecs = np.vstack([index.reconstruct(i) for i in range(n)]).astype(np.float32)
    return _normalize_rows(vecs)


@dataclass(slots=True)
class VideoPrismRetrievalResult:
    """Output of parallel VideoPrism whole-clip + sub-clip retrieval (Step D / Step F)."""

    status: str = "unavailable"
    query_source: str = "none"
    ranked_videos: list[tuple[str, float]] = field(default_factory=list)
    combined_scores: Mapping[str, float] = field(default_factory=dict)
    whole_scores: Mapping[str, float] = field(default_factory=dict)
    subclip_max_scores: Mapping[str, float] = field(default_factory=dict)
    best_subclip_windows: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    z_scores: Mapping[str, float] = field(default_factory=dict)
    corpus_mean: float = 0.0
    corpus_std: float = 1.0
    clause_hit_count: int = 0
    subclip_score_array: np.ndarray | None = None


@dataclass(slots=True)
class VideoPrismCandidateSignals:
    """Rich VideoPrism Top-20 reranking & LTR signals for one candidate video (Step H)."""

    whole_cos: float = 0.0
    subclip_max_cos: float = 0.0
    subclip_topk_mean: float = 0.0
    rerank_score: float = 0.0
    vp_z: float = 0.0
    csls_score: float = 0.0
    clause_min_score: float = 0.0
    clause_order_score: float = 0.0
    best_window_start: float | None = None
    best_window_end: float | None = None
    has_subclips: bool = False


class VideoPrismClipRetriever:
    """Dedicated VideoPrism-LvT-B (768d) clip/subclip retriever and Top-20 reranker."""

    def __init__(
        self,
        processed_dir: Path | str = Path("data/processed"),
        cache_path: Path | str = Path("data/cache/query_videoprism_768d.npz"),
        clip_faiss_path: Path | str | None = None,
        clip_manifest_path: Path | str | None = None,
        subclip_vecs_path: Path | str | None = None,
        subclip_map_path: Path | str | None = None,
        query_encoder: Callable[[str], np.ndarray] | None = None,
        # Optional in-memory injection for unit tests / shared server state
        video_vectors: np.ndarray | None = None,
        video_ids: Sequence[str] | None = None,
        subclip_vectors: np.ndarray | None = None,
        subclip_rows: Sequence[dict[str, Any]] | None = None,
        query_cache_map: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.cache_path = Path(cache_path)
        self.clip_faiss_path = Path(clip_faiss_path) if clip_faiss_path else self.processed_dir / "faiss_videoprism_clip_768.bin"
        self.clip_manifest_path = Path(clip_manifest_path) if clip_manifest_path else self.processed_dir / "videoprism_clip_manifest.json"
        self.subclip_vecs_path = Path(subclip_vecs_path) if subclip_vecs_path else self.processed_dir / "videoprism_subclips_768.npy"
        self.subclip_map_path = Path(subclip_map_path) if subclip_map_path else self.processed_dir / "videoprism_subclips_map.jsonl"
        self.query_encoder = query_encoder

        self.available: bool = False
        self.status: str = "not_built"
        self.model_info: dict[str, Any] = {}
        self.source_receipt_sha256: str | None = None

        self.video_ids: list[str] = []
        self.vid_to_idx: dict[str, int] = {}
        self.video_vectors: np.ndarray | None = None
        self.gallery_hub_k10: np.ndarray | None = None

        self.subclip_vectors: np.ndarray | None = None
        self.subclip_rows: list[dict[str, Any]] = []
        self.subclip_vid_indices: np.ndarray = np.empty(0, dtype=np.int32)
        self.subclip_starts: np.ndarray = np.empty(0, dtype=np.float32)
        self.subclip_ends: np.ndarray = np.empty(0, dtype=np.float32)
        self.video_subclip_positions: dict[str, np.ndarray] = {}
        self.has_subclip_mask: np.ndarray = np.empty(0, dtype=bool)

        self._query_cache: dict[str, np.ndarray] = {}
        self._disk_cached_keys: set[str] = set()
        self.cache_revision: str | None = None

        self._load_all(
            video_vectors=video_vectors,
            video_ids=video_ids,
            subclip_vectors=subclip_vectors,
            subclip_rows=subclip_rows,
            query_cache_map=query_cache_map,
        )

    def _load_all(
        self,
        *,
        video_vectors: np.ndarray | None,
        video_ids: Sequence[str] | None,
        subclip_vectors: np.ndarray | None,
        subclip_rows: Sequence[dict[str, Any]] | None,
        query_cache_map: Mapping[str, np.ndarray] | None,
    ) -> None:
        if video_vectors is not None and video_ids is not None:
            vecs = _normalize_rows(video_vectors)
            vids = [str(v) for v in video_ids]
            if len(vids) > 0 and vecs.shape == (len(vids), VIDEOPRISM_DIM):
                self.video_ids = vids
                self.video_vectors = np.ascontiguousarray(vecs, dtype=np.float32)
                self.vid_to_idx = {v: i for i, v in enumerate(self.video_ids)}
                self.available = True
                self.status = "ready_in_memory"
        elif self.clip_faiss_path.is_file() and self.clip_manifest_path.is_file():
            try:
                manifest = json.loads(self.clip_manifest_path.read_text(encoding="utf-8"))
                vids = [str(v) for v in (manifest.get("video_ids") or [])]
                index = faiss.read_index(str(self.clip_faiss_path))
                vecs = _reconstruct_faiss_vectors(index, len(vids))
                if vecs is not None and len(vids) > 0 and vecs.shape[1] == VIDEOPRISM_DIM:
                    self.video_ids = vids
                    self.video_vectors = np.ascontiguousarray(vecs, dtype=np.float32)
                    self.vid_to_idx = {v: i for i, v in enumerate(self.video_ids)}
                    self.model_info = dict(manifest.get("model") or {})
                    self.source_receipt_sha256 = manifest.get("source_receipt_sha256")
                    self.available = True
                    self.status = "ready"
                else:
                    self.status = "invalid_manifest"
            except Exception:
                self.status = "load_error"

        if not self.available or self.video_vectors is None:
            return

        # Precompute corpus gallery top-K hubness once (~1ms for 1,324x1,324)
        n_vids = len(self.video_ids)
        if n_vids > 1:
            sim_mat = self.video_vectors @ self.video_vectors.T
            np.fill_diagonal(sim_mat, -np.inf)
            k_nn = min(10, n_vids - 1)
            part = np.partition(sim_mat, -k_nn, axis=1)[:, -k_nn:]
            self.gallery_hub_k10 = np.mean(part, axis=1).astype(np.float32)
        else:
            self.gallery_hub_k10 = np.zeros(n_vids, dtype=np.float32)

        self.has_subclip_mask = np.zeros(n_vids, dtype=bool)
        self._load_subclips(subclip_vectors=subclip_vectors, subclip_rows=subclip_rows)
        self._load_query_cache(query_cache_map=query_cache_map)

    def _load_subclips(
        self,
        *,
        subclip_vectors: np.ndarray | None,
        subclip_rows: Sequence[dict[str, Any]] | None,
    ) -> None:
        vecs: np.ndarray | None = None
        rows: list[dict[str, Any]] = []
        if subclip_vectors is not None and subclip_rows is not None:
            vecs = _normalize_rows(subclip_vectors)
            rows = [dict(r) for r in subclip_rows]
        elif self.subclip_vecs_path.is_file() and self.subclip_map_path.is_file():
            try:
                vecs = _normalize_rows(np.load(str(self.subclip_vecs_path)))
                with self.subclip_map_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        s = line.strip()
                        if s:
                            rows.append(json.loads(s))
            except Exception:
                vecs = None
                rows = []

        if vecs is None or len(rows) == 0 or vecs.shape != (len(rows), VIDEOPRISM_DIM):
            return

        self.subclip_vectors = np.ascontiguousarray(vecs, dtype=np.float32)
        self.subclip_rows = rows
        vid_indices = np.full(len(rows), -1, dtype=np.int32)
        starts = np.zeros(len(rows), dtype=np.float32)
        ends = np.zeros(len(rows), dtype=np.float32)
        grouped: dict[str, list[tuple[float, int]]] = {}

        for i, row in enumerate(rows):
            vid = str(row.get("video_id") or "")
            v_idx = self.vid_to_idx.get(vid, -1)
            st = float(row.get("clip_start_sec", 0.0) or 0.0)
            et = float(row.get("clip_end_sec", st) or st)
            vid_indices[i] = v_idx
            starts[i] = st
            ends[i] = et
            if v_idx >= 0:
                self.has_subclip_mask[v_idx] = True
                grouped.setdefault(vid, []).append((st, i))

        self.subclip_vid_indices = vid_indices
        self.subclip_starts = starts
        self.subclip_ends = ends
        self.video_subclip_positions = {
            vid: np.array([idx for _, idx in sorted(items, key=lambda x: x[0])], dtype=np.int32)
            for vid, items in grouped.items()
        }

    def _load_query_cache(self, *, query_cache_map: Mapping[str, np.ndarray] | None) -> None:
        if query_cache_map is not None:
            for k, vec in query_cache_map.items():
                nv = _normalize_vec(vec)
                if nv is not None:
                    ks = str(k)
                    for key_v in (ks, _sha256_text(ks), _sha256_text(unicodedata.normalize("NFC", ks.strip()))):
                        self._query_cache[key_v] = nv
                        self._disk_cached_keys.add(key_v)
            return

        if not self.cache_path.is_file():
            return
        try:
            with np.load(str(self.cache_path), allow_pickle=False) as data:
                keys = data["keys"]
                vectors = data["vectors"]
                if "model_revision" in data and len(data["model_revision"]) > 0:
                    self.cache_revision = str(data["model_revision"][0])
                expected_rev = str(self.model_info.get("hf_revision") or "")
                if expected_rev and self.cache_revision and self.cache_revision != expected_rev:
                    self.status = "cache_revision_mismatch"
                    return
                for k, vec in zip(keys, vectors):
                    nv = _normalize_vec(vec)
                    if nv is not None:
                        ks = str(k)
                        self._query_cache[ks] = nv
                        self._disk_cached_keys.add(ks)
        except Exception:
            self._query_cache = {}
            self._disk_cached_keys = set()

    def clear_runtime_cache(self) -> None:
        """Drop dynamically added runtime entries while preserving frozen disk NPZ cache."""
        if not self._disk_cached_keys:
            self._query_cache.clear()
            return
        for key in list(self._query_cache.keys()):
            if key not in self._disk_cached_keys:
                self._query_cache.pop(key, None)

    def encode_text(self, text: str) -> tuple[np.ndarray | None, str]:
        """Look up normalized 768d VideoPrism text embedding (strict cache or injected encoder)."""
        raw = str(text or "")
        if not raw.strip():
            return None, "empty"

        for variant in (raw, raw.strip(), unicodedata.normalize("NFC", raw.strip())):
            q_sha = _sha256_text(variant)
            cached = self._query_cache.get(q_sha)
            if cached is not None:
                return cached, "cache_hit"

        if self.query_encoder is not None:
            try:
                nv = _normalize_vec(self.query_encoder(raw))
                if nv is not None:
                    self._query_cache[_sha256_text(raw)] = nv
                    return nv, "live_encoder"
            except Exception:
                return None, "encoder_error"

        return None, "cache_miss"

    def encode_clauses(self, clauses: Sequence[str]) -> list[tuple[int, np.ndarray]]:
        """Encode distinct non-empty clauses available in cache or live encoder."""
        encoded: list[tuple[int, np.ndarray]] = []
        seen_idx: set[str] = set()
        for idx, cl in enumerate(clauses):
            cs = str(cl or "").strip()
            if not cs or cs in seen_idx:
                continue
            seen_idx.add(cs)
            vec, _ = self.encode_text(cs)
            if vec is not None:
                encoded.append((idx, vec))
        return encoded

    def retrieve_parallel(
        self,
        query: str,
        clauses: Sequence[str] = (),
        top_n: int = 80,
        allowed_video_mask: np.ndarray | None = None,
        allowed_video_ids: set[str] | None = None,
        whole_weight: float = 0.55,
        subclip_weight: float = 0.45,
        clause_bonus_weight: float = 0.10,
    ) -> VideoPrismRetrievalResult:
        """Execute parallel whole-clip + sub-clip VideoPrism retrieval (Step D)."""
        if not self.available or self.video_vectors is None:
            return VideoPrismRetrievalResult(status=self.status, query_source="unavailable")

        clause_vecs = self.encode_clauses(clauses)
        q_vec, q_src = self.encode_text(query)
        if q_vec is None and clause_vecs:
            mean_v = np.mean(np.stack([v for _, v in clause_vecs], axis=0), axis=0)
            q_vec = _normalize_vec(mean_v)
            if q_vec is not None:
                q_src = "clause_mean_fallback"

        if q_vec is None:
            return VideoPrismRetrievalResult(status="cache_miss", query_source=q_src)

        n_vids = len(self.video_ids)
        whole_arr = (self.video_vectors @ q_vec).astype(np.float32)

        sub_max_arr = whole_arr.copy()
        best_sub_idx = np.full(n_vids, -1, dtype=np.int32)
        sub_sims: np.ndarray | None = None

        if self.subclip_vectors is not None and len(self.subclip_vectors) > 0:
            sub_sims = (self.subclip_vectors @ q_vec).astype(np.float32)
            # Vectorized segment-max aggregation over valid subclip->video mappings
            valid_mask = self.subclip_vid_indices >= 0
            if bool(np.any(valid_mask)):
                v_idx_valid = self.subclip_vid_indices[valid_mask]
                s_valid = sub_sims[valid_mask]
                orig_idx_valid = np.flatnonzero(valid_mask)
                order = np.argsort(s_valid, kind="mergesort")
                # Assign from lowest to highest similarity so highest wins per video_id
                v_ordered = v_idx_valid[order]
                s_ordered = s_valid[order]
                idx_ordered = orig_idx_valid[order]
                sub_max_arr[v_ordered] = np.maximum(sub_max_arr[v_ordered], s_ordered)
                best_sub_idx[v_ordered] = idx_ordered.astype(np.int32)

        blend = (whole_weight * whole_arr + subclip_weight * sub_max_arr).astype(np.float32)
        combined_arr = np.where(self.has_subclip_mask, np.maximum(whole_arr, blend), whole_arr).astype(np.float32)

        if len(clause_vecs) >= 2:
            c_mat = np.stack([v for _, v in clause_vecs], axis=0)  # (C, 768)
            c_whole = c_mat @ self.video_vectors.T  # (C, N_vid)
            c_best = c_whole.copy()
            if self.subclip_vectors is not None and len(self.subclip_vectors) > 0:
                valid_mask = self.subclip_vid_indices >= 0
                if bool(np.any(valid_mask)):
                    v_idx_valid = self.subclip_vid_indices[valid_mask]
                    c_sub = c_mat @ self.subclip_vectors[valid_mask].T  # (C, N_sub_valid)
                    for c_i in range(len(clause_vecs)):
                        order_c = np.argsort(c_sub[c_i], kind="mergesort")
                        v_ord = v_idx_valid[order_c]
                        c_best[c_i, v_ord] = np.maximum(c_best[c_i, v_ord], c_sub[c_i, order_c])
            # Geometric-friendly soft-min + mean balance across clauses
            clause_agg = (0.55 * np.mean(c_best, axis=0) + 0.45 * np.min(c_best, axis=0)).astype(np.float32)
            combined_arr = (combined_arr + clause_bonus_weight * clause_agg).astype(np.float32)

        active_mask = np.ones(n_vids, dtype=bool)
        if allowed_video_mask is not None and len(allowed_video_mask) == n_vids:
            active_mask &= np.asarray(allowed_video_mask, dtype=bool)
        if allowed_video_ids is not None:
            id_mask = np.array([v in allowed_video_ids for v in self.video_ids], dtype=bool)
            active_mask &= id_mask

        active_vals = combined_arr[active_mask]
        c_mean = float(np.mean(active_vals)) if active_vals.size > 0 else 0.0
        c_std = float(np.std(active_vals)) if active_vals.size > 1 else 1.0
        denom_std = max(c_std, 1e-4)

        masked_scores = np.where(active_mask, combined_arr, -np.inf)
        top_k = min(max(1, int(top_n)), int(np.sum(active_mask))) if np.any(active_mask) else 0
        if top_k <= 0:
            return VideoPrismRetrievalResult(
                status="empty_filtered",
                query_source=q_src,
                corpus_mean=c_mean,
                corpus_std=denom_std,
                clause_hit_count=len(clause_vecs),
                subclip_score_array=sub_sims,
            )

        if top_k == n_vids:
            top_indices = np.argsort(-masked_scores, kind="mergesort")
        else:
            part_idx = np.argpartition(masked_scores, -top_k)[-top_k:]
            top_indices = part_idx[np.argsort(-masked_scores[part_idx], kind="mergesort")]

        ranked_videos: list[tuple[str, float]] = []
        for idx in top_indices:
            if not np.isfinite(masked_scores[idx]):
                continue
            ranked_videos.append((self.video_ids[int(idx)], float(masked_scores[idx])))

        combined_map: Mapping[str, float]
        whole_map: Mapping[str, float]
        sub_max_map: Mapping[str, float]
        best_windows: dict[str, tuple[float, float, float]] = {}
        z_map: Mapping[str, float]

        if allowed_video_mask is None and allowed_video_ids is None:
            z_arr = ((combined_arr - c_mean) / denom_std).astype(np.float32)
            combined_map = _VideoScoreMap(self.vid_to_idx, self.video_ids, combined_arr)
            whole_map = _VideoScoreMap(self.vid_to_idx, self.video_ids, whole_arr)
            sub_max_map = _VideoScoreMap(self.vid_to_idx, self.video_ids, sub_max_arr)
            z_map = _VideoScoreMap(self.vid_to_idx, self.video_ids, z_arr)
            if sub_sims is not None:
                for idx in np.flatnonzero(best_sub_idx >= 0).tolist():
                    b_idx = int(best_sub_idx[idx])
                    best_windows[self.video_ids[idx]] = (
                        float(self.subclip_starts[b_idx]),
                        float(self.subclip_ends[b_idx]),
                        float(sub_sims[b_idx]),
                    )
        else:
            combined_dict: dict[str, float] = {}
            whole_dict: dict[str, float] = {}
            sub_max_dict: dict[str, float] = {}
            z_dict: dict[str, float] = {}
            active_indices = np.flatnonzero(active_mask)
            for idx in active_indices.tolist():
                vid = self.video_ids[idx]
                comb = float(combined_arr[idx])
                combined_dict[vid] = comb
                whole_dict[vid] = float(whole_arr[idx])
                sub_max_dict[vid] = float(sub_max_arr[idx])
                z_dict[vid] = (comb - c_mean) / denom_std
                b_idx = int(best_sub_idx[idx])
                if b_idx >= 0 and sub_sims is not None:
                    best_windows[vid] = (
                        float(self.subclip_starts[b_idx]),
                        float(self.subclip_ends[b_idx]),
                        float(sub_sims[b_idx]),
                    )
            combined_map = combined_dict
            whole_map = whole_dict
            sub_max_map = sub_max_dict
            z_map = z_dict

        return VideoPrismRetrievalResult(
            status="ok",
            query_source=q_src,
            ranked_videos=ranked_videos,
            combined_scores=combined_map,
            whole_scores=whole_map,
            subclip_max_scores=sub_max_map,
            best_subclip_windows=best_windows,
            z_scores=z_map,
            corpus_mean=c_mean,
            corpus_std=denom_std,
            clause_hit_count=len(clause_vecs),
            subclip_score_array=sub_sims,
        )

    def rerank_top_candidates(
        self,
        query: str,
        clauses: Sequence[str],
        candidate_video_ids: Sequence[str],
        retrieval_result: VideoPrismRetrievalResult | None = None,
    ) -> dict[str, VideoPrismCandidateSignals]:
        """Compute Step H Top-20 VideoPrism reranking & feature signals for candidates."""
        if not self.available or self.video_vectors is None or not candidate_video_ids:
            return {}

        if retrieval_result is None or retrieval_result.status != "ok":
            retrieval_result = self.retrieve_parallel(
                query=query,
                clauses=clauses,
                top_n=max(20, len(candidate_video_ids)),
            )
        if retrieval_result.status != "ok":
            return {}

        q_vec, _ = self.encode_text(query)
        clause_vecs = self.encode_clauses(clauses)
        if q_vec is None and clause_vecs:
            q_vec = _normalize_vec(np.mean(np.stack([v for _, v in clause_vecs], axis=0), axis=0))
        if q_vec is None:
            return {}

        sub_sims_all = retrieval_result.subclip_score_array
        if sub_sims_all is None and self.subclip_vectors is not None and len(self.subclip_vectors) > 0:
            sub_sims_all = (self.subclip_vectors @ q_vec).astype(np.float32)

        c_mat = np.stack([v for _, v in clause_vecs], axis=0) if len(clause_vecs) >= 2 else None
        out: dict[str, VideoPrismCandidateSignals] = {}
        raw_rerank_vals: list[float] = []
        ordered_vids: list[str] = []

        for vid in candidate_video_ids:
            idx = self.vid_to_idx.get(vid)
            if idx is None:
                continue
            whole_cos = float(self.video_vectors[idx] @ q_vec)
            hub_val = float(self.gallery_hub_k10[idx]) if self.gallery_hub_k10 is not None else 0.0
            csls = float(2.0 * whole_cos - hub_val)

            sub_pos = self.video_subclip_positions.get(vid)
            has_sub = sub_pos is not None and len(sub_pos) > 0 and sub_sims_all is not None
            if has_sub and sub_pos is not None and sub_sims_all is not None:
                s_vals = sub_sims_all[sub_pos]
                best_local = int(np.argmax(s_vals))
                sub_max = float(s_vals[best_local])
                k_top = min(2, len(s_vals))
                sub_topk = float(np.mean(np.sort(s_vals)[-k_top:]))
                best_global = int(sub_pos[best_local])
                win_start: float | None = float(self.subclip_starts[best_global])
                win_end: float | None = float(self.subclip_ends[best_global])
            else:
                sub_max = whole_cos
                sub_topk = whole_cos
                win_start = None
                win_end = None

            clause_min = whole_cos
            clause_order = 0.0
            if c_mat is not None:
                c_w = (c_mat @ self.video_vectors[idx]).astype(np.float32)
                if has_sub and sub_pos is not None and self.subclip_vectors is not None:
                    c_s = (c_mat @ self.subclip_vectors[sub_pos].T).astype(np.float32)  # (C, W)
                    c_vid_best = np.maximum(c_w, np.max(c_s, axis=1))
                    clause_min = float(np.min(c_vid_best))
                    # Verify chronological alignment across subclips (first half vs later clauses)
                    if len(sub_pos) >= 2:
                        peaks = np.argmax(c_s, axis=1)
                        in_order = 0
                        pairs = 0
                        for i in range(len(peaks) - 1):
                            for j in range(i + 1, len(peaks)):
                                pairs += 1
                                if peaks[i] <= peaks[j]:
                                    in_order += 1
                        clause_order = (in_order / pairs) if pairs > 0 else 0.0
                else:
                    clause_min = float(np.min(c_w))

            rerank_raw = (
                0.48 * whole_cos
                + 0.34 * sub_max
                + 0.14 * clause_min
                + 0.08 * csls
                + 0.06 * clause_order
            )
            vp_z = (
                (max(whole_cos, sub_max) - retrieval_result.corpus_mean) / retrieval_result.corpus_std
                if retrieval_result.corpus_std > 1e-6
                else 0.0
            )
            out[vid] = VideoPrismCandidateSignals(
                whole_cos=whole_cos,
                subclip_max_cos=sub_max,
                subclip_topk_mean=sub_topk,
                rerank_score=float(rerank_raw),
                vp_z=float(vp_z),
                csls_score=csls,
                clause_min_score=clause_min,
                clause_order_score=clause_order,
                best_window_start=win_start,
                best_window_end=win_end,
                has_subclips=bool(has_sub),
            )
            raw_rerank_vals.append(float(rerank_raw))
            ordered_vids.append(vid)

        # Normalize pool-level vp_z to combine corpus calibration and Top-M discrimination cleanly
        if len(raw_rerank_vals) > 1:
            arr = np.asarray(raw_rerank_vals, dtype=np.float32)
            pool_std = float(np.std(arr))
            if pool_std > 1e-4:
                pool_mean = float(np.mean(arr))
                for vid, val in zip(ordered_vids, raw_rerank_vals):
                    out[vid].vp_z = 0.5 * out[vid].vp_z + 0.5 * ((val - pool_mean) / pool_std)

        return out

    def get_subclip_curve(
        self,
        video_id: str,
        query: str,
        clauses: Sequence[str] = (),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Return (center_times, subclip_scores, clause_subclip_scores_or_None) for Step J."""
        if not self.available or self.subclip_vectors is None:
            return None
        sub_pos = self.video_subclip_positions.get(video_id)
        if sub_pos is None or len(sub_pos) == 0:
            return None
        q_vec, _ = self.encode_text(query)
        clause_vecs = self.encode_clauses(clauses)
        if q_vec is None and clause_vecs:
            q_vec = _normalize_vec(np.mean(np.stack([v for _, v in clause_vecs], axis=0), axis=0))
        if q_vec is None:
            return None

        sub_vecs = self.subclip_vectors[sub_pos]
        centers = 0.5 * (self.subclip_starts[sub_pos] + self.subclip_ends[sub_pos])
        scores = (sub_vecs @ q_vec).astype(np.float32)
        clause_matrix: np.ndarray | None = None
        if len(clause_vecs) >= 2:
            c_mat = np.stack([v for _, v in clause_vecs], axis=0)
            clause_matrix = (c_mat @ sub_vecs.T).astype(np.float32)
        return centers.astype(np.float32), scores, clause_matrix
