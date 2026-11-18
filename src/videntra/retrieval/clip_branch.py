"""SigLIP 2 frame + clip mean retrieval branch and clip reserve pool helpers.

Implements Section 2.1 and Section 2.2 of VIDENTRA_CLIP_RETRIEVAL_VIDEOPRISM_KAGGLE_RUNBOOK.md:
- L2-normalized frame bank and per-video clip mean matrix built once per snapshot.
- Dual-prompt query vector validation/construction (0.75 * raw + 0.25 * frame prompt).
- Stable 1-based descending ranks with deterministic `video_id` tie-breaking.
- Reciprocal Rank Fusion (`1/(60 + rank_frame) + 1/(60 + rank_clip)`).
- Optional presentation-order sliding window (e.g. 3-keyframe) mean+max scoring.
- Clip reserve pool policy (`K >= 2`) preserving `a5_pool[:K-1]` and filling the tail slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MIN_NORM: float = 1e-8
DEFAULT_RRF_K: float = 60.0
CLIP_BRANCH_SCHEMA_VERSION: str = "clip_branch_v2"


class SnapshotMismatchError(ValueError):
    """Raised when a cached ClipBranchIndex does not match the active snapshot."""


@dataclass(frozen=True)
class ClipBranchScoreResult:
    """Per-query scoring output of the SigLIP 2 frame + clip branch."""

    branch_order: list[str]
    branch_scores: dict[str, float]
    frame_scores: dict[str, float]
    clip_scores: dict[str, float]
    rank_frame: dict[str, int]
    rank_clip: dict[str, int]
    best_frame_fid: dict[str, int]
    best_frame_pts: dict[str, float]
    window_scores: dict[str, float] = field(default_factory=dict)
    valid_gallery_count: int = 0
    invalid_videos_skipped: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipReserveResult:
    """Result and audit metadata of applying the clip reserve policy (K >= 2)."""

    clip_pool: list[str]
    a5_pool: list[str]
    protected: list[str]
    added_video: str | None
    displaced_video: str | None
    duplicate_skipped: int
    branch_rank: int | None
    fallback_reason: str | None
    pool_size: int
    requested_k: int
    valid_gallery_count: int | None = None


def l2_normalize_vector(vec: np.ndarray, *, min_norm: float = DEFAULT_MIN_NORM, name: str = "vector") -> np.ndarray:
    """Validate finiteness/norm and return an L2-normalized 1D float32 vector."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= min_norm:
        raise ValueError(f"{name} has near-zero norm ({norm})")
    return (arr / norm).astype(np.float32, copy=False)


def build_dual_prompt_query_vector(
    original_vec: np.ndarray,
    frame_prompt_vec: np.ndarray,
    *,
    min_norm: float = DEFAULT_MIN_NORM,
) -> np.ndarray:
    """Build `L2_normalize(0.75 * encode(original_query) + 0.25 * encode(frame_prompt))`."""
    v1 = l2_normalize_vector(original_vec, min_norm=min_norm, name="original_query")
    v2 = l2_normalize_vector(frame_prompt_vec, min_norm=min_norm, name="frame_prompt")
    if v1.shape != v2.shape:
        raise ValueError(f"Query vector dimension mismatch: {v1.shape} vs {v2.shape}")
    blended = 0.75 * v1 + 0.25 * v2
    return l2_normalize_vector(blended, min_norm=min_norm, name="dual_prompt_query")


def compute_embedding_signature(all_embeddings: np.ndarray) -> str:
    """Compute a full deterministic SHA-256 checksum over the entire embedding matrix."""
    emb = np.asarray(all_embeddings)
    h = hashlib.sha256()
    h.update(f"shape={emb.shape};dtype={emb.dtype};".encode("utf-8"))
    if emb.size > 0:
        h.update(np.ascontiguousarray(emb, dtype=np.float32).tobytes())
    return h.hexdigest()


def compute_fids_signature(video_kf_fids: Mapping[str, Sequence[int]]) -> str:
    """Compute deterministic SHA-256 over video_id -> presentation-ordered faiss_id sequences."""
    h = hashlib.sha256()
    for vid in sorted(video_kf_fids.keys()):
        fids = video_kf_fids[vid]
        h.update(f"{vid}:{','.join(str(int(x)) for x in fids)};".encode("utf-8"))
    return h.hexdigest()


def compute_mapping_signature(
    per_video_frames: Mapping[str, Sequence[tuple[int, float, int]]],
) -> str:
    """Compute deterministic SHA-256 over video_id -> (ordinal, faiss_id, pts_time) mappings."""
    h = hashlib.sha256()
    for vid in sorted(per_video_frames.keys()):
        items = per_video_frames[vid]
        parts = [f"{int(ord_idx)}:{int(fid)}:{float(pts):.6f}" for fid, pts, ord_idx in items]
        h.update(f"{vid}:{'|'.join(parts)};".encode("utf-8"))
    return h.hexdigest()


def compute_snapshot_signature(
    all_embeddings: np.ndarray,
    video_kf_fids: Mapping[str, Sequence[int]] | None = None,
    *,
    video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    keyframe_map: Mapping[int, Mapping[str, Any]] | None = None,
    schema_version: str = CLIP_BRANCH_SCHEMA_VERSION,
) -> str:
    """Compute a complete deterministic signature covering vectors, fids, ordinals, and PTS."""
    emb_sig = compute_embedding_signature(all_embeddings)
    if video_keyframes is not None or keyframe_map is not None:
        ordered = _extract_ordered_video_frames_with_ordinals(video_keyframes, video_kf_fids, keyframe_map)
        fids_sig = compute_fids_signature({vid: [fid for fid, _, _ in pairs] for vid, pairs in ordered.items()})
        map_sig = compute_mapping_signature(ordered)
    elif video_kf_fids is not None:
        fids_sig = compute_fids_signature(video_kf_fids)
        map_sig = "fids_only"
    else:
        fids_sig = "no_mapping"
        map_sig = "no_mapping"
    payload = f"{schema_version}|{emb_sig}|{fids_sig}|{map_sig}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ClipBranchIndex:
    """Cached L2-normalized frame bank and per-video clip mean index for one snapshot."""

    dim: int
    num_source_rows: int
    snapshot_signature: str
    video_ids: list[str]
    video_to_pos: dict[str, int]
    clip_mean_matrix: np.ndarray
    frame_vectors_norm: np.ndarray
    frame_video_pos: np.ndarray
    frame_fids: np.ndarray
    frame_pts: np.ndarray
    video_frame_ranges: dict[str, tuple[int, int]]
    invalid_videos: dict[str, str]
    embedding_signature: str = ""
    fids_signature: str = ""
    mapping_signature: str = ""
    schema_version: str = CLIP_BRANCH_SCHEMA_VERSION
    video_starts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    video_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    video_ids_sorted: bool = False

    def validate_snapshot(
        self,
        all_embeddings: np.ndarray,
        video_kf_fids: Mapping[str, Sequence[int]] | None = None,
        *,
        video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        keyframe_map: Mapping[int, Mapping[str, Any]] | None = None,
        expected_signature: str | None = None,
    ) -> None:
        """Raise SnapshotMismatchError if `all_embeddings`, fids, ordinals, or PTS differ from this index."""
        emb = np.asarray(all_embeddings)
        if emb.ndim != 2 or emb.shape[1] != self.dim or emb.shape[0] != self.num_source_rows:
            raise SnapshotMismatchError(
                f"Snapshot shape mismatch: expected ({self.num_source_rows}, {self.dim}), got {emb.shape}"
            )
        if expected_signature is not None and expected_signature != self.snapshot_signature:
            raise SnapshotMismatchError(
                f"Snapshot signature mismatch: expected {self.snapshot_signature}, got {expected_signature}"
            )
        emb_sig = compute_embedding_signature(emb)
        if self.embedding_signature and emb_sig != self.embedding_signature:
            raise SnapshotMismatchError(
                f"Snapshot vector content mismatch: expected {self.embedding_signature}, got {emb_sig}"
            )
        if video_keyframes is not None or keyframe_map is not None:
            ordered = _extract_ordered_video_frames_with_ordinals(video_keyframes, video_kf_fids, keyframe_map)
            map_sig = compute_mapping_signature(ordered)
            if self.mapping_signature and map_sig != self.mapping_signature:
                raise SnapshotMismatchError(
                    f"Snapshot mapping/PTS mismatch: expected {self.mapping_signature}, got {map_sig}"
                )
        elif video_kf_fids is not None:
            fids_sig = compute_fids_signature(video_kf_fids)
            if self.fids_signature and fids_sig != self.fids_signature:
                raise SnapshotMismatchError(
                    f"Snapshot faiss_id mapping mismatch: expected {self.fids_signature}, got {fids_sig}"
                )

    def matches(
        self,
        all_embeddings: np.ndarray,
        video_kf_fids: Mapping[str, Sequence[int]] | None = None,
        *,
        video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        keyframe_map: Mapping[int, Mapping[str, Any]] | None = None,
        expected_signature: str | None = None,
    ) -> bool:
        """Return True iff `all_embeddings` and mappings/PTS match this cached index."""
        try:
            self.validate_snapshot(
                all_embeddings,
                video_kf_fids,
                video_keyframes=video_keyframes,
                keyframe_map=keyframe_map,
                expected_signature=expected_signature,
            )
            return True
        except SnapshotMismatchError:
            return False


def _extract_ordered_video_frames_with_ordinals(
    video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    video_kf_fids: Mapping[str, Sequence[int]] | None,
    keyframe_map: Mapping[int, Mapping[str, Any]] | None,
) -> dict[str, list[tuple[int, float, int]]]:
    """Resolve per-video presentation-order `(faiss_id, pts_time, presentation_ordinal)` tuples."""
    per_video: dict[str, list[tuple[int, float, int]]] = {}

    if video_keyframes:
        for vid in sorted(video_keyframes.keys()):
            kfs = video_keyframes[vid]
            triples: list[tuple[int, float, int]] = []
            for pos, kf in enumerate(kfs):
                if "faiss_id" not in kf:
                    continue
                fid = int(kf["faiss_id"])
                pts = float(kf.get("pts_time", 0.0))
                raw_ord = kf.get("presentation_ordinal", kf.get("original_frame_index", pos))
                ord_idx = int(raw_ord) if raw_ord is not None else pos
                triples.append((fid, pts, ord_idx))
            per_video[vid] = triples
        return per_video

    if video_kf_fids:
        for vid in sorted(video_kf_fids.keys()):
            triples = []
            for pos, fid_raw in enumerate(video_kf_fids[vid]):
                fid = int(fid_raw)
                pts = 0.0
                ord_idx = pos
                if keyframe_map and fid in keyframe_map:
                    pts = float(keyframe_map[fid].get("pts_time", 0.0))
                    raw_ord = keyframe_map[fid].get("presentation_ordinal", keyframe_map[fid].get("original_frame_index", pos))
                    ord_idx = int(raw_ord) if raw_ord is not None else pos
                triples.append((fid, pts, ord_idx))
            per_video[vid] = triples
        return per_video

    if keyframe_map:
        grouped: dict[str, list[tuple[int, float, int]]] = {}
        for fid_raw, info in keyframe_map.items():
            vid = info.get("video_id")
            if not vid:
                continue
            fid = int(fid_raw)
            raw_ord = info.get("presentation_ordinal", info.get("original_frame_index", fid))
            orig_idx = int(raw_ord) if raw_ord is not None else fid
            pts = float(info.get("pts_time", 0.0))
            grouped.setdefault(str(vid), []).append((orig_idx, pts, fid))
        for vid in sorted(grouped.keys()):
            grouped[vid].sort(key=lambda item: (item[0], item[1], item[2]))
            per_video[vid] = [(fid, pts, orig_idx) for orig_idx, pts, fid in grouped[vid]]

    return per_video


def _extract_ordered_video_frames(
    video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    video_kf_fids: Mapping[str, Sequence[int]] | None,
    keyframe_map: Mapping[int, Mapping[str, Any]] | None,
) -> dict[str, list[tuple[int, float]]]:
    """Resolve per-video presentation-order `(faiss_id, pts_time)` pairs without label leakage."""
    ordered = _extract_ordered_video_frames_with_ordinals(video_keyframes, video_kf_fids, keyframe_map)
    return {vid: [(fid, pts) for fid, pts, _ in triples] for vid, triples in ordered.items()}


def build_clip_branch_index(
    all_embeddings: np.ndarray,
    *,
    video_keyframes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    video_kf_fids: Mapping[str, Sequence[int]] | None = None,
    keyframe_map: Mapping[int, Mapping[str, Any]] | None = None,
    min_norm: float = DEFAULT_MIN_NORM,
    snapshot_signature: str | None = None,
) -> ClipBranchIndex:
    """Build L2-normalized frame bank and `clip_mean[v]` matrix once for a snapshot."""
    emb = np.asarray(all_embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] == 0:
        raise ValueError(f"all_embeddings must have shape (N, D) with D > 0, got {emb.shape}")
    n_rows, dim = emb.shape

    per_video_ordered = _extract_ordered_video_frames_with_ordinals(video_keyframes, video_kf_fids, keyframe_map)
    resolved_fids_map: dict[str, list[int]] = {
        vid: [fid for fid, _, _ in pairs] for vid, pairs in per_video_ordered.items()
    }
    emb_sig = compute_embedding_signature(emb)
    fids_sig = compute_fids_signature(resolved_fids_map)
    map_sig = compute_mapping_signature(per_video_ordered)
    sig = snapshot_signature or compute_snapshot_signature(
        emb,
        video_kf_fids=resolved_fids_map,
        video_keyframes=video_keyframes,
        keyframe_map=keyframe_map,
    )
    per_video_raw: dict[str, list[tuple[int, float]]] = {
        vid: [(fid, pts) for fid, pts, _ in pairs] for vid, pairs in per_video_ordered.items()
    }

    video_ids: list[str] = []
    video_to_pos: dict[str, int] = {}
    clip_means: list[np.ndarray] = []
    frame_vecs_list: list[np.ndarray] = []
    frame_vpos_list: list[int] = []
    frame_fids_list: list[int] = []
    frame_pts_list: list[float] = []
    video_frame_ranges: dict[str, tuple[int, int]] = {}
    video_starts_list: list[int] = []
    video_lengths_list: list[int] = []
    invalid_videos: dict[str, str] = {}

    frame_offset = 0
    for vid in sorted(per_video_raw.keys()):
        pairs = per_video_raw[vid]
        if not pairs:
            invalid_videos[vid] = "empty_frames"
            continue

        valid_vecs: list[np.ndarray] = []
        valid_fids: list[int] = []
        valid_pts: list[float] = []
        for fid, pts in pairs:
            if fid < 0 or fid >= n_rows:
                continue
            vec = emb[fid]
            if vec.shape != (dim,) or not np.isfinite(vec).all():
                continue
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= min_norm:
                continue
            valid_vecs.append((vec / norm).astype(np.float32, copy=False))
            valid_fids.append(fid)
            valid_pts.append(pts)

        if not valid_vecs:
            invalid_videos[vid] = "no_valid_finite_frames"
            continue

        mat_v = np.vstack(valid_vecs).astype(np.float32, copy=False)
        mean_v = np.mean(mat_v, axis=0)
        if not np.isfinite(mean_v).all():
            invalid_videos[vid] = "non_finite_clip_mean"
            continue
        mean_norm = float(np.linalg.norm(mean_v))
        if not np.isfinite(mean_norm) or mean_norm <= min_norm:
            invalid_videos[vid] = "low_norm_clip_mean"
            continue

        vpos = len(video_ids)
        video_ids.append(vid)
        video_to_pos[vid] = vpos
        clip_means.append((mean_v / mean_norm).astype(np.float32, copy=False))

        n_v = len(valid_vecs)
        frame_vecs_list.append(mat_v)
        frame_vpos_list.extend([vpos] * n_v)
        frame_fids_list.extend(valid_fids)
        frame_pts_list.extend(valid_pts)
        video_frame_ranges[vid] = (frame_offset, frame_offset + n_v)
        video_starts_list.append(frame_offset)
        video_lengths_list.append(n_v)
        frame_offset += n_v

    if clip_means:
        clip_mean_matrix = np.ascontiguousarray(np.vstack(clip_means), dtype=np.float32)
        frame_vectors_norm = np.ascontiguousarray(np.vstack(frame_vecs_list), dtype=np.float32)
    else:
        clip_mean_matrix = np.zeros((0, dim), dtype=np.float32)
        frame_vectors_norm = np.zeros((0, dim), dtype=np.float32)

    if invalid_videos:
        logger.debug("ClipBranchIndex skipped %d invalid videos: %s", len(invalid_videos), invalid_videos)

    return ClipBranchIndex(
        dim=dim,
        num_source_rows=n_rows,
        snapshot_signature=sig,
        video_ids=video_ids,
        video_to_pos=video_to_pos,
        clip_mean_matrix=clip_mean_matrix,
        frame_vectors_norm=frame_vectors_norm,
        frame_video_pos=np.asarray(frame_vpos_list, dtype=np.int64),
        frame_fids=np.asarray(frame_fids_list, dtype=np.int64),
        frame_pts=np.asarray(frame_pts_list, dtype=np.float64),
        video_frame_ranges=video_frame_ranges,
        invalid_videos=invalid_videos,
        embedding_signature=emb_sig,
        fids_signature=fids_sig,
        mapping_signature=map_sig,
        schema_version=CLIP_BRANCH_SCHEMA_VERSION,
        video_starts=np.asarray(video_starts_list, dtype=np.int64),
        video_lengths=np.asarray(video_lengths_list, dtype=np.int64),
        video_ids_sorted=True,
    )


def score_presentation_window_max(
    frame_vectors_norm: np.ndarray,
    query_vec_norm: np.ndarray,
    *,
    window_size: int = 3,
    min_norm: float = DEFAULT_MIN_NORM,
) -> tuple[float, int]:
    """Compute max cosine similarity over L2-normalized presentation-order sliding windows.

    Returns `(best_window_score, best_anchor_offset)` where `best_anchor_offset` is the
    0-based index within `frame_vectors_norm` of the highest-scoring frame inside the
    winning window (used for timestamp anchoring without any PTS label leakage).
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    n_frames = frame_vectors_norm.shape[0]
    if n_frames == 0:
        raise ValueError("frame_vectors_norm must be non-empty")

    frame_sims = frame_vectors_norm @ query_vec_norm
    if n_frames <= window_size:
        win_mean = np.mean(frame_vectors_norm, axis=0)
        norm = float(np.linalg.norm(win_mean))
        if not np.isfinite(norm) or norm <= min_norm:
            best_pos = int(np.argmax(frame_sims))
            return float(frame_sims[best_pos]), best_pos
        score = float(np.dot(win_mean / norm, query_vec_norm))
        best_pos = int(np.argmax(frame_sims))
        return score, best_pos

    # Sliding window mean vectors via prefix sum for O(L * D) without Python inner loops.
    cumsum = np.zeros((n_frames + 1, frame_vectors_norm.shape[1]), dtype=np.float32)
    np.cumsum(frame_vectors_norm, axis=0, out=cumsum[1:])
    win_sums = cumsum[window_size:] - cumsum[:-window_size]
    win_norms = np.linalg.norm(win_sums, axis=1)
    valid_mask = np.isfinite(win_norms) & (win_norms > min_norm)
    if not np.any(valid_mask):
        best_pos = int(np.argmax(frame_sims))
        return float(frame_sims[best_pos]), best_pos

    win_scores = np.full(win_sums.shape[0], -np.inf, dtype=np.float64)
    raw_dots = (win_sums[valid_mask] @ query_vec_norm).astype(np.float64)
    win_scores[valid_mask] = raw_dots / win_norms[valid_mask].astype(np.float64)

    best_win_start = int(np.argmax(win_scores))
    best_score = float(win_scores[best_win_start])
    local_best = int(np.argmax(frame_sims[best_win_start : best_win_start + window_size]))
    return best_score, best_win_start + local_best


def _dense_descending_ranks(
    scores: Mapping[str, float],
) -> dict[str, int]:
    """Assign 1-based descending ranks with deterministic ascending `video_id` tie-break."""
    ordered = sorted(scores.keys(), key=lambda vid: (-float(scores[vid]), str(vid)))
    return {vid: idx for idx, vid in enumerate(ordered, start=1)}


def score_clip_branch(
    index: ClipBranchIndex,
    query_vec: np.ndarray,
    *,
    valid_video_ids: set[str] | Sequence[str] | None = None,
    window_size: int | None = None,
    rrf_k: float = DEFAULT_RRF_K,
    min_norm: float = DEFAULT_MIN_NORM,
    precomputed_frame_dots: np.ndarray | None = None,
    consensus_only_a5_vid: str | None = None,
) -> ClipBranchScoreResult:
    """Score the filtered valid gallery using frame max, clip mean, and RRF fusion."""
    if rrf_k <= 0.0:
        raise ValueError("rrf_k must be > 0")
    if window_size is not None and window_size < 1:
        raise ValueError("window_size must be >= 1 when provided")

    q = l2_normalize_vector(query_vec, min_norm=min_norm, name="query_vec")
    if q.shape[0] != index.dim:
        raise ValueError(f"Query dim {q.shape[0]} does not match index dim {index.dim}")

    if valid_video_ids is None:
        active_positions = list(range(len(index.video_ids)))
    else:
        valid_set = set(valid_video_ids)
        active_positions = [
            pos for pos, vid in enumerate(index.video_ids) if vid in valid_set
        ]

    if not active_positions or len(index.frame_vectors_norm) == 0:
        return ClipBranchScoreResult(
            branch_order=[],
            branch_scores={},
            frame_scores={},
            clip_scores={},
            rank_frame={},
            rank_clip={},
            best_frame_fid={},
            best_frame_pts={},
            window_scores={},
            valid_gallery_count=0,
            invalid_videos_skipped=dict(index.invalid_videos),
        )

    # Compute (or reuse) dot products across the normalized frame bank and clip mean matrix.
    if (
        precomputed_frame_dots is not None
        and len(precomputed_frame_dots) >= index.num_source_rows
    ):
        all_frame_dots = np.asarray(precomputed_frame_dots, dtype=np.float64)[index.frame_fids]
        if not np.isfinite(all_frame_dots).all():
            all_frame_dots = (index.frame_vectors_norm @ q).astype(np.float64, copy=False)
    else:
        all_frame_dots = (index.frame_vectors_norm @ q).astype(np.float64, copy=False)
    all_clip_dots = (index.clip_mean_matrix @ q).astype(np.float64, copy=False)

    n_all_videos = len(index.video_ids)
    can_vectorize = (
        window_size is None
        and index.video_ids_sorted
        and len(index.video_starts) == n_all_videos
        and len(index.video_lengths) == n_all_videos
        and n_all_videos > 0
        and np.all(np.isfinite(all_frame_dots))
    )

    if can_vectorize:
        max_all = np.maximum.reduceat(all_frame_dots, index.video_starts)
        max_expanded = np.repeat(max_all, index.video_lengths)
        eq_mask = all_frame_dots >= max_expanded
        eq_idx = np.flatnonzero(eq_mask)
        eq_vpos = index.frame_video_pos[eq_idx]
        first_mask = np.empty(eq_vpos.size, dtype=bool)
        first_mask[0] = True
        if eq_vpos.size > 1:
            first_mask[1:] = eq_vpos[1:] != eq_vpos[:-1]
        best_globals_all = eq_idx[first_mask]
        if len(best_globals_all) == n_all_videos:
            if valid_video_ids is None or len(active_positions) == n_all_videos:
                active_pos_arr = np.arange(n_all_videos, dtype=np.int64)
                active_vids = index.video_ids
            else:
                active_pos_arr = np.asarray(active_positions, dtype=np.int64)
                active_vids = [index.video_ids[i] for i in active_positions]

            f_arr = max_all[active_pos_arr]
            c_arr = all_clip_dots[active_pos_arr]
            g_arr = best_globals_all[active_pos_arr]

            n_act = len(active_vids)
            order_f = np.argsort(-f_arr, kind="stable")
            ranks_f = np.empty(n_act, dtype=np.int64)
            ranks_f[order_f] = np.arange(1, n_act + 1, dtype=np.int64)

            order_c = np.argsort(-c_arr, kind="stable")
            ranks_c = np.empty(n_act, dtype=np.int64)
            ranks_c[order_c] = np.arange(1, n_act + 1, dtype=np.int64)

            rrf_k_f = float(rrf_k)
            b_arr = (1.0 / (rrf_k_f + ranks_f)) + (1.0 / (rrf_k_f + ranks_c))
            if consensus_only_a5_vid is not None and valid_video_ids is None:
                b_pos = int(np.argmax(b_arr))
                b_v1 = active_vids[b_pos]
                keep_positions = [b_pos]
                a5_pos = index.video_to_pos.get(consensus_only_a5_vid)
                if a5_pos is not None and a5_pos != b_pos:
                    keep_positions.append(a5_pos)
                return ClipBranchScoreResult(
                    branch_order=[b_v1],
                    branch_scores={active_vids[p]: float(b_arr[p]) for p in keep_positions},
                    frame_scores={active_vids[p]: float(f_arr[p]) for p in keep_positions},
                    clip_scores={active_vids[p]: float(c_arr[p]) for p in keep_positions},
                    rank_frame={active_vids[p]: int(ranks_f[p]) for p in keep_positions},
                    rank_clip={active_vids[p]: int(ranks_c[p]) for p in keep_positions},
                    best_frame_fid={},
                    best_frame_pts={},
                    window_scores={},
                    valid_gallery_count=n_act,
                    invalid_videos_skipped={},
                )

            order_b = np.argsort(-b_arr, kind="stable")

            return ClipBranchScoreResult(
                branch_order=[active_vids[i] for i in order_b.tolist()],
                branch_scores=dict(zip(active_vids, b_arr.tolist())),
                frame_scores=dict(zip(active_vids, f_arr.tolist())),
                clip_scores=dict(zip(active_vids, c_arr.tolist())),
                rank_frame=dict(zip(active_vids, ranks_f.tolist())),
                rank_clip=dict(zip(active_vids, ranks_c.tolist())),
                best_frame_fid=dict(zip(active_vids, index.frame_fids[g_arr].tolist())),
                best_frame_pts=dict(zip(active_vids, index.frame_pts[g_arr].tolist())),
                window_scores={},
                valid_gallery_count=n_act,
                invalid_videos_skipped=dict(index.invalid_videos),
            )

    frame_scores: dict[str, float] = {}
    clip_scores: dict[str, float] = {}
    window_scores: dict[str, float] = {}
    best_frame_fid: dict[str, int] = {}
    best_frame_pts: dict[str, float] = {}

    for pos in active_positions:
        vid = index.video_ids[pos]
        start_i, end_i = index.video_frame_ranges[vid]
        vid_dots = all_frame_dots[start_i:end_i]
        best_local = int(np.argmax(vid_dots))
        best_global = start_i + best_local

        frame_scores[vid] = float(vid_dots[best_local])
        clip_scores[vid] = float(all_clip_dots[pos])
        best_frame_fid[vid] = int(index.frame_fids[best_global])
        best_frame_pts[vid] = float(index.frame_pts[best_global])

        if window_size is not None:
            w_score, w_anchor_local = score_presentation_window_max(
                index.frame_vectors_norm[start_i:end_i],
                q,
                window_size=window_size,
                min_norm=min_norm,
            )
            window_scores[vid] = w_score
            w_global = start_i + w_anchor_local
            best_frame_fid[vid] = int(index.frame_fids[w_global])
            best_frame_pts[vid] = float(index.frame_pts[w_global])

    rank_frame = _dense_descending_ranks(frame_scores)
    rank_clip = _dense_descending_ranks(clip_scores)

    branch_scores = {
        vid: (1.0 / (rrf_k + rank_frame[vid])) + (1.0 / (rrf_k + rank_clip[vid]))
        for vid in frame_scores
    }
    branch_order = sorted(
        branch_scores.keys(),
        key=lambda vid: (-branch_scores[vid], str(vid)),
    )

    return ClipBranchScoreResult(
        branch_order=branch_order,
        branch_scores=branch_scores,
        frame_scores=frame_scores,
        clip_scores=clip_scores,
        rank_frame=rank_frame,
        rank_clip=rank_clip,
        best_frame_fid=best_frame_fid,
        best_frame_pts=best_frame_pts,
        window_scores=window_scores,
        valid_gallery_count=len(active_positions),
        invalid_videos_skipped=dict(index.invalid_videos),
    )


def _dedupe_and_filter(
    video_ids: Sequence[str],
    valid_gallery: set[str] | None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for vid in video_ids:
        if valid_gallery is not None and vid not in valid_gallery:
            continue
        if vid in seen:
            continue
        seen.add(vid)
        out.append(vid)
    return out


def build_clip_reserve_pool(
    a5_pool: Sequence[str],
    branch_order: Sequence[str],
    *,
    k: int,
    valid_gallery: set[str] | Sequence[str] | None = None,
) -> ClipReserveResult:
    """Apply clip reserve policy (K >= 2): protect `a5_pool[:K-1]` and fill tail from branch."""
    if k < 2:
        raise ValueError(f"clip reserve policy requires k >= 2, got {k}")

    valid_set = set(valid_gallery) if valid_gallery is not None else None
    a5_filtered = _dedupe_and_filter(a5_pool, valid_set)
    branch_filtered = _dedupe_and_filter(branch_order, valid_set)

    a5_k = a5_filtered[:k]
    protected = a5_k[: k - 1]
    protected_set = set(protected)

    tail: str | None = None
    tail_branch_rank: int | None = None
    duplicate_skipped = 0
    for rank_idx, vid in enumerate(branch_filtered, start=1):
        if vid in protected_set:
            duplicate_skipped += 1
            continue
        tail = vid
        tail_branch_rank = rank_idx
        break

    if tail is not None:
        clip_pool = protected + [tail]
        if len(a5_k) >= k and tail == a5_k[k - 1]:
            added_video = None
            displaced_video = None
            fallback_reason = "tail_matches_a5_kth"
        else:
            added_video = tail
            displaced_video = a5_k[k - 1] if len(a5_k) >= k else None
            fallback_reason = None
    else:
        clip_pool = list(a5_k)
        added_video = None
        displaced_video = None
        fallback_reason = "no_eligible_tail_candidate"

    logger.debug(
        "ClipReservePool(k=%d, pool_size=%d, added=%s, displaced=%s, dup_skipped=%d, branch_rank=%s, fallback=%s, valid_gallery=%s)",
        k,
        len(clip_pool),
        added_video,
        displaced_video,
        duplicate_skipped,
        tail_branch_rank,
        fallback_reason,
        len(valid_set) if valid_set is not None else None,
    )

    return ClipReserveResult(
        clip_pool=clip_pool,
        a5_pool=a5_k,
        protected=protected,
        added_video=added_video,
        displaced_video=displaced_video,
        duplicate_skipped=duplicate_skipped,
        branch_rank=tail_branch_rank,
        fallback_reason=fallback_reason,
        pool_size=len(clip_pool),
        requested_k=k,
        valid_gallery_count=len(valid_set) if valid_set is not None else None,
    )
