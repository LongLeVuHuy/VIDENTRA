"""Cross-Keyframe Sequential Temporal Alignment & Research Localization Modules."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class ViterbiLocalizationResult:
    """Output of Sub-Event Viterbi Chronological Localization (SE-Viterbi)."""

    best_score: float
    avg_score: float
    selected_indices: tuple[int, ...]
    start_sec: float
    end_sec: float


def align_sequential_keyframe_clauses(
    cand_matrix: np.ndarray,
    cand_timestamps: Sequence[float],
    clause_vectors: Sequence[np.ndarray],
    max_transition_sec: float = 35.0,
    precomputed_clause_curves: np.ndarray | None = None,
) -> tuple[float, tuple[float, float] | None]:
    """Aligns sequential clauses (e.g. C_1 -> C_2) across keyframes within a video candidate.
    
    Guarantees:
    - Ordered time matching: t_a <= t_b <= t_a + max_transition_sec (no reverse-order matches)
    - Max average sequence score across ordered keyframe transitions
    - Anchors the transition window [t_a, t_b] for subsequent temporal moment refinement
    
    Args:
        cand_matrix: np.ndarray of shape (K_v, D) where K_v is number of keyframes, D is vector dimension
        cand_timestamps: Sequence of length K_v, ascending float seconds
        clause_vectors: Sequence of M clause query vectors, each shape (D,)
        max_transition_sec: Maximum permitted gap between consecutive clause transitions (default: 35.0s)
        precomputed_clause_curves: Optional (M, K_v) similarity matrix to skip redundant GEMM
        
    Returns:
        tuple (best_sequence_score, anchor_window)
        where anchor_window is (t_a, t_b) or None if matrix is empty
    """
    k_v = cand_matrix.shape[0] if cand_matrix is not None else 0
    m_clauses = len(clause_vectors)

    if k_v == 0 or m_clauses == 0:
        return 0.0, None

    ts = list(map(float, cand_timestamps))

    if k_v == 1:
        if precomputed_clause_curves is not None and precomputed_clause_curves.shape == (m_clauses, 1):
            avg_score = float(np.mean(precomputed_clause_curves[:, 0]))
        else:
            avg_score = float(np.mean([np.dot(cand_matrix[0], cv) for cv in clause_vectors]))
        return avg_score, (ts[0], ts[0])

    if m_clauses == 1:
        if precomputed_clause_curves is not None and precomputed_clause_curves.shape == (1, k_v):
            sims = precomputed_clause_curves[0].ravel()
        else:
            sims = (cand_matrix @ clause_vectors[0]).flatten()
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), (ts[best_idx], ts[best_idx])

    if (
        precomputed_clause_curves is not None
        and np.asarray(precomputed_clause_curves).shape == (m_clauses, k_v)
    ):
        clause_curves = np.asarray(precomputed_clause_curves, dtype=np.float32)
    else:
        # Compute similarity curves for each clause: shape (M, K_v) via single GEMM
        clause_mat = np.vstack(clause_vectors)
        clause_curves = (cand_matrix @ clause_mat.T).T.astype(np.float32)

    # For 2 clauses: exact O(K_v) monotonic deque for non-decreasing timestamps, vectorized NumPy fallback
    if m_clauses == 2:
        s1 = np.asarray(clause_curves[0], dtype=np.float32)
        s2 = np.asarray(clause_curves[1], dtype=np.float32)
        is_nondecreasing = all(ts[i] <= ts[i + 1] for i in range(k_v - 1))
        if is_nondecreasing and max_transition_sec >= 0.0:
            best_score = -float("inf")
            best_pair: tuple[float, float] | None = None
            dq: deque[int] = deque()
            right = 0
            for a in range(k_v):
                t_a = ts[a]
                diag_base = 0.5 * (float(s1[a]) + float(s2[a]))
                diag_score = 0.92 * diag_base if k_v >= 2 else diag_base
                if diag_score > best_score:
                    best_score = diag_score
                    best_pair = (t_a, t_a)
                if right <= a:
                    right = a + 1
                while right < k_v and (ts[right] - t_a) <= max_transition_sec:
                    val_r = float(s2[right])
                    while dq and float(s2[dq[-1]]) < val_r:
                        dq.pop()
                    dq.append(right)
                    right += 1
                while dq and dq[0] <= a:
                    dq.popleft()
                if dq:
                    b_best = dq[0]
                    cand_score = 0.5 * (float(s1[a]) + float(s2[b_best]))
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pair = (t_a, ts[b_best])
            if best_pair is not None:
                return float(best_score), best_pair

        ts_arr = np.asarray(ts, dtype=np.float64)
        mat = 0.5 * (s1[:, None].astype(np.float64) + s2[None, :].astype(np.float64))
        if k_v >= 2:
            np.fill_diagonal(mat, 0.92 * np.diag(mat))
        tri = np.triu(np.ones((k_v, k_v), dtype=bool))
        diffs = ts_arr[None, :] - ts_arr[:, None]
        valid = tri & (diffs >= 0.0) & (diffs <= max_transition_sec)
        if bool(np.any(valid)):
            masked = np.where(valid, mat, -np.inf)
            ia, ib = divmod(int(np.argmax(masked)), k_v)
            return float(masked[ia, ib]), (ts[ia], ts[ib])

        # Fallback if no pair satisfies max_transition_sec: pick best upper-triangular pair
        masked_fb = np.where(tri, mat, -np.inf)
        ia, ib = divmod(int(np.argmax(masked_fb)), k_v)
        return float(masked_fb[ia, ib]), (ts[ia], ts[ib])

    # General M >= 3: vectorized Dynamic Programming across sequential stages
    # dp[m, i] = max sum of scores for clauses 0..m ending at keyframe i
    dp = np.full((m_clauses, k_v), -np.inf, dtype=np.float32)
    parent = np.full((m_clauses, k_v), -1, dtype=np.int32)
    dp[0] = clause_curves[0]

    ts_arr = np.asarray(ts, dtype=np.float64)
    tril = np.tril(np.ones((k_v, k_v), dtype=bool))
    diffs = ts_arr[:, None] - ts_arr[None, :]
    valid_dp = tril & (diffs >= 0.0) & (diffs <= max_transition_sec)
    eye_mask = np.eye(k_v, dtype=bool) if k_v >= 2 else None
    row_idx = np.arange(k_v, dtype=np.int32)

    for m in range(1, m_clauses):
        prev = dp[m - 1]
        s_curr = clause_curves[m]
        step_mat = np.broadcast_to(s_curr[:, None], (k_v, k_v)).copy()
        if eye_mask is not None:
            step_mat[eye_mask] *= np.float32(0.92)
        cand_mat = np.where(
            valid_dp & (prev[None, :] != -np.inf),
            prev[None, :] + step_mat,
            np.float32(-np.inf),
        )
        best_j = np.argmax(cand_mat, axis=1)
        best_val = cand_mat[row_idx, best_j]
        has_valid = best_val > -np.inf
        dp[m] = np.where(has_valid, best_val, np.float32(-np.inf))
        parent[m] = np.where(has_valid, best_j, -1)

    last_scores = dp[m_clauses - 1]
    best_last_idx = int(np.argmax(last_scores))

    if last_scores[best_last_idx] == -np.inf:
        # Fallback: diagonal or single best
        sims_full = (cand_matrix @ clause_vectors[0]).flatten()
        idx_max = int(np.argmax(sims_full))
        return float(sims_full[idx_max]), (ts[idx_max], ts[idx_max])

    # Traceback to find anchor window [t_start, t_end]
    curr_idx = best_last_idx
    t_end = ts[curr_idx]
    for m in range(m_clauses - 1, 0, -1):
        curr_idx = int(parent[m, curr_idx])
    t_start = ts[curr_idx]

    best_avg_score = float(last_scores[best_last_idx]) / float(m_clauses)
    return best_avg_score, (t_start, t_end)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Safely L2-normalize a 2D array row-wise without NaN on zero rows."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-8 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    return arr / safe_norms


def shot_aware_graph_diffusion(
    embeddings_768: np.ndarray,
    timestamps_sec: Sequence[float],
    initial_scores: Sequence[float] | np.ndarray,
    *,
    ocr_scores: Sequence[float] | np.ndarray | None = None,
    asr_scores: Sequence[float] | np.ndarray | None = None,
    eta_ocr: float = 0.25,
    eta_asr: float = 0.20,
    sigma_t: float = 4.0,
    shot_sim_threshold: float = 0.65,
    alpha_d: float = 0.25,
) -> np.ndarray:
    """Shot-Aware Cross-Modal Graph Diffusion within a candidate video (CM-GD).

    Propagates multimodal keyframe scores only across temporally close keyframes
    that belong to the same visual shot (cosine similarity >= ``shot_sim_threshold``),
    preventing score leakage across hard shot cuts.
    """
    s_0 = np.asarray(initial_scores, dtype=np.float32).ravel()
    m_v = s_0.size
    if m_v == 0:
        return np.zeros(0, dtype=np.float32)

    if ocr_scores is not None:
        s_0 = s_0 + float(eta_ocr) * np.asarray(ocr_scores, dtype=np.float32).ravel()
    if asr_scores is not None:
        s_0 = s_0 + float(eta_asr) * np.asarray(asr_scores, dtype=np.float32).ravel()

    if m_v == 1:
        return s_0.astype(np.float32)

    emb = _l2_normalize_rows(embeddings_768)
    ts = np.asarray(timestamps_sec, dtype=np.float32).ravel()
    if emb.shape[0] != m_v or ts.size != m_v:
        raise ValueError(
            f"Shape mismatch in shot_aware_graph_diffusion: "
            f"emb={emb.shape[0]}, ts={ts.size}, scores={m_v}"
        )

    sim_mat = emb @ emb.T
    dt = ts[:, None] - ts[None, :]
    sigma_sq = max(1e-6, float(sigma_t) ** 2)
    temporal_kernel = np.exp(-(dt * dt) / (2.0 * sigma_sq)).astype(np.float32)
    shot_mask = (sim_mat >= float(shot_sim_threshold)).astype(np.float32)

    W = temporal_kernel * shot_mask
    np.fill_diagonal(W, 1.0)
    deg = np.sum(W, axis=1, keepdims=True)
    P = W / np.maximum(deg, 1e-8)

    alpha = float(np.clip(alpha_d, 0.0, 1.0))
    s_diff = (1.0 - alpha) * s_0 + alpha * (P @ s_0)
    return s_diff.astype(np.float32)


def compute_query_adaptive_csls(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    *,
    reference_query_vecs: np.ndarray | None = None,
    k_neighbors: int = 15,
) -> np.ndarray:
    """Compute Query-Adaptive Cross-Domain Similarity Local Scaling (CSLS).

    Suppresses query-specific local hubs among top candidate keyframes:
        CSLS(q, v_i) = 2 * <q, v_i> - 0.5 * r_S(q) - 0.5 * r_T(v_i)
    """
    cands = np.asarray(candidate_vecs, dtype=np.float32)
    if cands.size == 0 or cands.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    q = _l2_normalize_rows(np.asarray(query_vec, dtype=np.float32).ravel())
    v = _l2_normalize_rows(cands)

    q_sims = (v @ q).astype(np.float32)
    k_s = max(1, min(int(k_neighbors), q_sims.size))
    r_s = float(np.mean(np.partition(q_sims, q_sims.size - k_s)[-k_s:]))

    if reference_query_vecs is not None and len(reference_query_vecs) > 0:
        ref_q = _l2_normalize_rows(np.asarray(reference_query_vecs, dtype=np.float32))
        cross = v @ ref_q.T
        k_t = max(1, min(int(k_neighbors), cross.shape[1]))
        top_cross = np.partition(cross, cross.shape[1] - k_t, axis=1)[:, -k_t:]
        r_t = np.mean(top_cross, axis=1).astype(np.float32)
    else:
        # Local candidate-neighborhood hubness proxy when no reference query bank is passed
        local_sim = v @ v.T
        k_t = max(1, min(int(k_neighbors), local_sim.shape[1]))
        top_local = np.partition(local_sim, local_sim.shape[1] - k_t, axis=1)[:, -k_t:]
        r_t = np.mean(top_local, axis=1).astype(np.float32)

    return (2.0 * q_sims - 0.5 * r_s - 0.5 * r_t).astype(np.float32)


def viterbi_subevent_localization(
    timestamps_sec: Sequence[float],
    event_vis_curves: Sequence[Sequence[float] | np.ndarray],
    *,
    event_ocr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_asr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_weights: Sequence[tuple[float, float, float]] | None = None,
    max_gap_sec: float = 35.0,
    lambda_gap: float = 0.02,
    lambda_same: float = 0.08,
) -> ViterbiLocalizationResult | None:
    """Multi-modal Sub-Event Viterbi Chronological Localization (SE-Viterbi).

    Solves monotonic DP over K ordered sub-events across M_v keyframes using
    fused per-event curves:
        Phi_k(t_i) = w_vis * S_vis(e_k, t_i) + w_ocr * S_ocr(e_k, t_i) + w_asr * S_asr(e_k, t_i)
    """
    ts = [float(t) for t in timestamps_sec]
    m_v = len(ts)
    k_events = len(event_vis_curves)
    if m_v == 0 or k_events == 0:
        return None

    vis_mat = np.asarray(event_vis_curves, dtype=np.float32)
    if vis_mat.shape != (k_events, m_v):
        raise ValueError(
            f"event_vis_curves must have shape ({k_events}, {m_v}), got {vis_mat.shape}"
        )

    ocr_mat = (
        np.asarray(event_ocr_curves, dtype=np.float32)
        if event_ocr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    asr_mat = (
        np.asarray(event_asr_curves, dtype=np.float32)
        if event_asr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    if ocr_mat.shape != (k_events, m_v) or asr_mat.shape != (k_events, m_v):
        raise ValueError("OCR/ASR event curve shapes must match event_vis_curves")

    phi = np.empty((k_events, m_v), dtype=np.float32)
    for k in range(k_events):
        w_vis, w_ocr, w_asr = (
            event_weights[k] if event_weights is not None else (1.0, 0.6, 0.5)
        )
        phi[k] = float(w_vis) * vis_mat[k] + float(w_ocr) * ocr_mat[k] + float(w_asr) * asr_mat[k]

    if k_events == 1:
        best_idx = int(np.argmax(phi[0]))
        best_sc = float(phi[0, best_idx])
        return ViterbiLocalizationResult(
            best_score=best_sc,
            avg_score=best_sc,
            selected_indices=(best_idx,),
            start_sec=ts[best_idx],
            end_sec=ts[best_idx],
        )

    dp = np.full((k_events, m_v), -np.inf, dtype=np.float64)
    parent = np.full((k_events, m_v), -1, dtype=np.int32)
    dp[0] = phi[0].astype(np.float64)

    for k in range(1, k_events):
        for i in range(m_v):
            t_curr = ts[i]
            emit = float(phi[k, i])
            best_val = -np.inf
            best_prev = -1
            for j in range(i + 1):
                prev_val = dp[k - 1, j]
                if prev_val == -np.inf:
                    continue
                dt = max(0.0, t_curr - ts[j])
                excess_gap = max(0.0, dt - float(max_gap_sec))
                same_pen = float(lambda_same) if (m_v >= 2 and j == i) else 0.0
                cand = prev_val + emit - float(lambda_gap) * excess_gap - same_pen
                if cand > best_val:
                    best_val = cand
                    best_prev = j
            dp[k, i] = best_val
            parent[k, i] = best_prev

    best_last = int(np.argmax(dp[k_events - 1]))
    best_total = float(dp[k_events - 1, best_last])
    if not np.isfinite(best_total):
        return None

    path = [best_last]
    curr = best_last
    for k in range(k_events - 1, 0, -1):
        curr = int(parent[k, curr])
        path.append(curr)
    path.reverse()

    return ViterbiLocalizationResult(
        best_score=best_total,
        avg_score=best_total / float(k_events),
        selected_indices=tuple(path),
        start_sec=ts[path[0]],
        end_sec=ts[path[-1]],
    )


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis`` in pure NumPy."""
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def temporal_sinkhorn_alignment_score(
    clause_frame_sims: np.ndarray,
    *,
    timestamps_sec: Sequence[float] | np.ndarray | None = None,
    max_gap_sec: float = 35.0,
    eta: float = 0.15,
    epsilon: float = 0.02,
    n_iters: int = 20,
) -> float:
    """Unbalanced / Partial Temporal Sinkhorn Optimal Transport (T-SOT) alignment score.

    Finds a soft transport plan ``P`` of shape ``(M, T)`` between ``M`` query
    clauses and ``T`` keyframes in a temporal window using stabilized log-domain
    Partial Sinkhorn iterations (exact row marginals ``1/M`` and upper-bounded
    column marginals ``<= 1/min(M, T)`` so irrelevant frames do not dilute
    the score while single-frame collapse is still penalized).

    When ``timestamps_sec`` is provided, restricts evaluation to the candidate
    window ``peak ± max_gap_sec`` and uses real timestamp gaps ``Δt`` for the
    chronological order penalty instead of uniform frame index spacing.
    Returns ``sum(P * clause_frame_sims)``.
    """
    if clause_frame_sims is None:
        return 0.0
    sims = np.asarray(clause_frame_sims, dtype=np.float64)
    if sims.size == 0:
        return 0.0
    if sims.ndim == 1:
        sims = sims.reshape(1, -1)
    elif sims.ndim != 2:
        raise ValueError(
            f"clause_frame_sims must be 1D or 2D, got shape {sims.shape}"
        )

    m_clauses, t_frames = sims.shape
    if m_clauses == 1:
        return float(np.max(sims))
    if t_frames == 1:
        return float(np.mean(sims))

    if timestamps_sec is not None:
        ts = np.asarray(timestamps_sec, dtype=np.float64).ravel()
        if ts.size != t_frames:
            raise ValueError(
                f"timestamps_sec length ({ts.size}) must match T ({t_frames})"
            )
        if max_gap_sec > 0.0 and t_frames > 2:
            peak_idx = int(np.argmax(np.max(sims, axis=0)))
            peak_t = float(ts[peak_idx])
            win_mask = np.abs(ts - peak_t) <= float(max_gap_sec)
            if int(np.sum(win_mask)) >= 2:
                sims = sims[:, win_mask]
                ts = ts[win_mask]
                t_frames = sims.shape[1]
        span = float(ts[-1] - ts[0])
        if span > 1e-6:
            t_pos = ((ts - ts[0]) / span)[None, :]
        else:
            t_pos = np.arange(t_frames, dtype=np.float64)[None, :] / float(max(1, t_frames - 1))
    else:
        t_pos = np.arange(t_frames, dtype=np.float64)[None, :] / float(max(1, t_frames - 1))

    m_pos = np.arange(m_clauses, dtype=np.float64)[:, None] / float(max(1, m_clauses - 1))
    penalty = float(eta) * (m_pos - t_pos) ** 2

    eps = max(1e-6, float(epsilon))
    log_k = (sims - penalty) / eps

    # Partial OT: each clause distributes 1/M mass; each frame absorbs at most 1/min(M, T)
    log_a = -np.log(float(m_clauses))
    log_b = -np.log(float(min(m_clauses, t_frames)))
    u = np.zeros(m_clauses, dtype=np.float64)
    v = np.zeros(t_frames, dtype=np.float64)

    for it in range(max(1, int(n_iters))):
        prev_u = u
        v = np.minimum(0.0, log_b - _logsumexp(log_k + u[:, None], axis=0))
        u = log_a - _logsumexp(log_k + v[None, :], axis=1)
        if it >= 3 and float(np.max(np.abs(u - prev_u))) < 1e-4:
            break

    plan = np.exp(log_k + u[:, None] + v[None, :])
    return float(np.sum(plan * sims))


def query_guided_temporal_attention_score(
    query_vec: np.ndarray,
    frame_vecs: np.ndarray,
    *,
    tau: float = 12.0,
    disp_penalty: float = 0.15,
    precomputed_cos_sims: np.ndarray | None = None,
    assume_unit_norm: bool = False,
) -> float:
    """Parameter-Free Query-Guided Temporal Attention Pooler.

    Computes attention weights ``alpha_k = softmax(tau * cos(q, v_k))`` and
    returns ``sum(alpha_k * cos(q, v_k)) - disp_penalty * (1 - ||sum(alpha_k * v_k)||_2)``.
    Rewards high query-aligned frames while penalizing high directional
    dispersion (e.g., unrelated frames across hard scene cuts).
    """
    if query_vec is None or frame_vecs is None:
        return 0.0

    f_arr = np.asarray(frame_vecs, dtype=np.float32)
    if f_arr.size == 0:
        return 0.0

    if f_arr.ndim == 1:
        f_arr = f_arr.reshape(1, -1)
    elif f_arr.ndim != 2:
        raise ValueError(f"frame_vecs must be 1D or 2D, got shape {f_arr.shape}")

    if f_arr.shape[0] == 1 and assume_unit_norm and precomputed_cos_sims is not None and precomputed_cos_sims.size == 1:
        return float(precomputed_cos_sims.flat[0])

    use_fast_f32 = bool(assume_unit_norm and precomputed_cos_sims is not None and precomputed_cos_sims.size == f_arr.shape[0])
    f_unit = None
    if use_fast_f32:
        cos_sims = np.asarray(precomputed_cos_sims, dtype=np.float64).ravel()
    else:
        f_unit = f_arr.astype(np.float64, copy=False) if assume_unit_norm else _l2_normalize_rows(f_arr).astype(np.float64)
        if precomputed_cos_sims is not None and precomputed_cos_sims.size == f_arr.shape[0]:
            cos_sims = np.asarray(precomputed_cos_sims, dtype=np.float64).ravel()
        else:
            q_arr = np.asarray(query_vec, dtype=np.float32).ravel()
            if q_arr.size == 0:
                return 0.0
            if f_arr.shape[1] != q_arr.size:
                raise ValueError(
                    f"Dimension mismatch: query_vec has {q_arr.size}, "
                    f"frame_vecs has {f_arr.shape[1]}"
                )
            q_norm = float(np.linalg.norm(q_arr))
            if q_norm <= 1e-8:
                return 0.0
            q_unit = (q_arr / q_norm).astype(np.float64)
            cos_sims = f_unit @ q_unit

    logits = float(tau) * cos_sims
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    alpha = exp_logits / np.sum(exp_logits)

    mean_cos = float(np.dot(alpha, cos_sims))
    v_att = (alpha.astype(np.float32) @ f_arr) if use_fast_f32 else (alpha @ f_unit)
    v_norm = float(np.linalg.norm(v_att))
    if v_norm <= 1e-8:
        return 0.0
    dispersion = max(0.0, 1.0 - v_norm)
    return float(mean_cos - float(disp_penalty) * dispersion)


_VALID_TEMPORAL_POOL_MODES = ("mean", "mean_top2", "mean_top3", "attention")


def compute_temporal_pool_score(
    query_vec: np.ndarray,
    frame_vecs: np.ndarray,
    *,
    mode: str = "attention",
    tau: float = 12.0,
    disp_penalty: float = 0.15,
) -> tuple[float, float]:
    """Compute a pooled temporal score and spike gap (max - mean_top2) over frame vectors.

    Supported modes: ``mean``, ``mean_top2``, ``mean_top3``, ``attention``.
    Returns ``(pool_score, spike_gap)``.
    """
    if mode not in _VALID_TEMPORAL_POOL_MODES:
        raise ValueError(
            f"Unsupported temporal pool mode {mode!r}; expected one of {_VALID_TEMPORAL_POOL_MODES}"
        )
    if query_vec is None or frame_vecs is None:
        return 0.0, 0.0

    q_arr = np.asarray(query_vec, dtype=np.float32).ravel()
    f_arr = np.asarray(frame_vecs, dtype=np.float32)
    if q_arr.size == 0 or f_arr.size == 0:
        return 0.0, 0.0

    if f_arr.ndim == 1:
        f_arr = f_arr.reshape(1, -1)
    elif f_arr.ndim != 2:
        raise ValueError(f"frame_vecs must be 1D or 2D, got shape {f_arr.shape}")

    if f_arr.shape[1] != q_arr.size:
        raise ValueError(
            f"Dimension mismatch: query_vec has {q_arr.size}, "
            f"frame_vecs has {f_arr.shape[1]}"
        )

    q_norm = float(np.linalg.norm(q_arr))
    if q_norm <= 1e-8:
        return 0.0, 0.0
    q_unit = (q_arr / q_norm).astype(np.float64)
    f_unit = _l2_normalize_rows(f_arr).astype(np.float64)
    cos_sims = f_unit @ q_unit

    sorted_sims = np.sort(cos_sims)
    top2_mean = float(np.mean(sorted_sims[-min(2, len(sorted_sims)):]))
    spike_gap = float(sorted_sims[-1] - top2_mean)

    if mode == "mean":
        pool_score = float(np.mean(cos_sims))
    elif mode == "mean_top2":
        pool_score = top2_mean
    elif mode == "mean_top3":
        pool_score = float(np.mean(sorted_sims[-min(3, len(sorted_sims)):]))
    else:
        pool_score = query_guided_temporal_attention_score(
            q_arr, f_arr, tau=tau, disp_penalty=disp_penalty
        )
    return float(pool_score), float(spike_gap)


@dataclass(frozen=True)
class EventSpanPrediction:
    """Per-event localized interval anchored by presentation-order PTS & 0-based frame ordinals."""

    clause_id: str
    clause_text: str
    relation_to_prev: str
    start_sec: float
    end_sec: float
    anchor_index: int
    anchor_sec: float
    start_frame_index: int
    end_frame_index: int
    score: float
    status: str = "supported"

    def to_dict(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "clause_text": self.clause_text,
            "relation_to_prev": self.relation_to_prev,
            "start_sec": round(float(self.start_sec), 4),
            "end_sec": round(float(self.end_sec), 4),
            "anchor_index": int(self.anchor_index),
            "anchor_sec": round(float(self.anchor_sec), 4),
            "start_frame_index": int(self.start_frame_index),
            "end_frame_index": int(self.end_frame_index),
            "score": round(float(self.score), 6),
            "status": self.status,
        }


@dataclass(frozen=True)
class EventChainLocalizationResult:
    """Multi-event Semi-Markov DP result with per-event spans and Directional Order Margin."""

    total_score: float
    avg_score: float
    reverse_avg_score: float
    directional_order_margin: float
    spans: tuple[EventSpanPrediction, ...]
    overall_start_sec: float
    overall_end_sec: float
    all_supported: bool

    def to_dict_list(self) -> tuple[dict[str, object], ...]:
        return tuple(sp.to_dict() for sp in self.spans)


def _solve_chain_dp_path(
    ts: Sequence[float],
    phi: np.ndarray,
    relations: Sequence[str],
    *,
    max_gap_sec: float,
    lambda_gap: float,
    lambda_same: float,
    simultaneous_max_delta_sec: float,
) -> tuple[float, list[int]] | None:
    """Solve forward DP path over `(K, M_v)` event curves respecting `before` vs `simultaneous`."""
    k_events, m_v = phi.shape
    if m_v == 0 or k_events == 0:
        return None
    if k_events == 1:
        best_idx = int(np.argmax(phi[0]))
        return float(phi[0, best_idx]), [best_idx]

    dp = np.full((k_events, m_v), -np.inf, dtype=np.float64)
    parent = np.full((k_events, m_v), -1, dtype=np.int32)
    dp[0] = phi[0].astype(np.float64)

    for k in range(1, k_events):
        rel = relations[k] if k < len(relations) else "before"
        for i in range(m_v):
            t_curr = ts[i]
            emit = float(phi[k, i])
            best_val = -np.inf
            best_prev = -1
            if rel == "simultaneous":
                for j in range(m_v):
                    dt_abs = abs(t_curr - ts[j])
                    if dt_abs > float(simultaneous_max_delta_sec):
                        continue
                    prev_val = dp[k - 1, j]
                    if prev_val == -np.inf:
                        continue
                    cand = prev_val + emit - 0.005 * dt_abs
                    if cand > best_val:
                        best_val = cand
                        best_prev = j
            if best_prev == -1:
                for j in range(i + 1):
                    prev_val = dp[k - 1, j]
                    if prev_val == -np.inf:
                        continue
                    dt = max(0.0, t_curr - ts[j])
                    excess_gap = max(0.0, dt - float(max_gap_sec))
                    same_pen = float(lambda_same) if (m_v >= 2 and j == i and rel != "simultaneous") else 0.0
                    cand = prev_val + emit - float(lambda_gap) * excess_gap - same_pen
                    if cand > best_val:
                        best_val = cand
                        best_prev = j
            dp[k, i] = best_val
            parent[k, i] = best_prev

    best_last = int(np.argmax(dp[k_events - 1]))
    best_total = float(dp[k_events - 1, best_last])
    if not np.isfinite(best_total):
        return None

    path = [best_last]
    curr = best_last
    for k in range(k_events - 1, 0, -1):
        curr = int(parent[k, curr])
        if curr < 0:
            return None
        path.append(curr)
    path.reverse()
    return best_total, path


def localize_event_chain_dp(
    timestamps_sec: Sequence[float],
    event_vis_curves: Sequence[Sequence[float] | np.ndarray],
    *,
    clause_ids: Sequence[str] | None = None,
    clause_texts: Sequence[str] | None = None,
    relations_to_prev: Sequence[str] | None = None,
    event_ocr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_asr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_weights: Sequence[tuple[float, float, float]] | None = None,
    frame_ordinals: Sequence[int] | None = None,
    v_dur_sec: float | None = None,
    max_gap_sec: float = 45.0,
    lambda_gap: float = 0.02,
    lambda_same: float = 0.08,
    simultaneous_max_delta_sec: float = 8.0,
    missing_evidence_threshold: float = 0.05,
) -> EventChainLocalizationResult | None:
    """Semi-Markov DP multi-event localizer emitting per-clause PTS intervals + Directional Order Margin."""
    from videntra.localization.refine import expand_boundary_contrast

    ts = [float(t) for t in timestamps_sec]
    m_v = len(ts)
    vis_mat = np.asarray(event_vis_curves, dtype=np.float32)
    if vis_mat.ndim == 1:
        vis_mat = vis_mat.reshape(1, -1)
    k_events = int(vis_mat.shape[0]) if vis_mat.size > 0 else 0
    if m_v == 0 or k_events == 0 or vis_mat.shape != (k_events, m_v):
        return None

    ocr_mat = (
        np.asarray(event_ocr_curves, dtype=np.float32)
        if event_ocr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    asr_mat = (
        np.asarray(event_asr_curves, dtype=np.float32)
        if event_asr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    phi = np.empty((k_events, m_v), dtype=np.float32)
    for k in range(k_events):
        w_vis, w_ocr, w_asr = (
            event_weights[k] if event_weights is not None else (1.0, 0.6, 0.5)
        )
        phi[k] = float(w_vis) * vis_mat[k] + float(w_ocr) * ocr_mat[k] + float(w_asr) * asr_mat[k]

    eff_cids = [str(clause_ids[k]) if (clause_ids and k < len(clause_ids)) else f"e{k}" for k in range(k_events)]
    eff_txts = [str(clause_texts[k]) if (clause_texts and k < len(clause_texts)) else "" for k in range(k_events)]
    eff_rels = [
        ("start" if k == 0 else str(relations_to_prev[k]))
        if (relations_to_prev and k < len(relations_to_prev))
        else ("start" if k == 0 else "before")
        for k in range(k_events)
    ]
    eff_ords = (
        [int(frame_ordinals[i]) for i in range(m_v)]
        if (frame_ordinals is not None and len(frame_ordinals) == m_v)
        else list(range(m_v))
    )

    fwd = _solve_chain_dp_path(
        ts,
        phi,
        eff_rels,
        max_gap_sec=max_gap_sec,
        lambda_gap=lambda_gap,
        lambda_same=lambda_same,
        simultaneous_max_delta_sec=simultaneous_max_delta_sec,
    )
    if fwd is None:
        return None
    fwd_total, path = fwd
    fwd_avg = fwd_total / float(k_events)

    rev_avg = fwd_avg
    has_order_edge = any(r == "before" for r in eff_rels[1:])
    if k_events >= 2 and has_order_edge:
        rev_phi = phi[::-1]
        rev_rels = ["start"] + ["before"] * (k_events - 1)
        rev_sol = _solve_chain_dp_path(
            ts,
            rev_phi,
            rev_rels,
            max_gap_sec=max_gap_sec,
            lambda_gap=lambda_gap,
            lambda_same=lambda_same,
            simultaneous_max_delta_sec=simultaneous_max_delta_sec,
        )
        if rev_sol is not None:
            rev_avg = float(rev_sol[0]) / float(k_events)
    order_margin = float(fwd_avg - rev_avg)

    eff_dur = float(v_dur_sec) if v_dur_sec is not None else max(ts[-1] + 2.0, 2.0)
    per_ev_max_dur_ratio = min(0.60, max(0.28, 1.1 / float(k_events)))
    raw_starts: list[float] = []
    raw_ends: list[float] = []
    for k in range(k_events):
        anchor_idx = path[k]
        st_k, et_k = expand_boundary_contrast(
            ts=ts,
            curve=phi[k].tolist(),
            v_dur=eff_dur,
            anchor_idx=anchor_idx,
            anchor_window=None,
            decay_lambda=0.015,
            c_thresh=0.45,
            max_gap_sec=4.0,
            max_dur_ratio=per_ev_max_dur_ratio,
        )
        anchor_t = ts[anchor_idx]
        st_k = min(float(st_k), anchor_t)
        et_k = max(float(et_k), anchor_t)
        if et_k <= st_k:
            st_k = max(0.0, anchor_t - 0.5)
            et_k = min(eff_dur, max(st_k + 0.25, anchor_t + 0.5))
        raw_starts.append(st_k)
        raw_ends.append(et_k)

    # Order-preserving midpoint valley clamping ONLY for adjacent sequential `before` edges
    for k in range(1, k_events):
        if eff_rels[k] != "before":
            continue
        if raw_ends[k - 1] <= raw_starts[k]:
            continue
        t_a = ts[path[k - 1]]
        t_b = ts[path[k]]
        if t_b <= t_a + 1e-6:
            cut_t = t_a
        else:
            between = [
                (m, ts[m], float(phi[k - 1, m] + phi[k, m]))
                for m in range(m_v)
                if t_a <= ts[m] <= t_b
            ]
            if between:
                cut_t = min(between, key=lambda item: item[2])[1]
            else:
                cut_t = 0.5 * (t_a + t_b)
            if t_b > t_a + 0.1:
                cut_t = min(max(cut_t, t_a + 0.05), t_b - 0.05)
            else:
                cut_t = 0.5 * (t_a + t_b)
        raw_ends[k - 1] = max(raw_starts[k - 1] + 0.05, min(raw_ends[k - 1], cut_t))
        raw_starts[k] = min(raw_ends[k] - 0.05, max(raw_starts[k], cut_t))

    spans: list[EventSpanPrediction] = []
    all_supported = True
    for k in range(k_events):
        anchor_idx = path[k]
        st_k = round(float(raw_starts[k]), 3)
        et_k = round(max(st_k + 0.05, float(raw_ends[k])), 3)
        s_pos = min(range(m_v), key=lambda idx: abs(ts[idx] - st_k))
        e_pos = min(range(m_v), key=lambda idx: abs(ts[idx] - et_k))
        s_ord = eff_ords[min(s_pos, anchor_idx)]
        e_ord = eff_ords[max(e_pos, anchor_idx)]
        ev_score = float(phi[k, anchor_idx])
        peak_k = float(np.max(phi[k]))
        status = (
            "supported"
            if (peak_k >= float(missing_evidence_threshold) and ev_score >= float(missing_evidence_threshold))
            else "missing"
        )
        if status != "supported":
            all_supported = False
        spans.append(
            EventSpanPrediction(
                clause_id=eff_cids[k],
                clause_text=eff_txts[k],
                relation_to_prev=eff_rels[k],
                start_sec=st_k,
                end_sec=et_k,
                anchor_index=int(anchor_idx),
                anchor_sec=round(float(ts[anchor_idx]), 3),
                start_frame_index=int(min(s_ord, e_ord)),
                end_frame_index=int(max(s_ord, e_ord)),
                score=ev_score,
                status=status,
            )
        )

    return EventChainLocalizationResult(
        total_score=float(fwd_total),
        avg_score=float(fwd_avg),
        reverse_avg_score=float(rev_avg),
        directional_order_margin=order_margin,
        spans=tuple(spans),
        overall_start_sec=min(sp.start_sec for sp in spans),
        overall_end_sec=max(sp.end_sec for sp in spans),
        all_supported=all_supported,
    )


def _voronoi_frame_edges(ts: Sequence[float], v_dur_sec: float | None = None) -> tuple[list[float], list[float]]:
    """Compute contiguous non-overlapping `[left_i, right_i]` intervals around sorted timestamps."""
    m_v = len(ts)
    if m_v == 0:
        return [], []
    eff_dur = float(v_dur_sec) if v_dur_sec is not None else max(float(ts[-1]) + 2.0, 2.0)
    if m_v == 1:
        half = min(2.0, max(0.5, 0.25 * eff_dur))
        return [max(0.0, float(ts[0]) - half)], [min(eff_dur, max(float(ts[0]) + 0.1, float(ts[0]) + half))]

    mids = [0.5 * (float(ts[i]) + float(ts[i + 1])) for i in range(m_v - 1)]
    first_half = max(0.25, 0.5 * (float(ts[1]) - float(ts[0])))
    last_half = max(0.25, 0.5 * (float(ts[-1]) - float(ts[-2])))
    lefts = [max(0.0, float(ts[0]) - first_half)] + mids
    rights = mids + [min(eff_dur, max(float(ts[-1]) + 0.1, float(ts[-1]) + last_half))]
    return lefts, rights


def _solve_exact_segment_dp(
    phi: np.ndarray,
    left_edges: Sequence[float],
    right_edges: Sequence[float],
    relations: Sequence[str],
    *,
    max_gap_sec: float = 45.0,
    lambda_bg: float = 0.30,
    lambda_peak: float = 0.35,
    lambda_len: float = 0.04,
    lambda_gap: float = 0.02,
    simultaneous_max_delta_sec: float = 8.0,
) -> tuple[float, list[tuple[int, int]]] | None:
    """Exact 2D Semi-Markov DP choosing contiguous index spans `[a_k, b_k]` per event via vectorized prefix sums."""
    k_events, m_v = phi.shape
    if k_events == 0 or m_v == 0:
        return None

    phi64 = np.asarray(phi, dtype=np.float64)
    left_arr = np.asarray(left_edges, dtype=np.float64)
    right_arr = np.asarray(right_edges, dtype=np.float64)
    pref = np.zeros((k_events, m_v + 1), dtype=np.float64)
    pref[:, 1:] = np.cumsum(phi64, axis=1)
    row_sums = pref[:, -1]

    eff_dur = max(1.0, float(right_arr[-1]) - float(left_arr[0]))
    l_prior = max(2.0, eff_dur / float(max(1, k_events)))

    idx_1d = np.arange(m_v)
    a_grid = idx_1d[:, None]
    b_grid = idx_1d[None, :]
    tri_mask = b_grid >= a_grid
    cnt_grid = np.where(tri_mask, (b_grid - a_grid + 1).astype(np.float64), 1.0)
    out_cnt_grid = float(m_v) - (b_grid - a_grid + 1).astype(np.float64)
    seg_dur = np.maximum(0.05, right_arr[None, :] - left_arr[:, None])
    len_pen = np.abs(np.log((seg_dur + 1.0) / (l_prior + 1.0)))

    # Precompute E[k, a, b] over valid intervals 0 <= a <= b < m_v
    emit_seg = np.full((k_events, m_v, m_v), -np.inf, dtype=np.float64)
    for k in range(k_events):
        row = phi64[k]
        in_sum = pref[k, 1:][None, :] - pref[k, :-1][:, None]
        mean_in = in_sum / cnt_grid
        mean_out = np.where(out_cnt_grid > 0.0, (row_sums[k] - in_sum) / np.maximum(1.0, out_cnt_grid), 0.0)
        peak_mat = np.empty((m_v, m_v), dtype=np.float64)
        for a in range(m_v):
            peak_mat[a, a:] = np.maximum.accumulate(row[a:])
        e_k = (
            mean_in
            - float(lambda_bg) * mean_out
            + float(lambda_peak) * peak_mat
            - float(lambda_len) * len_pen
        )
        emit_seg[k] = np.where(tri_mask, e_k, -np.inf)

    if k_events == 1:
        best_a, best_b = np.unravel_index(int(np.argmax(emit_seg[0])), (m_v, m_v))
        return float(emit_seg[0, best_a, best_b]), [(int(best_a), int(best_b))]

    # Precompute invariant transition costs over (a, bp) for non-simultaneous hops
    bp_grid = idx_1d[None, :]
    allow_bp_mask = (bp_grid < a_grid) | ((a_grid == 0) & (bp_grid == 0))
    same_pen = np.where((a_grid == 0) & (bp_grid == 0), 0.12, 0.0)
    gap_sec = np.maximum(0.0, left_arr[:, None] - right_arr[None, :])
    excess_gap = np.maximum(0.0, gap_sec - float(max_gap_sec))
    trans_cost = np.where(allow_bp_mask, float(lambda_gap) * excess_gap + same_pen, np.inf)

    dp = np.full((k_events, m_v, m_v), -np.inf, dtype=np.float64)
    parent_a = np.full((k_events, m_v, m_v), -1, dtype=np.int32)
    parent_b = np.full((k_events, m_v, m_v), -1, dtype=np.int32)
    dp[0] = emit_seg[0]
    mid_grid = 0.5 * (left_arr[:, None] + right_arr[None, :])

    for k in range(1, k_events):
        rel = relations[k] if k < len(relations) else "before"
        prev_best_a = np.argmax(dp[k - 1], axis=0)  # shape (m_v,)
        prev_best_val = dp[k - 1, prev_best_a, idx_1d]

        # Non-simultaneous best predecessor per `a` (independent of `b`)
        cand_trans = np.where(
            np.isfinite(prev_best_val[None, :]) & allow_bp_mask,
            prev_best_val[None, :] - trans_cost,
            -np.inf,
        )
        best_pb_a = np.argmax(cand_trans, axis=1)
        best_trans_a = cand_trans[idx_1d, best_pb_a]
        finite_a = np.isfinite(best_trans_a)
        best_pa_a = np.where(finite_a, prev_best_a[best_pb_a], -1)

        if rel != "simultaneous":
            cand_dp = best_trans_a[:, None] + emit_seg[k]
            valid_cell = tri_mask & finite_a[:, None] & np.isfinite(emit_seg[k])
            dp[k] = np.where(valid_cell, cand_dp, -np.inf)
            parent_a[k] = np.where(valid_cell, best_pa_a[:, None], -1)
            parent_b[k] = np.where(valid_cell, best_pb_a[:, None], -1)
            continue

        valid_prev = np.argwhere(np.isfinite(dp[k - 1]))
        for a in range(m_v):
            for b in range(a, m_v):
                e_val = emit_seg[k, a, b]
                if not np.isfinite(e_val):
                    continue
                mid_curr = float(mid_grid[a, b])
                best_val = -np.inf
                best_pa = -1
                best_pb = -1
                for ap, bp in valid_prev:
                    dt_c = abs(mid_curr - float(mid_grid[ap, bp]))
                    if dt_c <= float(simultaneous_max_delta_sec):
                        cand = float(dp[k - 1, ap, bp]) + float(e_val) - 0.01 * dt_c
                        if cand > best_val:
                            best_val = cand
                            best_pa = int(ap)
                            best_pb = int(bp)
                if best_pa == -1 and finite_a[a]:
                    best_val = float(best_trans_a[a] + e_val)
                    best_pa = int(best_pa_a[a])
                    best_pb = int(best_pb_a[a])
                dp[k, a, b] = best_val
                parent_a[k, a, b] = best_pa
                parent_b[k, a, b] = best_pb

    flat_idx = int(np.argmax(dp[k_events - 1]))
    ca, cb = np.unravel_index(flat_idx, (m_v, m_v))
    best_total = float(dp[k_events - 1, ca, cb])
    if not np.isfinite(best_total):
        return None

    spans_idx: list[tuple[int, int]] = [(int(ca), int(cb))]
    cur_a, cur_b = int(ca), int(cb)
    for k in range(k_events - 1, 0, -1):
        pa = int(parent_a[k, cur_a, cur_b])
        pb = int(parent_b[k, cur_a, cur_b])
        if pa < 0 or pb < 0:
            return None
        spans_idx.append((pa, pb))
        cur_a, cur_b = pa, pb
    spans_idx.reverse()
    return best_total, spans_idx


def localize_exact_segment_chain_dp(
    timestamps_sec: Sequence[float],
    event_vis_curves: Sequence[Sequence[float] | np.ndarray],
    *,
    clause_ids: Sequence[str] | None = None,
    clause_texts: Sequence[str] | None = None,
    relations_to_prev: Sequence[str] | None = None,
    event_ocr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_asr_curves: Sequence[Sequence[float] | np.ndarray] | None = None,
    event_weights: Sequence[tuple[float, float, float]] | None = None,
    frame_ordinals: Sequence[int] | None = None,
    v_dur_sec: float | None = None,
    max_gap_sec: float = 45.0,
    lambda_bg: float = 0.30,
    lambda_peak: float = 0.35,
    lambda_len: float = 0.04,
    lambda_gap: float = 0.02,
    simultaneous_max_delta_sec: float = 8.0,
    missing_evidence_threshold: float = 0.05,
) -> EventChainLocalizationResult | None:
    """Exact 2D Semi-Markov Segment DP optimizing contiguous event spans + Directional Order Margin."""
    import math as _math  # noqa: F401 (used cleanly below)

    ts = [float(t) for t in timestamps_sec]
    m_v = len(ts)
    vis_mat = np.asarray(event_vis_curves, dtype=np.float32)
    if vis_mat.ndim == 1:
        vis_mat = vis_mat.reshape(1, -1)
    k_events = int(vis_mat.shape[0]) if vis_mat.size > 0 else 0
    if m_v == 0 or k_events == 0 or vis_mat.shape != (k_events, m_v):
        return None

    ocr_mat = (
        np.asarray(event_ocr_curves, dtype=np.float32)
        if event_ocr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    asr_mat = (
        np.asarray(event_asr_curves, dtype=np.float32)
        if event_asr_curves is not None
        else np.zeros((k_events, m_v), dtype=np.float32)
    )
    phi = np.empty((k_events, m_v), dtype=np.float32)
    for k in range(k_events):
        w_vis, w_ocr, w_asr = (
            event_weights[k] if event_weights is not None else (1.0, 0.6, 0.5)
        )
        phi[k] = float(w_vis) * vis_mat[k] + float(w_ocr) * ocr_mat[k] + float(w_asr) * asr_mat[k]

    eff_cids = [str(clause_ids[k]) if (clause_ids and k < len(clause_ids)) else f"e{k}" for k in range(k_events)]
    eff_txts = [str(clause_texts[k]) if (clause_texts and k < len(clause_texts)) else "" for k in range(k_events)]
    eff_rels = [
        ("start" if k == 0 else str(relations_to_prev[k]))
        if (relations_to_prev and k < len(relations_to_prev))
        else ("start" if k == 0 else "before")
        for k in range(k_events)
    ]
    eff_ords = (
        [int(frame_ordinals[i]) for i in range(m_v)]
        if (frame_ordinals is not None and len(frame_ordinals) == m_v)
        else list(range(m_v))
    )

    left_edges, right_edges = _voronoi_frame_edges(ts, v_dur_sec=v_dur_sec)
    fwd = _solve_exact_segment_dp(
        phi,
        left_edges,
        right_edges,
        eff_rels,
        max_gap_sec=max_gap_sec,
        lambda_bg=lambda_bg,
        lambda_peak=lambda_peak,
        lambda_len=lambda_len,
        lambda_gap=lambda_gap,
        simultaneous_max_delta_sec=simultaneous_max_delta_sec,
    )
    if fwd is None:
        return None
    fwd_total, spans_idx = fwd
    fwd_avg = fwd_total / float(k_events)

    rev_avg = fwd_avg
    has_order_edge = any(r == "before" for r in eff_rels[1:])
    if k_events >= 2 and has_order_edge:
        rev_rels = ["start"] + ["before"] * (k_events - 1)
        rev_sol = _solve_exact_segment_dp(
            phi[::-1],
            left_edges,
            right_edges,
            rev_rels,
            max_gap_sec=max_gap_sec,
            lambda_bg=lambda_bg,
            lambda_peak=lambda_peak,
            lambda_len=lambda_len,
            lambda_gap=lambda_gap,
            simultaneous_max_delta_sec=simultaneous_max_delta_sec,
        )
        if rev_sol is not None:
            rev_avg = float(rev_sol[0]) / float(k_events)

    spans: list[EventSpanPrediction] = []
    all_supported = True
    for k, (a, b) in enumerate(spans_idx):
        sub = phi[k, a : b + 1]
        local_peak_off = int(np.argmax(sub))
        anchor_idx = a + local_peak_off
        st_k = round(float(left_edges[a]), 3)
        et_k = round(max(st_k + 0.05, float(right_edges[b])), 3)
        ev_score = float(np.mean(sub))
        peak_k = float(np.max(phi[k]))
        status = (
            "supported"
            if (peak_k >= float(missing_evidence_threshold) and float(sub[local_peak_off]) >= float(missing_evidence_threshold))
            else "missing"
        )
        if status != "supported":
            all_supported = False
        spans.append(
            EventSpanPrediction(
                clause_id=eff_cids[k],
                clause_text=eff_txts[k],
                relation_to_prev=eff_rels[k],
                start_sec=st_k,
                end_sec=et_k,
                anchor_index=int(anchor_idx),
                anchor_sec=round(float(ts[anchor_idx]), 3),
                start_frame_index=int(eff_ords[a]),
                end_frame_index=int(eff_ords[b]),
                score=ev_score,
                status=status,
            )
        )

    return EventChainLocalizationResult(
        total_score=float(fwd_total),
        avg_score=float(fwd_avg),
        reverse_avg_score=float(rev_avg),
        directional_order_margin=float(fwd_avg - rev_avg),
        spans=tuple(spans),
        overall_start_sec=min(sp.start_sec for sp in spans),
        overall_end_sec=max(sp.end_sec for sp in spans),
        all_supported=all_supported,
    )


def _soft_com_order_concordance(
    timestamps_sec: Sequence[float],
    vis_mat: np.ndarray,
    relations: Sequence[str],
    *,
    tau: float = 0.08,
) -> float:
    """Compute pairwise chronological rank concordance minus inversion rate via softmax centers-of-mass."""
    k_events, m_v = vis_mat.shape
    if k_events < 2 or m_v < 2:
        return 0.0
    ts_arr = np.asarray(timestamps_sec, dtype=np.float64)
    scaled = vis_mat.astype(np.float64) / max(1e-4, float(tau))
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= np.sum(probs, axis=1, keepdims=True)
    com_t = probs @ ts_arr  # shape (k_events,)

    dur = max(1e-3, float(ts_arr[-1] - ts_arr[0]))
    signed_sum = 0.0
    pair_cnt = 0
    for i in range(k_events):
        for j in range(i + 1, k_events):
            # Only enforce strict temporal precedence if no purely-simultaneous block sits between i..j
            if all((relations[r] if r < len(relations) else "before") == "simultaneous" for r in range(i + 1, j + 1)):
                continue
            signed_sum += math.tanh(2.5 * float(com_t[j] - com_t[i]) / dur)
            pair_cnt += 1
    return float(signed_sum / float(pair_cnt)) if pair_cnt > 0 else 0.0


def score_videocomp_order(
    timestamps_sec: Sequence[float],
    event_vis_curves: Sequence[Sequence[float] | np.ndarray],
    *,
    relations_to_prev: Sequence[str] | None = None,
    global_score: float = 0.0,
    max_gap_sec: float = 45.0,
    beta_order: float = 1.0,
    beta_com: float = 0.35,
    beta_min: float = 0.45,
    beta_cov: float = 0.25,
    beta_global: float = 0.50,
    coverage_floor: float = 0.12,
    use_segment_dp: bool = True,
) -> dict[str, float]:
    """3-Axis Compositional Scorer targeting Action Replace, Temporal Reordering & Segment Mismatch."""
    vis_mat = np.asarray(event_vis_curves, dtype=np.float32)
    if vis_mat.ndim == 1:
        vis_mat = vis_mat.reshape(1, -1)
    if vis_mat.size == 0 or len(timestamps_sec) != vis_mat.shape[1]:
        return {
            "event_max_mean": 0.0,
            "min_clause_sim": 0.0,
            "bottom2_clause_mean": 0.0,
            "active_clause_coverage": 0.0,
            "com_concordance": 0.0,
            "forward_avg": 0.0,
            "reverse_avg": 0.0,
            "directional_order_margin": 0.0,
            "global_score": float(global_score),
            "combined_score": float(global_score),
        }

    k_events = int(vis_mat.shape[0])
    eff_rels = [
        ("start" if k == 0 else str(relations_to_prev[k]))
        if (relations_to_prev and k < len(relations_to_prev))
        else ("start" if k == 0 else "before")
        for k in range(k_events)
    ]

    per_clause_max = np.max(vis_mat, axis=1).astype(np.float64)
    sorted_peaks = np.sort(per_clause_max)
    event_max_mean = float(np.mean(per_clause_max))
    min_clause_sim = float(sorted_peaks[0])
    bottom2_clause_mean = float(np.mean(sorted_peaks[: min(2, k_events)]))

    # Soft active coverage across all decomposed clauses (flags mismatched/unobserved clauses in sliced clips)
    active_probs = 1.0 / (1.0 + np.exp(-15.0 * (per_clause_max - float(coverage_floor))))
    active_clause_coverage = float(np.mean(active_probs))

    com_concordance = _soft_com_order_concordance(timestamps_sec, vis_mat, eff_rels)

    chain_res = (
        localize_exact_segment_chain_dp(
            timestamps_sec=timestamps_sec,
            event_vis_curves=vis_mat,
            relations_to_prev=eff_rels,
            max_gap_sec=max_gap_sec,
        )
        if use_segment_dp
        else localize_event_chain_dp(
            timestamps_sec=timestamps_sec,
            event_vis_curves=vis_mat,
            relations_to_prev=eff_rels,
            max_gap_sec=max_gap_sec,
        )
    )
    if chain_res is None:
        fwd_avg = event_max_mean
        rev_avg = event_max_mean
        order_margin = 0.0
    else:
        fwd_avg = float(chain_res.avg_score)
        rev_avg = float(chain_res.reverse_avg_score)
        order_margin = float(chain_res.directional_order_margin)

    comb = (
        float(beta_global) * float(global_score)
        + 0.50 * event_max_mean
        + 0.50 * fwd_avg
        + float(beta_min) * (0.5 * min_clause_sim + 0.5 * bottom2_clause_mean)
        + float(beta_order) * order_margin
        + float(beta_com) * com_concordance
        + float(beta_cov) * active_clause_coverage
    )
    return {
        "event_max_mean": event_max_mean,
        "min_clause_sim": min_clause_sim,
        "bottom2_clause_mean": bottom2_clause_mean,
        "active_clause_coverage": active_clause_coverage,
        "com_concordance": com_concordance,
        "forward_avg": fwd_avg,
        "reverse_avg": rev_avg,
        "directional_order_margin": order_margin,
        "global_score": float(global_score),
        "combined_score": float(comb),
    }
