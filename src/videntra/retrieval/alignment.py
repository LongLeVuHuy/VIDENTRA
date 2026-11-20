"""Cross-Keyframe Sequential Temporal Alignment & Research Localization Modules."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
