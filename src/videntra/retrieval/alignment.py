"""Cross-Keyframe Sequential Temporal Alignment for multi-clause video retrieval."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def align_sequential_keyframe_clauses(
    cand_matrix: np.ndarray,
    cand_timestamps: Sequence[float],
    clause_vectors: Sequence[np.ndarray],
    max_transition_sec: float = 35.0,
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
        avg_score = float(np.mean([np.dot(cand_matrix[0], cv) for cv in clause_vectors]))
        return avg_score, (ts[0], ts[0])

    if m_clauses == 1:
        sims = (cand_matrix @ clause_vectors[0]).flatten()
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), (ts[best_idx], ts[best_idx])

    # Compute similarity curves for each clause: shape (M, K_v) via single GEMM
    clause_mat = np.vstack(clause_vectors)
    clause_curves = (cand_matrix @ clause_mat.T).T.astype(np.float32)

    # For 2 clauses: vectorized optimization over valid ordered pairs (a <= b and ts[b] - ts[a] <= max_transition)
    if m_clauses == 2:
        s1 = clause_curves[0]
        s2 = clause_curves[1]
        best_score = -float("inf")
        best_pair = None

        for a in range(k_v):
            t_a = ts[a]
            s_a = s1[a]
            for b in range(a, k_v):
                t_b = ts[b]
                diff = t_b - t_a
                if diff < 0.0 or diff > max_transition_sec:
                    if diff > max_transition_sec:
                        break
                    continue
                # B11: Apply same-frame penalty 0.92 when both clauses map to the exact same keyframe
                base_score = (s_a + s2[b]) / 2.0
                pair_score = 0.92 * base_score if (k_v >= 2 and b == a) else base_score
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (t_a, t_b)

        if best_pair is not None:
            return float(best_score), best_pair

        # Fallback if no pair satisfies max_transition_sec: pick closest temporal pair
        for a in range(k_v):
            for b in range(a, k_v):
                base_score = (s1[a] + s2[b]) / 2.0
                pair_score = 0.92 * base_score if (k_v >= 2 and b == a) else base_score
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (ts[a], ts[b])
        return float(best_score), best_pair

    # General M >= 3: Dynamic Programming across sequential stages
    # dp[m, i] = max sum of scores for clauses 0..m ending at keyframe i
    dp = np.full((m_clauses, k_v), -np.inf, dtype=np.float32)
    parent = np.full((m_clauses, k_v), -1, dtype=np.int32)

    dp[0] = clause_curves[0]

    for m in range(1, m_clauses):
        for i in range(k_v):
            t_curr = ts[i]
            s_curr = clause_curves[m, i]
            for j in range(i + 1):  # j <= i
                diff = t_curr - ts[j]
                if 0.0 <= diff <= max_transition_sec:
                    prev_sum = dp[m - 1, j]
                    if prev_sum != -np.inf:
                        # B11: Apply same-frame penalty 0.92 when step transition stays on same keyframe (j == i)
                        step_score = (0.92 * s_curr) if (k_v >= 2 and j == i) else s_curr
                        cand_sum = prev_sum + step_score
                        if cand_sum > dp[m, i]:
                            dp[m, i] = cand_sum
                            parent[m, i] = j

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
