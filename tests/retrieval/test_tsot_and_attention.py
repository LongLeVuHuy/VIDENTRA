"""Unit tests for Temporal Sinkhorn Optimal Transport (T-SOT) and Temporal Attention Pooler."""
from __future__ import annotations

import numpy as np
import pytest

from videntra.retrieval.alignment import (
    query_guided_temporal_attention_score,
    temporal_sinkhorn_alignment_score,
)


def test_tsot_prefers_distinct_chronological_over_single_frame_peak() -> None:
    # 2 clauses, 5 frames:
    # Ordered distinct: clause 0 peaks at t=0, clause 1 peaks at t=4.
    ordered = np.array(
        [
            [0.90, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.90],
        ],
        dtype=np.float32,
    )
    # Single-frame collapse: both clauses peak at the exact same frame t=0.
    same_frame = np.array(
        [
            [0.90, 0.10, 0.10, 0.10, 0.10],
            [0.90, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    s_ordered = temporal_sinkhorn_alignment_score(ordered)
    s_same = temporal_sinkhorn_alignment_score(same_frame)
    assert s_ordered > s_same


def test_tsot_prefers_chronological_over_reversed_order() -> None:
    # Clause 0 peaks early, clause 1 peaks late.
    ordered = np.array(
        [
            [0.35, 0.15, 0.10, 0.10],
            [0.10, 0.10, 0.15, 0.35],
        ],
        dtype=np.float32,
    )
    # Reversed order: clause 0 peaks late, clause 1 peaks early.
    reversed_order = np.array(
        [
            [0.10, 0.10, 0.15, 0.35],
            [0.35, 0.15, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    s_ordered = temporal_sinkhorn_alignment_score(ordered, eta=0.15, epsilon=0.08, n_iters=3)
    s_reversed = temporal_sinkhorn_alignment_score(
        reversed_order, eta=0.15, epsilon=0.08, n_iters=3
    )
    assert s_ordered > s_reversed


def test_tsot_three_clauses_chronological_and_stability() -> None:
    ordered_3 = np.array(
        [
            [0.85, 0.10, 0.05, 0.05, 0.05],
            [0.05, 0.10, 0.85, 0.10, 0.05],
            [0.05, 0.05, 0.05, 0.10, 0.85],
        ],
        dtype=np.float32,
    )
    collapsed_3 = np.array(
        [
            [0.85, 0.10, 0.05, 0.05, 0.05],
            [0.85, 0.10, 0.05, 0.05, 0.05],
            [0.85, 0.10, 0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )

    s_ord = temporal_sinkhorn_alignment_score(ordered_3, epsilon=0.01, n_iters=5)
    s_col = temporal_sinkhorn_alignment_score(collapsed_3, epsilon=0.01, n_iters=5)
    assert np.isfinite(s_ord)
    assert np.isfinite(s_col)
    assert s_ord > s_col


def test_tsot_edge_cases() -> None:
    assert temporal_sinkhorn_alignment_score(np.zeros((0, 5))) == 0.0
    assert temporal_sinkhorn_alignment_score(np.zeros((2, 0))) == 0.0
    assert temporal_sinkhorn_alignment_score(None) == 0.0  # type: ignore[arg-type]

    # 1D array treated as 1 clause across T frames
    assert temporal_sinkhorn_alignment_score(np.array([0.2, 0.7, 0.4])) == pytest.approx(0.7)
    # 1 clause (2D)
    assert temporal_sinkhorn_alignment_score(np.array([[0.2, 0.7, 0.4]])) == pytest.approx(0.7)
    # 1 frame (M, 1)
    assert temporal_sinkhorn_alignment_score(np.array([[0.2], [0.8]])) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        temporal_sinkhorn_alignment_score(np.zeros((2, 2, 2)))


def test_tsot_partial_ot_event_present_vs_no_event_acceptance() -> None:
    # 2 clauses, 20 frames over [0, 95]s: event present at t=20s & t=30s (within max_gap_sec=35)
    ts = np.linspace(0.0, 95.0, 20, dtype=np.float64)
    event_present = np.full((2, 20), 0.10, dtype=np.float32)
    event_present[0, 4] = 0.85  # t=20s
    event_present[1, 6] = 0.85  # t=30s

    no_event = np.full((2, 20), 0.10, dtype=np.float32)

    s_present = temporal_sinkhorn_alignment_score(
        event_present, timestamps_sec=ts, max_gap_sec=35.0, epsilon=0.02, n_iters=20
    )
    s_absent = temporal_sinkhorn_alignment_score(
        no_event, timestamps_sec=ts, max_gap_sec=35.0, epsilon=0.02, n_iters=20
    )
    mean_of_max_gap = float(np.mean(np.max(event_present, axis=1))) - float(
        np.mean(np.max(no_event, axis=1))
    )
    assert (s_present - s_absent) >= 0.5 * mean_of_max_gap


def test_tsot_real_timestamp_gaps_matter() -> None:
    # Uneven timestamps: frame 1 is at 1.0s (near t=0) vs 99.0s (far from t=0)
    sims = np.array(
        [
            [0.80, 0.20, 0.20],
            [0.20, 0.80, 0.20],
        ],
        dtype=np.float32,
    )
    tight_ts = [0.0, 1.0, 100.0]
    wide_ts = [0.0, 99.0, 100.0]
    s_tight = temporal_sinkhorn_alignment_score(
        sims, timestamps_sec=tight_ts, max_gap_sec=120.0, eta=0.15, epsilon=0.02, n_iters=20
    )
    s_wide = temporal_sinkhorn_alignment_score(
        sims, timestamps_sec=wide_ts, max_gap_sec=120.0, eta=0.15, epsilon=0.02, n_iters=20
    )
    # In wide_ts, frame 1 is at 0.99 of span (matches clause 1 at 1.0 better than 0.01 in tight_ts)
    assert s_wide > s_tight


def test_attention_pooler_penalizes_dispersion_and_rewards_coherence() -> None:
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Coherent frames: both have cos(q, v_k) = 0.6 and point in the exact same direction.
    coherent = np.array(
        [
            [0.6, 0.8, 0.0],
            [0.6, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    # Diverse / scene-cut frames: equal per-frame cos(q, v_k) = 0.6, but orthogonal components oppose.
    diverse = np.array(
        [
            [0.6, 0.8, 0.0],
            [0.6, -0.8, 0.0],
        ],
        dtype=np.float32,
    )

    s_coherent = query_guided_temporal_attention_score(q, coherent, tau=12.0, disp_penalty=0.15)
    s_diverse = query_guided_temporal_attention_score(q, diverse, tau=12.0, disp_penalty=0.15)

    assert s_coherent == pytest.approx(0.6, abs=1e-5)
    assert s_diverse == pytest.approx(0.6 - 0.15 * (1.0 - 0.6), abs=1e-5)
    assert s_coherent >= s_diverse
    assert s_coherent > s_diverse


def test_attention_pooler_edge_cases() -> None:
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Empty / None
    assert query_guided_temporal_attention_score(np.zeros(0), np.ones((2, 3))) == 0.0
    assert query_guided_temporal_attention_score(q, np.zeros((0, 3))) == 0.0
    assert query_guided_temporal_attention_score(None, np.ones((2, 3))) == 0.0  # type: ignore[arg-type]

    # Zero query or zero frames
    assert query_guided_temporal_attention_score(np.zeros(3), np.ones((2, 3))) == 0.0
    assert query_guided_temporal_attention_score(q, np.zeros((3, 3))) == 0.0

    # Single frame (2D and 1D)
    f_1_2d = np.array([[0.6, 0.8, 0.0]], dtype=np.float32)
    f_1_1d = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    assert query_guided_temporal_attention_score(q, f_1_2d) == pytest.approx(0.6, abs=1e-5)
    assert query_guided_temporal_attention_score(q, f_1_1d) == pytest.approx(0.6, abs=1e-5)

    # Unnormalized inputs are safely normalized
    assert query_guided_temporal_attention_score(2.0 * q, 5.0 * f_1_2d) == pytest.approx(
        0.6, abs=1e-5
    )

    with pytest.raises(ValueError):
        query_guided_temporal_attention_score(q, np.ones((2, 4)))
