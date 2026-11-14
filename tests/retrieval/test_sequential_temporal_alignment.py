"""Comprehensive unit tests for Cross-Keyframe Sequential Temporal Alignment.

Tests:
- Sequential temporal order: C_1 -> C_2 with t_a <= t_b <= t_a + 35.0s gives (S_1 + S_2)/2
- Single keyframe consistency K_v=1: averages all clause vectors, matching K_v=2 baseline
- Time inversion prevention: rejecting reverse order t_a > t_b even on unsorted input
- Temporal gap exceeding 35.0s: filtering out gaps > 35.0s in favor of valid transitions
- Multi-clause Dynamic Programming (M >= 3): ordered chaining C_1 -> C_2 -> C_3 and traceback
- Single-clause degenerate case (M = 1)
- Edge cases: empty candidates, empty clauses, fallback when no pair satisfies threshold
"""
import numpy as np
import pytest

from videntra.retrieval.alignment import align_sequential_keyframe_clauses


class TestSequentialTemporalAlignment:
    """Tests for align_sequential_keyframe_clauses function."""

    def test_sequential_two_clause_happy_path(self):
        """Verifies C_1 -> C_2 with t_a <= t_b <= t_a + 35.0s gives max average score."""
        dim = 4
        # 4 keyframes at t = 0.0, 10.0, 25.0, 70.0
        ts = [0.0, 10.0, 25.0, 70.0]

        # Keyframe representations
        # kf 1 (10.0s) is aligned with c1 [1, 0, 0, 0]
        # kf 2 (25.0s) is aligned with c2 [0, 1, 0, 0]
        cand_matrix = np.array([
            [0.1, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # t=10.0: c1 score = 0.9
            [0.1, 0.8, 0.0, 0.0],  # t=25.0: c2 score = 0.8
            [0.0, 0.2, 0.0, 0.0],  # t=70.0: c2 score = 0.2
        ], dtype=np.float32)

        c1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts,
            clause_vectors=[c1, c2],
            max_transition_sec=35.0,
        )

        expected_score = (0.9 + 0.8) / 2.0  # 0.85
        assert abs(score - expected_score) < 1e-5
        assert anchor_window == (10.0, 25.0)

    def test_single_keyframe_consistency_kv_1(self):
        """Verifies K_v=1 calculates the average across ALL clause vectors, consistent with K_v=2."""
        dim = 2
        # Single keyframe matching c1 fully (1.0) and c2 not at all (0.0)
        cand_matrix_1 = np.array([[1.0, 0.0]], dtype=np.float32)
        ts_1 = [10.0]

        # Two identical keyframes separated by a tiny epsilon
        cand_matrix_2 = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ts_2 = [10.0, 10.001]

        c1 = np.array([1.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0], dtype=np.float32)

        score_1, win_1 = align_sequential_keyframe_clauses(cand_matrix_1, ts_1, [c1, c2])
        score_2, win_2 = align_sequential_keyframe_clauses(cand_matrix_2, ts_2, [c1, c2])

        # K_v=1 score must average all clause similarities: (1.0 + 0.0) / 2 = 0.5
        assert abs(score_1 - 0.5) < 1e-5
        # K_v=1 score must equal K_v=2 score exactly
        assert abs(score_1 - score_2) < 1e-5
        # Anchor window for K_v=1 is anchored at the single keyframe timestamp
        assert win_1 == (10.0, 10.0)

    def test_time_inversion_prevention_on_unsorted_timestamps(self):
        """Verifies that reverse-order transitions (t_a > t_b) are never accepted."""
        cand_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # Unsorted: keyframe 0 is at 20.0s, keyframe 1 is at 2.0s
        ts_unsorted = [20.0, 2.0]

        c1 = np.array([1.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts_unsorted,
            clause_vectors=[c1, c2],
            max_transition_sec=35.0,
        )

        # Reverse window (20.0, 2.0) MUST NOT be selected
        assert anchor_window != (20.0, 2.0)
        assert anchor_window[0] <= anchor_window[1]

    def test_temporal_gap_exceeding_max_transition(self):
        """Verifies that pairs with gap > 35.0s are rejected in favor of valid transitions."""
        # Timestamps: 10.0s, 30.0s, 70.0s
        ts = [10.0, 30.0, 70.0]

        # Keyframe 0 (10.0s) matches c1 strongly (0.8)
        # Keyframe 1 (30.0s) matches c2 moderately (0.7) -> valid pair (10.0, 30.0), diff=20s <= 35s, score=0.75
        # Keyframe 2 (70.0s) matches c2 strongly (0.95) -> pair (10.0, 70.0), diff=60s > 35s (exceeds gap!)
        cand_matrix = np.array([
            [0.8, 0.0],
            [0.1, 0.7],
            [0.0, 0.95],
        ], dtype=np.float32)

        c1 = np.array([1.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts,
            clause_vectors=[c1, c2],
            max_transition_sec=35.0,
        )

        # Must pick (10.0, 30.0) with score (0.8 + 0.7)/2 = 0.75, NOT (10.0, 70.0)
        assert anchor_window == (10.0, 30.0)
        assert abs(score - 0.75) < 1e-5

    def test_fallback_when_no_pair_satisfies_max_transition(self):
        """Verifies graceful fallback when all keyframe pairs exceed max_transition_sec."""
        # Gaps are all 50s > 35.0s
        ts = [0.0, 50.0, 100.0]
        cand_matrix = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.8],
        ], dtype=np.float32)

        c1 = np.array([1.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts,
            clause_vectors=[c1, c2],
            max_transition_sec=35.0,
        )

        # Fallback should select the best available pair without crashing
        assert anchor_window is not None
        assert anchor_window[0] <= anchor_window[1]
        assert score > 0.0

    def test_multi_clause_dynamic_programming_three_clauses(self):
        """Verifies M=3 sequential stages C_1 -> C_2 -> C_3 via dynamic programming."""
        ts = [5.0, 15.0, 25.0, 40.0]
        # Clause 1 peaks at t=5.0 (score 0.9)
        # Clause 2 peaks at t=15.0 (score 0.8)
        # Clause 3 peaks at t=25.0 (score 0.85)
        # All pairwise transitions: 15 - 5 = 10 <= 35, 25 - 15 = 10 <= 35
        cand_matrix = np.array([
            [0.9, 0.1, 0.0],   # t=5.0
            [0.1, 0.8, 0.1],   # t=15.0
            [0.0, 0.1, 0.85],  # t=25.0
            [0.1, 0.1, 0.2],   # t=40.0
        ], dtype=np.float32)

        c1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        c3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts,
            clause_vectors=[c1, c2, c3],
            max_transition_sec=35.0,
        )

        expected_avg_score = (0.9 + 0.8 + 0.85) / 3.0  # 0.85
        assert abs(score - expected_avg_score) < 1e-4
        assert anchor_window == (5.0, 25.0)

    def test_single_clause_degenerate_case(self):
        """Verifies M=1 clause returns highest similarity keyframe and point window."""
        ts = [2.0, 6.0, 12.0]
        cand_matrix = np.array([
            [0.3],
            [0.85],
            [0.4],
        ], dtype=np.float32)
        c1 = np.array([1.0], dtype=np.float32)

        score, anchor_window = align_sequential_keyframe_clauses(
            cand_matrix=cand_matrix,
            cand_timestamps=ts,
            clause_vectors=[c1],
        )

        assert abs(score - 0.85) < 1e-5
        assert anchor_window == (6.0, 6.0)

    def test_empty_candidates_or_clauses_returns_zero(self):
        """Verifies empty cand_matrix or empty clause_vectors safely returns (0.0, None)."""
        c1 = np.array([1.0, 0.0], dtype=np.float32)

        # Empty matrix
        score, win = align_sequential_keyframe_clauses(np.zeros((0, 2)), [], [c1])
        assert score == 0.0
        assert win is None

        # None matrix
        score, win = align_sequential_keyframe_clauses(None, [], [c1])
        assert score == 0.0
        assert win is None

        # Empty clauses
        score, win = align_sequential_keyframe_clauses(np.ones((2, 2)), [0.0, 1.0], [])
        assert score == 0.0
        assert win is None

    def test_b11_same_frame_penalty_two_clauses(self):
        """Verifies B11: When K_v >= 2, pairs where b == a receive a 0.92 penalty factor."""
        c1 = np.array([1.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0], dtype=np.float32)

        # Video with 2 keyframes:
        # kf 0 (10.0s) matches c1 (0.8) and c2 (0.8) -> same-frame score = 0.92 * (0.8 + 0.8) / 2 = 0.736
        # kf 1 (20.0s) matches c2 (0.75) -> genuine transition (10.0, 20.0) score = (0.8 + 0.75) / 2 = 0.775
        cand_matrix = np.array([
            [0.8, 0.8],   # t=10.0
            [0.1, 0.75],  # t=20.0
        ], dtype=np.float32)
        ts = [10.0, 20.0]

        score, win = align_sequential_keyframe_clauses(cand_matrix, ts, [c1, c2])

        # Without 0.92 penalty, same-frame (10.0, 10.0) would score 0.80 > 0.775.
        # With 0.92 penalty, same-frame scores 0.736 < 0.775, so genuine transition (10.0, 20.0) wins!
        assert win == (10.0, 20.0)
        assert abs(score - 0.775) < 1e-5

        # If only a static frame is available in a 2-keyframe video:
        cand_matrix_static = np.array([
            [0.8, 0.8],
            [0.0, 0.0],
        ], dtype=np.float32)
        score_static, win_static = align_sequential_keyframe_clauses(cand_matrix_static, ts, [c1, c2])
        # Receives exactly 0.92 * (0.8 + 0.8) / 2.0 = 0.736
        assert win_static == (10.0, 10.0)
        assert abs(score_static - 0.736) < 1e-5

    def test_b11_same_frame_penalty_dp_three_clauses(self):
        """Verifies B11: DP with M=3 applies 0.92 penalty when transitions stay on the same frame."""
        c1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        c3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 3 keyframes
        # kf 0 (5.0s): c1=0.8, c2=0.8, c3=0.8
        # kf 1 (15.0s): c1=0.2, c2=0.75, c3=0.1
        # kf 2 (25.0s): c1=0.1, c2=0.1, c3=0.75
        cand_matrix = np.array([
            [0.8, 0.8, 0.8],
            [0.2, 0.75, 0.1],
            [0.1, 0.1, 0.75],
        ], dtype=np.float32)
        ts = [5.0, 15.0, 25.0]

        score, win = align_sequential_keyframe_clauses(cand_matrix, ts, [c1, c2, c3])
        # Genuine transition (5.0, 25.0) scores (0.8 + 0.75 + 0.75) / 3 = 0.7667
        # Static transition on kf 0 scores (0.8 + 0.92*0.8 + 0.92*0.8) / 3 = 2.272 / 3 = 0.7573
        assert win == (5.0, 25.0)
        assert abs(score - (0.8 + 0.75 + 0.75) / 3.0) < 1e-4
