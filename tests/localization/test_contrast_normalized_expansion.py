"""Comprehensive unit tests for Contrast-Normalized Dynamic Temporal Boundary Expansion.

Tests:
- Real SigLIP 2 Mode (s_max <= 0.45 and mu > 0.08):
  - High positive baseline curve (mu approx 0.18, sigma approx 0.015, s_max approx 0.20):
    expands across full 30s-80s event without collapsing to 6-12s stubs.
- Uniform Long-Event Mode (sigma / mu <= 0.075 or s_max - mu <= 0.016):
  - Expands up to 98% of video duration without artificial 65% capping.
- Backward Compatibility in Synthetic Mode (s_max > 0.45):
  - Preserves legacy hysteresis thresholding, 0.02 decay, and 65% duration capping.
- Edge cases and invariants:
  - Empty inputs, single keyframe, all-zero/negative scores, half-step padding.
"""
import math
import numpy as np
import pytest

from videntra.localization.refine import (
    adaptive_temporal_boundary_expansion,
    smooth_1d_temporal_scores,
    compute_temporal_affinity_decay,
)


class TestContrastNormalizedExpansion:
    """Tests for adaptive_temporal_boundary_expansion with contrast normalization."""

    def test_real_siglip2_high_baseline_expansion(self):
        """Verifies real SigLIP 2 cosine curve (s_max <= 0.45, mu > 0.08) expands across 30-80s.
        
        Under legacy absolute thresholding, high baseline curves (mu ~ 0.18, peak ~ 0.21)
        rapidly choked down to 6-12s because decay_lambda=0.02 decayed the peak quickly.
        With contrast normalization and gentle decay lambda=0.004, the event expands
        across the full genuine duration (e.g. 50s).
        """
        # 120s video with keyframes every 2.0s (61 keyframes)
        v_dur = 120.0
        ts = [float(i * 2.0) for i in range(61)]  # 0.0, 2.0, ..., 120.0

        # Background cosine similarity around 0.14
        # Event active from 25.0s to 85.0s (60s event) with scores around 0.20..0.22
        scores = []
        for t in ts:
            if 25.0 <= t <= 85.0:
                # Active event frames: 0.20 to 0.22
                scores.append(0.20 + 0.02 * math.sin((t - 25.0) / 60.0 * math.pi))
            else:
                # Background frames: 0.13 to 0.15
                scores.append(0.14)

        # Check properties of this realistic curve
        s_max = max(scores)
        mu = float(np.mean(scores))
        sigma = float(np.std(scores))

        assert s_max <= 0.45, f"s_max {s_max} should be <= 0.45 for real SigLIP 2 mode"
        assert mu > 0.08, f"mu {mu} should be > 0.08"
        assert sigma / mu > 0.075, "Non-uniform mode"

        t_start, t_end = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=v_dur,
        )

        moment_duration = t_end - t_start

        # The expansion must cover the genuine 60s event (e.g. from ~24s to ~86s)
        # and MUST NOT be strangled down to a short stub (6s - 12s)
        assert moment_duration >= 50.0, f"Expected expanded duration >= 50.0s, got {moment_duration}s"
        assert t_start <= 26.0
        assert t_end >= 84.0

    def test_uniform_long_event_mode_expands_to_98_percent(self):
        """Verifies uniform long-event mode (sigma / mu <= 0.075) expands up to 98% of duration."""
        # 100s video with keyframes every 2.0s
        v_dur = 100.0
        ts = [float(i * 2.0) for i in range(51)]  # 0.0, 2.0, ..., 100.0

        # Continuous activity across the entire video with minimal variance: mu ~ 0.22, sigma ~ 0.0014
        # with a natural peak at the center (e.g. video of continuous cooking or dancing)
        # sigma / mu <= 0.075 and s_max - mu <= 0.016
        scores = [0.22] * len(ts)
        scores[len(ts) // 2] = 0.23  # Natural center peak

        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        s_max = max(scores)

        assert s_max <= 0.45
        assert mu > 0.08
        assert (sigma / mu <= 0.075) or ((s_max - mu) <= 0.016)

        t_start, t_end = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=v_dur,
        )

        moment_dur = t_end - t_start
        # In legacy mode, max_duration_ratio=0.65 would limit duration to 65.0s.
        # Uniform mode expands up to 98% - 100% of video duration (>= 90.0s)
        assert moment_dur >= 90.0, f"Uniform mode should expand >= 90s, got {moment_dur}s"
        assert t_start <= 4.0
        assert t_end >= 96.0

    def test_backward_compatibility_synthetic_mode(self):
        """Verifies s_max > 0.45 executes legacy Mode B (hysteresis thresholding & 65% duration cap)."""
        v_dur = 100.0
        ts = [float(i * 5.0) for i in range(21)]  # 0, 5, ..., 100

        # Synthetic curve: peak at t=25.0s (s=0.92 > 0.45), background is 0.10
        scores = [0.10] * len(ts)
        # Event active only from t=20 to t=30
        scores[4] = 0.75  # t=20.0
        scores[5] = 0.92  # t=25.0 (peak)
        scores[6] = 0.72  # t=30.0

        t_start, t_end = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=v_dur,
            min_relative_peak_ratio=0.58,
            max_duration_ratio=0.65,
        )

        # In synthetic mode, high peak is tightly localized around t=20..30
        assert t_start >= 15.0
        assert t_end <= 35.0
        assert (t_end - t_start) <= 20.0

    def test_backward_compatibility_synthetic_duration_cap(self):
        """Verifies synthetic mode caps non-uniform events at max_duration_ratio = 0.65."""
        v_dur = 100.0
        ts = [float(i * 2.0) for i in range(51)]  # 0 to 100s

        # Synthetic high scores spanning 80% of video (t=10 to t=90)
        # With s_max = 0.88 > 0.45
        scores = [0.10 if (t < 10.0 or t > 90.0) else 0.88 for t in ts]

        t_start, t_end = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=v_dur,
            max_duration_ratio=0.65,
        )

        moment_dur = t_end - t_start
        # Cap at 65% of 100s = 65.0s (+ epsilon for rounding)
        assert moment_dur <= 65.01

    def test_edge_case_empty_inputs(self):
        """Verifies empty inputs return default window [0.0, min(v_dur, 4.0)]."""
        st, et = adaptive_temporal_boundary_expansion([], [], video_duration=50.0)
        assert st == 0.0
        assert et == 4.0

        st_short, et_short = adaptive_temporal_boundary_expansion([], [], video_duration=2.5)
        assert st_short == 0.0
        assert et_short == 2.5

    def test_edge_case_single_keyframe(self):
        """Verifies single keyframe centers a minimum window around ts[0]."""
        st, et = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=[20.0],
            keyframe_scores=[0.85],
            video_duration=100.0,
        )
        # Delta = max(4.0, 0.12 * 100) = 12.0s
        # Centered at 20.0: [14.0, 26.0]
        assert abs(st - 14.0) < 1e-4
        assert abs(et - 26.0) < 1e-4
        assert abs((et - st) - 12.0) < 1e-4

    def test_edge_case_all_negative_or_zero_scores(self):
        """Verifies scores <= 0.0 gracefully returns centered window around peak."""
        ts = [0.0, 10.0, 20.0, 30.0]
        scores = [-0.5, 0.0, -0.2, -0.8]  # peak is at index 1 (t=10.0, s=0.0)

        st, et = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=50.0,
        )
        assert st >= 0.0
        assert et <= 50.0
        assert et > st

    def test_video_duration_none_defaults_to_max_ts_plus_two(self):
        """Verifies video_duration=None derives duration from max keyframe timestamp."""
        ts = [5.0, 15.0, 30.0]
        scores = [0.2, 0.9, 0.3]

        st, et = adaptive_temporal_boundary_expansion(
            keyframe_timestamps=ts,
            keyframe_scores=scores,
            video_duration=None,
        )
        # Derived v_dur = 30.0 + 2.0 = 32.0
        assert et <= 32.0
        assert st >= 0.0
