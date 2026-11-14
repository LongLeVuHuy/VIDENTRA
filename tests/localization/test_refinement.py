"""Tests for proposal generation and temporal boundary refinement."""
import pytest

from videntra.localization.proposals import generate_proposals
from videntra.localization.refine import refine_moment_boundaries


def test_generate_proposals():
    seeds = [5.0, 10.0]
    ordinals = [150, 300]
    props = generate_proposals(
        video_id="vid_test",
        seed_timestamps=seeds,
        seed_ordinals=ordinals,
        window_duration=4.0,
        max_video_duration=60.0,
    )
    assert len(props) == 2
    assert props[0].start_seconds == 3.0
    assert props[0].end_seconds == 7.0
    assert props[0].duration == 4.0


def test_refine_moment_boundaries_snaps_to_nearest():
    # Shot cuts at 1.8s and 5.2s
    shot_cuts = [1.8, 5.2, 10.0]
    refined_start, refined_end = refine_moment_boundaries(
        moment_start=2.0,
        moment_end=5.0,
        shot_cuts=shot_cuts,
        tolerance=0.4,
    )
    assert refined_start == 1.8
    assert refined_end == 5.2


def test_refine_moment_boundaries_respects_tolerance_and_min_duration():
    # No shot cuts nearby
    shot_cuts = [10.0, 20.0]
    refined_start, refined_end = refine_moment_boundaries(
        moment_start=1.0,
        moment_end=2.0,
        shot_cuts=shot_cuts,
        tolerance=0.3,
        min_duration=0.5,
    )
    # Stays unchanged since distance > tolerance
    assert refined_start == 1.0
    assert refined_end == 2.0


def test_smooth_1d_temporal_scores():
    from videntra.localization.refine import smooth_1d_temporal_scores
    scores = [0.1, 0.9, 0.2]
    smoothed = smooth_1d_temporal_scores(scores)
    assert len(smoothed) == 3
    # Center = 0.25*0.1 + 0.5*0.9 + 0.25*0.2 = 0.025 + 0.45 + 0.05 = 0.525
    assert abs(smoothed[1] - 0.525) < 1e-4


def test_adaptive_temporal_boundary_expansion_region_growing():
    from videntra.localization.refine import adaptive_temporal_boundary_expansion
    # A video with 10 keyframes every 2.0s
    ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    # Event active from 4.0s to 12.0s with a small dip at 8.0s
    scores = [0.1, 0.15, 0.82, 0.88, 0.65, 0.85, 0.79, 0.2, 0.1, 0.05]
    t_start, t_end = adaptive_temporal_boundary_expansion(
        keyframe_timestamps=ts,
        keyframe_scores=scores,
        video_duration=20.0,
        max_gap_seconds=4.0,
    )
    # The event should start around frame index 2 (4.0s) and end around index 6 (12.0s)
    # With half-step padding (approx 1.0s), start <= 3.5s and end >= 12.5s
    assert t_start <= 4.0
    assert t_end >= 12.0
    assert t_end - t_start >= 8.0


def test_adaptive_boundary_refiner_class():
    from videntra.localization.refine import AdaptiveBoundaryRefiner
    refiner = AdaptiveBoundaryRefiner()
    ts = [0.0, 2.0, 4.0, 6.0, 8.0]
    scores = [0.2, 0.85, 0.9, 0.2, 0.1]
    st, et = refiner.refine_moment(
        video_id="v1",
        keyframe_timestamps=ts,
        keyframe_scores=scores,
        video_duration=10.0,
    )
    assert st >= 0.0
    assert et > st
    assert et <= 10.0


def test_adaptive_boundary_expansion_stricter_threshold():
    from videntra.localization.refine import adaptive_temporal_boundary_expansion
    # Main peak at 4.0s (score 0.95), minor ripple at 14.0s (score 0.45)
    ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    scores = [0.1, 0.70, 0.95, 0.72, 0.15, 0.15, 0.20, 0.45, 0.10]

    # Stricter ratio (0.58): theta_keep >= 0.58 * 0.95 = 0.551
    # Minor ripple at 14.0s (0.45) is below theta_keep, so expansion terminates at main event
    st_strict, et_strict = adaptive_temporal_boundary_expansion(
        keyframe_timestamps=ts,
        keyframe_scores=scores,
        video_duration=20.0,
        min_relative_peak_ratio=0.58,
        max_gap_seconds=3.0,
    )
    assert et_strict <= 8.0


def test_temporal_affinity_decay_weighting():
    import numpy as np
    from videntra.localization.refine import compute_temporal_affinity_decay, adaptive_temporal_boundary_expansion

    # 1. Direct mathematical test
    decay_val = compute_temporal_affinity_decay(delta_seconds=10.0, decay_lambda=0.02)
    expected = float(np.exp(-0.02 * 10.0))
    assert abs(decay_val - expected) < 1e-5

    # 2. Integration test: distant keyframe (50s away) with moderate raw score
    # Peak at 10.0s (0.90), distant frame at 60.0s (0.65)
    ts = [0.0, 5.0, 10.0, 15.0, 20.0, 60.0]
    scores = [0.2, 0.85, 0.90, 0.85, 0.2, 0.65]

    st, et = adaptive_temporal_boundary_expansion(
        keyframe_timestamps=ts,
        keyframe_scores=scores,
        video_duration=100.0,
        decay_lambda=0.02,
        max_gap_seconds=50.0,  # Even with large gap allowance
    )
    # The decayed score of frame at 60s is 0.65 * exp(-0.02 * 50) = 0.65 * 0.368 = 0.239
    # which is well below theta_keep (~0.58 * 0.9 = 0.522), so it is not included
    assert et < 30.0


def test_max_moment_duration_cap():
    from videntra.localization.refine import adaptive_temporal_boundary_expansion

    # Video duration = 100.0s, high scores spanning 80 seconds (ts 10.0 to 90.0)
    ts = list(range(0, 105, 5))  # 0, 5, ..., 100
    # Frames from 10.0s to 90.0s have high scores
    scores = [0.1 if (t < 10 or t > 90) else 0.88 for t in ts]

    st, et = adaptive_temporal_boundary_expansion(
        keyframe_timestamps=ts,
        keyframe_scores=scores,
        video_duration=100.0,
        max_duration_ratio=0.65,
    )
    # Duration must be capped to <= 65% of video duration (<= 65.0s)
    dur = et - st
    assert dur <= 65.01

    # Exception: when variance is extremely low (uniform high score across whole video)
    uniform_scores = [0.90] * len(ts)
    st_u, et_u = adaptive_temporal_boundary_expansion(
        keyframe_timestamps=ts,
        keyframe_scores=uniform_scores,
        video_duration=100.0,
        max_duration_ratio=0.65,
    )
    # Uniform video is not artificially truncated by duration cap
    assert (et_u - st_u) > 65.0

