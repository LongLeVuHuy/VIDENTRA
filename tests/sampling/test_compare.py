"""Unit tests for baseline comparison and structural metrics."""
from fractions import Fraction
import pytest

from videntra.ingestion.decode import DecodedFrame
from videntra.sampling.compare import (
    generate_uniform_baseline,
    compute_structural_metrics,
)


def _make_dummy_frame(ordinal: int, t_sec: float) -> DecodedFrame:
    pts = int(round(t_sec * 1000))
    return DecodedFrame(
        ordinal=ordinal,
        pts=pts,
        dts=pts,
        time_base_num=1,
        time_base_den=1000,
        origin=Fraction(0),
        is_keyframe=(ordinal == 0),
        width=64,
        height=48,
    )


def test_uniform_baseline_fixed_rate():
    # 50 frames at 10 fps (0.0 to 4.9s)
    frames = [_make_dummy_frame(i, i * 0.1) for i in range(50)]
    # 1 fps -> 5 frames (0.0, 1.0, 2.0, 3.0, 4.0)
    selected = generate_uniform_baseline(frames, fps=1.0)
    assert len(selected) == 5
    assert selected == [0, 10, 20, 30, 40]


def test_uniform_baseline_matched_budget():
    frames = [_make_dummy_frame(i, i * 0.1) for i in range(50)]
    # Matched budget of 10 frames
    selected = generate_uniform_baseline(frames, budget_count=10)
    assert len(selected) == 10
    assert selected[0] == 0
    assert selected[-1] == 49


def test_structural_metrics_calculation():
    frames = [_make_dummy_frame(i, i * 0.5) for i in range(20)]
    # Select every 2nd frame (interval 1.0s)
    sel = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    metrics = compute_structural_metrics(frames, sel, max_allowed_gap_seconds=2.0, mandatory_count=2)
    assert metrics.selected_count == 10
    assert metrics.mandatory_count == 2
    assert metrics.soft_count == 8
    assert metrics.max_gap_violations == 0
    assert abs(metrics.inter_sample_gap_mean - 1.0) < 1e-4
