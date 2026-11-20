"""Tests for P5 localize_moments_multimodal_dp and Soft-NMS secondary proposals."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.localization.refine import (
    localize_moments_multimodal_dp,
    propose_soft_nms_secondary_windows,
)


def _tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(1e-6, max(a[1], b[1]) - min(a[0], b[0]))
    return inter / union


def test_localize_moments_multimodal_dp_shot_snap_and_subclip() -> None:
    # 20 frames spaced every 2.0s (0..38s), shot cut between frame 4 (t=8) and 5 (t=10),
    # and another cut between frame 11 (t=22) and 12 (t=24) -> true event at [10..22]
    n = 20
    ts = [float(i * 2.0) for i in range(n)]
    rng = np.random.default_rng(7)
    v_bg = rng.standard_normal(768).astype(np.float32)
    v_ev = rng.standard_normal(768).astype(np.float32)
    v_bg /= np.linalg.norm(v_bg)
    v_ev /= np.linalg.norm(v_ev)

    vecs = np.tile(v_bg, (n, 1))
    vecs[5:12] = v_ev
    vis_scores = np.array([0.82 if 5 <= i <= 11 else 0.18 for i in range(n)], dtype=np.float32)

    sub_centers = np.array([4.0, 14.0, 22.0, 32.0], dtype=np.float32)
    sub_scores = np.array([0.20, 0.90, 0.72, 0.15], dtype=np.float32)

    st, et, fused = localize_moments_multimodal_dp(
        timestamps=ts,
        visual_scores=vis_scores,
        v_dur=40.0,
        frame_vectors_768=vecs,
        subclip_curve=(sub_centers, sub_scores, None),
        lexical_timestamps=[14.0, 18.0],
        rep_timestamp=16.0,
    )
    assert len(fused) == n
    assert _tiou((st, et), (10.0, 22.0)) >= 0.70


def test_localize_moments_multimodal_dp_viterbi_multiclause_and_soft_nms() -> None:
    ts = [float(i * 2.0) for i in range(24)]
    # Two strong separated peaks: primary at [4..10], secondary at [32..38]
    vis_scores = np.array([
        0.92 if 2 <= i <= 5 else (0.88 if 16 <= i <= 19 else 0.12)
        for i in range(24)
    ], dtype=np.float32)
    clause_curve = np.vstack([
        np.array([0.90 if i in (2, 3) else 0.10 for i in range(24)], dtype=np.float32),
        np.array([0.90 if i in (4, 5) else 0.10 for i in range(24)], dtype=np.float32),
    ])

    st, et, fused = localize_moments_multimodal_dp(
        timestamps=ts,
        visual_scores=vis_scores,
        v_dur=48.0,
        clause_curves=clause_curve,
    )
    assert 2.0 <= st <= 6.0 and 8.0 <= et <= 14.0

    sec = propose_soft_nms_secondary_windows(
        ts=ts,
        curve=fused,
        v_dur=48.0,
        primary_start=st,
        primary_end=et,
        base_ranking_score=0.95,
        iou_thresh=0.30,
    )
    assert len(sec) >= 1
    assert sec[0][0] >= 28.0 and sec[0][1] <= 42.0
    assert sec[0][3] < 0.95


def test_localize_moments_multimodal_dp_dense_curve_suppresses_single_frame_spike() -> None:
    n = 150
    ts = [float(i) for i in range(n)]
    vis_scores = np.full(n, 0.15, dtype=np.float64)
    vis_scores[64:83] = 0.72
    vis_scores[45] = 0.78  # Isolated 1-frame transient spike outside the plateau

    st, et, fused = localize_moments_multimodal_dp(
        timestamps=ts,
        visual_scores=vis_scores,
        v_dur=150.0,
        rep_timestamp=45.0,
    )
    assert len(fused) == n
    assert _tiou((st, et), (64.0, 82.0)) >= 0.55
