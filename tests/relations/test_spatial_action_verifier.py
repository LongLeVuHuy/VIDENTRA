"""Tests for P4 SpatialActionVerifier and YOLO single-keyframe deduplication."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.indexing.augmented_vector import deduplicate_and_filter_frame_objects
from videntra.relations.spatial_action_verifier import SpatialActionVerifier


def test_deduplicate_frame_objects_nms_and_thresholds() -> None:
    objs = [
        {"keyframe_id": 10, "class_name": "person", "confidence": 0.90, "bbox": [10, 10, 100, 200]},
        # Duplicate overlapping person box (IoU > 0.65) on same keyframe -> suppressed
        {"keyframe_id": 10, "class_name": "person", "confidence": 0.72, "bbox": [12, 12, 98, 198]},
        # Second distinct person on same keyframe -> kept
        {"keyframe_id": 10, "class_name": "person", "confidence": 0.85, "bbox": [220, 10, 320, 200]},
        # Low-conf standard object (< 0.35) -> dropped
        {"keyframe_id": 10, "class_name": "car", "confidence": 0.29, "bbox": [0, 0, 50, 50]},
        # Small COCO object with conf=0.30 (>= 0.28) -> kept
        {"keyframe_id": 10, "class_name": "cup", "confidence": 0.30, "bbox": [140, 140, 170, 180]},
    ]
    kept = deduplicate_and_filter_frame_objects(objs)
    classes = [k["class_name"] for k in kept]
    assert classes.count("person") == 2
    assert "car" not in classes
    assert classes.count("cup") == 1


def test_spatial_action_verifier_cardinality_spatial_and_order() -> None:
    verifier = SpatialActionVerifier(window_pad_sec=2.0)
    query = "two dogs on the left of a car running then jumping"
    ctx = verifier.prepare_query_context(query, ["two dogs left of a car", "dogs jumping"])

    # Covered matching video: 2 distinct dogs on same keyframe left of car, chronological clauses
    good_objs = [
        {"keyframe_id": 1, "pts_sec": 5.0, "class_name": "dog", "confidence": 0.9, "bbox": [10, 50, 90, 150]},
        {"keyframe_id": 1, "pts_sec": 5.0, "class_name": "dog", "confidence": 0.88, "bbox": [100, 60, 180, 160]},
        {"keyframe_id": 1, "pts_sec": 5.0, "class_name": "car", "confidence": 0.92, "bbox": [300, 40, 500, 180]},
        {"keyframe_id": 2, "pts_sec": 8.0, "class_name": "dog", "confidence": 0.85, "bbox": [40, 40, 120, 140]},
        {"keyframe_id": 3, "pts_sec": 11.0, "class_name": "dog", "confidence": 0.85, "bbox": [240, 40, 320, 140]},
    ]
    good_curve = np.array([[0.85, 0.20, 0.15], [0.10, 0.40, 0.88]], dtype=np.float32)
    res_good = verifier.verify_candidate(
        query_ctx=ctx,
        video_id="v_good",
        window_start=4.0,
        window_end=12.0,
        video_objects=good_objs,
        video_obj_pts=[o["pts_sec"] for o in good_objs],
        detector_ran=True,
        clause_curve=good_curve,
    )
    assert res_good.cardinality_score > 0.5
    assert res_good.spatial_predicate_score > 0.5
    assert res_good.action_order_score > 0.3
    assert res_good.score > 0.4
    assert res_good.status == "supported"

    # Bad video: only 1 dog, dog on the RIGHT of car, reversed action order
    bad_objs = [
        {"keyframe_id": 1, "pts_sec": 5.0, "class_name": "dog", "confidence": 0.9, "bbox": [350, 50, 430, 150]},
        {"keyframe_id": 1, "pts_sec": 5.0, "class_name": "car", "confidence": 0.92, "bbox": [50, 40, 200, 180]},
    ]
    bad_curve = np.array([[0.10, 0.25, 0.85], [0.88, 0.30, 0.15]], dtype=np.float32)
    res_bad = verifier.verify_candidate(
        query_ctx=ctx,
        video_id="v_bad",
        window_start=4.0,
        window_end=12.0,
        video_objects=bad_objs,
        video_obj_pts=[o["pts_sec"] for o in bad_objs],
        detector_ran=True,
        clause_curve=bad_curve,
    )
    assert res_bad.score < 0.0
    assert res_bad.score < res_good.score
    assert res_bad.status in ("partial", "contradicted")

    # Pure contradiction: asks only for two dogs when only 1 dog exists
    res_count_bad = verifier.verify_candidate(
        query_ctx=verifier.prepare_query_context("two dogs", []),
        video_id="v_one_dog",
        window_start=4.0,
        window_end=12.0,
        video_objects=[bad_objs[0]],
        video_obj_pts=[bad_objs[0]["pts_sec"]],
        detector_ran=True,
    )
    assert res_count_bad.cardinality_score < 0.0
    assert res_count_bad.status == "contradicted"

    # Uncovered video with no detector run -> cardinality/spatial remain unknown (0.0), never false negative
    res_uncovered = verifier.verify_candidate(
        query_ctx=verifier.prepare_query_context("two dogs on the left of a car", []),
        video_id="v_uncovered",
        window_start=0.0,
        window_end=5.0,
        video_objects=[],
        video_obj_pts=[],
        detector_ran=False,
    )
    assert res_uncovered.cardinality_score == 0.0
    assert res_uncovered.spatial_predicate_score == 0.0
    assert res_uncovered.status == "unknown"
