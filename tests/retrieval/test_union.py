"""Unit tests for independent branch union and candidate rescue rate."""
import pytest
from videntra.retrieval.branches import BranchCandidate
from videntra.retrieval.union import IndependentBranchUnion


def test_independent_union_preserves_seeds():
    vis = [
        BranchCandidate("v1", "video_A", "visual", 1.0, 3.0, 10, 30, 0.9),
        BranchCandidate("v2", "video_B", "visual", 2.0, 4.0, 20, 40, 0.8),
    ]
    ocr = [
        # video_C is completely missing from visual results!
        BranchCandidate("o1", "video_C", "ocr", 5.0, 6.0, 50, 60, 0.95, matched_text="SALE"),
    ]
    asr = [
        # video_D is also missing from visual results!
        BranchCandidate("a1", "video_D", "asr", 7.0, 9.0, 70, 90, 0.92, matched_text="cảm ơn"),
    ]

    combined = IndependentBranchUnion.collect(vis, ocr, asr)
    assert len(combined) == 4
    vids = set(c.video_id for c in combined)
    assert vids == {"video_A", "video_B", "video_C", "video_D"}


def test_rescue_rate_metric():
    vis_candidates = [
        BranchCandidate(f"v_{i}", f"video_{i}", "visual", 0.0, 2.0, 0, 20, 1.0 - i * 0.1)
        for i in range(10)
    ]

    # Final fused results contain 2 videos from top visual (0, 1) and 2 rescued videos (video_99, video_100)
    fused_vids = ["video_0", "video_1", "video_99", "video_100"]

    rescue_rate = IndependentBranchUnion.compute_rescue_rate(fused_vids, vis_candidates)
    # 2 out of 4 videos were rescued (not in top-5 visual candidates video_0..video_4)
    assert rescue_rate == 0.5
