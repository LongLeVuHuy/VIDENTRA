"""Tests for Reciprocal Rank Fusion, temporal clustering, and Result schema parity."""
import pytest
from videntra.contracts import validate
from videntra.retrieval.branches import BranchCandidate
from videntra.retrieval.fusion import MultimodalMomentFusion


def test_rrf_multimodal_synergy():
    fusion = MultimodalMomentFusion(k_rrf=60)

    # Video 1 has all 3 modalities agreeing around t = 10.0s - 12.0s
    vis_1 = BranchCandidate("v_1", "vid_1", "visual", 10.0, 12.0, 100, 120, 0.95)
    ocr_1 = BranchCandidate("o_1", "vid_1", "ocr", 10.2, 11.8, 102, 118, 0.90, matched_text="SALE")
    asr_1 = BranchCandidate("a_1", "vid_1", "asr", 10.5, 12.5, 105, 125, 0.88, matched_text="giảm giá")

    # Video 2 has only Visual match at rank 1, but no OCR/ASR
    vis_2 = BranchCandidate("v_2", "vid_2", "visual", 5.0, 7.0, 50, 70, 0.99)

    moments = fusion.fuse(
        visual_candidates=[vis_2, vis_1],  # vis_2 is rank 1, vis_1 is rank 2
        ocr_candidates=[ocr_1],
        asr_candidates=[asr_1],
        top_k=5,
    )

    assert len(moments) == 2
    # Video 1 should rank higher because of multi-branch RRF synergy (Visual + OCR + ASR)
    top_moment = moments[0]
    assert top_moment.video_id == "vid_1"
    assert top_moment.match_status == "supported"
    assert top_moment.ranking_score > moments[1].ranking_score
    assert moments[1].video_id == "vid_2"
    assert moments[1].match_status == "partial"


def test_fusion_temporal_separation_no_overmerge():
    fusion = MultimodalMomentFusion(k_rrf=60, temporal_iou_threshold=0.3)

    # In the same video, candidate 1 is at t = 2.0s, candidate 2 is at t = 20.0s
    c1 = BranchCandidate("c1", "vid_A", "visual", 2.0, 4.0, 20, 40, 0.8)
    c2 = BranchCandidate("c2", "vid_A", "ocr", 20.0, 22.0, 200, 220, 0.9, matched_text="EXIT")

    moments = fusion.fuse([c1], [c2], [])
    # Must produce 2 separate moments for vid_A, NOT merge them into [2.0s, 22.0s]!
    assert len(moments) == 2
    times = sorted([m.moment_start_seconds for m in moments])
    assert times[0] == 2.0
    assert times[1] == 20.0


def test_fused_moment_result_contract_validation():
    fusion = MultimodalMomentFusion()
    c = BranchCandidate("c1", "vid_test", "visual", 1.0, 3.0, 10, 30, 0.85)
    moments = fusion.fuse([c], [], [], top_k=1)
    assert len(moments) == 1

    res_dict = moments[0].to_result_dict()
    # Contract validation must pass with zero schema errors
    validate("result", res_dict)


def test_aggregate_video_moments_deduplication():
    from videntra.retrieval.fusion import aggregate_video_moments, RankedMoment

    m1 = RankedMoment("m1", "vid_1", 2.0, 4.0, 20, 40, 20, 2.0, 0.90)
    m2 = RankedMoment("m2", "vid_1", 6.0, 8.0, 60, 80, 60, 6.0, 0.85)  # Adjacent to m1
    m3 = RankedMoment("m3", "vid_2", 1.0, 3.0, 10, 30, 10, 1.0, 0.88)
    m4 = RankedMoment("m4", "vid_2", 20.0, 22.0, 200, 220, 200, 20.0, 0.50)

    # 4 moments across 2 videos
    fused = aggregate_video_moments([m1, m2, m3, m4], top_k=5)
    # Output must have exactly 2 distinct video_ids (deduplicated)
    assert len(fused) == 2
    video_ids = [m.video_id for m in fused]
    assert set(video_ids) == {"vid_1", "vid_2"}
    # vid_1 has density bonus because m1 and m2 are adjacent within 6.0s
    assert fused[0].video_id == "vid_1"


def test_aggregate_video_moments_negative_score_safety_and_custom_factor():
    from videntra.retrieval.fusion import aggregate_video_moments, RankedMoment

    # Negative scores: density bonus must not be negative
    m_neg1 = RankedMoment("n1", "vid_neg", 1.0, 3.0, 10, 30, 10, 1.0, -0.5)
    m_neg2 = RankedMoment("n2", "vid_neg", 4.0, 6.0, 40, 60, 40, 4.0, -0.6)
    fused_neg = aggregate_video_moments([m_neg1, m_neg2], top_k=1)
    assert len(fused_neg) == 1
    # Without non-negative clamp, bonus would be 0.02 * (-0.5) * 1 = -0.01
    # With clamp, bonus is 0.0
    expected_score = 0.65 * (-0.5) + 0.25 * ((-0.5 - 0.6) / 2) + 0.10 * ((-0.5 - 0.6) / 2)
    assert abs(fused_neg[0].ranking_score - round(expected_score, 5)) < 1e-4

    # Custom factor
    m_pos1 = RankedMoment("p1", "vid_pos", 1.0, 3.0, 10, 30, 10, 1.0, 1.0)
    m_pos2 = RankedMoment("p2", "vid_pos", 4.0, 6.0, 40, 60, 40, 4.0, 1.0)
    fused_custom = aggregate_video_moments([m_pos1, m_pos2], density_bonus_factor=0.05, top_k=1)
    # bonus = 0.05 * 1.0 * 1 = 0.05 -> score = 1.0 + 0.05 = 1.05
    assert abs(fused_custom[0].ranking_score - 1.05) < 1e-4

