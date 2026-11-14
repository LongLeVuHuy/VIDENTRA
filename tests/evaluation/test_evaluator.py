"""Unit tests for offline evaluation metrics against known mathematical fixtures."""
import pytest
from videntra.evaluation.evaluator import evaluate_predictions, EvaluationReport


def test_cli_evaluation_uses_unique_video_ranking_and_labeled_moment_denominator():
    gt = [{"query_id": "a", "video_id": "v1", "relevant_windows": [[0, 10]]},
          {"query_id": "b", "video_id": "v2"}]
    preds = [{"query_id": "a", "video_id": "v1", "moment_start_seconds": 0,
              "moment_end_seconds": 10, "ranking_score": 1.0}]
    preds += [{"query_id": "b", "video_id": "other", "ranking_score": 1.0 - i / 100}
              for i in range(5)]
    preds += [{"query_id": "b", "video_id": "v2", "ranking_score": 0.5}]
    report = evaluate_predictions(preds, gt)
    assert report.moment_labeled_queries == 1
    assert report.moment_hit_at_1_iou05 == 1.0
    assert report.mean_reciprocal_rank == 0.75
    assert report.video_hit_at_5 == 1.0


def test_cli_evaluation_does_not_invent_moment_scores_and_limits_mrr():
    gt = [{"query_id": "a", "video_id": "target"}]
    preds = [{"query_id": "a", "video_id": f"v{i}", "ranking_score": 1.0 - i / 100}
             for i in range(10)]
    preds.append({"query_id": "a", "video_id": "target", "ranking_score": 0.5})
    preds.append({"query_id": "unknown", "video_id": "target", "ranking_score": 1.0})
    report = evaluate_predictions(preds, gt)
    assert report.evaluated_queries == 1
    assert report.mean_reciprocal_rank == 0.0
    assert report.moment_labeled_queries == 0
    assert report.moment_hit_at_1_iou05 is None
    assert report.moment_ndcg_at_5 is None


def test_evaluator_exact_metrics():
    # Ground truth: 2 queries
    # Q1: target video_1, relevant window [10.0, 14.0]
    # Q2: target video_2, relevant window [20.0, 25.0]
    gt = [
        {"query_id": "q1", "video_id": "v1", "start_seconds": 10.0, "end_seconds": 14.0},
        {"query_id": "q2", "video_id": "v2", "start_seconds": 20.0, "end_seconds": 25.0},
    ]

    # Predictions:
    # Q1:
    #   Rank 1: v1, [10.5, 14.0] -> inter=3.5, union=4.0 -> IoU = 3.5/4.0 = 0.875 (matches IoU 0.3, 0.5, 0.7!)
    # Q2:
    #   Rank 1: v3 (wrong video!), [0.0, 5.0]
    #   Rank 2: v2, [21.0, 24.0] -> inter=3.0, union=5.0 -> IoU = 3.0/5.0 = 0.60 (matches IoU 0.3, 0.5; misses 0.7)
    preds = [
        {"query_id": "q1", "video_id": "v1", "moment_start_seconds": 10.5, "moment_end_seconds": 14.0, "ranking_score": 0.95},
        {"query_id": "q2", "video_id": "v3", "moment_start_seconds": 0.0, "moment_end_seconds": 5.0, "ranking_score": 0.80},
        {"query_id": "q2", "video_id": "v2", "moment_start_seconds": 21.0, "moment_end_seconds": 24.0, "ranking_score": 0.70},
    ]

    report = evaluate_predictions(preds, gt)

    assert report.total_queries == 2
    assert report.evaluated_queries == 2

    # VideoHit@1: Q1 hit (v1 at rank 1), Q2 miss (v3 at rank 1) -> 1/2 = 0.5
    assert report.video_hit_at_1 == 0.5
    # VideoHit@5: Q1 hit (v1 at rank 1), Q2 hit (v2 at rank 2) -> 2/2 = 1.0
    assert report.video_hit_at_5 == 1.0

    # MRR: Q1 rank 1 (1.0), Q2 rank 2 (0.5) -> (1.0 + 0.5)/2 = 0.75
    assert report.mean_reciprocal_rank == 0.75

    # MomentHit@1 at IoU 0.5: Q1 hit (0.875 >= 0.5), Q2 miss (v3 wrong) -> 1/2 = 0.5
    assert report.moment_hit_at_1_iou05 == 0.5
    # MomentHit@1 at IoU 0.7: Q1 hit (0.875 >= 0.7), Q2 miss -> 1/2 = 0.5
    assert report.moment_hit_at_1_iou07 == 0.5
    # MomentHit@5 at IoU 0.5: Q1 hit, Q2 hit (rank 2 IoU=0.6 >= 0.5) -> 2/2 = 1.0
    assert report.moment_hit_at_5_iou05 == 1.0

    # Video nDCG@K:
    # Q1: v1 at rank 1 -> nDCG@1 = 1.0, @5 = 1.0, @10 = 1.0
    # Q2: v3 at rank 1 (wrong), v2 at rank 2 (target) ->
    #     nDCG@1 = 0.0
    #     nDCG@5 = (1/log2(3)) / (1/log2(2)) = 1/1.58496 = 0.6309
    # Mean @1 = (1.0 + 0.0)/2 = 0.5000
    # Mean @5 = (1.0 + 0.6309)/2 = 0.8155
    assert report.video_ndcg_at_1 == 0.5
    assert report.video_ndcg_at_5 == 0.8155
    assert report.video_ndcg_at_10 == 0.8155

    # Moment nDCG@K:
    # Q1: rank 1 IoU=0.875 >= 0.70 -> grade 3 (gain 7.0), IDCG=7.0 -> nDCG=1.0
    # Q2: rank 1 wrong video (gain 0.0), rank 2 IoU=0.60 (grade 2, gain 3.0) ->
    #     DCG@5 = 3.0 / log2(3) = 1.8928, IDCG@5 = 7.0 / log2(2) = 7.0 -> nDCG = 0.2704
    # Mean moment nDCG@5 = (1.0 + 0.2704)/2 = 0.6352
    assert report.moment_ndcg_at_5 == 0.6352
    assert report.moment_ndcg_at_10 == 0.6352


def test_binary_ndcg_empty_inputs():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    assert compute_binary_ndcg_at_k([], {"v1"}, k=5) == 0.0
    assert compute_binary_ndcg_at_k(["v1"], set(), k=5) == 0.0
    assert compute_binary_ndcg_at_k(["v1"], {"v1"}, k=0) == 0.0
    assert compute_binary_ndcg_at_k(["v1"], {"v1"}, k=-1) == 0.0


def test_binary_ndcg_single_target_perfect():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    preds = ["v1", "v2", "v3"]
    targets = {"v1"}
    assert compute_binary_ndcg_at_k(preds, targets, k=1) == 1.0
    assert compute_binary_ndcg_at_k(preds, targets, k=5) == 1.0
    assert compute_binary_ndcg_at_k(preds, targets, k=10) == 1.0


def test_binary_ndcg_multi_target_perfect():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    preds = ["v1", "v2", "v3"]
    targets = {"v1", "v2", "v3"}
    assert compute_binary_ndcg_at_k(preds, targets, k=3) == 1.0


def test_binary_ndcg_inverted_ranking():
    import math
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    preds = ["v_wrong", "v1"]
    targets = {"v1"}
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    result = compute_binary_ndcg_at_k(preds, targets, k=5)
    assert abs(result - expected) < 1e-4


def test_binary_ndcg_deduplication():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    # Duplicate video in predictions should be deduplicated preserving first occurrence
    preds = ["v1", "v1", "v2"]
    targets = {"v1", "v2"}
    assert compute_binary_ndcg_at_k(preds, targets, k=2) == 1.0


def test_binary_ndcg_target_larger_than_k():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    targets = {"v1", "v2", "v3", "v4", "v5"}
    preds = ["v1", "v2"]
    # At k=2, only 2 target items are considered in IDCG
    assert compute_binary_ndcg_at_k(preds, targets, k=2) == 1.0


def test_binary_ndcg_indexed_video_ids_filtering():
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k
    targets = {"v1", "v_unindexed"}
    indexed = {"v1"}
    preds = ["v1"]
    # v_unindexed should be filtered out from effective targets, so IDCG is based only on v1
    assert compute_binary_ndcg_at_k(preds, targets, k=5, indexed_video_ids=indexed) == 1.0


def test_graded_moment_ndcg_perfect_single():
    from videntra.evaluation.evaluator import compute_graded_moment_ndcg_at_k
    gt = [{"video_id": "v1", "start": 10.0, "end": 20.0}]
    preds = [{"video_id": "v1", "start": 10.0, "end": 20.0}]
    assert compute_graded_moment_ndcg_at_k(preds, gt, k=5) == 1.0


def test_graded_moment_ndcg_four_levels():
    import math
    from videntra.evaluation.evaluator import compute_graded_moment_ndcg_at_k
    # 4 distinct GT windows for v1
    gt = [
        {"video_id": "v1", "start": 0.0, "end": 10.0},
        {"video_id": "v1", "start": 20.0, "end": 30.0},
        {"video_id": "v1", "start": 40.0, "end": 50.0},
        {"video_id": "v1", "start": 60.0, "end": 70.0},
    ]
    # Pred 1: [0.0, 10.0] -> IoU = 1.0 >= 0.70 -> Grade 3, Gain = 7.0
    # Pred 2: [20.0, 26.0] -> Inter=6.0, Union=10.0 -> IoU = 0.60 (0.50..0.70) -> Grade 2, Gain = 3.0
    # Pred 3: [40.0, 44.0] -> Inter=4.0, Union=10.0 -> IoU = 0.40 (0.30..0.50) -> Grade 1, Gain = 1.0
    # Pred 4: [60.0, 62.0] -> Inter=2.0, Union=10.0 -> IoU = 0.20 (< 0.30) -> Grade 0, Gain = 0.0
    preds = [
        {"video_id": "v1", "start": 0.0, "end": 10.0},
        {"video_id": "v1", "start": 20.0, "end": 26.0},
        {"video_id": "v1", "start": 40.0, "end": 44.0},
        {"video_id": "v1", "start": 60.0, "end": 62.0},
    ]

    dcg = (
        7.0 / math.log2(2)
        + 3.0 / math.log2(3)
        + 1.0 / math.log2(4)
        + 0.0 / math.log2(5)
    )
    idcg = (
        7.0 / math.log2(2)
        + 7.0 / math.log2(3)
        + 7.0 / math.log2(4)
        + 7.0 / math.log2(5)
    )
    expected = dcg / idcg

    res = compute_graded_moment_ndcg_at_k(preds, gt, k=4)
    assert abs(res - expected) < 1e-4


def test_graded_moment_ndcg_anti_gaming_duplicate_gt():
    from videntra.evaluation.evaluator import compute_graded_moment_ndcg_at_k
    # Only 1 GT window
    gt = [("v1", 10.0, 20.0)]
    # 3 duplicate predictions targeting the exact same window
    preds = [
        ("v1", 10.0, 20.0),
        ("v1", 10.0, 20.0),
        ("v1", 10.0, 20.0),
    ]
    # First prediction claims the GT window (gain 7.0). Subsequent predictions get gain 0.0.
    # DCG@3 = 7.0 / log2(2) = 7.0
    # IDCG@3 = 7.0 / log2(2) = 7.0
    # nDCG@3 = 1.0
    assert compute_graded_moment_ndcg_at_k(preds, gt, k=3) == 1.0


def test_graded_moment_ndcg_wrong_video_zero_gain():
    from videntra.evaluation.evaluator import compute_graded_moment_ndcg_at_k
    gt = [("v1", 10.0, 20.0)]
    # Same temporal window but wrong video
    preds = [("v_wrong", 10.0, 20.0)]
    assert compute_graded_moment_ndcg_at_k(preds, gt, k=5) == 0.0


def test_graded_moment_ndcg_empty_or_zero_k():
    from videntra.evaluation.evaluator import compute_graded_moment_ndcg_at_k
    gt = [("v1", 10.0, 20.0)]
    preds = [("v1", 10.0, 20.0)]
    assert compute_graded_moment_ndcg_at_k([], gt, k=5) == 0.0
    assert compute_graded_moment_ndcg_at_k(preds, [], k=5) == 0.0
    assert compute_graded_moment_ndcg_at_k(preds, gt, k=0) == 0.0
    assert compute_graded_moment_ndcg_at_k(preds, gt, k=-2) == 0.0


def test_binary_ndcg_avc_single_target_normalization():
    """Verifies binary nDCG@5 calculation when AVC target is normalized to 1 video via known_target_window_id.

    In AVC benchmark queries, candidate_plan_inference.json may list 15 candidate video IDs.
    If all 15 candidates are incorrectly treated as ground truth targets, ideal_hits at k=5 is min(5, 15)=5,
    yielding IDCG@5 = 1 + 1/log2(3) + 1/log2(4) + 1/log2(5) + 1/log2(6) = 2.9485.
    Even when the engine correctly predicts the true target at Rank 1 (DCG=1.0), nDCG@5 is artificially
    diluted down to 1.0 / 2.9485 = 0.3392.

    When normalized with known_target_window_id to exactly 1 target video, IDCG@5 is based on ideal_hits=1
    (IDCG = 1.0), and a Rank 1 correct prediction achieves nDCG@5 = 1.0 (undiluted).
    """
    from videntra.evaluation.evaluator import compute_binary_ndcg_at_k

    known_target_vid = "avc_target_video_001"
    distractor_candidates = [f"avc_cand_distractor_{i}" for i in range(1, 15)]
    all_15_candidates = [known_target_vid] + distractor_candidates

    # Engine predicts the correct known target at Rank 1, followed by non-target videos
    predicted_rankings = [known_target_vid, "other_video_1", "other_video_2"]

    # 1. Unnormalized (flawed) scenario: treating all 15 candidates as ground truth targets
    # Ideal hits at k=5 is min(5, 15) = 5, yielding IDCG@5 = 2.9485
    # Since only Rank 1 is in targets, DCG@5 = 1.0 -> nDCG@5 = 1.0 / 2.9485 = 0.3392
    unnormalized_targets = set(all_15_candidates)
    diluted_ndcg_at_5 = compute_binary_ndcg_at_k(predicted_rankings, unnormalized_targets, k=5)
    # nDCG@5 is severely depressed below 0.35 despite Rank 1 hit
    assert diluted_ndcg_at_5 < 0.35
    assert abs(diluted_ndcg_at_5 - (1.0 / 2.9485)) < 1e-3

    # 2. Normalized scenario: exactly 1 true target from known_target_window_id
    # Ideal hits at k=5 is min(5, 1) = 1, yielding IDCG@5 = 1.0
    # Since Rank 1 is the true target, DCG@5 = 1.0 -> nDCG@5 = 1.0 / 1.0 = 1.0
    normalized_target = {known_target_vid}
    normalized_ndcg_at_1 = compute_binary_ndcg_at_k(predicted_rankings, normalized_target, k=1)
    normalized_ndcg_at_5 = compute_binary_ndcg_at_k(predicted_rankings, normalized_target, k=5)
    normalized_ndcg_at_10 = compute_binary_ndcg_at_k(predicted_rankings, normalized_target, k=10)

    # With single target normalization, Rank 1 hit achieves perfect 1.0 across all K
    assert normalized_ndcg_at_1 == 1.0
    assert normalized_ndcg_at_5 == 1.0
    assert normalized_ndcg_at_10 == 1.0


def test_evaluate_predictions_avc_single_target_integration():
    """End-to-end evaluate_predictions test with AVC normalized single target query."""
    known_target_vid = "avc_vid_true"
    gt = [
        {"query_id": "avc_q1", "video_id": known_target_vid, "start_seconds": 12.0, "end_seconds": 18.0}
    ]
    preds = [
        {"query_id": "avc_q1", "video_id": known_target_vid, "moment_start_seconds": 12.5, "moment_end_seconds": 18.0, "ranking_score": 0.92},
        {"query_id": "avc_q1", "video_id": "avc_vid_distractor", "moment_start_seconds": 0.0, "moment_end_seconds": 5.0, "ranking_score": 0.50},
    ]

    report = evaluate_predictions(preds, gt)

    assert report.total_queries == 1
    assert report.evaluated_queries == 1
    assert report.video_hit_at_1 == 1.0
    assert report.video_ndcg_at_1 == 1.0
    assert report.video_ndcg_at_5 == 1.0
    assert report.video_ndcg_at_10 == 1.0


def test_moment_map_and_bootstrap_ci_and_sign_test():
    """Tests E4 (Moment mAP@0.5, @0.75, @[0.5:0.95:0.05]), E5 (Bootstrap 95% CI, Paired Sign Test), and E10 (Failure Taxonomy)."""
    from videntra.evaluation.evaluator import (
        compute_moment_ap_at_iou,
        compute_moment_map_range,
        compute_bootstrap_ci,
        compute_paired_sign_test_pvalue,
        classify_query_failure,
    )

    gt = [("v1", 10.0, 20.0), ("v1", 30.0, 40.0)]
    preds = [
        ("v1", 10.0, 20.0),  # Exact match with first GT -> IoU=1.0
        ("v2", 10.0, 20.0),  # Wrong video -> FP
        ("v1", 31.0, 39.0),  # Inter=8.0, Union=10.0 -> IoU=0.80
    ]
    ap_05 = compute_moment_ap_at_iou(preds, gt, iou_threshold=0.5)
    # Rank 1: TP (prec=1/1), Rank 2: FP, Rank 3: TP (prec=2/3) -> AP = (1.0 + 2/3)/2 = 5/6 = 0.833333
    assert abs(ap_05 - (5.0 / 6.0)) < 1e-4

    map_range = compute_moment_map_range(preds, gt)
    assert 0.5 < map_range <= (5.0 / 6.0)

    # Bootstrap CI
    vals = [1.0] * 80 + [0.0] * 20
    mean_v, ci_low, ci_high = compute_bootstrap_ci(vals, n_resamples=200, seed=42)
    assert abs(mean_v - 0.80) < 1e-6
    assert 0.70 <= ci_low <= 0.80 <= ci_high <= 0.90

    # Paired sign test
    sys_a = [1.0] * 15 + [0.0] * 5
    sys_b = [0.0] * 20
    p_val = compute_paired_sign_test_pvalue(sys_a, sys_b)
    assert p_val < 0.001

    # Failure taxonomy F1..F5
    assert classify_query_failure(1, {"v1"}, "v1", "visual", {"v1"}, 1) == "PASS"
    assert classify_query_failure(1, {"v1"}, "v1", "visual", {"v1"}, 0) == "F5_BOUNDARY_LOCALIZATION_DRIFT"
    assert classify_query_failure(0, {"v1"}, "v2", "asr", {"v1"}) == "F2_LEXICAL_DISTRACTOR_OVERTHROW"
    assert classify_query_failure(0, {"v1"}, "v2", "visual", {"v1"}) == "F4_FINE_GRAINED_VISUAL_CONFUSION"
    assert classify_query_failure(0, {"v1"}, "v2", "visual", set()) == "F1_VISUAL_CANDIDATE_MISS"
    assert classify_query_failure(0, {"abcdefghijk_1"}, "abcdefghijk_2", "visual") == "F3_SAME_SOURCE_CLIP_CONFUSION"


def test_b5_b6_b8_b11_algorithmic_fixes():
    """Verifies B5 decoupled subspace scaling, B6 word-boundary OCR flag, B8 plural/multi-person extraction, B11 reverse after clause."""
    import numpy as np
    from videntra.indexing.augmented_vector import build_augmented_864d_embeddings, build_query_864d_vector
    from videntra.query.rules import decompose_query_clauses, extract_object_requirements
    from videntra.retrieval.alignment import align_sequential_keyframe_clauses

    # B6: "sign" vs "signing"/"design"
    v_768 = np.ones((1, 768), dtype=np.float32)
    v_768 /= np.linalg.norm(v_768)
    q_signing = build_query_864d_vector(v_768, "a woman is signing a contract at a desk", {})
    q_sign = build_query_864d_vector(v_768, 'a street sign saying "STOP"', {})
    # Dimension 768 + 80 + 8 = 856 is the OCR cue flag
    assert q_signing[856] == 0.0
    assert q_sign[856] > 0.0

    # B5: Decoupled 864d inner product equals 1.0 * visual_ip + alpha_obj * obj_ip + alpha_meta * meta_ip
    raw_embs = np.random.RandomState(42).randn(4, 768).astype(np.float32)
    kf_map = {i: {"video_id": f"v{i}", "pts_time": 5.0} for i in range(4)}
    f_objs = {"v0": [{"class_name": "dog", "pts_sec": 5.0, "confidence": 0.9}]}
    v_durs = {f"v{i}": 20.0 for i in range(4)}
    aug_mat, _, idf_map = build_augmented_864d_embeddings(raw_embs, kf_map, f_objs, v_durs, {}, {}, alpha_obj=0.25, alpha_meta=0.15)
    # Both index and query use the same fixed denominator, independent of metadata.
    vis_norms = np.linalg.norm(aug_mat[:, :768], axis=1)
    denominator = np.sqrt(1.0 + 0.25**2 + 0.15**2)
    assert np.allclose(vis_norms, 1.0 / denominator, atol=1e-4)
    q = build_query_864d_vector(raw_embs[0], "a dog", idf_map)
    assert np.isclose(np.linalg.norm(q[:768]), 1.0 / denominator, atol=1e-4)

    # B8: Plural nouns & paired singular persons
    reqs_plural = extract_object_requirements("two puppies and three couches")
    req_map = {r["class_name"]: r["cardinality"] for r in reqs_plural}
    assert req_map.get("dog") == 2
    assert req_map.get("couch") == 3

    reqs_pair = extract_object_requirements("a man and a woman talking")
    pair_map = {r["class_name"]: r["cardinality"] for r in reqs_pair}
    assert pair_map.get("person") == 2

    # B11: Reverse temporal conjunction "after" swaps clauses to chronological order
    clauses = decompose_query_clauses("a man laughs after a dog barks loudly")
    assert len(clauses) == 3
    assert "dog barks" in clauses[1]
    assert "man laughs" in clauses[2]

    # B11: Same-frame penalty 0.92 in align_sequential_keyframe_clauses
    cand_mat = np.eye(2, 4, dtype=np.float32)
    c1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    c2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    score_same, win_same = align_sequential_keyframe_clauses(cand_mat, [1.0, 5.0], [c1, c2])
    assert abs(score_same - 0.92) < 1e-4
    assert win_same == (1.0, 1.0)



