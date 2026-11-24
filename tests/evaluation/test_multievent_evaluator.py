"""Tests for ActivityNet full-chain evaluator, VideoComp evaluator, split lock & Modal validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.modal_videntra_worker import (
    is_shard_validated,
    run_vlm_rerank_pairs_local_or_gpu,
    shard_artifact_dir,
    summarize_pilot_costs,
    write_validated_shard_output,
)
from scripts.prepare_activitynet_videocomp_splits import assign_source_disjoint_splits
from videntra.evaluation.multievent_evaluator import (
    evaluate_activitynet_full_chain,
    evaluate_videocomp,
    paired_source_group_bootstrap_ci,
)


def test_activitynet_full_chain_strict_cases() -> None:
    locked_queries = [
        # q1: 2/3 match strong + 3rd event tIoU=0.31 (<0.5) => MUST be full-chain miss (0)
        {
            "query_id": "q1",
            "video_id": "v1",
            "source_group_id": "g1",
            "events": [
                {"clause_id": "e0", "start_sec": 0.0, "end_sec": 10.0},
                {"clause_id": "e1", "start_sec": 10.0, "end_sec": 20.0},
                {"clause_id": "e2", "start_sec": 20.0, "end_sec": 30.0},
            ],
        },
        # q2: timestamps match perfectly, WRONG video at rank 1 (right at rank 3 => Hit@5=1, Hit@1=0)
        {
            "query_id": "q2",
            "video_id": "v2",
            "source_group_id": "g2",
            "events": [
                {"clause_id": "e0", "start_sec": 2.0, "end_sec": 8.0},
                {"clause_id": "e1", "start_sec": 10.0, "end_sec": 18.0},
            ],
        },
        # q3: target video missing from index => MUST be miss AND stay in fixed denominator
        {
            "query_id": "q3",
            "video_id": "v3_missing",
            "source_group_id": "g3",
            "events": [
                {"clause_id": "e0", "start_sec": 0.0, "end_sec": 5.0},
                {"clause_id": "e1", "start_sec": 6.0, "end_sec": 12.0},
            ],
        },
        # q4: >=4 simultaneous + sequential chain matching full chain cleanly => Hit@1 = 1
        {
            "query_id": "q4",
            "video_id": "v4",
            "source_group_id": "g4",
            "events": [
                {"clause_id": "e0", "start_sec": 0.0, "end_sec": 6.0},
                {"clause_id": "e1", "start_sec": 8.0, "end_sec": 16.0},
                {"clause_id": "e2", "start_sec": 10.0, "end_sec": 18.0},  # overlaps e1 (simultaneous)
                {"clause_id": "e3", "start_sec": 20.0, "end_sec": 28.0},
            ],
        },
        # q5: swapped prediction order (pred e1 > pred e0, though each tIoU? actually fails order)
        {
            "query_id": "q5",
            "video_id": "v5",
            "source_group_id": "g5",
            "events": [
                {"clause_id": "e0", "start_sec": 0.0, "end_sec": 10.0},
                {"clause_id": "e1", "start_sec": 8.0, "end_sec": 20.0},  # overlapping GT, order pair explicit
            ],
            "order_pairs": [("e0", "e1")],
        },
    ]

    preds = {
        "q1": {
            "top_k_results": [
                {
                    "video_id": "v1",
                    "event_spans": [
                        {"clause_id": "e0", "start_sec": 0.5, "end_sec": 9.5},   # tIoU=0.9
                        {"clause_id": "e1", "start_sec": 10.5, "end_sec": 19.5}, # tIoU=0.9
                        {"clause_id": "e2", "start_sec": 20.0, "end_sec": 23.1}, # tIoU=0.31 < 0.5
                    ],
                }
            ]
        },
        "q2": {
            "top_k_results": [
                {
                    "video_id": "v_wrong",
                    "event_spans": [
                        {"clause_id": "e0", "start_sec": 2.0, "end_sec": 8.0},
                        {"clause_id": "e1", "start_sec": 10.0, "end_sec": 18.0},
                    ],
                },
                {"video_id": "v_wrong2", "event_spans": []},
                {
                    "video_id": "v2",
                    "event_spans": [
                        {"clause_id": "e0", "start_sec": 2.0, "end_sec": 8.0},
                        {"clause_id": "e1", "start_sec": 10.0, "end_sec": 18.0},
                    ],
                },
            ]
        },
        "q3": {
            "top_k_results": [
                {
                    "video_id": "v3_missing",
                    "event_spans": [
                        {"clause_id": "e0", "start_sec": 0.0, "end_sec": 5.0},
                        {"clause_id": "e1", "start_sec": 6.0, "end_sec": 12.0},
                    ],
                }
            ]
        },
        "q4": {
            "top_k_results": [
                {
                    "video_id": "v4",
                    "event_spans": [
                        {"clause_id": "e0", "start_sec": 0.0, "end_sec": 5.8},
                        {"clause_id": "e1", "start_sec": 8.2, "end_sec": 16.0},
                        {"clause_id": "e2", "start_sec": 10.0, "end_sec": 17.8},
                        {"clause_id": "e3", "start_sec": 20.2, "end_sec": 27.5},
                    ],
                }
            ]
        },
        "q5": {
            "top_k_results": [
                {
                    "video_id": "v5",
                    "event_spans": [
                        # Swapped relative start compared to explicit ("e0", "e1") constraint
                        {"clause_id": "e0", "start_sec": 10.0, "end_sec": 18.0},
                        {"clause_id": "e1", "start_sec": 8.0, "end_sec": 20.0},
                    ],
                }
            ]
        },
    }

    indexed = {"v1", "v2", "v4", "v5", "v_wrong", "v_wrong2"}
    res = evaluate_activitynet_full_chain(locked_queries, preds, indexed_video_ids=indexed)
    assert res["total_queries"] == 5
    assert res["missing_from_index"] == 1
    assert res["full_chain_hit_at_1_count"] == 1  # ONLY q4 hits at Top-1!
    assert pytest.approx(res["full_chain_hit_at_1"], abs=1e-6) == 0.2
    assert pytest.approx(res["full_chain_hit_at_5"], abs=1e-6) == 0.4  # q2 at rank 3 + q4
    assert pytest.approx(res["given_video_full_chain_hit_at_1"], abs=1e-6) == 0.4
    assert res["by_chain_length"][">=4"]["queries"] == 1
    assert res["by_chain_length"][">=4"]["full_chain_hit_at_1"] == 1.0
    assert res["by_chain_length"]["3"]["full_chain_hit_at_1"] == 0.0


def test_videocomp_domains_types_ties_and_all_product() -> None:
    pairs = [
        {"pair_id": "p1", "domain": "activitynet_comp", "type": "temporal_reordering"},
        {"pair_id": "p2", "domain": "activitynet_comp", "type": "temporal_reordering"},
        {"pair_id": "p3", "domain": "activitynet_comp", "type": "action_replacement"},
        {"pair_id": "p4", "domain": "activitynet_comp", "type": "segment_mismatch"},
        {"pair_id": "p5", "domain": "youcook2_comp", "type": "temporal_reordering"},
    ]
    scores = {
        "p1": {"positive_score": 0.9, "negative_score": 0.4},
        "p2": {"positive_score": 0.5, "negative_score": 0.5},  # tie -> counts wrong
        "p3": {"positive_score": 0.8, "negative_score": 0.2},
        "p4": {"positive_score": 0.7, "negative_score": 0.3},
        "p5": {"positive_score": 0.9, "negative_score": 0.1},
    }
    res = evaluate_videocomp(pairs, scores)
    anet = res["by_domain"]["activitynet_comp"]
    assert anet["temporal_reordering"]["accuracy"] == 0.5
    assert anet["temporal_reordering"]["ties"] == 1
    assert anet["action_replacement"]["accuracy"] == 1.0
    assert anet["segment_mismatch"]["accuracy"] == 1.0
    assert pytest.approx(anet["all"], abs=1e-6) == 0.5 * 1.0 * 1.0
    assert res["by_domain"]["youcook2_comp"]["temporal_reordering"]["accuracy"] == 1.0


def test_paired_source_group_bootstrap_ci() -> None:
    hits_a = [1, 1, 1, 1, 0, 1]
    hits_b = [0, 0, 1, 0, 0, 0]
    gids = ["g1", "g1", "g2", "g3", "g4", "g5"]
    ci = paired_source_group_bootstrap_ci(hits_a, hits_b, gids, n_resamples=500, seed=20261124)
    assert ci["n_queries"] == 6
    assert ci["n_groups"] == 5
    assert ci["rescued"] == 4
    assert ci["lost"] == 0
    assert ci["ci95_low"] <= ci["delta"] <= ci["ci95_high"]
    assert ci["significant_gt_zero"] is True


def test_prepare_activitynet_videocomp_splits_zero_overlap_and_dedup() -> None:
    records = []
    # 10 train groups & 5 val groups in ActivityNet; plus duplicate cut + hash dup + cross-pool contaminant
    for i in range(10):
        records.append({"query_id": f"cap_tr_{i}", "video_id": f"v_anet_tr{i}", "public_pool": "train", "domain": "activitynet"})
        records.append({"query_id": f"comp_tr_{i}", "video_id": f"anet_tr{i}_seg0_10", "public_pool": "train", "domain": "activitynet"})
    for i in range(5):
        records.append({"query_id": f"cap_va_{i}", "video_id": f"v_anet_va{i}", "public_pool": "validation", "domain": "activitynet"})
    # Cross-pool contaminant sharing same video ID across train & validation
    records.append({"query_id": "bad_1", "video_id": "v_leak", "public_pool": "train", "domain": "activitynet"})
    records.append({"query_id": "bad_2", "video_id": "v_leak", "public_pool": "validation", "domain": "activitynet"})
    # Duplicate video by file hash with another train ID
    hashes = {"v_anet_tr0": "same_sha", "v_reupload": "same_sha"}
    records.append({"query_id": "dup_1", "video_id": "v_reupload", "public_pool": "train", "domain": "activitynet"})

    bundle1 = assign_source_disjoint_splits(
        records, seed=20261124, max_groups_per_domain=10, file_sha256_by_video_id=hashes
    )
    bundle2 = assign_source_disjoint_splits(
        records, seed=20261124, max_groups_per_domain=10, file_sha256_by_video_id=hashes
    )
    assert bundle1["overlap_verified"] is True
    assert bundle1["quarantined_count"] == 1
    assert bundle1["quarantined_groups"][0]["source_group_id"] == "leak"
    assert bundle1["splits"]["tune"]["counts"]["source_groups"] == 4
    assert bundle1["splits"]["valid_confirm"]["counts"]["source_groups"] == 4
    assert bundle1["splits"]["test"]["counts"]["source_groups"] == 2
    # v_reupload merges into anet_tr0 -> never split apart from cap_tr_0 / comp_tr_0
    assert (
        bundle1["splits"]["tune"]["checksums"]
        == bundle2["splits"]["tune"]["checksums"]
    )


def test_modal_worker_shard_validation_and_pilot_cost(tmp_path: Path) -> None:
    sdir = shard_artifact_dir(tmp_path, run_id="run01", model_tag="qwen3b", shard_id="shard0")
    receipt = write_validated_shard_output(
        sdir,
        output_filename="vlm_pair_scores.jsonl",
        rows=[{"query_id": "q1", "scores": {"v1": 1.2}, "oom_fallback": False}],
        expected_rows=1,
        extra_meta={"elapsed_sec": 12.5},
    )
    assert receipt["status"] == "validated"
    ok, loaded = is_shard_validated(sdir, output_filename="vlm_pair_scores.jsonl", expected_rows=1)
    assert ok is True and loaded is not None

    # Tamper detection invalidates shard immediately
    (sdir / "vlm_pair_scores.jsonl").write_text('{"query_id": "q1", "scores": {"v1": 9.9}}\n', encoding="utf-8")
    ok_bad, _ = is_shard_validated(sdir, output_filename="vlm_pair_scores.jsonl", expected_rows=1)
    assert ok_bad is False

    # Re-running already validated shard skips work immediately
    sdir2 = shard_artifact_dir(tmp_path, run_id="run01", model_tag="qwen3b", shard_id="shard1")
    write_validated_shard_output(
        sdir2,
        output_filename="vlm_pair_scores.jsonl",
        rows=[{"query_id": "q2", "scores": {}, "oom_fallback": False}],
        expected_rows=1,
        extra_meta={"elapsed_sec": 10.0},
    )
    res_reuse_exact = run_vlm_rerank_pairs_local_or_gpu(
        [{"query_id": "q2"}], video_paths_by_id={}, shard_dir=sdir2
    )
    assert res_reuse_exact["reused_validated_shard"] is True

    # Mismatched expected_rows does not reuse stale 1-row receipt when 2 rows are requested
    res_mismatch = run_vlm_rerank_pairs_local_or_gpu(
        [{"query_id": "q2"}, {"query_id": "q3"}], video_paths_by_id={}, shard_dir=sdir2
    )
    assert res_mismatch["reused_validated_shard"] is False

    cost = summarize_pilot_costs([receipt], usd_per_gpu_sec=0.000222, videos_processed=25, pairs_processed=100)
    assert cost["total_billed_sec"] == 12.5
    assert cost["usd_per_100_videos"] is not None
    assert cost["usd_per_1000_pairs"] is not None
