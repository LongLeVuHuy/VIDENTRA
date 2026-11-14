"""Unit tests for Query-Adaptive Z-Score, 18D LTR features, LearnedRankerLTR, and GroupKFold CV trainer."""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pytest

from scripts.train_ltr_ranker_cv import (
    assert_no_heldout_leakage,
    group_kfold_indices,
    load_ltr_jsonl_dataset,
    run_group_kfold_cv,
    save_ltr_model_json,
)
from videntra.ranking.features import (
    LTR_FEATURE_NAMES,
    compute_query_z_scores,
    extract_ltr_features_18d,
    extract_ltr_features_batch,
    extract_moment_features,
)
from videntra.ranking.learned_ranker import LearnedRankerLTR
from videntra.retrieval.visual import RankedMoment


def test_ltr_feature_names_exact_order():
    expected = (
        "s_vis_blend_max",
        "z_vis_max",
        "delta_z_vis",
        "s_vis_top3_mean",
        "s_vis_top5_mean",
        "s_vis_ctx_coherence",
        "s_late_interaction",
        "spike_drop_ratio",
        "rrf_vis",
        "rrf_ocr",
        "rrf_asr",
        "bm25_log_score",
        "high_spec_token_coverage",
        "anchor_lexical_match",
        "number_consistency",
        "yolo_object_verification_boost",
        "graph_relation_score",
        "llm_modality_alignment",
    )
    assert LTR_FEATURE_NAMES == expected
    assert len(LTR_FEATURE_NAMES) == 18


def test_compute_query_z_scores_top120_and_edge_cases():
    # Empty
    empty_res = compute_query_z_scores([])
    assert empty_res.z_scores.size == 0
    assert empty_res.mean == 0.0

    # Constant scores -> std == 0, eps prevents NaN
    const_res = compute_query_z_scores([0.25, 0.25, 0.25])
    assert np.allclose(const_res.z_scores, 0.0)
    assert np.allclose(const_res.delta_z_scores, 0.0)

    # Top-K subsetting (top_k=3 out of 5)
    scores = [0.10, 0.20, 0.30, 0.40, 0.50]
    res = compute_query_z_scores(scores, top_k=3)
    expected_mu = float(np.mean([0.30, 0.40, 0.50]))
    expected_std = float(np.std([0.30, 0.40, 0.50]))
    assert pytest.approx(res.mean, abs=1e-7) == expected_mu
    assert pytest.approx(res.std, abs=1e-7) == expected_std
    assert pytest.approx(res.delta_z_scores[-1], abs=1e-7) == 0.0
    assert np.all(res.delta_z_scores >= 0.0)


def test_extract_ltr_features_18d_and_derived_fields():
    z_stats = compute_query_z_scores([0.20, 0.30, 0.40])
    cand = {
        "s_vis_blend_max": 0.40,
        "s_vis_top3_mean": 0.35,
        "s_vis_top5_mean": 0.32,
        "s_vis_ctx_coherence": 0.31,
        "s_late_interaction": 0.44,
        "kf1_score": 0.40,
        "kf2_score": 0.28,
        "rrf_vis": 0.016,
        "rrf_ocr": 0.012,
        "rrf_asr": 0.0,
        "fts_score": 9.0,
        "m_high": 3,
        "k_high": 4,
        "anchor_lexical_match": True,
        "number_consistency": 1.0,
        "yolo_object_verification_boost": 0.05,
        "graph_relation_score": 0.8,
        "llm_modality_alignment": 0.9,
    }
    vec = extract_ltr_features_18d(cand, z_stats=z_stats)
    feat_dict = extract_ltr_features_18d(cand, z_stats=z_stats, return_dict=True)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (18,)
    assert vec.dtype == np.float32
    assert list(feat_dict.keys()) == list(LTR_FEATURE_NAMES)
    assert pytest.approx(feat_dict["spike_drop_ratio"], abs=1e-5) == 0.12
    assert pytest.approx(feat_dict["bm25_log_score"], abs=1e-5) == math.log1p(9.0)
    assert pytest.approx(feat_dict["high_spec_token_coverage"], abs=1e-5) == 0.75
    assert feat_dict["anchor_lexical_match"] == 1.0
    assert pytest.approx(feat_dict["delta_z_vis"], abs=1e-5) == 0.0

    batch = extract_ltr_features_batch([cand, {"s_vis_blend_max": float("nan")}])
    assert batch.shape == (2, 18)
    assert np.all(np.isfinite(batch))


def test_learned_ranker_fallback_and_latency(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.json"
    ranker = LearnedRankerLTR(model_path=missing_path)
    assert ranker.is_loaded is False
    assert ranker.model_type == "fallback_linear"

    rng = np.random.default_rng(7)
    X = rng.standard_normal((120, 18)).astype(np.float32)

    # Warm up and benchmark pure-NumPy scoring latency (< 0.3 ms target)
    _ = ranker.score_matrix(X)
    t0 = time.perf_counter()
    for _ in range(20):
        scores = ranker.score_matrix(X)
    elapsed_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
    assert scores.shape == (120,)
    assert elapsed_ms < 5.0  # Generous CI guard; typically < 0.05 ms


def test_learned_ranker_tree_ensemble_json(tmp_path: Path):
    model_path = tmp_path / "tree_ltr.json"
    tree_payload = {
        "model_type": "tree_ensemble",
        "feature_names": list(LTR_FEATURE_NAMES),
        "bias": 0.5,
        "learning_rate": 0.1,
        "trees": [
            {
                "feature_idx": 0,
                "threshold": 0.3,
                "left": {"leaf_value": -1.0},
                "right": {"leaf_value": 2.0},
            }
        ],
    }
    model_path.write_text(__import__("json").dumps(tree_payload), encoding="utf-8")

    ranker = LearnedRankerLTR(model_path=model_path)
    assert ranker.is_loaded is True
    X = np.zeros((2, 18), dtype=np.float32)
    X[0, 0] = 0.2  # <= 0.3 -> 0.5 + 0.1 * (-1.0) = 0.4
    X[1, 0] = 0.5  # > 0.3 -> 0.5 + 0.1 * (2.0) = 0.7
    scores = ranker.score_matrix(X)
    assert np.allclose(scores, [0.4, 0.7])


def test_group_kfold_cv_no_leakage_and_training(tmp_path: Path):
    with pytest.raises(ValueError, match="held-out"):
        assert_no_heldout_leakage("data/heldout_test_201.jsonl")
    with pytest.raises(ValueError, match="held-out"):
        assert_no_heldout_leakage("test_queries/msrvtt.json")

    rng = np.random.default_rng(42)
    n_queries = 25
    cands_per_query = 6
    rows = []
    X_list = []
    y_list = []
    qids = []
    vids = []

    for q_idx in range(n_queries):
        qid = f"q_{q_idx:03d}"
        # Assign 10 distinct source_video_ids across queries
        pos_vid = f"vid_{q_idx % 10:02d}"
        for c_idx in range(cands_per_query):
            is_pos = 1.0 if c_idx == 0 else 0.0
            feat = rng.normal(0.0, 0.2, size=18)
            if is_pos > 0:
                feat[0] += 1.5  # s_vis_blend_max
                feat[6] += 1.0  # s_late_interaction
            cand_vid = pos_vid if is_pos > 0 else f"vid_{(q_idx + c_idx + 1) % 10:02d}"
            X_list.append(feat)
            y_list.append(is_pos)
            qids.append(qid)
            vids.append(pos_vid)
            rows.append({
                "query_id": qid,
                "source_video_id": pos_vid,
                "video_id": cand_vid,
                "label": is_pos,
                "features": [float(v) for v in feat],
            })

    # Verify GroupKFold strictly separates source_video_id
    splits = group_kfold_indices(vids, n_splits=5)
    assert len(splits) == 5
    for tr_idx, val_idx in splits:
        tr_groups = {vids[i] for i in tr_idx}
        val_groups = {vids[i] for i in val_idx}
        assert tr_groups.isdisjoint(val_groups)

    # Test JSONL loading & end-to-end CV + artifact save + reload
    ds_path = tmp_path / "dev859_synth.jsonl"
    ds_path.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")
    X, y, loaded_qids, loaded_vids = load_ltr_jsonl_dataset(ds_path)

    artifact = run_group_kfold_cv(
        X=X,
        y=y,
        query_ids=loaded_qids,
        source_video_ids=loaded_vids,
        n_splits=5,
        l2_reg=0.5,
        backend="pairwise_logistic",
    )
    assert artifact["metadata"]["cv_hit_at_1"] >= 0.85

    out_model = tmp_path / "videntra_ltr_v1.json"
    save_ltr_model_json(artifact, out_model)

    ranker = LearnedRankerLTR(model_path=out_model)
    assert ranker.is_loaded is True
    assert ranker.score_matrix(X[:6]).shape == (6,)


def test_backwards_compatible_moment_features():
    moment = RankedMoment(
        moment_id="m1",
        video_id="v1",
        moment_start_seconds=1.0,
        moment_end_seconds=5.0,
        start_original_frame_index=10,
        end_original_frame_index=50,
        representative_original_frame_index=30,
        representative_timestamp_seconds=3.0,
        ranking_score=0.75,
        score_type="fused",
    )
    mf = extract_moment_features(moment)
    assert mf.duration_seconds == 4.0
    assert len(mf.to_feature_vector()) == 6
