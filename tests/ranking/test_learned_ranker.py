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
    LTR_FEATURE_NAMES_18D,
    LTR_FEATURE_NAMES_22D,
    compute_query_z_scores,
    extract_ltr_features_18d,
    extract_ltr_features_22d,
    extract_ltr_features_24d,
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
        "vis_lexical_synergy",
        "unverified_lexical_hazard",
        "subclause_weakest_link",
        "window_energy_concentration",
        "temporal_attention_coherence",
        "rrf_vis_caption",
    )
    assert LTR_FEATURE_NAMES == expected
    assert len(LTR_FEATURE_NAMES) == 24
    assert len(LTR_FEATURE_NAMES_22D) == 22
    assert len(LTR_FEATURE_NAMES_18D) == 18


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
    assert list(feat_dict.keys()) == list(LTR_FEATURE_NAMES_18D)
    assert pytest.approx(feat_dict["spike_drop_ratio"], abs=1e-5) == 0.12
    assert pytest.approx(feat_dict["bm25_log_score"], abs=1e-5) == math.log1p(9.0)
    assert pytest.approx(feat_dict["high_spec_token_coverage"], abs=1e-5) == 0.75
    assert feat_dict["anchor_lexical_match"] == 1.0
    assert pytest.approx(feat_dict["delta_z_vis"], abs=1e-5) == 0.0

    vec22 = extract_ltr_features_22d(cand, z_stats=z_stats)
    dict22 = extract_ltr_features_22d(cand, z_stats=z_stats, return_dict=True)
    assert vec22.shape == (22,)
    assert list(dict22.keys()) == list(LTR_FEATURE_NAMES_22D)
    assert pytest.approx(dict22["vis_lexical_synergy"], abs=1e-5) == (0.40 - 0.12) * 0.012
    assert pytest.approx(dict22["subclause_weakest_link"], abs=1e-5) == 0.40
    assert pytest.approx(dict22["window_energy_concentration"], abs=1e-5) == 0.35 / 0.40

    cand24 = dict(cand, temporal_attention_coherence=0.55, rrf_vis_caption=0.015)
    vec24 = extract_ltr_features_24d(cand24, z_stats=z_stats)
    dict24 = extract_ltr_features_24d(cand24, z_stats=z_stats, return_dict=True)
    assert vec24.shape == (24,)
    assert list(dict24.keys()) == list(LTR_FEATURE_NAMES)
    assert pytest.approx(dict24["temporal_attention_coherence"], abs=1e-5) == 0.55
    assert pytest.approx(dict24["rrf_vis_caption"], abs=1e-5) == 0.015

    low_vis_cand = dict(cand, s_vis_blend_max=0.11)
    low_vis_dict = extract_ltr_features_22d(low_vis_cand, z_stats=z_stats, return_dict=True)
    assert low_vis_dict["vis_lexical_synergy"] == 0.0

    zero_vis_dict = extract_ltr_features_22d(dict(cand, s_vis_blend_max=0.0), z_stats=z_stats, return_dict=True)
    assert zero_vis_dict["window_energy_concentration"] == 0.0

    tiny_vis_dict = extract_ltr_features_22d(
        dict(cand, s_vis_blend_max=0.02, s_vis_top3_mean=0.01),
        z_stats=z_stats,
        return_dict=True,
    )
    assert pytest.approx(tiny_vis_dict["window_energy_concentration"], abs=1e-5) == 0.01 / 0.05

    batch = extract_ltr_features_batch([cand24, {"s_vis_blend_max": float("nan")}])
    assert batch.shape == (2, 24)
    assert np.all(np.isfinite(batch))


def test_learned_ranker_fallback_and_latency(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.json"
    ranker = LearnedRankerLTR(model_path=missing_path)
    assert ranker.is_loaded is False
    assert ranker.model_type == "fallback_linear"

    rng = np.random.default_rng(7)
    X = rng.standard_normal((120, 24)).astype(np.float32)

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
    X = np.zeros((2, 24), dtype=np.float32)
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
    assert artifact["metadata"]["trained_split"] == "dev_full_859"
    assert len(artifact["metadata"]["trained_query_ids"]) == n_queries
    assert len(artifact["fold_weights"]) == 5
    for fw in artifact["fold_weights"]:
        assert "eval_query_ids" in fw and len(fw["eval_query_ids"]) > 0
        assert len(fw["weights"]) == 24

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


def test_ranking_features_standalone_import_no_circular_import():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from videntra.ranking.features import LTR_FEATURE_NAMES, extract_ltr_features_18d, extract_ltr_features_22d, extract_ltr_features_24d; assert len(LTR_FEATURE_NAMES) == 24",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Standalone import failed: {proc.stderr}"


def test_base_anchored_residual_pairwise_training(tmp_path: Path):
    rng = np.random.default_rng(123)
    n_queries = 25
    cands_per_query = 6
    rows = []

    for q_idx in range(n_queries):
        qid = f"rq_{q_idx:03d}"
        pos_vid = f"rvid_{q_idx % 10:02d}"
        for c_idx in range(cands_per_query):
            is_pos = 1.0 if c_idx == 0 else 0.0
            feat = rng.normal(0.0, 0.2, size=len(LTR_FEATURE_NAMES))
            if is_pos > 0:
                feat[0] += 2.0  # s_vis_blend_max
                feat[6] += 1.5  # s_late_interaction
            # Ambiguous base_video_score: negative slightly ahead of positive until residual fixes it
            base_score = 0.500 if is_pos > 0 else (0.503 if c_idx == 1 else 0.35)
            cand_vid = pos_vid if is_pos > 0 else f"rvid_{(q_idx + c_idx + 1) % 10:02d}"
            rows.append({
                "query_id": qid,
                "source_video_id": pos_vid,
                "video_id": cand_vid,
                "label": is_pos,
                "base_video_score": base_score,
                "features": [float(v) for v in feat],
            })

    ds_path = tmp_path / "residual_synth.jsonl"
    ds_path.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")

    X, y, qids, vids, base_scores = load_ltr_jsonl_dataset(ds_path, return_base_scores=True)
    assert base_scores is not None
    assert base_scores.shape == (n_queries * cands_per_query,)

    artifact = run_group_kfold_cv(
        X=X,
        y=y,
        query_ids=qids,
        source_video_ids=vids,
        n_splits=5,
        l2_reg=1.0,
        backend="pairwise_logistic",
        base_scores=base_scores,
        blend_weight=0.015,
    )
    assert artifact["metadata"]["residual_training"] is True
    assert artifact["metadata"]["blend_weight"] == 0.015
    assert artifact["metadata"]["cv_hit_at_1"] >= 0.85
    assert artifact["metadata"]["trained_split"] == "dev_full_859"
    assert len(artifact["metadata"]["trained_query_ids"]) == n_queries
    assert len(artifact["fold_weights"]) == 5
    for fw in artifact["fold_weights"]:
        assert "eval_query_ids" in fw and len(fw["eval_query_ids"]) > 0
        assert len(fw["weights"]) == 24

    # Legacy JSONL without base_video_score falls back cleanly
    legacy_rows = [{k: v for k, v in r.items() if k != "base_video_score"} for r in rows]
    legacy_path = tmp_path / "legacy_synth.jsonl"
    legacy_path.write_text("\n".join(__import__("json").dumps(r) for r in legacy_rows), encoding="utf-8")
    _, _, _, _, legacy_base = load_ltr_jsonl_dataset(legacy_path, return_base_scores=True)
    assert legacy_base is None


def test_oof_fold_schema_per_fold_normalization_and_leakage_guards():
    from scripts.train_ltr_ranker_cv import assign_connected_source_groups

    rng = np.random.default_rng(99)
    # Construct 10 queries where two AVC clips share source xvAvN5xn5oY
    query_targets = [
        ("q_avc_a", ["xvAvN5xn5oY__s140__e150"]),
        ("q_avc_b", ["xvAvN5xn5oY__s560__e570"]),
        ("q_multi", ["grpX__s0__e10", "grpY__s10__e20"]),
        ("q_grpY", ["grpY__s30__e40"]),
        ("q_3", ["video1"]),
        ("q_4", ["video2"]),
        ("q_5", ["video3"]),
        ("q_6", ["video4"]),
        ("q_7", ["video5"]),
        ("q_8", ["video6"]),
    ]
    assigned = assign_connected_source_groups(query_targets)
    fg = assigned["fold_group_by_query"]

    X_rows, y_rows, qids, grps = [], [], [], []
    for idx, (qid, _) in enumerate(query_targets):
        for c_idx in range(4):
            feat = rng.normal(loc=float(idx * 2), scale=0.3, size=len(LTR_FEATURE_NAMES))
            if c_idx == 0:
                feat[0] += 3.0
            X_rows.append(feat)
            y_rows.append(1.0 if c_idx == 0 else 0.0)
            qids.append(qid)
            grps.append(fg[qid])

    X = np.vstack(X_rows)
    y = np.asarray(y_rows)
    global_mean = np.mean(X, axis=0)

    artifact = run_group_kfold_cv(
        X=X,
        y=y,
        query_ids=qids,
        source_video_ids=grps,
        n_splits=5,
        l2_reg=1.0,
        backend="pairwise_logistic",
        manifest_hash=assigned["manifest_hash"],
        raw_groups_by_query=assigned["raw_groups_by_query"],
    )

    assert artifact["manifest_hash"] == assigned["manifest_hash"]
    assert artifact["metadata"]["manifest_hash"] == assigned["manifest_hash"]
    seen_oof_qids = set()
    for fw in artifact["fold_weights"]:
        assert "fold_id" in fw
        assert fw["manifest_hash"] == assigned["manifest_hash"]
        tr_g = set(fw["train_group_ids"])
        ev_g = set(fw["eval_group_ids"])
        assert tr_g and ev_g
        assert tr_g.isdisjoint(ev_g)
        ev_q = set(fw["eval_query_ids"])
        assert seen_oof_qids.isdisjoint(ev_q)
        seen_oof_qids |= ev_q
        # Same-source AVC pair must stay in the same fold
        if "q_avc_a" in ev_q or "q_avc_b" in ev_q:
            assert {"q_avc_a", "q_avc_b"}.issubset(ev_q)
        # Multi-target connected component must stay in the same fold
        if "q_multi" in ev_q or "q_grpY" in ev_q:
            assert {"q_multi", "q_grpY"}.issubset(ev_q)
        # Per-fold feature_mean must be fit strictly on train split, not global
        assert not np.allclose(fw["feature_mean"], global_mean, atol=1e-3)

    assert seen_oof_qids == {qid for qid, _ in query_targets}

    # Query spanning multiple unmerged groups must be rejected
    bad_grps = list(grps)
    bad_grps[0] = "different_unmerged_group"
    with pytest.raises(ValueError, match="multiple unmerged source groups"):
        run_group_kfold_cv(
            X=X,
            y=y,
            query_ids=qids,
            source_video_ids=bad_grps,
            n_splits=5,
            backend="pairwise_logistic",
        )


def test_learned_ranker_snapshot_restore_apply_and_lightgbm_cv():
    import argparse

    ranker = LearnedRankerLTR(model_path=None)
    orig = ranker.snapshot_state()
    fold_payload = {
        "model_type": "pairwise_logistic",
        "weights": [0.25] * len(LTR_FEATURE_NAMES),
        "bias": 0.5,
        "feature_mean": [0.0] * len(LTR_FEATURE_NAMES),
        "feature_std": [1.0] * len(LTR_FEATURE_NAMES),
    }
    ranker.apply_fold_payload(fold_payload)
    assert ranker.is_loaded is True
    assert float(ranker.weights[0]) == 0.25
    assert ranker.bias == 0.5
    ranker.restore_state(orig)
    assert ranker.is_loaded is False

    # Verify invalid tree_ensemble fold without trees raises ValueError
    with pytest.raises(ValueError, match="trees"):
        ranker.apply_fold_payload({"model_type": "tree_ensemble", "trees": []})

    import scripts.train_ltr_ranker_cv as ltr_cv
    import sys
    import types

    class FakeBooster:
        def dump_model(self):
            return {
                "tree_info": [
                    {
                        "tree_structure": {
                            "split_feature": 0,
                            "threshold": 0.5,
                            "left_child": {"leaf_value": -0.5},
                            "right_child": {"leaf_value": 1.2},
                        }
                    }
                ]
            }

    class FakeLGBMRanker:
        def __init__(self, **_kwargs):
            self.booster_ = FakeBooster()

        def fit(self, _X, _y, group=None):
            assert group is not None and sum(group) > 0
            return self

    fake_lgb = types.SimpleNamespace(LGBMRanker=FakeLGBMRanker)
    orig_lgb = sys.modules.get("lightgbm")
    sys.modules["lightgbm"] = fake_lgb  # type: ignore[assignment]
    try:
        rng = np.random.default_rng(7)
        X_rows, y_rows, qids, grps = [], [], [], []
        for i in range(10):
            qid = f"q_{i}"
            grp = f"g_{i}"
            for c in range(4):
                f = rng.normal(0.0, 0.2, size=len(LTR_FEATURE_NAMES))
                if c == 0:
                    f[0] += 2.0
                X_rows.append(f)
                y_rows.append(1.0 if c == 0 else 0.0)
                qids.append(qid)
                grps.append(grp)
        artifact = run_group_kfold_cv(
            X=np.vstack(X_rows),
            y=np.asarray(y_rows),
            query_ids=qids,
            source_video_ids=grps,
            n_splits=5,
            backend="lightgbm",
        )
    finally:
        if orig_lgb is None:
            sys.modules.pop("lightgbm", None)
        else:
            sys.modules["lightgbm"] = orig_lgb

    assert artifact["model_type"] == "tree_ensemble"
    for fw in artifact["fold_weights"]:
        assert fw["model_type"] == "tree_ensemble"
        assert len(fw["trees"]) > 0
        assert "weights" not in fw
