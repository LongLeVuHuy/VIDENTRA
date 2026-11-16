"""Tests for LTR leakage detection, OOF evaluation, and benchmark provenance metadata (B5, B9, B11)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_validation_benchmark as rvb
from videntra.evaluation.provenance import (
    evaluate_provenance_gate,
    load_ltr_model_info,
    sha256_file,
)
from videntra.ranking.features import LTR_FEATURE_NAMES
from videntra.ranking.learned_ranker import LearnedRankerLTR


def _write_ltr_artifact(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return sha256_file(path)


def test_overlap_detection_catches_dev150_subset_of_dev_full_859(tmp_path):
    ltr_path = tmp_path / "ltr.json"
    _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "metadata": {"dataset": "dev_full_859"},
        },
    )
    ltr_info = load_ltr_model_info(ltr_path)
    gate = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={
            "dev_150": ["q1", "q2"],
            "dev_full_859": ["q1", "q2", "q3"],
            "heldout_test_201": ["t1", "t2"],
        },
    )
    assert gate["overlap_with_eval"] == {"dev_150": 2, "dev_full_859": 3, "heldout_test_201": 0}
    assert gate["eval_mode"] == "fit_all_leakage_detected"
    assert gate["leakage_free"] is False
    assert gate["gate_passed"] is False


def test_heldout_overlap_always_fails_gate_even_with_allow_insample(tmp_path):
    ltr_path = tmp_path / "ltr.json"
    _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1", "t1"],
        },
    )
    ltr_info = load_ltr_model_info(ltr_path)
    gate = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={"dev_150": ["q1"], "heldout_test_201": ["t1", "t2"]},
        allow_insample_eval=True,
    )
    assert gate["overlap_with_eval"]["heldout_test_201"] == 1
    assert gate["leakage_free"] is False
    assert gate["gate_passed"] is False


def test_oof_mode_passes_gate_when_fold_weights_cover_overlapping_queries(tmp_path):
    ltr_path = tmp_path / "ltr_oof.json"
    expected_sha = _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1", "q2", "q3"],
            "fold_weights": [
                {"fold": 0, "val_query_ids": ["q1", "q2"], "weights": [0.1] * len(LTR_FEATURE_NAMES)},
                {"fold": 1, "val_query_ids": ["q3"], "weights": [0.2] * len(LTR_FEATURE_NAMES)},
            ],
            "metadata": {"dataset": "dev_full_859"},
        },
    )
    ltr_info = load_ltr_model_info(ltr_path)
    gate = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={"dev_150": ["q1", "q2"], "heldout_test_201": ["t1"]},
        used_oof_splits={"dev_150"},
        degraded_channels=["e5_text", "caption_index", "e5_text"],
    )
    assert gate["ltr_model_sha256"] == expected_sha
    assert gate["overlap_with_eval"] == {"dev_150": 2, "heldout_test_201": 0}
    assert gate["eval_mode"] == "oof"
    assert gate["leakage_free"] is True
    assert gate["gate_passed"] is True
    assert gate["degraded_channels"] == ["caption_index", "e5_text"]


def test_evaluate_query_set_switches_oof_fold_weights_and_collects_degraded():
    n_feat = len(LTR_FEATURE_NAMES)
    ranker = LearnedRankerLTR(model_path=None)
    ranker.is_loaded = True
    ranker.weights = np.full(n_feat, 9.0, dtype=np.float64)

    observed_weights: list[float] = []

    class FakePipeline:
        ltr_ranker = ranker

    class FakeRunner:
        indexed_videos = {"v1", "v2"}
        dataset_videos = {"synthetic": indexed_videos}
        pipeline = FakePipeline()
        timings = {
            "t_total_ms": 2.0,
            "t_non_neural_ms": 1.0,
            "t_faiss_ms": 0.5,
            "t_fts5_ms": 0.1,
            "t_fusion_ms": 0.2,
            "t_boundary_ms": 0.2,
        }
        diagnostics = {"degraded": ["e5_caption_unavailable"]}

        def search_multimodal(self, query, top_k, ablation_config):
            observed_weights.append(float(self.pipeline.ltr_ranker.weights[0]))
            return [{
                "video_id": "v1",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "ranking_score": 1.0,
                "modality_attribution": "visual",
            }]

    fold_map = {
        "q1": {"weights": [0.25] * n_feat, "bias": 0.0},
        "q2": {"weights": [0.75] * n_feat, "bias": 0.0},
    }
    queries = [
        {"query_id": "q1", "query_text": "first", "dataset": "synthetic", "target_videos": ["v1"], "target_windows": []},
        {"query_id": "q2", "query_text": "second", "dataset": "synthetic", "target_videos": ["v1"], "target_windows": []},
    ]
    summary, _ = rvb.evaluate_query_set(FakeRunner(), queries, show_progress=False, fold_query_map=fold_map)
    assert observed_weights == [0.25, 0.75]
    assert float(ranker.weights[0]) == 9.0
    assert summary["degraded_channels"] == ["e5_caption_unavailable"]


def test_run_benchmark_writes_provenance_and_fails_on_insample_overlap(tmp_path, monkeypatch):
    ltr_path = tmp_path / "videntra_ltr_v1.json"
    ltr_sha = _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1"],
            "metadata": {"dataset": "dev_full_859"},
        },
    )

    class StubRunner:
        indexed_videos = {"v1"}
        dataset_videos = {"synthetic": {"v1"}}
        device = "cpu"
        timings = {
            "t_total_ms": 2.0,
            "t_non_neural_ms": 1.0,
            "t_faiss_ms": 0.5,
            "t_fts5_ms": 0.1,
            "t_fusion_ms": 0.2,
            "t_boundary_ms": 0.2,
        }
        diagnostics = {"degraded": ["ocr_fts_disabled"]}

        def __init__(self, ltr_model_path=None, **_kwargs):
            ranker = LearnedRankerLTR(model_path=ltr_model_path)
            self.pipeline = type("P", (), {"ltr_ranker": ranker, "close": lambda _s: None})()

        def search_multimodal(self, query, top_k, ablation_config):
            return [{
                "video_id": "v1",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "ranking_score": 1.0,
                "modality_attribution": "visual",
            }]

    monkeypatch.setattr(rvb, "MultimodalBenchmarkRunner", lambda **kwargs: StubRunner(**kwargs))
    monkeypatch.setattr(
        rvb,
        "load_validation_queries",
        lambda *_args, **_kwargs: [
            {"query_id": "q1", "query_text": "hello", "dataset": "synthetic", "target_videos": ["v1"], "target_windows": []}
        ],
    )
    monkeypatch.setattr(rvb, "get_git_sha", lambda: "abc1234")

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    out_dir = tmp_path / "bench_out"
    with pytest.raises(RuntimeError, match="Provenance leakage gate failed"):
        rvb.run_benchmark(
            val_dir=val_dir,
            test_dir=tmp_path / "missing_test",
            out_dir=out_dir,
            run_all_splits=False,
            allow_insample_eval=False,
            ltr_model_path=ltr_path,
        )

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    report_md = (out_dir / "benchmark_report.md").read_text(encoding="utf-8")

    assert prov["git_sha"] == "abc1234"
    assert prov["ltr_model_sha256"] == ltr_sha
    assert prov["ltr_trained_on"]["trained_query_ids"] == ["q1"]
    assert prov["overlap_with_eval"] == {"dev_150": 1}
    assert prov["eval_mode"] == "fit_all_leakage_detected"
    assert prov["leakage_free"] is False
    assert prov["degraded_channels"] == ["ocr_fts_disabled"]

    for key in ("git_sha", "ltr_model_sha256", "run_id", "eval_mode", "overlap_with_eval", "degraded_channels", "config"):
        assert key in summary
    assert summary["git_sha"] == "abc1234"
    assert summary["ltr_model_sha256"] == ltr_sha
    assert summary["eval_mode"] == "fit_all_leakage_detected"
    assert "abc1234" in report_md
    assert ltr_sha in report_md
    assert "fit_all_leakage_detected" in report_md


def test_oof_supports_eval_query_ids_schema_from_train_ltr_cv(tmp_path):
    ltr_path = tmp_path / "ltr_cv.json"
    _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "tree_ensemble",
            "feature_names": list(LTR_FEATURE_NAMES),
            "trees": [{"tree": "dummy"}],
            "fold_weights": [
                {
                    "fold": 0,
                    "eval_query_ids": ["q1", "q2"],
                    "weights": [0.1] * len(LTR_FEATURE_NAMES),
                    "bias": 0.0,
                    "feature_mean": [0.0] * len(LTR_FEATURE_NAMES),
                    "feature_std": [1.0] * len(LTR_FEATURE_NAMES),
                },
            ],
            "metadata": {
                "dataset": "dev_full_859",
                "trained_query_ids": ["q1", "q2"],
            },
        },
    )
    ltr_info = load_ltr_model_info(ltr_path)
    assert ltr_info["has_fold_weights"] is True
    assert set(ltr_info["fold_query_map"].keys()) == {"q1", "q2"}

    gate = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={"dev_150": ["q1"], "heldout_test_201": ["t1"]},
        used_oof_splits={"dev_150"},
    )
    assert gate["eval_mode"] == "oof"
    assert gate["gate_passed"] is True

    ranker = LearnedRankerLTR(model_path=None)
    ranker.model_type = "tree_ensemble"
    rvb._apply_ltr_fold_payload(ranker, ltr_info["fold_query_map"]["q1"])
    assert ranker.model_type == "pairwise_logistic"
    assert ranker.is_loaded is True


def test_oof_blocks_group_overlap_duplicate_missing_and_manifest_mismatch(tmp_path):
    base_fold = {
        "fold_id": 0,
        "train_group_ids": ["g1"],
        "eval_group_ids": ["g2"],
        "eval_query_ids": ["q1"],
        "weights": [0.1] * len(LTR_FEATURE_NAMES),
    }

    # 1. Train/eval group overlap
    p_overlap = tmp_path / "ltr_group_overlap.json"
    _write_ltr_artifact(
        p_overlap,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1"],
            "manifest_hash": "abc",
            "fold_weights": [{**base_fold, "eval_group_ids": ["g1", "g2"]}],
        },
    )
    gate_overlap = evaluate_provenance_gate(
        load_ltr_model_info(p_overlap),
        {"dev_150": ["q1"]},
        used_oof_splits={"dev_150"},
    )
    assert gate_overlap["leakage_free"] is False
    assert gate_overlap["gate_passed"] is False
    assert any("train_eval_group_overlap" in e for e in gate_overlap["oof_validation_errors"])

    # 2. Duplicate OOF query across folds
    p_dup = tmp_path / "ltr_dup.json"
    _write_ltr_artifact(
        p_dup,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1"],
            "fold_weights": [
                base_fold,
                {**base_fold, "fold_id": 1, "train_group_ids": ["g3"], "eval_group_ids": ["g4"]},
            ],
        },
    )
    gate_dup = evaluate_provenance_gate(
        load_ltr_model_info(p_dup),
        {"dev_150": ["q1"]},
        used_oof_splits={"dev_150"},
    )
    assert gate_dup["leakage_free"] is False
    assert any("duplicate_oof_query:q1" in e for e in gate_dup["oof_validation_errors"])

    # 3. Missing OOF query
    p_miss = tmp_path / "ltr_miss.json"
    _write_ltr_artifact(
        p_miss,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1", "q2"],
            "fold_weights": [base_fold],
        },
    )
    gate_miss = evaluate_provenance_gate(
        load_ltr_model_info(p_miss),
        {"dev_150": ["q1", "q2"]},
        used_oof_splits={"dev_150"},
    )
    assert gate_miss["leakage_free"] is False
    assert any("missing_oof_queries" in e for e in gate_miss["oof_validation_errors"])

    # 4. Manifest hash mismatch
    p_ok = tmp_path / "ltr_ok.json"
    _write_ltr_artifact(
        p_ok,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["q1"],
            "manifest_hash": "hash_v1",
            "fold_weights": [base_fold],
        },
    )
    gate_mm = evaluate_provenance_gate(
        load_ltr_model_info(p_ok),
        {"dev_150": ["q1"]},
        used_oof_splits={"dev_150"},
        expected_manifest_hash="hash_v2",
    )
    assert gate_mm["leakage_free"] is False
    assert any("manifest_hash_mismatch" in e for e in gate_mm["oof_validation_errors"])


def test_custom_ltr_model_path_changes_score_and_provenance_across_cwd(tmp_path, monkeypatch):
    ltr_a = tmp_path / "ltr_a.json"
    ltr_b = tmp_path / "ltr_b.json"
    sha_a = _write_ltr_artifact(
        ltr_a,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [1.0] + [0.0] * (len(LTR_FEATURE_NAMES) - 1),
            "trained_query_ids": ["train_only"],
        },
    )
    sha_b = _write_ltr_artifact(
        ltr_b,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [-1.0] + [0.0] * (len(LTR_FEATURE_NAMES) - 1),
            "trained_query_ids": ["train_only"],
        },
    )
    assert sha_a != sha_b

    class ScoreRunner:
        indexed_videos = {"v1"}
        dataset_videos = {"synthetic": {"v1"}}
        device = "cpu"
        timings = {
            "t_total_ms": 2.0, "t_non_neural_ms": 1.0, "t_faiss_ms": 0.5,
            "t_fts5_ms": 0.1, "t_fusion_ms": 0.2, "t_boundary_ms": 0.2,
        }
        diagnostics = {
            "degraded": {
                "e5_text_index": {"status": "missing_artifact", "reason": "e5 missing"},
                "e5_caption_index": {"status": "disabled_by_config", "reason": "disabled"},
                "spatial_patches": {"status": "ready", "reason": ""},
            }
        }

        def __init__(self, ltr_model_path=None):
            self.ltr_ranker = LearnedRankerLTR(model_path=ltr_model_path)
            self.pipeline = type("P", (), {"ltr_ranker": self.ltr_ranker, "close": lambda _s: None})()

        def search_multimodal(self, query, top_k, ablation_config):
            w0 = float(self.ltr_ranker.weights[0])
            return [{
                "video_id": "v1", "start_sec": 0.0, "end_sec": 5.0,
                "ranking_score": 0.5 + 0.1 * w0, "modality_attribution": "visual",
            }]

    monkeypatch.setattr(rvb, "MultimodalBenchmarkRunner", lambda ltr_model_path=None: ScoreRunner(ltr_model_path))
    monkeypatch.setattr(
        rvb,
        "load_validation_queries",
        lambda *_a, **_kw: [{"query_id": "eval_q", "query_text": "q", "dataset": "synthetic", "target_videos": ["v1"], "target_windows": []}],
    )
    monkeypatch.setattr(rvb, "hash_paths", lambda paths: {str(p): "stub" for p in paths})
    monkeypatch.setattr(rvb, "verify_hashes", lambda _h: True)

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    workdir = tmp_path / "other_cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    rvb.run_benchmark(val_dir=val_dir, test_dir=tmp_path / "no_test", out_dir=out_a, run_all_splits=False, ltr_model_path=ltr_a)
    rvb.run_benchmark(val_dir=val_dir, test_dir=tmp_path / "no_test", out_dir=out_b, run_all_splits=False, ltr_model_path=ltr_b)

    prov_a = json.loads((out_a / "provenance.json").read_text(encoding="utf-8"))
    prov_b = json.loads((out_b / "provenance.json").read_text(encoding="utf-8"))
    sum_a = json.loads((out_a / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert prov_a["ltr_model_sha256"] == sha_a
    assert prov_b["ltr_model_sha256"] == sha_b
    assert prov_a["ltr_model_resolved_path"] == str(ltr_a.resolve())
    assert sum_a["degraded_channels"] == ["e5_text_index"]
    assert sum_a["disabled_channels"] == ["e5_caption_index"]
    assert sum_a["channel_status"]["e5_text_index"]["status"] == "missing_artifact"
    assert sum_a["channel_status"]["e5_caption_index"]["status"] == "disabled_by_config"
    assert sum_a["channel_status"]["spatial_patches"]["status"] == "ready"

    rec_a = json.loads((out_a / "per_query_evaluation.jsonl").read_text(encoding="utf-8").strip())
    rec_b = json.loads((out_b / "per_query_evaluation.jsonl").read_text(encoding="utf-8").strip())
    score_a = json.loads(rec_a["ranked_predictions"])[0]["ranking_score"]
    score_b = json.loads(rec_b["ranked_predictions"])[0]["ranking_score"]
    assert score_a > score_b


def test_missing_or_corrupt_ltr_fails_early(tmp_path):
    with pytest.raises(RuntimeError, match="Invalid LTR model artifact"):
        rvb.run_benchmark(
            val_dir=tmp_path,
            out_dir=tmp_path / "out_missing",
            run_all_splits=False,
            ltr_model_path=tmp_path / "does_not_exist.json",
        )

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid LTR model artifact"):
        rvb.run_benchmark(
            val_dir=tmp_path,
            out_dir=tmp_path / "out_corrupt",
            run_all_splits=False,
            ltr_model_path=corrupt,
        )


def test_splits_excludes_heldout_and_records_feature_flags_in_config_and_provenance(tmp_path, monkeypatch):
    ltr_path = tmp_path / "videntra_ltr_v1.json"
    _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["train_only"],
        },
    )

    captured_runner_kwargs: list[dict] = []
    loaded_dirs: list[Path] = []

    class FlagRunner:
        indexed_videos = {"v1"}
        dataset_videos = {"msrvtt": {"v1"}}
        device = "cpu"
        timings = {
            "t_total_ms": 2.0, "t_non_neural_ms": 1.0, "t_faiss_ms": 0.5,
            "t_fts5_ms": 0.1, "t_fusion_ms": 0.2, "t_boundary_ms": 0.2,
        }
        diagnostics = {
            "channel_status": {
                "spatial_patches": {"status": "disabled_by_config", "reason": "disabled"},
                "vlm_caption_lexical": {"status": "disabled_by_config", "reason": "disabled"},
                "vlm_caption_dense": {"status": "disabled_by_config", "reason": "disabled"},
            },
            "visual_candidate_ids": ["v1"],
            "union_candidate_ids": ["v1"],
            "pre_rerank_video_ids": ["v1"],
            "reranker_input_ids": ["v1"],
            "ranked_video_ids": ["v1"],
        }

        def __init__(self, **kwargs):
            captured_runner_kwargs.append(kwargs)
            ranker = LearnedRankerLTR(model_path=kwargs.get("ltr_model_path"))
            self.pipeline = type("P", (), {"ltr_ranker": ranker, "close": lambda _s: None})()

        def search_multimodal(self, query, top_k, ablation_config):
            return [{
                "video_id": "v1", "start_sec": 0.0, "end_sec": 5.0,
                "ranking_score": 0.8, "modality_attribution": "visual",
            }]

    def fake_load_queries(qdir, indexed, max_msrvtt=None):
        loaded_dirs.append(Path(qdir))
        return [{"query_id": f"q_{max_msrvtt}", "query_text": "q", "dataset": "msrvtt", "target_videos": ["v1"], "target_windows": []}]

    monkeypatch.setattr(rvb, "MultimodalBenchmarkRunner", FlagRunner)
    monkeypatch.setattr(rvb, "load_validation_queries", fake_load_queries)
    monkeypatch.setattr(rvb, "hash_paths", lambda paths: {str(p): "stub" for p in paths})
    monkeypatch.setattr(rvb, "verify_hashes", lambda _h: True)

    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test_queries"
    val_dir.mkdir()
    test_dir.mkdir()
    out_dir = tmp_path / "out_flags"
    monkeypatch.chdir(tmp_path)

    rvb.run_benchmark(
        val_dir=val_dir,
        test_dir=test_dir,
        out_dir=out_dir,
        ltr_model_path=ltr_path,
        splits="dev_150,dev_full_859",
        temporal_rerank_mode="mean_top2",
        temporal_rerank_lambda=0.10,
        temporal_rerank_top_k=10,
        enable_spatial_patches=False,
        enable_vlm_caption_lexical=False,
        enable_vlm_caption_dense=False,
    )

    assert test_dir not in loaded_dirs
    assert len(captured_runner_kwargs) == 1
    kw = captured_runner_kwargs[0]
    assert kw["temporal_rerank_mode"] == "mean_top2"
    assert kw["temporal_rerank_lambda"] == 0.10
    assert kw["temporal_rerank_top_k"] == 10
    assert kw["enable_temporal_rerank"] is True
    assert kw["enable_spatial_patches"] is False
    assert kw["enable_vlm_caption_lexical"] is False
    assert kw["enable_vlm_caption_dense"] is False

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert "heldout_test_201" not in summary["multi_split_evaluation"]
    assert set(summary["multi_split_evaluation"].keys()) == {"dev_150", "dev_full_859"}
    assert summary["degraded_channels"] == []
    assert sorted(summary["disabled_channels"]) == ["spatial_patches", "vlm_caption_dense", "vlm_caption_lexical"]
    assert summary["config"]["splits"] == ["dev_150", "dev_full_859"]
    assert prov["selected_splits"] == ["dev_150", "dev_full_859"]
    assert prov["feature_flags"] == summary["config"]["feature_flags"]
    assert prov["feature_flags"]["temporal_rerank_mode"] == "mean_top2"
    assert prov["feature_flags"]["temporal_rerank_lambda"] == 0.10
    assert prov["feature_flags"]["enable_vlm_caption_dense"] is False


def test_provenance_blocks_heldout_source_and_media_overlap_and_resolves_shared_group_oof(tmp_path):
    from videntra.evaluation.provenance import resolve_oof_fold_for_query

    ltr_path = tmp_path / "ltr_source.json"
    _write_ltr_artifact(
        ltr_path,
        {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [0.0] * len(LTR_FEATURE_NAMES),
            "trained_query_ids": ["dev_q1", "dev_q2"],
            "fold_weights": [
                {
                    "fold_id": 0,
                    "model_type": "pairwise_logistic",
                    "train_group_ids": ["src_B"],
                    "eval_group_ids": ["src_A"],
                    "eval_query_ids": ["dev_q1"],
                    "weights": [0.1] * len(LTR_FEATURE_NAMES),
                },
                {
                    "fold_id": 1,
                    "model_type": "pairwise_logistic",
                    "train_group_ids": ["src_A"],
                    "eval_group_ids": ["src_B"],
                    "eval_query_ids": ["dev_q2"],
                    "weights": [0.2] * len(LTR_FEATURE_NAMES),
                },
            ],
            "metadata": {"dataset": "dev_full_859"},
        },
    )
    ltr_info = load_ltr_model_info(ltr_path)

    # Shared-source query not in trained_query_ids resolves OOF via eval_group_ids
    fold_p, status = resolve_oof_fold_for_query("dev_unseen_q3", ["src_A"], ltr_info)
    assert status == "oof_source_group"
    assert fold_p is not None and fold_p["fold_id"] == 0

    # Held-out query sharing canonical source group (AVC sibling window) must fail
    gate_src_leak = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={"heldout_test_201": ["test_t1"]},
        split_queries={
            "heldout_test_201": [
                {"query_id": "test_t1", "dataset": "avc", "target_videos": ["src_A__s30__e40"]}
            ]
        },
    )
    assert gate_src_leak["heldout_source_leakage"] is True
    assert gate_src_leak["eval_mode"] == "source_leakage_detected"
    assert gate_src_leak["gate_passed"] is False

    # Media SHA-256 overlap must fail even with disjoint IDs
    gate_media_leak = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids={"dev_full_859": ["dev_q1", "dev_q2"], "heldout_test_201": ["test_t2"]},
        used_oof_splits={"dev_full_859"},
        split_queries={
            "dev_full_859": [
                {"query_id": "dev_q1", "dataset": "msrvtt", "target_videos": ["src_A__s0__e10"]},
                {"query_id": "dev_q2", "dataset": "msrvtt", "target_videos": ["src_B__s0__e10"]},
            ],
            "heldout_test_201": [
                {"query_id": "test_t2", "dataset": "msrvtt", "target_videos": ["video9999"]}
            ],
        },
        media_sha256_by_video={"src_A__s0__e10": "same_sha256", "video9999": "same_sha256"},
    )
    assert gate_media_leak["media_sha256_overlap_with_eval"]["heldout_test_201"] == 1
    assert gate_media_leak["gate_passed"] is False


def test_oof_rejects_fold_model_type_mismatch_and_dirty_publish_reference(tmp_path, monkeypatch):
    bad_ltr = tmp_path / "bad_fold_type.json"
    _write_ltr_artifact(
        bad_ltr,
        {
            "model_type": "tree_ensemble",
            "feature_names": list(LTR_FEATURE_NAMES),
            "trees": [{"leaf_value": 0.0}],
            "trained_query_ids": ["q1"],
            "fold_weights": [
                {
                    "fold_id": 0,
                    "model_type": "pairwise_logistic",
                    "train_group_ids": ["g2"],
                    "eval_group_ids": ["g1"],
                    "eval_query_ids": ["q1"],
                    "weights": [0.1] * len(LTR_FEATURE_NAMES),
                }
            ],
        },
    )
    bad_info = load_ltr_model_info(bad_ltr)
    assert any("fold_model_type_mismatch" in e for e in bad_info["oof_validation_errors"])

    monkeypatch.setattr(
        rvb,
        "get_git_worktree_status",
        lambda: {"git_dirty": True, "git_dirty_count": 1, "git_dirty_files": ["M src/videntra/foo.py"]},
    )
    with pytest.raises(RuntimeError, match="dirty/unknown git worktree"):
        rvb.run_benchmark(
            out_dir=tmp_path / "pub_out",
            publish_reference=True,
        )
