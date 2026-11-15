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

    class StubPipeline:
        ltr_ranker = LearnedRankerLTR(model_path=None)

        def close(self):
            pass

    class StubRunner:
        indexed_videos = {"v1"}
        dataset_videos = {"synthetic": {"v1"}}
        device = "cpu"
        pipeline = StubPipeline()
        timings = {
            "t_total_ms": 2.0,
            "t_non_neural_ms": 1.0,
            "t_faiss_ms": 0.5,
            "t_fts5_ms": 0.1,
            "t_fusion_ms": 0.2,
            "t_boundary_ms": 0.2,
        }
        diagnostics = {"degraded": ["ocr_fts_disabled"]}

        def search_multimodal(self, query, top_k, ablation_config):
            return [{
                "video_id": "v1",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "ranking_score": 1.0,
                "modality_attribution": "visual",
            }]

    monkeypatch.setattr(rvb, "MultimodalBenchmarkRunner", lambda: StubRunner())
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
