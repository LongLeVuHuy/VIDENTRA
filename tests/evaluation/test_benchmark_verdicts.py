import json
from pathlib import Path

import pytest

from scripts.run_validation_benchmark import acceptance_metric, generate_markdown_report
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


def _minimal_benchmark_summary() -> dict:
    return {
        "benchmark_timestamp": "2026-11-23 00:00:00",
        "total_queries_evaluated": 150,
        "primary_split": "dev_150",
        "acceptance_metrics": [],
        "per_dataset_metrics": {},
        "metrics_overall": {
            "video_hit_at_1": 0.64,
            "video_hit_at_1_ci95": [0.56, 0.72],
            "video_hit_at_5": 0.85,
            "video_hit_at_10": 0.90,
            "mean_reciprocal_rank_mrr": 0.72,
            "mrr_ci95": [0.65, 0.78],
            "video_ndcg_at_1": 0.64,
            "video_ndcg_at_5": 0.71,
            "video_ndcg_at_10": 0.75,
            "video_ndcg_at_10_ci95": [0.68, 0.81],
            "moment_hit_at_1_iou03": 0.72,
            "moment_hit_at_1_iou05": 0.60,
            "moment_hit_at_1_iou05_ci95": [0.40, 0.78],
            "moment_hit_at_1_iou07": 0.45,
            "moment_ndcg_at_5": 0.52,
            "moment_ndcg_at_5_ci95": [0.35, 0.68],
            "moment_map_at_05": 0.50,
            "moment_map_at_075": 0.35,
            "moment_map_range_05_095": 0.32,
            "branch_recall": {
                "visual_recall_at_30": 0.80,
                "visual_recall_at_120": 0.92,
                "lexical_recall_at_30": 0.30,
                "fused_recall_at_10": 0.90,
            },
        },
    }


@pytest.mark.parametrize(("value", "expected"), [(20.0, "PASS"), (20.1, "FAIL"), (None, "NOT_MEASURED")])
def test_acceptance_verdict(value, expected):
    assert acceptance_metric("latency", value, 20, "max", "fixture", 100)["verdict"] == expected


def test_report_uses_same_verdict_as_summary(tmp_path):
    summary = _minimal_benchmark_summary()
    summary["acceptance_metrics"] = [acceptance_metric("non_neural_latency_p50_ms", 26.46, 20, "max", "fixture", 150)]
    path = tmp_path / "report.md"
    generate_markdown_report(path, summary, [])
    line = next(line for line in path.read_text(encoding="utf-8").splitlines() if "**Non-Neural Latency" in line)
    assert summary["acceptance_metrics"][0]["verdict"] in line


def test_missing_encoder_fails_explicitly(tmp_path):
    pipeline = UnifiedMultimodalSearchPipeline(model_dir=tmp_path / "missing", lazy_load=True, use_disk_cache=False)
    with pytest.raises(RuntimeError, match="Text encoder unavailable"):
        pipeline.encode_texts(["example"])


def test_report_renders_denominators_latency_separation_and_not_measured(tmp_path):
    summary = _minimal_benchmark_summary()
    summary["metrics_overall"]["moment_labeled_n"] = 22
    summary["metrics_overall"]["pure_asr_top1_count"] = 2
    summary["metrics_overall"]["pure_asr_fp_top1_count"] = 0
    summary["metrics_overall"]["cold_latency_mean_ms"] = None
    summary["metrics_overall"]["cold_latency_p50_ms"] = None
    summary["metrics_overall"]["cold_latency_p95_ms"] = None
    summary["metrics_overall"]["ram_rss_mb"] = None
    summary["metrics_overall"]["ram_peak_mb"] = None
    summary["metrics_overall"].pop("pvalue_vs_a0_hit1", None)
    summary["metrics_overall"].pop("pvalue_vs_a0_mrr", None)
    summary["metrics_overall"]["branch_recall"]["union_recall"] = 0.85
    summary["metrics_overall"]["branch_recall"]["shortlist_recall"] = 0.82
    summary["metrics_overall"]["branch_recall"]["shortlist_not_top1_count"] = 31
    summary["metrics_overall"]["branch_recall"]["shortlist_not_top1_rate"] = 0.2067

    path = tmp_path / "report.md"
    generate_markdown_report(path, summary, [])
    text = path.read_text(encoding="utf-8")
    assert "N=22" in text
    assert "0/2 pure-ASR" in text
    assert "End-to-End" in text
    assert "Non-Neural Latency" in text
    assert "Union Recall" in text
    assert "Shortlist Recall" in text
    assert "NOT_MEASURED" in text
    assert "None ms" not in text
    assert "None MB" not in text
