import json
from pathlib import Path

import pytest

from scripts.run_validation_benchmark import acceptance_metric, generate_markdown_report
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


@pytest.mark.parametrize(("value", "expected"), [(20.0, "PASS"), (20.1, "FAIL"), (None, "NOT_MEASURED")])
def test_acceptance_verdict(value, expected):
    assert acceptance_metric("latency", value, 20, "max", "fixture", 100)["verdict"] == expected


def test_report_uses_same_verdict_as_summary(tmp_path):
    summary = json.loads(Path("benchmark/benchmark_summary.json").read_text(encoding="utf-8"))
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
    summary = json.loads(Path("benchmark/benchmark_summary.json").read_text(encoding="utf-8"))
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
