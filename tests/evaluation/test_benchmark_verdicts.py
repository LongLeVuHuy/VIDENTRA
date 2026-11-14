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
