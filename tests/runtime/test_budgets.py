"""Tests for deadline budgets and telemetry collectors."""
import time
import pytest

from videntra.runtime.budgets import DeadlineBudget
from videntra.runtime.telemetry import TelemetryCollector


def test_deadline_budget():
    budget = DeadlineBudget(max_duration_seconds=0.05)
    assert not budget.is_expired()
    assert budget.remaining_seconds > 0.0

    time.sleep(0.06)
    assert budget.is_expired()
    with pytest.raises(TimeoutError, match="Execution budget exceeded"):
        budget.check()


def test_telemetry_collector_percentiles():
    collector = TelemetryCollector()
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    for lat in latencies:
        collector.record_query(lat)

    report = collector.summarize()
    assert report.total_queries == 10
    assert report.mean_latency_ms == 55.0
    assert report.p50_latency_ms == 55.0
    assert report.p95_latency_ms > 90.0
    assert report.deadline_misses == 0
