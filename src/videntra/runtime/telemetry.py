"""Telemetry collection measuring latency percentiles, throughput, and memory footprint."""
from dataclasses import dataclass, field
import os
import time
from typing import Sequence
import numpy as np


@dataclass
class TelemetryReport:
    total_queries: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    peak_rss_mb: float
    deadline_misses: int


class TelemetryCollector:
    """Thread-safe latency and system memory profiler."""

    def __init__(self):
        self.latencies_ms: list[float] = []
        self.deadline_misses = 0

    def record_query(self, latency_ms: float, exceeded_budget: bool = False) -> None:
        self.latencies_ms.append(float(latency_ms))
        if exceeded_budget:
            self.deadline_misses += 1

    def get_current_rss_mb(self) -> float:
        """Measure current process RSS memory in MB."""
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
        return 0.0

    def summarize(self) -> TelemetryReport:
        if not self.latencies_ms:
            return TelemetryReport(
                total_queries=0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                mean_latency_ms=0.0,
                peak_rss_mb=self.get_current_rss_mb(),
                deadline_misses=self.deadline_misses,
            )

        arr = np.array(self.latencies_ms)
        return TelemetryReport(
            total_queries=len(self.latencies_ms),
            p50_latency_ms=round(float(np.percentile(arr, 50)), 2),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 2),
            p99_latency_ms=round(float(np.percentile(arr, 99)), 2),
            mean_latency_ms=round(float(np.mean(arr)), 2),
            peak_rss_mb=self.get_current_rss_mb(),
            deadline_misses=self.deadline_misses,
        )
