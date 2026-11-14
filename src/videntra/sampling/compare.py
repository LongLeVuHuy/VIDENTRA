"""Uniform baseline generation and structural sampling comparison."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from videntra.ingestion.decode import DecodedFrame
from videntra.sampling.hybrid import SampleManifest


@dataclass(frozen=True)
class StructuralMetrics:
    total_frames: int
    selected_count: int
    duration_seconds: float
    density_fps: float
    inter_sample_gap_mean: float
    inter_sample_gap_max: float
    max_gap_violations: int
    mandatory_count: int
    soft_count: int


def generate_uniform_baseline(
    frames: Sequence[DecodedFrame],
    fps: float = 1.0,
    budget_count: int | None = None,
) -> list[int]:
    """Generate uniform baseline frame ordinals snapped to nearest available decoded frames."""
    if not frames:
        return []

    timestamps = [f.timestamp_seconds or 0.0 for f in frames]
    duration = timestamps[-1] - timestamps[0]

    if budget_count is not None:
        target_count = max(1, budget_count)
        if target_count == 1:
            grid_times = [timestamps[0]]
        else:
            grid_times = [timestamps[0] + (i / (target_count - 1)) * duration for i in range(target_count)]
    else:
        step = 1.0 / fps
        target_count = int(np.floor(duration * fps)) + 1
        grid_times = [timestamps[0] + i * step for i in range(target_count)]

    # Snap each grid time to the nearest frame
    selected_ordinals = []
    ts_arr = np.array(timestamps)
    for gt in grid_times:
        nearest_idx = int(np.argmin(np.abs(ts_arr - gt)))
        ord_val = frames[nearest_idx].ordinal
        if not selected_ordinals or selected_ordinals[-1] != ord_val:
            selected_ordinals.append(ord_val)

    return selected_ordinals


def compute_structural_metrics(
    frames: Sequence[DecodedFrame],
    selected_ordinals: Sequence[int],
    max_allowed_gap_seconds: float = 2.0,
    mandatory_count: int = 0,
) -> StructuralMetrics:
    """Compute structural metrics on selected frame ordinals."""
    if not selected_ordinals:
        return StructuralMetrics(
            total_frames=len(frames),
            selected_count=0,
            duration_seconds=0.0,
            density_fps=0.0,
            inter_sample_gap_mean=0.0,
            inter_sample_gap_max=0.0,
            max_gap_violations=0,
            mandatory_count=0,
            soft_count=0,
        )

    ord_to_ts = {f.ordinal: (f.timestamp_seconds or 0.0) for f in frames}
    sel_ts = [ord_to_ts[ord_val] for ord_val in sorted(set(selected_ordinals))]

    duration = (frames[-1].timestamp_seconds or 0.0) - (frames[0].timestamp_seconds or 0.0) if len(frames) > 1 else 0.0
    density = len(sel_ts) / duration if duration > 0 else float(len(sel_ts))

    if len(sel_ts) > 1:
        gaps = np.diff(sel_ts)
        gap_mean = float(np.mean(gaps))
        gap_max = float(np.max(gaps))
        tolerance = max_allowed_gap_seconds + 0.25
        violations = int(np.sum(gaps > tolerance))
    else:
        gap_mean = 0.0
        gap_max = 0.0
        violations = 0

    return StructuralMetrics(
        total_frames=len(frames),
        selected_count=len(sel_ts),
        duration_seconds=duration,
        density_fps=density,
        inter_sample_gap_mean=gap_mean,
        inter_sample_gap_max=gap_max,
        max_gap_violations=violations,
        mandatory_count=mandatory_count,
        soft_count=len(sel_ts) - mandatory_count,
    )
