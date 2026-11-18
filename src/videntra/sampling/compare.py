"""Uniform baseline generation, structural metrics, and multi-strategy keyframe quality comparison."""
from dataclasses import dataclass, replace
import time
from typing import Sequence

import numpy as np

from videntra.config.ingestion import SamplingConfig, VALID_SAMPLING_STRATEGIES
from videntra.ingestion.decode import DecodedFrame
from videntra.sampling.hybrid import (
    SampleManifest,
    _compute_phash_64_from_luma,
    _compute_sharpness_from_luma,
    _rgb_to_luma,
    compute_grid_hist_512,
    phash_hamming_distance,
    resize_lowres,
    sample_frames_hybrid,
)


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


@dataclass(frozen=True)
class QualityBenchmarkMetrics:
    strategy: str
    total_frames: int
    selected_count: int
    duration_seconds: float
    density_fps: float
    inter_sample_gap_mean: float
    inter_sample_gap_max: float
    max_gap_violations: int
    mandatory_count: int
    soft_count: int
    mean_representativeness: float
    p10_representativeness: float
    mean_sharpness: float
    sharpness_gain_ratio: float
    blank_frame_rate: float
    soft_redundancy_rate: float
    elapsed_ms: float = 0.0


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

    selected_ordinals: list[int] = []
    ts_arr = np.array(timestamps, dtype=np.float64)
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
    """Compute structural coverage and gap metrics on selected frame ordinals."""
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
    sel_ts = [ord_to_ts[ord_val] for ord_val in sorted(set(selected_ordinals)) if ord_val in ord_to_ts]

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
        soft_count=max(0, len(sel_ts) - mandatory_count),
    )


def precompute_eval_signals(
    frames: Sequence[DecodedFrame],
    *,
    lowres_max_dim: int = 256,
    min_luma_std: float = 8.0,
) -> dict[int, dict[str, object]]:
    """Precompute lowres histograms, sharpness, blank flags, and pHashes keyed by frame ordinal."""
    signals: dict[int, dict[str, object]] = {}
    for f in frames:
        if f.pixels is None:
            signals[f.ordinal] = {
                "hist": np.zeros(16 * 512, dtype=np.float32),
                "sharpness": 0.0,
                "is_blank": True,
                "phash": 0,
            }
            continue
        lowres = resize_lowres(f.pixels, lowres_max_dim)
        luma = _rgb_to_luma(lowres)
        luma_std = float(np.std(luma))
        hist = compute_grid_hist_512(lowres).reshape(-1)
        signals[f.ordinal] = {
            "hist": hist,
            "sharpness": _compute_sharpness_from_luma(luma),
            "is_blank": luma_std < min_luma_std,
            "phash": _compute_phash_64_from_luma(luma),
        }
    return signals


def evaluate_keyframe_quality(
    frames: Sequence[DecodedFrame],
    selected: SampleManifest | Sequence[int],
    *,
    strategy_name: str | None = None,
    max_allowed_gap_seconds: float = 2.0,
    min_luma_std: float = 8.0,
    elapsed_ms: float = 0.0,
    precomputed: dict[int, dict[str, object]] | None = None,
    max_eval_frames: int | None = None,
) -> QualityBenchmarkMetrics:
    """Evaluate representativeness, sharpness gain, blank rate, and redundancy for selected keyframes."""
    if isinstance(selected, SampleManifest):
        ordinals = [s.original_decoded_frame_index for s in selected.samples]
        mandatory_map = {s.original_decoded_frame_index: s.is_mandatory for s in selected.samples}
        mandatory_count = selected.mandatory_anchor_count
        strat_label = strategy_name or selected.sampler_revision
    else:
        ordinals = sorted(set(int(o) for o in selected))
        mandatory_map = {}
        mandatory_count = 0
        strat_label = strategy_name or "custom"

    struct = compute_structural_metrics(
        frames,
        ordinals,
        max_allowed_gap_seconds=max_allowed_gap_seconds,
        mandatory_count=mandatory_count,
    )
    if not frames or not ordinals:
        return QualityBenchmarkMetrics(
            strategy=strat_label,
            total_frames=struct.total_frames,
            selected_count=struct.selected_count,
            duration_seconds=struct.duration_seconds,
            density_fps=struct.density_fps,
            inter_sample_gap_mean=struct.inter_sample_gap_mean,
            inter_sample_gap_max=struct.inter_sample_gap_max,
            max_gap_violations=struct.max_gap_violations,
            mandatory_count=struct.mandatory_count,
            soft_count=struct.soft_count,
            mean_representativeness=0.0,
            p10_representativeness=0.0,
            mean_sharpness=0.0,
            sharpness_gain_ratio=1.0,
            blank_frame_rate=0.0,
            soft_redundancy_rate=0.0,
            elapsed_ms=elapsed_ms,
        )

    sigs = precomputed if precomputed is not None else precompute_eval_signals(frames, min_luma_std=min_luma_std)

    valid_ordinals = [o for o in ordinals if o in sigs]
    eval_frames = frames
    if max_eval_frames is not None and max_eval_frames > 0 and len(frames) > max_eval_frames:
        step = max(1, len(frames) // max_eval_frames)
        eval_frames = frames[::step]
    all_hists = np.stack([sigs[f.ordinal]["hist"] for f in eval_frames], axis=0)  # (N_eval, 8192)
    sel_hists = np.stack([sigs[o]["hist"] for o in valid_ordinals], axis=0)  # (K, 8192)

    # Chunked exact spatial histogram intersection coverage: max_{s in sel} (1 - hist_dist(f, s))
    max_cov = np.zeros(len(eval_frames), dtype=np.float32)
    chunk_sz = 16
    for st in range(0, len(eval_frames), chunk_sz):
        ed = min(len(eval_frames), st + chunk_sz)
        f_chunk = all_hists[st:ed]
        inter = np.minimum(f_chunk[:, None, :], sel_hists[None, :, :]).sum(axis=-1) / 16.0
        max_cov[st:ed] = np.max(inter, axis=1)

    mean_rep = float(np.mean(max_cov))
    p10_rep = float(np.percentile(max_cov, 10))

    all_sharp = [float(sigs[f.ordinal]["sharpness"]) for f in frames]
    sel_sharp = [float(sigs[o]["sharpness"]) for o in valid_ordinals]
    mean_all_sharp = float(np.mean(all_sharp)) if all_sharp else 0.0
    mean_sel_sharp = float(np.mean(sel_sharp)) if sel_sharp else 0.0
    sharp_gain = (mean_sel_sharp / mean_all_sharp) if mean_all_sharp > 1e-6 else 1.0

    blank_rate = sum(1 for o in valid_ordinals if sigs[o]["is_blank"]) / max(1, len(valid_ordinals))

    # Redundancy among non-mandatory (or all trailing) samples against prior 3 samples
    recent_phashes: list[int] = []
    soft_total = 0
    soft_dup = 0
    for o in valid_ordinals:
        ph = int(sigs[o]["phash"])
        is_mand = mandatory_map.get(o, False)
        if not is_mand and recent_phashes:
            soft_total += 1
            if min(phash_hamming_distance(prev_ph, ph) for prev_ph in recent_phashes[-3:]) <= 4:
                soft_dup += 1
        recent_phashes.append(ph)
    soft_redun = (soft_dup / soft_total) if soft_total > 0 else 0.0

    return QualityBenchmarkMetrics(
        strategy=strat_label,
        total_frames=struct.total_frames,
        selected_count=struct.selected_count,
        duration_seconds=round(struct.duration_seconds, 4),
        density_fps=round(struct.density_fps, 4),
        inter_sample_gap_mean=round(struct.inter_sample_gap_mean, 4),
        inter_sample_gap_max=round(struct.inter_sample_gap_max, 4),
        max_gap_violations=struct.max_gap_violations,
        mandatory_count=struct.mandatory_count,
        soft_count=struct.soft_count,
        mean_representativeness=round(mean_rep, 4),
        p10_representativeness=round(p10_rep, 4),
        mean_sharpness=round(mean_sel_sharp, 3),
        sharpness_gain_ratio=round(sharp_gain, 4),
        blank_frame_rate=round(blank_rate, 4),
        soft_redundancy_rate=round(soft_redun, 4),
        elapsed_ms=round(elapsed_ms, 3),
    )


def compare_sampling_strategies(
    frames: Sequence[DecodedFrame],
    *,
    base_config: SamplingConfig | None = None,
    strategies: Sequence[str] | None = None,
    video_id: str = "compare_video",
    timeline_revision: str = "rev_1",
    include_uniform_matched_budget: bool = True,
) -> list[QualityBenchmarkMetrics]:
    """Benchmark multiple keyframe sampling strategies on the same decoded stream."""
    cfg = base_config or SamplingConfig()
    strat_list = list(strategies or VALID_SAMPLING_STRATEGIES)
    precomputed = precompute_eval_signals(frames, lowres_max_dim=cfg.lowres_max_dim, min_luma_std=cfg.min_luma_std)

    max_gap_limit = cfg.max_static_gap_seconds or cfg.max_gap_seconds
    results: list[QualityBenchmarkMetrics] = []
    adaptive_budget: int | None = None

    for strat in strat_list:
        s_cfg = replace(cfg, strategy=strat)
        t0 = time.perf_counter()
        manifest = sample_frames_hybrid(frames, s_cfg, video_id=video_id, timeline_revision=timeline_revision)
        elapsed = (time.perf_counter() - t0) * 1000.0
        metrics = evaluate_keyframe_quality(
            frames,
            manifest,
            strategy_name=strat,
            max_allowed_gap_seconds=max_gap_limit,
            min_luma_std=cfg.min_luma_std,
            elapsed_ms=elapsed,
            precomputed=precomputed,
        )
        results.append(metrics)
        if strat == "hybrid_v2_adaptive":
            adaptive_budget = metrics.selected_count

    if include_uniform_matched_budget and adaptive_budget is not None and adaptive_budget > 0:
        t0 = time.perf_counter()
        matched_ords = generate_uniform_baseline(frames, budget_count=adaptive_budget)
        elapsed = (time.perf_counter() - t0) * 1000.0
        results.append(
            evaluate_keyframe_quality(
                frames,
                matched_ords,
                strategy_name="uniform_matched_budget",
                max_allowed_gap_seconds=max_gap_limit,
                min_luma_std=cfg.min_luma_std,
                elapsed_ms=elapsed,
                precomputed=precomputed,
            )
        )

    return results
