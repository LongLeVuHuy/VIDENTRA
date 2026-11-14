"""Generate measured E01 (probe/decode) and E02 (sampling comparison) benchmark reports with native RSS."""
import ctypes
from ctypes import wintypes
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from videntra.config.ingestion import IngestionConfig
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import probe_media
from videntra.sampling.compare import compute_structural_metrics, generate_uniform_baseline
from videntra.sampling.hybrid import sample_frames_hybrid
from videntra.sampling.probes import select_ocr_probes


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
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
        ("PrivateUsage", ctypes.c_size_t),
    ]


_mem_fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
_mem_fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
_mem_fn.restype = wintypes.BOOL


def get_native_memory_mb() -> dict[str, float]:
    """Measure actual process working set and private bytes via Win32 API."""
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(pmc)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if _mem_fn(handle, ctypes.byref(pmc), ctypes.sizeof(pmc)):
        return {
            "working_set_rss_mb": round(pmc.WorkingSetSize / (1024**2), 2),
            "peak_working_set_mb": round(pmc.PeakWorkingSetSize / (1024**2), 2),
            "private_bytes_mb": round(pmc.PrivateUsage / (1024**2), 2),
        }
    return {}


def get_ffmpeg_version() -> str:
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        return res.stdout.splitlines()[0]
    except Exception as exc:
        return f"unknown ({exc})"


def run_benchmarks(run_id: str = "run_m1_hardening_01"):
    # Ensure fixtures exist
    fixture_dir = Path("work/m1_fixtures")
    if not (fixture_dir / "receipt.json").exists():
        if fixture_dir.exists():
            import shutil
            shutil.rmtree(fixture_dir, ignore_errors=True)
        build_fixtures(fixture_dir, include_extended=True)

    receipt = json.loads((fixture_dir / "receipt.json").read_text("utf-8"))
    fixtures_list = receipt["fixtures"]

    cfg = IngestionConfig()
    e01_items = []
    e02_items = []

    total_decode_wall = 0.0
    total_sample_wall = 0.0
    total_frames = 0
    total_samples = 0

    pilot_dir = Path("work/m1/pilot_synthetic")

    for f_info in fixtures_list:
        name = f_info["name"]
        fn = f_info["file"]
        media_path = fixture_dir / fn
        if not media_path.exists() or f_info.get("truncated"):
            # Skip truncated corrupted tail for standard decode benchmark
            continue

        vid = f"synthetic:{name}"

        # E01 measurement
        t0 = time.perf_counter()
        probe_res = probe_media(media_path)
        t_probe = time.perf_counter() - t0

        t1 = time.perf_counter()
        frames = list(decode_frames(media_path, probe_res, include_pixels=True))
        t_decode = time.perf_counter() - t1
        total_decode_wall += t_decode
        total_frames += len(frames)

        safe_vid = vid.replace(":", "_")
        parquet_path = pilot_dir / "timeline" / safe_vid / "rev_m1_1" / "frames.parquet"
        file_bytes = parquet_path.stat().st_size if parquet_path.exists() else 0

        pts_list = [f.pts for f in frames if f.pts is not None]
        repeated_pts = sum(1 for f in frames if f.pts_occurrence > 0)
        missing_pts = sum(1 for f in frames if f.pts is None)
        epochs = sorted(list(set(f.epoch_id for f in frames)))

        audio_offset_us = None
        if probe_res.has_audio:
            shared_origin = frames[0].origin if frames else Fraction(0)
            from videntra.timeline.maps import build_audio_map
            amap = build_audio_map(media_path, probe_res, shared_origin)
            audio_offset_us = amap.audio_delay_us

        dur_sec = frames[-1].timestamp_seconds if frames else 0.0

        e01_items.append({
            "video_id": vid,
            "file": media_path.name,
            "duration_seconds": dur_sec,
            "decoded_frame_count": len(frames),
            "probe_elapsed_seconds": t_probe,
            "decode_elapsed_seconds": t_decode,
            "decode_fps": len(frames) / t_decode if t_decode > 0 else 0.0,
            "missing_pts_count": missing_pts,
            "repeated_pts_count": repeated_pts,
            "epochs": epochs,
            "audio_offset_us": audio_offset_us,
            "parquet_ledger_bytes": file_bytes,
        })

        # E02 measurement
        t2 = time.perf_counter()
        manifest_hybrid = sample_frames_hybrid(
            frames,
            cfg.sampling,
            vid,
            "rev_m1_1",
            "videntra_hybrid_v1",
        )
        t_sample = time.perf_counter() - t2
        total_sample_wall += t_sample
        total_samples += len(manifest_hybrid.samples)

        ocr_probes = select_ocr_probes(
            frames,
            vid,
            probe_fps=cfg.sampling.ocr_probe_fps,
            shot_cut_ordinals=manifest_hybrid.shot_cut_ordinals,
        )

        hybrid_ordinals = [s.original_decoded_frame_index for s in manifest_hybrid.samples]
        budget_count = len(hybrid_ordinals)

        uniform_1fps_ordinals = generate_uniform_baseline(frames, fps=1.0)
        uniform_matched_ordinals = generate_uniform_baseline(frames, budget_count=budget_count)

        m_hybrid = compute_structural_metrics(frames, hybrid_ordinals, max_allowed_gap_seconds=cfg.sampling.max_gap_seconds, mandatory_count=manifest_hybrid.mandatory_anchor_count)
        m_u1 = compute_structural_metrics(frames, uniform_1fps_ordinals, max_allowed_gap_seconds=cfg.sampling.max_gap_seconds)
        m_umatched = compute_structural_metrics(frames, uniform_matched_ordinals, max_allowed_gap_seconds=cfg.sampling.max_gap_seconds)

        reasons_dist = {}
        for s in manifest_hybrid.samples:
            for reason in s.reasons:
                reasons_dist[reason] = reasons_dist.get(reason, 0) + 1

        e02_items.append({
            "video_id": vid,
            "duration_seconds": dur_sec,
            "total_frames": len(frames),
            "hybrid_sampling": {
                "selected_count": len(manifest_hybrid.samples),
                "density_fps": m_hybrid.density_fps,
                "mandatory_anchors": manifest_hybrid.mandatory_anchor_count,
                "soft_samples": manifest_hybrid.soft_sample_count,
                "max_gap_violations": m_hybrid.max_gap_violations,
                "reasons_distribution": reasons_dist,
                "ocr_probe_count": len(ocr_probes.probes),
                "sampler_overhead_seconds": t_sample,
            },
            "uniform_1fps_baseline": {
                "selected_count": len(uniform_1fps_ordinals),
                "density_fps": m_u1.density_fps,
                "max_gap_violations": m_u1.max_gap_violations,
            },
            "uniform_matched_budget_baseline": {
                "budget_matched_count": budget_count,
                "selected_count": len(uniform_matched_ordinals),
                "density_fps": m_umatched.density_fps,
                "max_gap_violations": m_umatched.max_gap_violations,
            },
            "event_visibility_recall": "NOT_EVALUATED",
            "event_visibility_note": "Synthetic fixtures lack human ground truth event spans; real event visibility pending D04 dataset.",
        })

    native_mem = get_native_memory_mb()

    e01_report = {
        "experiment_id": "E01",
        "title": "Probe and Decoding Baseline Measurement Report",
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "working_set_rss_mb": native_mem.get("working_set_rss_mb"),
            "peak_working_set_mb": native_mem.get("peak_working_set_mb"),
            "private_bytes_mb": native_mem.get("private_bytes_mb"),
            "ffmpeg_version": get_ffmpeg_version(),
            "pyav_version": av.__version__,
            "pyarrow_version": pa.__version__,
            "numpy_version": np.__version__,
        },
        "summary": {
            "total_media_files": len(e01_items),
            "total_decoded_frames": total_frames,
            "total_decode_wall_seconds": round(total_decode_wall, 4),
            "aggregate_decode_fps": round(total_frames / total_decode_wall, 2) if total_decode_wall > 0 else 0.0,
        },
        "items": e01_items,
    }

    e02_report = {
        "experiment_id": "E02",
        "title": "Adaptive Hybrid Sampling vs Uniform Baselines Report",
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling_config": cfg.sampling.to_dict() if hasattr(cfg.sampling, "to_dict") else cfg.sampling.__dict__,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "working_set_rss_mb": native_mem.get("working_set_rss_mb"),
            "peak_working_set_mb": native_mem.get("peak_working_set_mb"),
        },
        "summary": {
            "total_videos": len(e02_items),
            "total_frames_analyzed": total_frames,
            "total_hybrid_samples": total_samples,
            "total_sampler_wall_seconds": round(total_sample_wall, 4),
            "rights_gate_status": "QVH pilot blocked by rights review",
            "rights_gate_details": "24 real QVH clips remain rights_status=unknown pending formal non-commercial licensing/rights activation review. Synthetic pilot verified end-to-end.",
        },
        "items": e02_items,
    }

    # Save to immutable experiment run directory
    exp_dir = Path("work/experiments") / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "e01_report.json", "w", encoding="utf-8") as f:
        json.dump(e01_report, f, indent=2)
    with open(exp_dir / "e02_report.json", "w", encoding="utf-8") as f:
        json.dump(e02_report, f, indent=2)

    # Also update work/m1 pointers
    out_e01 = Path("work/m1/e01_report.json")
    out_e02 = Path("work/m1/e02_report.json")
    out_e01.parent.mkdir(parents=True, exist_ok=True)
    with open(out_e01, "w", encoding="utf-8") as f:
        json.dump(e01_report, f, indent=2)
    with open(out_e02, "w", encoding="utf-8") as f:
        json.dump(e02_report, f, indent=2)

    print(f"Generated reports in {exp_dir} and updated {out_e01}, {out_e02}")


if __name__ == "__main__":
    run_benchmarks()
