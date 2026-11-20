"""VIDENTRA Multimodal Search Benchmark & Evaluation Protocol (P0-P4).

Evaluates:
- Video Hit@1, Hit@5, Hit@10, MRR, Video nDCG@1, nDCG@5, nDCG@10
- Moment Hit@1 @ tIoU 0.3, 0.5, 0.7; Moment mAP@0.5, mAP@0.75, mAP@[0.5:0.95:0.05]
- Graded Moment nDCG@5, nDCG@10; Temporal Boundary MAE (seconds)
- Dual Candidate Pool Reporting (Global 1,324 Videos vs Global-then-dataset-filter) (E3)
- Branch Recall@K (Visual Recall@30/120, Lexical Recall@30, Fused Recall@10) (E8)
- Bootstrap 95% Confidence Intervals & Paired Sign/Permutation Test p-values (E4)
- Automated Failure Taxonomy (F1..F5) & Modality Attribution (E10)
- 6-Step Ablation Matrix (A0 -> A5) on dev_150
- Multi-split evaluation: dev_150, dev_full_859, heldout_test_201 (E1, E2)
- Cold vs Warm / Non-Neural Latency & Peak RAM Footprint (C3)
- Immutable Experiment Provenance Logging in work/experiments/ (E9)

Outputs:
- benchmark/per_query_evaluation.csv
- benchmark/per_query_evaluation.jsonl
- benchmark/benchmark_summary.json
- benchmark/benchmark_report.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import platform
import uuid
from typing import Any, Sequence

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None

from videntra.config.device import get_device_info, resolve_device
from videntra.retrieval.unified_pipeline import (
    UnifiedMultimodalSearchPipeline,
    analyze_query_lexical_tokens,
)
from videntra.evaluation.evaluator import (
    classify_query_failure,
    compute_binary_ndcg_at_k,
    compute_bootstrap_ci,
    compute_graded_moment_ndcg_at_k,
    compute_moment_ap_at_iou,
    compute_moment_map_range,
    compute_paired_sign_test_pvalue,
)
from videntra.evaluation.provenance import (
    compute_code_sha256,
    compute_effective_config_sha256,
    evaluate_provenance_gate,
    get_cpu_thread_settings,
    hash_paths,
    load_ltr_model_info,
    resolve_a5_preset,
    resolve_oof_fold_for_query,
    resolve_query_source_groups,
    verify_hashes,
)

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def get_git_worktree_status() -> dict[str, Any]:
    """Return git worktree dirty status and changed file paths."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [line.rstrip() for line in out.splitlines() if line.strip()]
        return {
            "git_dirty": bool(lines),
            "git_dirty_count": len(lines),
            "git_dirty_files": lines[:50],
        }
    except Exception:
        return {
            "git_dirty": None,
            "git_dirty_count": None,
            "git_dirty_files": [],
        }


def get_hardware_info(device: str = "cpu") -> dict[str, Any]:
    """Collect local CPU/GPU/VRAM/RAM/platform hardware metadata."""
    dev_meta = get_device_info(device)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": dev_meta.get("ram_total_gb"),
        "device": device,
        "cuda_available": bool(dev_meta.get("cuda_available", False)),
        "gpu_name": dev_meta.get("gpu_name"),
        "cuda_version": dev_meta.get("cuda_version"),
        "vram_total_mb": dev_meta.get("vram_total_mb"),
    }


def get_process_memory_mb() -> tuple[float | None, float | None]:
    """Returns (rss_mb, peak_mb) using psutil or Windows ctypes GetProcessMemoryInfo fallback."""
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            rss_mb = round(mem_info.rss / (1024.0 * 1024.0), 1)
            peak_mb = round(mem_info.peak_wset / (1024.0 * 1024.0), 1) if hasattr(mem_info, "peak_wset") else None
            return rss_mb, peak_mb
        except Exception:
            pass
    if sys.platform == "win32":
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

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                rss_mb = round(pmc.WorkingSetSize / (1024.0 * 1024.0), 1)
                peak_mb = round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 1)
                return rss_mb, peak_mb
        except Exception:
            pass
    return None, None


def print_progress(current: int, total: int, split_label: str, hit1: float, mrr: float, lat: float) -> None:
    bar_len = 22
    pct = (current / total) * 100.0 if total > 0 else 100.0
    filled = int(bar_len * (pct / 100.0))
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(
        f"\r[{split_label:<16}] [{bar}] {pct:5.1f}% | {current:3d}/{total} | Hit@1: {hit1*100:5.1f}% | MRR: {mrr:5.3f} | {lat:4.1f}ms"
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


class MultimodalBenchmarkRunner:
    """Benchmark runner delegating 100% of retrieval logic to UnifiedMultimodalSearchPipeline (SSOT)."""

    def __init__(
        self,
        db_path: Path = Path("data/processed/metadata.db"),
        faiss_path: Path = Path("data/processed/faiss_index.bin"),
        model_dir: Path = Path("models/siglip2"),
        device: str = "cpu",
        use_disk_cache: bool = False,
        ltr_model_path: Path | str | None = Path("models/ranking/videntra_ltr_v1.json"),
        enable_spatial_patches: bool = True,
        enable_vlm_caption_lexical: bool = False,
        enable_vlm_caption_dense: bool = False,
        enable_temporal_rerank: bool | None = None,
        temporal_rerank_mode: str = "attention",
        temporal_rerank_lambda: float = 0.0,
        temporal_rerank_top_k: int = 10,
        enable_clip_consensus_boost: bool = True,
        ltr_blend_weight: float = 0.015,
        stage2_selection_mode: str = "exact",
        video_proposal_mode: str = "frame_topk",
        video_proposal_top_n: int = 60,
        video_proposal_anchors_per_video: int = 2,
        video_proposal_min_pts_gap: float = 1.0,
        video_proposal_clip_reserve_ratio: float = 0.0,
        disable_pure_visual_lexical_bonus: bool = False,
    ):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.model_dir = model_dir
        self.device = device
        self.ltr_model_path = Path(ltr_model_path) if ltr_model_path is not None else None
        self.ranked_videos: list[Any] | None = None
        resolved_enable_temporal = (
            bool(float(temporal_rerank_lambda) > 0.0)
            if enable_temporal_rerank is None
            else bool(enable_temporal_rerank)
        )
        self.feature_flags = {
            "stage2_selection_mode": str(stage2_selection_mode),
            "video_proposal_mode": str(video_proposal_mode),
            "video_proposal_top_n": int(video_proposal_top_n),
            "video_proposal_anchors_per_video": int(video_proposal_anchors_per_video),
            "video_proposal_min_pts_gap": float(video_proposal_min_pts_gap),
            "video_proposal_clip_reserve_ratio": float(video_proposal_clip_reserve_ratio),
            "disable_pure_visual_lexical_bonus": bool(disable_pure_visual_lexical_bonus),
            "enable_spatial_patches": bool(enable_spatial_patches),
            "enable_vlm_caption_lexical": bool(enable_vlm_caption_lexical),
            "enable_vlm_caption_dense": bool(enable_vlm_caption_dense),
            "enable_temporal_rerank": resolved_enable_temporal,
            "temporal_rerank_mode": str(temporal_rerank_mode),
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
            "temporal_rerank_top_k": int(temporal_rerank_top_k),
            "enable_clip_consensus_boost": bool(enable_clip_consensus_boost),
            "ltr_blend_weight": float(ltr_blend_weight),
        }

        print("1. Đang nạp UnifiedMultimodalSearchPipeline (SSOT)...")
        t0 = time.time()
        self.pipeline = UnifiedMultimodalSearchPipeline(
            db_path=db_path,
            faiss_path=faiss_path,
            model_dir=model_dir,
            device=device,
            use_disk_cache=use_disk_cache,
            memory_cache_size=0,
            ltr_model_path=ltr_model_path,
            ltr_blend_weight=float(ltr_blend_weight),
            enable_spatial_patches=bool(enable_spatial_patches),
            enable_vlm_caption_lexical=bool(enable_vlm_caption_lexical),
            enable_vlm_caption_dense=bool(enable_vlm_caption_dense),
            enable_temporal_rerank=resolved_enable_temporal,
            temporal_rerank_mode=str(temporal_rerank_mode),
            temporal_rerank_lambda=float(temporal_rerank_lambda),
            temporal_rerank_top_k=int(temporal_rerank_top_k),
            enable_clip_consensus_boost=bool(enable_clip_consensus_boost),
            stage2_selection_mode=str(stage2_selection_mode),
            video_proposal_mode=str(video_proposal_mode),
            video_proposal_top_n=int(video_proposal_top_n),
            video_proposal_anchors_per_video=int(video_proposal_anchors_per_video),
            video_proposal_min_pts_gap=float(video_proposal_min_pts_gap),
            video_proposal_clip_reserve_ratio=float(video_proposal_clip_reserve_ratio),
            disable_pure_visual_lexical_bonus=bool(disable_pure_visual_lexical_bonus),
        )
        self.indexed_videos = set(self.pipeline.video_durations.keys()) if self.pipeline.video_durations else set()
        if not self.indexed_videos and self.pipeline.keyframe_map:
            self.indexed_videos = {k["video_id"] for k in self.pipeline.keyframe_map.values()}

        # Partition indexed videos by dataset for Within-Dataset Candidate Pool reporting (E3)
        self.dataset_videos: dict[str, set[str]] = {
            "msrvtt": {v for v in self.indexed_videos if v.startswith("video")},
            "avc": {v for v in self.indexed_videos if "__s" in v},
            "qvhighlights": {v for v in self.indexed_videos if not v.startswith("video") and "__s" not in v},
        }
        print(
            f"   UnifiedMultimodalSearchPipeline sẵn sàng sau {time.time() - t0:.2f}s "
            f"(Tổng: {len(self.indexed_videos)} videos | QVH: {len(self.dataset_videos['qvhighlights'])}, "
            f"AVC: {len(self.dataset_videos['avc'])}, MSR-VTT: {len(self.dataset_videos['msrvtt'])})."
        )

    def search_multimodal(
        self,
        query: str,
        top_k: int = 10,
        ablation_config: str = "A5",
    ) -> list[dict[str, Any]]:
        response = self.pipeline.search_with_metadata(
            snapshot_id="unverified_local_snapshot",
            query=query,
            top_k=top_k,
            mode="accurate",
            ranking_mode="diverse_video",
            ablation_config=ablation_config,
        )
        self.timings = response["timings"]
        self.diagnostics = response["diagnostics"]
        self.ranked_videos = response.get("ranked_videos")
        moments = response["moments"]
        return [
            {
                "video_id": m.video_id,
                "start_sec": float(m.moment_start_seconds),
                "end_sec": float(m.moment_end_seconds),
                "ranking_score": float(m.ranking_score),
                "modality_attribution": m.match_status,
            }
            for m in moments
        ]


def load_validation_queries(
    val_dir: Path,
    indexed_videos: set[str],
    max_msrvtt: int | None = 100,
) -> list[dict[str, Any]]:
    """Loads queries from validation/ directory (max_msrvtt=100 -> dev_150; max_msrvtt=None -> dev_full_859)."""
    queries: list[dict[str, Any]] = []

    # 1. QVHighlights
    qvh_file = val_dir / "qvhighlights" / "highlight_val_release.jsonl"
    if qvh_file.exists():
        with open(qvh_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("vid") in indexed_videos:
                        windows = item.get("relevant_windows", [])
                        queries.append({
                            "dataset": "qvhighlights",
                            "query_id": f"qvh_{item['qid']}",
                            "query_text": item["query"],
                            "target_videos": [item["vid"]],
                            "target_windows": windows,
                        })

    # 2. AVC (Normalized with pilot_queries ground-truth target)
    avc_file = val_dir / "avc" / "candidate_plan_inference.json"
    pilot_file = Path("data/raw/avc/provenance/data/processed/avc_1.8m/manifests/pilot_queries_eval_only.json")
    pilot_targets: dict[str, str] = {}
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as pf:
            pilot_data = json.load(pf)
            for p_item in pilot_data:
                pilot_targets[p_item["query_id"]] = p_item["known_target_window_id"]

    if avc_file.exists():
        with open(avc_file, "r", encoding="utf-8") as f:
            avc_data = json.load(f)
            unlabeled = 0
            for item in avc_data.get("queries", []):
                target_vid = pilot_targets.get(item["query_id"])
                if not target_vid:
                    unlabeled += 1
                    continue
                valid_cands = [c for c in item.get("candidate_ids", []) if c in indexed_videos]
                if valid_cands and target_vid in indexed_videos:
                    tgt_vids = [target_vid]
                    queries.append({
                        "dataset": "avc",
                        "query_id": f"avc_{item['query_id']}",
                        "query_text": item["query_text"],
                        "target_videos": tgt_vids,
                        "target_windows": [],
                    })
            print(f"AVC unlabeled queries excluded: {unlabeled}")

    # 3. MSR-VTT
    msr_file = val_dir / "msrvtt" / "test_videodatainfo.json"
    if msr_file.exists():
        with open(msr_file, "r", encoding="utf-8") as f:
            msr_data = json.load(f)
            sentences = msr_data.get("sentences", [])
            valid_sentences = [s for s in sentences if s.get("video_id") in indexed_videos]

            by_vid: dict[str, dict[str, Any]] = {}
            for s in valid_sentences:
                vid = s["video_id"]
                if vid not in by_vid:
                    by_vid[vid] = s

            sample_sentences = list(by_vid.values())
            rng = random.Random(42)
            rng.shuffle(sample_sentences)
            if max_msrvtt is not None and max_msrvtt > 0:
                sample_sentences = sample_sentences[:max_msrvtt]

            for s in sample_sentences:
                queries.append({
                    "dataset": "msrvtt",
                    "query_id": f"msr_{s.get('sen_id', s['video_id'])}",
                    "query_text": s["caption"],
                    "target_videos": [s["video_id"]],
                    "target_windows": [],
                })

    return queries


def _snapshot_ltr_ranker_state(ltr_ranker: Any) -> dict[str, Any]:
    if hasattr(ltr_ranker, "snapshot_state"):
        return ltr_ranker.snapshot_state()
    return {
        "is_loaded": ltr_ranker.is_loaded,
        "model_type": ltr_ranker.model_type,
        "weights": ltr_ranker.weights.copy(),
        "bias": ltr_ranker.bias,
        "feature_mean": ltr_ranker.feature_mean.copy(),
        "feature_std": ltr_ranker.feature_std.copy(),
        "trees": list(ltr_ranker.trees),
    }


def _restore_ltr_ranker_state(ltr_ranker: Any, state: dict[str, Any]) -> None:
    if hasattr(ltr_ranker, "restore_state"):
        ltr_ranker.restore_state(state)
        return
    ltr_ranker.is_loaded = state["is_loaded"]
    ltr_ranker.model_type = state["model_type"]
    ltr_ranker.weights = state["weights"].copy()
    ltr_ranker.bias = state["bias"]
    ltr_ranker.feature_mean = state["feature_mean"].copy()
    ltr_ranker.feature_std = state["feature_std"].copy()
    ltr_ranker.trees = list(state["trees"])


def _apply_ltr_fold_payload(ltr_ranker: Any, fold_payload: dict[str, Any]) -> None:
    if hasattr(ltr_ranker, "apply_fold_payload"):
        ltr_ranker.apply_fold_payload(fold_payload)
        return
    if "model_type" in fold_payload:
        ltr_ranker.model_type = str(fold_payload["model_type"]).lower()
    elif "trees" in fold_payload:
        ltr_ranker.model_type = "tree_ensemble"
    elif "weights" in fold_payload:
        ltr_ranker.model_type = "pairwise_logistic"
    if "weights" in fold_payload:
        ltr_ranker.weights = np.asarray(fold_payload["weights"], dtype=np.float64)
    if "bias" in fold_payload:
        ltr_ranker.bias = float(fold_payload["bias"])
    if "feature_mean" in fold_payload:
        ltr_ranker.feature_mean = np.asarray(fold_payload["feature_mean"], dtype=np.float64)
    if "feature_std" in fold_payload:
        ltr_ranker.feature_std = np.asarray(fold_payload["feature_std"], dtype=np.float64)
    if "trees" in fold_payload:
        ltr_ranker.trees = list(fold_payload["trees"])
    ltr_ranker.is_loaded = True


def _collect_channel_diagnostics(
    diag: dict[str, Any],
    channel_agg: dict[str, dict[str, Any]],
) -> None:
    """Aggregate channel_status and degraded diagnostics (dict or legacy list/str)."""
    def _record(ch: str, status: str, reason: str) -> None:
        entry = channel_agg.setdefault(
            ch,
            {"status": status, "affected_queries": 0, "sample_reason": reason},
        )
        entry["affected_queries"] += 1
        # Prefer more severe status if mixed across queries
        severity = {"failed": 3, "missing_artifact": 2, "degraded": 2, "disabled_by_config": 1, "ready": 0}
        if severity.get(status, 2) > severity.get(entry["status"], 0):
            entry["status"] = status
        if reason and not entry["sample_reason"]:
            entry["sample_reason"] = reason

    raw_status = diag.get("channel_status")
    if isinstance(raw_status, dict):
        for ch, info in raw_status.items():
            if not ch:
                continue
            if isinstance(info, dict):
                status = str(info.get("status") or "ready")
                reason = str(info.get("reason") or "")
            else:
                status = str(info or "ready")
                reason = ""
            _record(str(ch), status, reason)

    raw_degraded = diag.get("degraded")
    if isinstance(raw_degraded, dict):
        for ch, info in raw_degraded.items():
            if not ch:
                continue
            if isinstance(info, dict):
                status = str(info.get("status") or "failed")
                reason = str(info.get("reason") or "")
            else:
                status = "failed"
                reason = str(info or "")
            _record(str(ch), status, reason)
    elif isinstance(raw_degraded, (list, tuple, set)):
        for item in raw_degraded:
            if item:
                _record(str(item), "failed", str(item))
    elif isinstance(raw_degraded, str) and raw_degraded:
        _record(raw_degraded, "failed", raw_degraded)


def _extract_ordered_stage_ids(diag: dict[str, Any], *keys: str) -> list[str] | None:
    """Extract ordered unique video_ids from the first present non-None stage trace key."""
    for key in keys:
        if key in diag and diag[key] is not None:
            raw = diag[key]
            seen: set[str] = set()
            ordered: list[str] = []
            for item in raw:
                vid = str(item["video_id"] if isinstance(item, dict) else getattr(item, "video_id", item))
                if vid and vid not in seen:
                    seen.add(vid)
                    ordered.append(vid)
            return ordered
    return None


def _stage_hit(ordered_ids: list[str] | None, target_vids: set[str], k: int | None = None) -> int | str:
    if ordered_ids is None:
        return "NOT_MEASURED"
    pool = ordered_ids[:k] if k is not None else ordered_ids
    return 1 if any(v in target_vids for v in pool) else 0


def _first_target_rank(ordered_ids: list[str] | None, target_vids: set[str]) -> int | str:
    if ordered_ids is None:
        return "NOT_MEASURED"
    for rank, vid in enumerate(ordered_ids, 1):
        if vid in target_vids:
            return rank
    return 0


def _mean_measured(values: Sequence[int | float | str]) -> float | str:
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return "NOT_MEASURED"
    return round(sum(numeric) / len(numeric), 4)


def _build_ranked_videos(
    preds: list[dict[str, Any]],
    raw_ranked_videos: Any = None,
) -> list[str]:
    """Build deduplicated video ranking (best representative score + stable tie-break)."""
    if isinstance(raw_ranked_videos, (list, tuple)) and raw_ranked_videos:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in raw_ranked_videos:
            vid = str(item["video_id"] if isinstance(item, dict) else getattr(item, "video_id", item))
            if vid and vid not in seen:
                seen.add(vid)
                ordered.append(vid)
        if ordered:
            return ordered

    best_by_vid: dict[str, tuple[float, int]] = {}
    for idx, p in enumerate(preds):
        vid = str(p["video_id"])
        score = float(p.get("ranking_score", 0.0))
        if vid not in best_by_vid or score > best_by_vid[vid][0]:
            best_by_vid[vid] = (score, idx)
    ranked = sorted(best_by_vid.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
    return [vid for vid, _ in ranked]


def evaluate_query_set(
    runner: MultimodalBenchmarkRunner,
    queries: list[dict[str, Any]],
    split_label: str = "dev_150",
    ablation_config: str = "A5",
    show_progress: bool = True,
    fold_query_map: dict[str, dict[str, Any]] | None = None,
    ltr_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Runs evaluation on a query list and returns (summary_dict, per_query_records)."""
    if not queries:
        raise ValueError("Cannot evaluate an empty query set")
    for query in queries:
        targets = query.get("target_videos")
        if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(v, str) or not v for v in targets):
            raise ValueError("Every evaluated query requires known target videos")
        if query.get("target_windows") and len(set(targets)) != 1:
            raise ValueError("Moment windows require an unambiguous target video")
    total_queries = len(queries)
    per_query_results: list[dict[str, Any]] = []

    hits1: list[int] = []
    hits5: list[int] = []
    hits10: list[int] = []
    mrrs: list[float] = []
    ndcgs1: list[float] = []
    ndcgs5: list[float] = []
    ndcgs10: list[float] = []

    # Within-dataset candidate pool metrics (E3)
    in_ds_hits1: list[int] = []
    in_ds_hits5: list[int] = []
    in_ds_hits10: list[int] = []
    in_ds_mrrs: list[float] = []
    in_ds_ndcgs10: list[float] = []

    # Moment metrics (E7)
    mom_hits03: list[int] = []
    mom_hits05: list[int] = []
    mom_hits07: list[int] = []
    mom_ndcgs5: list[float] = []
    mom_ndcgs10: list[float] = []
    mom_ap05: list[float] = []
    mom_ap075: list[float] = []
    mom_map_range: list[float] = []
    boundary_errors: list[float] = []

    # Branch & Stage Recall@K (E8 / Section 2.2)
    vis_rec30: list[int] = []
    vis_rec120: list[int] = []
    lex_rec30: list[int] = []
    fused_rec10: list[int] = []
    vis_cand_hits: list[int | str] = []
    union_hits: list[int | str] = []
    shortlist_hits: list[int | str] = []
    shortlist_hits10: list[int | str] = []
    shortlist_hits20: list[int | str] = []
    shortlist_hits50: list[int | str] = []
    rerank_in_hits: list[int | str] = []
    ranked_stage_hits: list[int] = []
    vis_pool_sizes: list[int | str] = []
    union_pool_sizes: list[int | str] = []
    pre_rerank_pool_sizes: list[int | str] = []
    rerank_in_pool_sizes: list[int | str] = []
    shortlist_not_top1_count = 0

    # Timings (C3)
    latencies_total: list[float] = []
    latencies_non_neural: list[float] = []
    latencies_faiss: list[float] = []
    latencies_fts5: list[float] = []
    latencies_fusion: list[float] = []
    latencies_boundary: list[float] = []

    pure_asr_top1_count = 0
    pure_asr_fp_top1_count = 0
    oof_applied_count = 0
    oof_fallback_count = 0
    ltr_applied_count = 0
    ltr_top1_flipped_count = 0
    ltr_max_abs_deltas: list[float] = []
    consensus_applied_count = 0
    consensus_top1_changed_count = 0
    taxonomy_counts: dict[str, int] = defaultdict(int)
    channel_agg: dict[str, dict[str, Any]] = {}
    ltr_ranker = getattr(getattr(runner, "pipeline", None), "ltr_ranker", None)
    use_oof_swapping = bool(fold_query_map or (ltr_info and ltr_info.get("has_fold_weights")))
    orig_ltr_state = _snapshot_ltr_ranker_state(ltr_ranker) if (ltr_ranker is not None and use_oof_swapping) else None

    ds_stats: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "hits1": [], "hits5": [], "hits10": [], "mrrs": [],
            "ndcgs1": [], "ndcgs5": [], "ndcgs10": [],
            "in_ds_hits1": [], "in_ds_hits5": [], "in_ds_hits10": [], "in_ds_mrrs": [], "in_ds_ndcgs10": [],
            "mom_hits03": [], "mom_hits05": [], "mom_hits07": [],
            "mom_ndcgs5": [], "mom_ndcgs10": [],
            "mom_ap05": [], "mom_ap075": [], "mom_map_range": [],
            "vis_rec30": [], "vis_rec120": [], "lex_rec30": [], "fused_rec10": [],
            "vis_cand_hits": [], "union_hits": [],
            "shortlist_hits": [], "shortlist_hits10": [], "shortlist_hits20": [], "shortlist_hits50": [],
            "rerank_in_hits": [],
            "latencies": [], "latencies_non_neural": [],
        }
    )

    for idx, q in enumerate(queries, 1):
        ds = q["dataset"]
        q_text = q["query_text"]
        target_vids = set(q["target_videos"])
        target_wins = q.get("target_windows", [])

        if ltr_ranker is not None and use_oof_swapping:
            qid_str = str(q["query_id"])
            payload_match: dict[str, Any] | None = None
            if ltr_info is not None:
                q_groups, _, _ = resolve_query_source_groups(q)
                payload_match, _status = resolve_oof_fold_for_query(qid_str, q_groups, ltr_info)
            if payload_match is None and fold_query_map and qid_str in fold_query_map:
                payload_match = fold_query_map[qid_str]
            if payload_match is not None:
                _apply_ltr_fold_payload(ltr_ranker, payload_match)
                oof_applied_count += 1
            else:
                oof_fallback_count += 1
                if orig_ltr_state is not None:
                    _restore_ltr_ranker_state(ltr_ranker, orig_ltr_state)

        try:
            preds = runner.search_multimodal(q_text, top_k=10, ablation_config=ablation_config)
        finally:
            if ltr_ranker is not None and orig_ltr_state is not None:
                _restore_ltr_ranker_state(ltr_ranker, orig_ltr_state)
        timings = runner.timings
        diag = runner.diagnostics
        _collect_channel_diagnostics(diag, channel_agg)
        if isinstance(diag, dict):
            ltr_st = diag.get("ltr_stats")
            if isinstance(ltr_st, dict):
                if ltr_st.get("applied"):
                    ltr_applied_count += 1
                if ltr_st.get("top1_flipped"):
                    ltr_top1_flipped_count += 1
                if isinstance(ltr_st.get("max_abs_delta"), (int, float)):
                    ltr_max_abs_deltas.append(float(ltr_st["max_abs_delta"]))
            cons_st = diag.get("clip_consensus_stats")
            if isinstance(cons_st, dict):
                if cons_st.get("applied"):
                    consensus_applied_count += 1
                if cons_st.get("top1_changed"):
                    consensus_top1_changed_count += 1

        required_timings = {"t_total_ms", "t_non_neural_ms", "t_faiss_ms", "t_fts5_ms",
                            "t_fusion_ms", "t_boundary_ms"}
        if required_timings - timings.keys():
            raise RuntimeError(f"Missing timing telemetry: {sorted(required_timings - timings.keys())}")

        t_total = float(timings["t_total_ms"])
        t_non_neural = float(timings["t_non_neural_ms"])

        latencies_total.append(t_total)
        latencies_non_neural.append(t_non_neural)
        latencies_faiss.append(float(timings["t_faiss_ms"]))
        latencies_fts5.append(float(timings["t_fts5_ms"]))
        latencies_fusion.append(float(timings["t_fusion_ms"]))
        latencies_boundary.append(float(timings["t_boundary_ms"]))

        ds_stats[ds]["latencies"].append(t_total)
        ds_stats[ds]["latencies_non_neural"].append(t_non_neural)

        # Separate ranked_videos (for Video@K/MRR/nDCG) from ranked_moments (for Moment@K/mAP/tIoU)
        ranked_moments = preds[:10]
        dedup_pred_vids = _build_ranked_videos(
            preds,
            getattr(runner, "ranked_videos", None) or diag.get("ranked_video_ids") or diag.get("ranked_videos"),
        )

        h1 = 1 if any(v in target_vids for v in dedup_pred_vids[:1]) else 0
        h5 = 1 if any(v in target_vids for v in dedup_pred_vids[:5]) else 0
        h10 = 1 if any(v in target_vids for v in dedup_pred_vids[:10]) else 0

        hits1.append(h1)
        hits5.append(h5)
        hits10.append(h10)
        ds_stats[ds]["hits1"].append(h1)
        ds_stats[ds]["hits5"].append(h5)
        ds_stats[ds]["hits10"].append(h10)

        rr = 0.0
        matched_rank = None
        for rank, v in enumerate(dedup_pred_vids[:10], 1):
            if v in target_vids:
                rr = 1.0 / rank
                matched_rank = rank
                break
        mrrs.append(rr)
        ds_stats[ds]["mrrs"].append(rr)

        ndcg1 = compute_binary_ndcg_at_k(dedup_pred_vids, target_vids, 1, runner.indexed_videos)
        ndcg5 = compute_binary_ndcg_at_k(dedup_pred_vids, target_vids, 5, runner.indexed_videos)
        ndcg10 = compute_binary_ndcg_at_k(dedup_pred_vids, target_vids, 10, runner.indexed_videos)

        ndcgs1.append(ndcg1)
        ndcgs5.append(ndcg5)
        ndcgs10.append(ndcg10)
        ds_stats[ds]["ndcgs1"].append(ndcg1)
        ds_stats[ds]["ndcgs5"].append(ndcg5)
        ds_stats[ds]["ndcgs10"].append(ndcg10)

        # Within-Dataset Candidate Pool Evaluation (E3)
        ds_pool = runner.dataset_videos.get(ds, runner.indexed_videos)
        all_cand_moments = diag.get("all_candidate_moments", [])
        in_ds_vids: list[str] = []
        seen_in_ds: set[str] = set()
        for m in all_cand_moments:
            if m.video_id in ds_pool and m.video_id not in seen_in_ds:
                seen_in_ds.add(m.video_id)
                in_ds_vids.append(m.video_id)
        if not in_ds_vids:
            in_ds_vids = [v for v in dedup_pred_vids if v in ds_pool]

        ih1 = 1 if any(v in target_vids for v in in_ds_vids[:1]) else 0
        ih5 = 1 if any(v in target_vids for v in in_ds_vids[:5]) else 0
        ih10 = 1 if any(v in target_vids for v in in_ds_vids[:10]) else 0
        irr = 0.0
        for r_idx, v in enumerate(in_ds_vids[:10], 1):
            if v in target_vids:
                irr = 1.0 / r_idx
                break
        indcg10 = compute_binary_ndcg_at_k(in_ds_vids[:10], target_vids, 10, ds_pool)

        in_ds_hits1.append(ih1)
        in_ds_hits5.append(ih5)
        in_ds_hits10.append(ih10)
        in_ds_mrrs.append(irr)
        in_ds_ndcgs10.append(indcg10)
        ds_stats[ds]["in_ds_hits1"].append(ih1)
        ds_stats[ds]["in_ds_hits5"].append(ih5)
        ds_stats[ds]["in_ds_hits10"].append(ih10)
        ds_stats[ds]["in_ds_mrrs"].append(irr)
        ds_stats[ds]["in_ds_ndcgs10"].append(indcg10)

        # Branch & Stage Recall@K (E8 / Section 2.2)
        vis30_set = set(diag.get("visual_top30_vids", set()))
        vis120_set = set(diag.get("visual_top120_vids", set()))
        lex30_set = set(diag.get("lexical_top30_vids", set()))
        v30 = 1 if (target_vids & vis30_set) else 0
        v120 = 1 if (target_vids & vis120_set) else 0
        l30 = 1 if (target_vids & lex30_set) else 0
        f10 = 1 if (target_vids & diag.get("fused_top10_vids", set())) else 0

        vis_ids = _extract_ordered_stage_ids(diag, "visual_candidate_ids")
        union_ids = _extract_ordered_stage_ids(diag, "union_candidate_ids")
        pre_rerank_ids = _extract_ordered_stage_ids(diag, "pre_rerank_video_ids")
        rerank_in_ids = _extract_ordered_stage_ids(diag, "reranker_input_ids")
        ranked_stage_ids = _extract_ordered_stage_ids(diag, "ranked_video_ids")
        if ranked_stage_ids is None:
            ranked_stage_ids = dedup_pred_vids

        vis_hit = _stage_hit(vis_ids, target_vids)
        union_hit = _stage_hit(union_ids, target_vids)
        shortlist_hit = _stage_hit(pre_rerank_ids, target_vids)
        shortlist_h10 = _stage_hit(pre_rerank_ids, target_vids, 10)
        shortlist_h20 = _stage_hit(pre_rerank_ids, target_vids, 20)
        shortlist_h50 = _stage_hit(pre_rerank_ids, target_vids, 50)
        rerank_in_hit = _stage_hit(rerank_in_ids, target_vids)
        ranked_stage_hit = 1 if any(v in target_vids for v in ranked_stage_ids) else 0

        target_rank_before = _first_target_rank(pre_rerank_ids, target_vids)
        target_rank_after = _first_target_rank(ranked_stage_ids, target_vids)
        vis_frame_depth = diag.get("visual_frame_depth") if diag.get("visual_frame_depth") is not None else "NOT_MEASURED"
        vis_distinct_videos = (
            diag.get("visual_distinct_videos")
            if diag.get("visual_distinct_videos") is not None
            else (len(vis_ids) if vis_ids is not None else "NOT_MEASURED")
        )
        vis_pool_size = len(vis_ids) if vis_ids is not None else "NOT_MEASURED"
        union_pool_size = len(union_ids) if union_ids is not None else "NOT_MEASURED"
        pre_rerank_pool_size = len(pre_rerank_ids) if pre_rerank_ids is not None else "NOT_MEASURED"
        rerank_in_pool_size = len(rerank_in_ids) if rerank_in_ids is not None else "NOT_MEASURED"
        ranked_pool_size = len(ranked_stage_ids)

        if shortlist_hit == 1 and h1 == 0:
            shortlist_not_top1_count += 1

        vis_rec30.append(v30)
        vis_rec120.append(v120)
        lex_rec30.append(l30)
        fused_rec10.append(f10)
        vis_cand_hits.append(vis_hit)
        union_hits.append(union_hit)
        shortlist_hits.append(shortlist_hit)
        shortlist_hits10.append(shortlist_h10)
        shortlist_hits20.append(shortlist_h20)
        shortlist_hits50.append(shortlist_h50)
        rerank_in_hits.append(rerank_in_hit)
        ranked_stage_hits.append(ranked_stage_hit)
        vis_pool_sizes.append(vis_pool_size)
        union_pool_sizes.append(union_pool_size)
        pre_rerank_pool_sizes.append(pre_rerank_pool_size)
        rerank_in_pool_sizes.append(rerank_in_pool_size)

        ds_stats[ds]["vis_rec30"].append(v30)
        ds_stats[ds]["vis_rec120"].append(v120)
        ds_stats[ds]["lex_rec30"].append(l30)
        ds_stats[ds]["fused_rec10"].append(f10)
        ds_stats[ds]["vis_cand_hits"].append(vis_hit)
        ds_stats[ds]["union_hits"].append(union_hit)
        ds_stats[ds]["shortlist_hits"].append(shortlist_hit)
        ds_stats[ds]["shortlist_hits10"].append(shortlist_h10)
        ds_stats[ds]["shortlist_hits20"].append(shortlist_h20)
        ds_stats[ds]["shortlist_hits50"].append(shortlist_h50)
        ds_stats[ds]["rerank_in_hits"].append(rerank_in_hit)

        # Moment Localization Metrics (E7) using ranked_moments with unrounded timestamps
        m_hit03: int | str = ""
        m_hit05: int | str = ""
        m_hit07: int | str = ""
        mom_ndcg5_val: float | str = ""
        mom_ndcg10_val: float | str = ""
        mom_ap05_val: float | str = ""
        mom_map_val: float | str = ""
        b_err: float | None = None

        if target_wins:
            m_hit03 = 0
            m_hit05 = 0
            m_hit07 = 0
            for p in ranked_moments[:1]:
                if p["video_id"] in target_vids:
                    p_s, p_e = float(p["start_sec"]), float(p["end_sec"])
                    best_iou = -1.0
                    best_window = None
                    for gt_s, gt_e in target_wins:
                        iou = temporal_iou(p_s, p_e, float(gt_s), float(gt_e))
                        if iou > best_iou:
                            best_iou = iou
                            best_window = (float(gt_s), float(gt_e))
                        if iou >= 0.3:
                            m_hit03 = 1
                        if iou >= 0.5:
                            m_hit05 = 1
                        if iou >= 0.7:
                            m_hit07 = 1
                    if m_hit05 and best_window is not None:
                        gt_s, gt_e = best_window
                        b_err = round((abs(p_s - gt_s) + abs(p_e - gt_e)) / 2, 2)
                        boundary_errors.append(b_err)

            mom_hits03.append(int(m_hit03))
            mom_hits05.append(int(m_hit05))
            mom_hits07.append(int(m_hit07))
            ds_stats[ds]["mom_hits03"].append(int(m_hit03))
            ds_stats[ds]["mom_hits05"].append(int(m_hit05))
            ds_stats[ds]["mom_hits07"].append(int(m_hit07))

            gt_windows_formatted = [(q["target_videos"][0], float(w[0]), float(w[1])) for w in target_wins]
            m_ndcg5 = compute_graded_moment_ndcg_at_k(ranked_moments, gt_windows_formatted, 5)
            m_ndcg10 = compute_graded_moment_ndcg_at_k(ranked_moments, gt_windows_formatted, 10)
            ap05 = compute_moment_ap_at_iou(ranked_moments, gt_windows_formatted, 0.50)
            ap075 = compute_moment_ap_at_iou(ranked_moments, gt_windows_formatted, 0.75)
            map_rng = compute_moment_map_range(ranked_moments, gt_windows_formatted)

            mom_ndcgs5.append(m_ndcg5)
            mom_ndcgs10.append(m_ndcg10)
            mom_ap05.append(ap05)
            mom_ap075.append(ap075)
            mom_map_range.append(map_rng)

            ds_stats[ds]["mom_ndcgs5"].append(m_ndcg5)
            ds_stats[ds]["mom_ndcgs10"].append(m_ndcg10)
            ds_stats[ds]["mom_ap05"].append(ap05)
            ds_stats[ds]["mom_ap075"].append(ap075)
            ds_stats[ds]["mom_map_range"].append(map_rng)

            mom_ndcg5_val = round(m_ndcg5, 4)
            mom_ndcg10_val = round(m_ndcg10, 4)
            mom_ap05_val = round(ap05, 4)
            mom_map_val = round(map_rng, 4)

        top1 = ranked_moments[0] if ranked_moments else {"video_id": "NONE", "start_sec": 0.0, "end_sec": 0.0, "modality_attribution": "none"}
        if top1["modality_attribution"] == "asr":
            pure_asr_top1_count += 1
            if h1 == 0:
                pure_asr_fp_top1_count += 1

        failure_tag = classify_query_failure(
            hit_at_1=h1,
            target_videos=target_vids,
            predicted_top1_video=top1["video_id"],
            predicted_top1_attribution=top1["modality_attribution"],
            visual_top120_vids=diag.get("visual_top120_vids", set()),
            moment_hit_iou05=m_hit05,
        )
        taxonomy_counts[failure_tag] += 1

        res_record = {
            "schema_version": 3,
            "query_id": q["query_id"],
            "dataset": ds,
            "query_text": q_text,
            "ranked_videos": json.dumps(dedup_pred_vids[:10], ensure_ascii=False),
            "ranked_predictions": json.dumps(ranked_moments, ensure_ascii=False),
            "target_videos": "|".join(target_vids),
            "target_windows": json.dumps(target_wins),
            "predicted_top1_video": top1["video_id"],
            "predicted_top1_moment": f"[{round(float(top1['start_sec']), 2)}-{round(float(top1['end_sec']), 2)}]",
            "hit_at_1": h1,
            "hit_at_5": h5,
            "hit_at_10": h10,
            "reciprocal_rank": round(rr, 4),
            "matched_rank": matched_rank if matched_rank else 0,
            "ndcg_at_1": round(ndcg1, 4),
            "ndcg_at_5": round(ndcg5, 4),
            "ndcg_at_10": round(ndcg10, 4),
            "in_ds_hit_at_1": ih1,
            "in_ds_mrr": round(irr, 4),
            "visual_hit": vis_hit,
            "union_hit": union_hit,
            "shortlist_hit": shortlist_hit,
            "shortlist_hit_at_10": shortlist_h10,
            "shortlist_hit_at_20": shortlist_h20,
            "shortlist_hit_at_50": shortlist_h50,
            "reranker_input_hit": rerank_in_hit,
            "target_rank_before": target_rank_before,
            "target_rank_after": target_rank_after,
            "visual_frame_depth": vis_frame_depth,
            "visual_distinct_videos": vis_distinct_videos,
            "visual_pool_size": vis_pool_size,
            "union_pool_size": union_pool_size,
            "pre_rerank_pool_size": pre_rerank_pool_size,
            "reranker_input_pool_size": rerank_in_pool_size,
            "ranked_pool_size": ranked_pool_size,
            "moment_hit_iou03": m_hit03,
            "moment_hit_iou05": m_hit05,
            "moment_hit_iou07": m_hit07,
            "moment_ndcg_at_5": mom_ndcg5_val,
            "moment_ndcg_at_10": mom_ndcg10_val,
            "moment_ap_iou05": mom_ap05_val,
            "moment_map_range": mom_map_val,
            "boundary_error_sec": b_err if b_err is not None else "",
            "modality_attribution": top1["modality_attribution"],
            "failure_taxonomy": failure_tag,
            "latency_ms": round(t_total, 2),
            "latency_non_neural_ms": round(t_non_neural, 2),
        }
        per_query_results.append(res_record)

        if show_progress:
            print_progress(idx, total_queries, f"{split_label}:{ablation_config}", sum(hits1) / len(hits1), sum(mrrs) / len(mrrs), t_non_neural)

    total_n = max(1, len(queries))
    h1_mean, h1_ci_l, h1_ci_h = compute_bootstrap_ci(hits1)
    mrr_mean, mrr_ci_l, mrr_ci_h = compute_bootstrap_ci(mrrs)
    ndcg10_mean, ndcg10_ci_l, ndcg10_ci_h = compute_bootstrap_ci(ndcgs10)
    m05_mean, m05_ci_l, m05_ci_h = compute_bootstrap_ci(mom_hits05) if mom_hits05 else (0.0, 0.0, 0.0)
    mndcg5_mean, mndcg5_ci_l, mndcg5_ci_h = compute_bootstrap_ci(mom_ndcgs5) if mom_ndcgs5 else (0.0, 0.0, 0.0)

    degraded_channels = sorted(
        ch for ch, info in channel_agg.items()
        if info["status"] not in {"ready", "disabled_by_config"}
    )
    disabled_channels = sorted(
        ch for ch, info in channel_agg.items()
        if info["status"] == "disabled_by_config"
    )
    degraded_channel_details = {ch: channel_agg[ch] for ch in degraded_channels}

    summary: dict[str, Any] = {
        "split_name": split_label,
        "ablation_config": ablation_config,
        "total_queries_evaluated": len(queries),
        "oof_stats": {
            "applied_count": oof_applied_count,
            "fallback_count": oof_fallback_count,
            "total_queries": len(queries),
        },
        "ltr_stats": {
            "applied_count": ltr_applied_count,
            "top1_flipped_count": ltr_top1_flipped_count,
            "max_abs_delta": round(max(ltr_max_abs_deltas), 6) if ltr_max_abs_deltas else 0.0,
            "p50_max_abs_delta": round(float(np.median(ltr_max_abs_deltas)), 6) if ltr_max_abs_deltas else 0.0,
        },
        "clip_consensus_stats": {
            "applied_count": consensus_applied_count,
            "top1_changed_count": consensus_top1_changed_count,
        },
        "metrics_overall": {
            "video_hit_at_1": round(sum(hits1) / total_n, 4),
            "video_hit_at_1_ci95": [round(h1_ci_l, 4), round(h1_ci_h, 4)],
            "video_hit_at_5": round(sum(hits5) / total_n, 4),
            "video_hit_at_10": round(sum(hits10) / total_n, 4),
            "mean_reciprocal_rank_mrr": round(sum(mrrs) / total_n, 4),
            "mrr_ci95": [round(mrr_ci_l, 4), round(mrr_ci_h, 4)],
            "video_ndcg_at_1": round(sum(ndcgs1) / total_n, 4),
            "video_ndcg_at_5": round(sum(ndcgs5) / total_n, 4),
            "video_ndcg_at_10": round(sum(ndcgs10) / total_n, 4),
            "video_ndcg_at_10_ci95": [round(ndcg10_ci_l, 4), round(ndcg10_ci_h, 4)],
            "within_dataset_hit_at_1": round(sum(in_ds_hits1) / total_n, 4),
            "within_dataset_hit_at_5": round(sum(in_ds_hits5) / total_n, 4),
            "within_dataset_hit_at_10": round(sum(in_ds_hits10) / total_n, 4),
            "within_dataset_mrr": round(sum(in_ds_mrrs) / total_n, 4),
            "within_dataset_ndcg_at_10": round(sum(in_ds_ndcgs10) / total_n, 4),
            "moment_labeled_n": len(mom_hits05),
            "moment_hit_at_1_iou03": round(sum(mom_hits03) / max(1, len(mom_hits03)), 4),
            "moment_hit_at_1_iou05": round(sum(mom_hits05) / max(1, len(mom_hits05)), 4),
            "moment_hit_at_1_iou05_ci95": [round(m05_ci_l, 4), round(m05_ci_h, 4)],
            "moment_hit_at_1_iou07": round(sum(mom_hits07) / max(1, len(mom_hits07)), 4),
            "moment_ndcg_at_5": round(sum(mom_ndcgs5) / max(1, len(mom_ndcgs5)), 4),
            "moment_ndcg_at_5_ci95": [round(mndcg5_ci_l, 4), round(mndcg5_ci_h, 4)],
            "moment_ndcg_at_10": round(sum(mom_ndcgs10) / max(1, len(mom_ndcgs10)), 4),
            "moment_map_at_05": round(sum(mom_ap05) / max(1, len(mom_ap05)), 4),
            "moment_map_at_075": round(sum(mom_ap075) / max(1, len(mom_ap075)), 4),
            "moment_map_range_05_095": round(sum(mom_map_range) / max(1, len(mom_map_range)), 4),
            "mean_boundary_mae_sec": round(sum(boundary_errors) / len(boundary_errors), 2) if boundary_errors else None,
            "boundary_mae_scope": "top1_correct_video_iou_at_least_0.5",
            "boundary_mae_n": len(boundary_errors),
            "pure_asr_top1_count": pure_asr_top1_count,
            "pure_asr_fp_top1_count": pure_asr_fp_top1_count,
            "pure_asr_fp_top1_rate": round(pure_asr_fp_top1_count / pure_asr_top1_count, 4) if pure_asr_top1_count else 0.0,
            "branch_recall": {
                "visual_recall_at_30": round(sum(vis_rec30) / total_n, 4),
                "visual_recall_at_120": round(sum(vis_rec120) / total_n, 4),
                "lexical_recall_at_30": round(sum(lex_rec30) / total_n, 4),
                "visual_candidate_recall": _mean_measured(vis_cand_hits),
                "union_recall": _mean_measured(union_hits),
                "shortlist_recall": _mean_measured(shortlist_hits),
                "shortlist_recall_at_10": _mean_measured(shortlist_hits10),
                "shortlist_recall_at_20": _mean_measured(shortlist_hits20),
                "shortlist_recall_at_50": _mean_measured(shortlist_hits50),
                "shortlist_recall_at_k": _mean_measured(shortlist_hits),
                "reranker_input_recall": _mean_measured(rerank_in_hits),
                "ranked_recall": round(sum(ranked_stage_hits) / total_n, 4),
                "fused_recall_at_10": round(sum(fused_rec10) / total_n, 4),
                "mean_visual_pool_size": _mean_measured(vis_pool_sizes),
                "mean_union_pool_size": _mean_measured(union_pool_sizes),
                "mean_pre_rerank_pool_size": _mean_measured(pre_rerank_pool_sizes),
                "mean_reranker_input_pool_size": _mean_measured(rerank_in_pool_sizes),
                "shortlist_not_top1_count": shortlist_not_top1_count,
                "shortlist_not_top1_rate": round(shortlist_not_top1_count / total_n, 4),
            },
            "failure_taxonomy": dict(taxonomy_counts),
            "end_to_end_latency_mean_ms": round(float(np.mean(latencies_total)), 2),
            "end_to_end_latency_p50_ms": round(float(np.median(latencies_total)), 2),
            "end_to_end_latency_p95_ms": round(float(np.percentile(latencies_total, 95)), 2),
            "latency_mean_ms": round(float(np.mean(latencies_total)), 2),
            "latency_p50_ms": round(float(np.median(latencies_total)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_total, 95)), 2),
            "non_neural_latency_mean_ms": round(float(np.mean(latencies_non_neural)), 2),
            "non_neural_latency_p50_ms": round(float(np.median(latencies_non_neural)), 2),
            "non_neural_latency_p95_ms": round(float(np.percentile(latencies_non_neural, 95)), 2),
            "stage_latencies_p50_ms": {
                "faiss_ms": round(float(np.median(latencies_faiss)), 2),
                "fts5_ms": round(float(np.median(latencies_fts5)), 2),
                "fusion_ms": round(float(np.median(latencies_fusion)), 2),
                "boundary_ms": round(float(np.median(latencies_boundary)), 2),
            },
        },
        "per_dataset_metrics": {},
        "degraded_channels": degraded_channels,
        "disabled_channels": disabled_channels,
        "degraded_channel_details": degraded_channel_details,
        "channel_status": channel_agg,
        "_raw_arrays": {
            "hits1": hits1,
            "mrrs": mrrs,
            "ndcgs10": ndcgs10,
        },
    }

    for ds, st in ds_stats.items():
        n_ds = max(1, len(st["hits1"]))
        ds_h1_m, ds_h1_l, ds_h1_h = compute_bootstrap_ci(st["hits1"])
        ds_mrr_m, ds_mrr_l, ds_mrr_h = compute_bootstrap_ci(st["mrrs"])
        summary["per_dataset_metrics"][ds] = {
            "queries": len(st["hits1"]),
            "hit_at_1": round(sum(st["hits1"]) / n_ds, 4),
            "hit_at_1_ci95": [round(ds_h1_l, 4), round(ds_h1_h, 4)],
            "hit_at_5": round(sum(st["hits5"]) / n_ds, 4),
            "hit_at_10": round(sum(st["hits10"]) / n_ds, 4),
            "mrr": round(sum(st["mrrs"]) / n_ds, 4),
            "mrr_ci95": [round(ds_mrr_l, 4), round(ds_mrr_h, 4)],
            "video_ndcg_at_1": round(sum(st["ndcgs1"]) / n_ds, 4),
            "video_ndcg_at_5": round(sum(st["ndcgs5"]) / n_ds, 4),
            "video_ndcg_at_10": round(sum(st["ndcgs10"]) / n_ds, 4),
            "within_dataset_hit_at_1": round(sum(st["in_ds_hits1"]) / n_ds, 4),
            "within_dataset_hit_at_5": round(sum(st["in_ds_hits5"]) / n_ds, 4),
            "within_dataset_hit_at_10": round(sum(st["in_ds_hits10"]) / n_ds, 4),
            "within_dataset_mrr": round(sum(st["in_ds_mrrs"]) / n_ds, 4),
            "within_dataset_ndcg_at_10": round(sum(st["in_ds_ndcgs10"]) / n_ds, 4),
            "moment_labeled_n": len(st["mom_hits05"]),
            "moment_hit_at_1_iou03": round(sum(st["mom_hits03"]) / len(st["mom_hits03"]), 4) if st["mom_hits03"] else 0.0,
            "moment_hit_at_1_iou05": round(sum(st["mom_hits05"]) / len(st["mom_hits05"]), 4) if st["mom_hits05"] else 0.0,
            "moment_hit_at_1_iou07": round(sum(st["mom_hits07"]) / len(st["mom_hits07"]), 4) if st["mom_hits07"] else 0.0,
            "moment_ndcg_at_5": round(sum(st["mom_ndcgs5"]) / len(st["mom_ndcgs5"]), 4) if st["mom_ndcgs5"] else 0.0,
            "moment_ndcg_at_10": round(sum(st["mom_ndcgs10"]) / len(st["mom_ndcgs10"]), 4) if st["mom_ndcgs10"] else 0.0,
            "moment_map_at_05": round(sum(st["mom_ap05"]) / len(st["mom_ap05"]), 4) if st["mom_ap05"] else 0.0,
            "moment_map_at_075": round(sum(st["mom_ap075"]) / len(st["mom_ap075"]), 4) if st["mom_ap075"] else 0.0,
            "moment_map_range_05_095": round(sum(st["mom_map_range"]) / len(st["mom_map_range"]), 4) if st["mom_map_range"] else 0.0,
            "visual_recall_at_30": round(sum(st["vis_rec30"]) / n_ds, 4),
            "visual_recall_at_120": round(sum(st["vis_rec120"]) / n_ds, 4),
            "lexical_recall_at_30": round(sum(st["lex_rec30"]) / n_ds, 4),
            "fused_recall_at_10": round(sum(st["fused_rec10"]) / n_ds, 4),
            "visual_candidate_recall": _mean_measured(st["vis_cand_hits"]),
            "union_recall": _mean_measured(st["union_hits"]),
            "shortlist_recall": _mean_measured(st["shortlist_hits"]),
            "shortlist_recall_at_10": _mean_measured(st["shortlist_hits10"]),
            "shortlist_recall_at_20": _mean_measured(st["shortlist_hits20"]),
            "shortlist_recall_at_50": _mean_measured(st["shortlist_hits50"]),
            "shortlist_recall_at_k": _mean_measured(st["shortlist_hits"]),
            "reranker_input_recall": _mean_measured(st["rerank_in_hits"]),
            "latency_p50_ms": round(float(np.median(st["latencies"])), 2),
            "non_neural_latency_p50_ms": round(float(np.median(st["latencies_non_neural"])), 2),
        }

    ds_metrics = summary["per_dataset_metrics"]
    n_ds_groups = max(1, len(ds_metrics))
    summary["dataset_averages"] = {
        "micro": {
            "queries": len(queries),
            "video_hit_at_1": summary["metrics_overall"]["video_hit_at_1"],
            "video_hit_at_5": summary["metrics_overall"]["video_hit_at_5"],
            "video_hit_at_10": summary["metrics_overall"]["video_hit_at_10"],
            "mrr": summary["metrics_overall"]["mean_reciprocal_rank_mrr"],
            "shortlist_recall": summary["metrics_overall"]["branch_recall"]["shortlist_recall"],
        },
        "macro": {
            "datasets": len(ds_metrics),
            "video_hit_at_1": round(sum(m["hit_at_1"] for m in ds_metrics.values()) / n_ds_groups, 4),
            "video_hit_at_5": round(sum(m["hit_at_5"] for m in ds_metrics.values()) / n_ds_groups, 4),
            "video_hit_at_10": round(sum(m["hit_at_10"] for m in ds_metrics.values()) / n_ds_groups, 4),
            "mrr": round(sum(m["mrr"] for m in ds_metrics.values()) / n_ds_groups, 4),
            "shortlist_recall": _mean_measured([m["shortlist_recall"] for m in ds_metrics.values()]),
        },
        "msrvtt": ds_metrics.get("msrvtt"),
    }

    return summary, per_query_results


def measure_cold_query_latency(runner: MultimodalBenchmarkRunner, sample_queries: list[dict[str, Any]]) -> dict[str, float | None]:
    """Measure real cold-cache query latency by clearing in-memory query caches before each timed query."""
    if not sample_queries:
        return {"cold_mean_ms": None, "cold_p50_ms": None, "cold_p95_ms": None}

    pipeline = getattr(runner, "pipeline", None)
    if pipeline is None:
        return {"cold_mean_ms": None, "cold_p50_ms": None, "cold_p95_ms": None}

    latencies: list[float] = []
    for q in sample_queries[:15]:
        q_text = str(q.get("query_text") or "").strip()
        if not q_text:
            continue
        if hasattr(pipeline, "_text_memory_cache"):
            pipeline._text_memory_cache.clear()
        if hasattr(pipeline, "_e5_memory_cache"):
            pipeline._e5_memory_cache.clear()
        vp_ret = getattr(pipeline, "videoprism_retriever", None)
        if vp_ret is not None and hasattr(vp_ret, "clear_runtime_cache"):
            vp_ret.clear_runtime_cache()
        decomposer = getattr(pipeline, "query_decomposer", None)
        if decomposer is not None and hasattr(decomposer, "_memory_cache"):
            decomposer._memory_cache.clear()
        t0 = time.perf_counter()
        runner.search_multimodal(q_text, top_k=10, ablation_config="A5")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    if not latencies:
        return {"cold_mean_ms": None, "cold_p50_ms": None, "cold_p95_ms": None}

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "cold_mean_ms": round(float(np.mean(arr)), 2),
        "cold_p50_ms": round(float(np.median(arr)), 2),
        "cold_p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def acceptance_metric(metric: str, value: float | None, threshold: float, direction: str, source: str, n: int | None) -> dict[str, Any]:
    import math
    valid = value is not None and math.isfinite(value)
    if direction not in {"max", "min", "lt"}:
        raise ValueError(f"Unsupported threshold direction: {direction}")
    passed = valid and (value <= threshold if direction == "max" else value < threshold if direction == "lt" else value >= threshold)
    return {"metric": metric, "value": value, "threshold": threshold, "source": source, "n": n,
            "verdict": "PASS" if passed else "FAIL" if valid else "NOT_MEASURED"}


def _construct_runner(runner_cls: Any, kwargs: dict[str, Any]) -> Any:
    """Construct MultimodalBenchmarkRunner (or test monkeypatch) passing only supported kwargs."""
    import inspect
    try:
        sig = inspect.signature(runner_cls)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            return runner_cls(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return runner_cls(**filtered)
    except (TypeError, ValueError):
        return runner_cls(**kwargs)


def run_benchmark(
    val_dir: Path = Path("validation"),
    test_dir: Path = Path("test_queries"),
    out_dir: Path = Path("benchmark"),
    db_path: Path = Path("data/processed/metadata.db"),
    faiss_path: Path = Path("data/processed/faiss_index.bin"),
    model_dir: Path = Path("models/siglip2"),
    max_msrvtt: int = 100,
    run_all_splits: bool = True,
    allow_insample_eval: bool = False,
    ltr_model_path: Path | None = Path("models/ranking/videntra_ltr_v1.json"),
    splits: Sequence[str] | str | None = None,
    temporal_rerank_mode: str = "attention",
    temporal_rerank_lambda: float = 0.0,
    temporal_rerank_top_k: int = 10,
    enable_spatial_patches: bool = True,
    enable_vlm_caption_lexical: bool = False,
    enable_vlm_caption_dense: bool = False,
    enable_clip_consensus_boost: bool | None = None,
    ltr_blend_weight: float | None = None,
    a5_preset: str | None = None,
    publish_reference: bool = False,
    stage2_selection_mode: str = "exact",
    video_proposal_mode: str = "frame_topk",
    video_proposal_top_n: int = 60,
    video_proposal_anchors_per_video: int = 2,
    video_proposal_min_pts_gap: float = 1.0,
    video_proposal_clip_reserve_ratio: float = 0.0,
    disable_pure_visual_lexical_bonus: bool = False,
    device: str = "cpu",
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out_dir}")
    git_worktree = get_git_worktree_status()
    if publish_reference and git_worktree.get("git_dirty") is not False:
        raise RuntimeError(
            f"Refusing --publish-reference with dirty/unknown git worktree: {git_worktree.get('git_dirty_files')}"
        )

    valid_splits = ("dev_150", "dev_full_859", "heldout_test_201")
    if splits is not None:
        raw_splits = [s.strip() for s in (splits.split(",") if isinstance(splits, str) else splits) if str(s).strip()]
        if not raw_splits:
            raise ValueError("--splits must specify at least one split")
        unknown = [s for s in raw_splits if s not in valid_splits]
        if unknown:
            raise ValueError(f"Unknown split(s) in --splits: {unknown}. Valid: {list(valid_splits)}")
        selected_splits = list(dict.fromkeys(raw_splits))
        run_ablation_matrix = False
    else:
        selected_splits = ["dev_150", "dev_full_859", "heldout_test_201"] if run_all_splits else ["dev_150"]
        run_ablation_matrix = bool(run_all_splits)

    resolved_consensus_boost, resolved_ltr_weight = resolve_a5_preset(
        a5_preset,
        enable_clip_consensus_boost=enable_clip_consensus_boost,
        ltr_blend_weight=ltr_blend_weight,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    ltr_info = load_ltr_model_info(ltr_model_path)
    if ltr_model_path is not None and (not ltr_info.get("exists") or ltr_info.get("error")):
        raise RuntimeError(
            f"Invalid LTR model artifact ({ltr_model_path}): {ltr_info.get('error') or 'missing_artifact'}"
        )

    feature_flags = {
        "stage2_selection_mode": str(stage2_selection_mode),
        "video_proposal_mode": str(video_proposal_mode),
        "video_proposal_top_n": int(video_proposal_top_n),
        "video_proposal_anchors_per_video": int(video_proposal_anchors_per_video),
        "video_proposal_min_pts_gap": float(video_proposal_min_pts_gap),
        "video_proposal_clip_reserve_ratio": float(video_proposal_clip_reserve_ratio),
        "disable_pure_visual_lexical_bonus": bool(disable_pure_visual_lexical_bonus),
        "enable_spatial_patches": bool(enable_spatial_patches),
        "enable_vlm_caption_lexical": bool(enable_vlm_caption_lexical),
        "enable_vlm_caption_dense": bool(enable_vlm_caption_dense),
        "enable_temporal_rerank": bool(float(temporal_rerank_lambda) > 0.0),
        "temporal_rerank_mode": str(temporal_rerank_mode),
        "temporal_rerank_lambda": float(temporal_rerank_lambda),
        "temporal_rerank_top_k": int(temporal_rerank_top_k),
        "enable_clip_consensus_boost": resolved_consensus_boost,
        "ltr_blend_weight": float(resolved_ltr_weight),
        "a5_preset": a5_preset,
    }

    db_path = Path(db_path)
    faiss_path = Path(faiss_path)
    model_dir = Path(model_dir)
    processed_dir = db_path.resolve().parent

    src_videntra_files = [
        p for p in sorted(Path("src/videntra").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    input_paths = [
        *src_videntra_files,
        Path("configs"),
        Path("scripts/run_validation_benchmark.py"),
        val_dir,
        test_dir,
        db_path,
        faiss_path,
        processed_dir / "faiss_text_e5_384.bin",
        processed_dir / "e5_text_manifest.json",
        processed_dir / "faiss_e5_caption_384.bin",
        processed_dir / "e5_caption_manifest.json",
        processed_dir / "spatial_patches_fp16.npy",
        processed_dir / "spatial_patches_map.jsonl",
        processed_dir / "p85_receipt.json",
        *( [ltr_model_path] if ltr_model_path is not None else [] ),
        model_dir,
        Path("data/raw/avc/provenance/data/processed/avc_1.8m/manifests/pilot_queries_eval_only.json"),
    ]
    input_hashes = hash_paths(input_paths)
    resolved_device = resolve_device(device, default="auto")
    runner = _construct_runner(
        MultimodalBenchmarkRunner,
        {
            "db_path": db_path,
            "faiss_path": faiss_path,
            "model_dir": model_dir,
            "device": resolved_device,
            "ltr_model_path": ltr_model_path,
            "enable_spatial_patches": bool(enable_spatial_patches),
            "enable_vlm_caption_lexical": bool(enable_vlm_caption_lexical),
            "enable_vlm_caption_dense": bool(enable_vlm_caption_dense),
            "enable_temporal_rerank": bool(float(temporal_rerank_lambda) > 0.0),
            "temporal_rerank_mode": str(temporal_rerank_mode),
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
            "temporal_rerank_top_k": int(temporal_rerank_top_k),
            "enable_clip_consensus_boost": resolved_consensus_boost,
            "ltr_blend_weight": float(resolved_ltr_weight),
            "stage2_selection_mode": str(stage2_selection_mode),
            "video_proposal_mode": str(video_proposal_mode),
            "video_proposal_top_n": int(video_proposal_top_n),
            "video_proposal_anchors_per_video": int(video_proposal_anchors_per_video),
            "video_proposal_min_pts_gap": float(video_proposal_min_pts_gap),
            "video_proposal_clip_reserve_ratio": float(video_proposal_clip_reserve_ratio),
            "disable_pure_visual_lexical_bonus": bool(disable_pure_visual_lexical_bonus),
        },
    )
    if ltr_model_path is not None:
        ranker = getattr(getattr(runner, "pipeline", None), "ltr_ranker", None)
        if ranker is not None:
            ranker_path = getattr(ranker, "model_path", None)
            if ranker_path is not None and Path(ranker_path).resolve() != Path(ltr_model_path).resolve():
                raise RuntimeError(
                    f"LTR model path mismatch: requested {Path(ltr_model_path).resolve()}, "
                    f"loaded {Path(ranker_path).resolve()}"
                )
            if not getattr(ranker, "is_loaded", False):
                raise RuntimeError(f"LTR ranker failed to load artifact: {ltr_model_path}")

    def _can_use_oof(queries_list: list[dict[str, Any]]) -> bool:
        if not ltr_info.get("has_fold_weights"):
            return False
        trained_set = {str(q) for q in ltr_info.get("trained_query_ids", [])}
        trained_groups = {str(g) for g in ltr_info.get("trained_group_ids", [])}
        relevant: list[tuple[str, list[str]]] = []
        for q_item in queries_list:
            qid_str = str(q_item["query_id"])
            q_groups, _, _ = resolve_query_source_groups(q_item)
            if (not trained_set and not trained_groups) or qid_str in trained_set or bool(set(q_groups) & trained_groups):
                relevant.append((qid_str, q_groups))
        return bool(relevant) and all(
            resolve_oof_fold_for_query(qid_item, q_grps, ltr_info)[0] is not None
            for qid_item, q_grps in relevant
        )

    used_oof_splits: set[str] = set()
    split_query_ids: dict[str, list[str]] = {}
    all_degraded_channels: set[str] = set()
    all_disabled_channels: set[str] = set()
    all_channel_status: dict[str, dict[str, Any]] = {}
    oof_stats_by_split: dict[str, dict[str, int]] = {}
    ltr_stats_by_split: dict[str, dict[str, Any]] = {}
    clip_consensus_stats_by_split: dict[str, dict[str, Any]] = {}

    def _merge_summary_channels(s_obj: dict[str, Any], split_key: str) -> None:
        all_degraded_channels.update(s_obj.get("degraded_channels", []))
        all_disabled_channels.update(s_obj.get("disabled_channels", []))
        for ch, info in (s_obj.get("channel_status") or {}).items():
            prev = all_channel_status.get(ch)
            if prev is None:
                all_channel_status[ch] = dict(info)
            else:
                prev["affected_queries"] = int(prev.get("affected_queries", 0)) + int(info.get("affected_queries", 0))
                if not prev.get("sample_reason") and info.get("sample_reason"):
                    prev["sample_reason"] = info["sample_reason"]
        if "oof_stats" in s_obj:
            oof_stats_by_split[split_key] = dict(s_obj["oof_stats"])
        if "ltr_stats" in s_obj:
            ltr_stats_by_split[split_key] = dict(s_obj["ltr_stats"])
        if "clip_consensus_stats" in s_obj:
            clip_consensus_stats_by_split[split_key] = dict(s_obj["clip_consensus_stats"])

    print(f"\n2. Đang thu thập câu truy vấn cho các tập đánh giá ({', '.join(selected_splits)})...")
    ablation_summaries: dict[str, dict[str, Any]] = {}
    multi_split_summaries: dict[str, dict[str, Any]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    split_queries_map: dict[str, list[dict[str, Any]]] = {}
    split_raw_summaries: dict[str, dict[str, Any]] = {}

    if "dev_150" in selected_splits:
        queries_dev150 = load_validation_queries(val_dir, runner.indexed_videos, max_msrvtt=max_msrvtt)
        split_queries_map["dev_150"] = queries_dev150
        split_query_ids["dev_150"] = [str(q["query_id"]) for q in queries_dev150]
        print(f"   - dev_150          : {len(queries_dev150)} queries")

        use_oof_dev150 = _can_use_oof(queries_dev150)
        dev150_fold_map = ltr_info["fold_query_map"] if use_oof_dev150 else None
        if use_oof_dev150:
            used_oof_splits.add("dev_150")

        print("\n3. BẮT ĐẦU CHẠY ĐÁNH GIÁ CHÍNH TRÊN TẬP DEV_150 (A5 FULL PIPELINE)...")
        print("=" * 95)
        dev150_summary, dev150_records = evaluate_query_set(
            runner,
            queries_dev150,
            split_label="dev_150",
            ablation_config="A5",
            show_progress=True,
            fold_query_map=dev150_fold_map,
            ltr_info=ltr_info if use_oof_dev150 else None,
        )
        _merge_summary_channels(dev150_summary, "dev_150")
        split_raw_summaries["dev_150"] = dev150_summary
        split_records["dev_150"] = dev150_records
        multi_split_summaries["dev_150"] = {k: v for k, v in dev150_summary.items() if k != "_raw_arrays"}

        if run_ablation_matrix:
            print("\n4. ĐANG CHẠY MA TRẬN ABLATION 6 BƯỚC (A0 -> A5) TRÊN TẬP DEV_150...")
            for cfg in ["A0", "A1", "A2", "A3", "A4"]:
                s_cfg, cfg_records = evaluate_query_set(
                    runner,
                    queries_dev150,
                    split_label="ablation",
                    ablation_config=cfg,
                    show_progress=True,
                    fold_query_map=dev150_fold_map,
                    ltr_info=ltr_info if use_oof_dev150 else None,
                )
                _merge_summary_channels(s_cfg, f"ablation_{cfg}")
                ablation_summaries[cfg] = s_cfg
                with (out_dir / f"ablation_{cfg}_evaluation.jsonl").open("w", encoding="utf-8") as stream:
                    for record in cfg_records:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            ablation_summaries["A5"] = dev150_summary

            # Compute paired sign-test p-values against A0 and between consecutive steps
            a0_h1 = ablation_summaries["A0"]["_raw_arrays"]["hits1"]
            a0_mrr = ablation_summaries["A0"]["_raw_arrays"]["mrrs"]
            a5_h1 = ablation_summaries["A5"]["_raw_arrays"]["hits1"]
            a5_mrr = ablation_summaries["A5"]["_raw_arrays"]["mrrs"]
            dev150_summary["metrics_overall"]["pvalue_vs_a0_hit1"] = round(
                compute_paired_sign_test_pvalue(a0_h1, a5_h1), 6
            )
            dev150_summary["metrics_overall"]["pvalue_vs_a0_mrr"] = round(
                compute_paired_sign_test_pvalue(a0_mrr, a5_mrr), 6
            )

    if "dev_full_859" in selected_splits:
        queries_dev859 = load_validation_queries(val_dir, runner.indexed_videos, max_msrvtt=None)
        split_queries_map["dev_full_859"] = queries_dev859
        split_query_ids["dev_full_859"] = [str(q["query_id"]) for q in queries_dev859]
        use_oof_dev859 = _can_use_oof(queries_dev859)
        dev859_fold_map = ltr_info["fold_query_map"] if use_oof_dev859 else None
        if use_oof_dev859:
            used_oof_splits.add("dev_full_859")
        print(f"\n5. ĐANG CHẠY TRÊN TOÀN BỘ TẬP VALIDATION (dev_full_859: {len(queries_dev859)} queries)...")
        dev859_summary, dev859_records = evaluate_query_set(
            runner,
            queries_dev859,
            split_label="dev_full_859",
            ablation_config="A5",
            show_progress=True,
            fold_query_map=dev859_fold_map,
            ltr_info=ltr_info if use_oof_dev859 else None,
        )
        _merge_summary_channels(dev859_summary, "dev_full_859")
        split_raw_summaries["dev_full_859"] = dev859_summary
        split_records["dev_full_859"] = dev859_records
        multi_split_summaries["dev_full_859"] = {k: v for k, v in dev859_summary.items() if k != "_raw_arrays"}
        with (out_dir / "dev_full_evaluation.jsonl").open("w", encoding="utf-8") as stream:
            for record in dev859_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if "heldout_test_201" in selected_splits and test_dir.exists():
        queries_test201 = load_validation_queries(test_dir, runner.indexed_videos, max_msrvtt=None)
        split_queries_map["heldout_test_201"] = queries_test201
        split_query_ids["heldout_test_201"] = [str(q["query_id"]) for q in queries_test201]
        print(f"\n6. ĐANG CHẠY TRÊN TẬP HELDOUT TEST (heldout_test_201: {len(queries_test201)} queries)...")
        test201_summary, test201_records = evaluate_query_set(
            runner, queries_test201, split_label="heldout_test_201", ablation_config="A5", show_progress=True
        )
        _merge_summary_channels(test201_summary, "heldout_test_201")
        split_raw_summaries["heldout_test_201"] = test201_summary
        split_records["heldout_test_201"] = test201_records
        multi_split_summaries["heldout_test_201"] = {k: v for k, v in test201_summary.items() if k != "_raw_arrays"}
        with open(out_dir / "heldout_test_201_evaluation.jsonl", "w", encoding="utf-8") as f:
            for r in test201_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    primary_split = "dev_150" if "dev_150" in split_raw_summaries else next(iter(split_raw_summaries))
    primary_summary = split_raw_summaries[primary_split]
    primary_records = split_records[primary_split]
    primary_queries = split_queries_map[primary_split]

    print("\n7. ĐANG ĐO COLD QUERY LATENCY & PEAK RAM FOOTPRINT (C3)...")
    cold_stats = measure_cold_query_latency(runner, primary_queries[:15])
    rss_ram_mb, peak_ram_mb = get_process_memory_mb()

    primary_summary["metrics_overall"]["cold_latency_mean_ms"] = cold_stats["cold_mean_ms"]
    primary_summary["metrics_overall"]["cold_latency_p50_ms"] = cold_stats["cold_p50_ms"]
    primary_summary["metrics_overall"]["cold_latency_p95_ms"] = cold_stats["cold_p95_ms"]
    primary_summary["metrics_overall"]["ram_rss_mb"] = rss_ram_mb
    primary_summary["metrics_overall"]["ram_peak_mb"] = peak_ram_mb

    # Export Per-Query Results for primary split
    csv_file = out_dir / "per_query_evaluation.csv"
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(primary_records[0].keys()))
        writer.writeheader()
        writer.writerows(primary_records)

    jsonl_file = out_dir / "per_query_evaluation.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in primary_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ablation_clean = {}
    for cfg, s_cfg in ablation_summaries.items():
        ov_c = s_cfg["metrics_overall"]
        ablation_clean[cfg] = {
            "video_hit_at_1": ov_c["video_hit_at_1"],
            "video_hit_at_5": ov_c["video_hit_at_5"],
            "video_hit_at_10": ov_c["video_hit_at_10"],
            "mrr": ov_c["mean_reciprocal_rank_mrr"],
            "video_ndcg_at_10": ov_c["video_ndcg_at_10"],
            "moment_hit_at_1_iou03": ov_c["moment_hit_at_1_iou03"],
            "moment_hit_at_1_iou05": ov_c["moment_hit_at_1_iou05"],
            "moment_ndcg_at_5": ov_c["moment_ndcg_at_5"],
            "moment_map_range_05_095": ov_c["moment_map_range_05_095"],
            "non_neural_latency_p50_ms": ov_c["non_neural_latency_p50_ms"],
        }

    gate = evaluate_provenance_gate(
        ltr_info=ltr_info,
        split_query_ids=split_query_ids,
        used_oof_splits=used_oof_splits,
        allow_insample_eval=allow_insample_eval,
        degraded_channels=all_degraded_channels,
        feature_flags=feature_flags,
        selected_splits=selected_splits,
        split_queries=split_queries_map,
    )
    run_config = {
        "max_msrvtt": max_msrvtt,
        "run_all_splits": run_all_splits,
        "splits": selected_splits,
        "allow_insample_eval": allow_insample_eval,
        "publish_reference": bool(publish_reference),
        "db_path": str(db_path),
        "faiss_path": str(faiss_path),
        "model_dir": str(model_dir),
        "ltr_model_path": str(ltr_model_path) if ltr_model_path is not None else None,
        "ltr_model_resolved_path": gate.get("ltr_model_resolved_path"),
        "ablation_config": "A5",
        "feature_flags": feature_flags,
        "cutoffs": {
            "top_k": 10,
            "ranking_mode": "diverse_video",
            "max_msrvtt": max_msrvtt,
        },
    }
    effective_config_sha256 = compute_effective_config_sha256(run_config)
    code_sha256 = compute_code_sha256([Path(__file__).resolve()])
    cpu_thread_settings = get_cpu_thread_settings()

    gallery_coverage_rate: float | None = None
    ensure_clip_fn = getattr(getattr(runner, "pipeline", None), "_ensure_clip_branch_index", None)
    if callable(ensure_clip_fn) and runner.indexed_videos:
        clip_idx = ensure_clip_fn()
        if clip_idx is not None:
            gallery_coverage_rate = round(len(clip_idx.video_ids) / max(1, len(runner.indexed_videos)), 6)

    git_sha = get_git_sha()
    hardware_info = get_hardware_info(device=getattr(runner, "device", "cpu"))
    artifact_tokens = (
        "metadata.db",
        db_path.name,
        faiss_path.name,
        "faiss_",
        "manifest",
        "spatial_patches",
        "videoprism",
        "p85_receipt",
        "videntra_ltr",
        "siglip2",
    )
    snapshot_artifacts = {
        k: v for k, v in input_hashes.items()
        if any(tok in k for tok in artifact_tokens)
    }
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    final_summary = {
        "status": "completed_leakage_free" if gate["leakage_free"] else str(gate.get("eval_mode") or "fit_all_leakage_detected"),
        "run_id": run_id,
        "benchmark_timestamp": ts_str,
        "git_sha": git_sha,
        "git_worktree": git_worktree,
        "code_sha256": code_sha256,
        "effective_config_sha256": effective_config_sha256,
        "cpu_thread_settings": cpu_thread_settings,
        "gallery_coverage_rate": gallery_coverage_rate,
        "ltr_model_sha256": gate["ltr_model_sha256"],
        "ltr_model_resolved_path": gate.get("ltr_model_resolved_path"),
        "ltr_feature_schema": gate.get("ltr_feature_schema"),
        "manifest_hash": gate.get("manifest_hash"),
        "source_group_rule": gate.get("source_group_rule"),
        "eval_mode": gate["eval_mode"],
        "overlap_with_eval": gate["overlap_with_eval"],
        "source_group_overlap_with_eval": gate.get("source_group_overlap_with_eval", {}),
        "heldout_source_leakage": gate.get("heldout_source_leakage", False),
        "oof_stats": oof_stats_by_split,
        "ltr_stats": ltr_stats_by_split,
        "clip_consensus_stats": clip_consensus_stats_by_split,
        "degraded_channels": gate["degraded_channels"],
        "disabled_channels": sorted(all_disabled_channels),
        "channel_status": all_channel_status,
        "leakage_free": gate["leakage_free"],
        "config": run_config,
        "feature_flags": feature_flags,
        "hardware": hardware_info,
        "snapshot_artifact_hashes": snapshot_artifacts,
        "primary_split": primary_split,
        "total_queries_evaluated": len(primary_queries),
        "metrics_overall": primary_summary["metrics_overall"],
        "per_dataset_metrics": primary_summary["per_dataset_metrics"],
        "dataset_averages": primary_summary["dataset_averages"],
        "multi_split_evaluation": multi_split_summaries,
        "ablation_matrix_dev150": ablation_clean,
    }

    ov = final_summary["metrics_overall"]
    specs = [("video_hit_at_1", 89/150, "min"), ("mean_reciprocal_rank_mrr", .645, "min"),
             ("video_ndcg_at_10", .675, "min"), ("moment_hit_at_1_iou05", 16/22, "min"),
             ("moment_ndcg_at_5", .53, "min"), ("pure_asr_fp_top1_count", 1, "max"),
             ("non_neural_latency_p50_ms", 20, "max"), ("non_neural_latency_p95_ms", 45, "max"),
             ("cold_latency_p50_ms", 220, "max"), ("cold_latency_p95_ms", 420, "max"),
             ("ram_peak_mb", 4000, "lt")]
    acceptance = []
    for k, threshold, direction in specs:
        n = (None if k.startswith(("cold_", "ram_")) else
             sum(bool(q.get("target_windows")) for q in primary_queries) if k.startswith("moment_") else
             sum(r["modality_attribution"] == "asr" for r in primary_records) if k == "pure_asr_fp_top1_count" else len(primary_queries))
        value = None if n == 0 or (k.endswith("p95_ms") and n is not None and n < 100) else ov.get(k)
        acceptance.append(acceptance_metric(k, value, threshold, direction,
            "process_memory" if k.startswith("ram_") else primary_split, n))
    final_summary["acceptance_metrics"] = acceptance
    summary_file = out_dir / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    report_md = out_dir / "benchmark_report.md"
    generate_markdown_report(report_md, final_summary, primary_records)

    inputs_unchanged = verify_hashes(input_hashes) and hash_paths(input_paths) == input_hashes
    output_hashes = hash_paths([p for p in out_dir.iterdir() if p.is_file()])
    provenance = {
        "run_id": run_id,
        "git_sha": git_sha,
        "git_worktree": git_worktree,
        "code_sha256": code_sha256,
        "effective_config_sha256": effective_config_sha256,
        "cpu_thread_settings": cpu_thread_settings,
        "gallery_coverage_rate": gallery_coverage_rate,
        "ltr_model_sha256": gate["ltr_model_sha256"],
        "ltr_model_resolved_path": gate.get("ltr_model_resolved_path"),
        "ltr_feature_schema": gate.get("ltr_feature_schema"),
        "ltr_trained_on": gate["ltr_trained_on"],
        "manifest_hash": gate.get("manifest_hash"),
        "source_group_rule": gate.get("source_group_rule"),
        "oof_validation_errors": gate.get("oof_validation_errors", []),
        "fold_summary": gate.get("fold_summary", []),
        "oof_stats": oof_stats_by_split,
        "ltr_stats": ltr_stats_by_split,
        "clip_consensus_stats": clip_consensus_stats_by_split,
        "overlap_with_eval": gate["overlap_with_eval"],
        "source_group_overlap_with_eval": gate.get("source_group_overlap_with_eval", {}),
        "heldout_source_leakage": gate.get("heldout_source_leakage", False),
        "eval_mode": gate["eval_mode"],
        "eval_mode_by_split": gate["eval_mode_by_split"],
        "leakage_free": gate["leakage_free"],
        "allow_insample_eval": gate["allow_insample_eval"],
        "gate_passed": gate["gate_passed"],
        "degraded_channels": gate["degraded_channels"],
        "disabled_channels": sorted(all_disabled_channels),
        "channel_status": all_channel_status,
        "feature_flags": feature_flags,
        "selected_splits": selected_splits,
        "hardware": hardware_info,
        "snapshot_artifact_hashes": snapshot_artifacts,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "inputs_unchanged": inputs_unchanged,
        "outputs_verified": verify_hashes(output_hashes),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "device": getattr(runner, "device", "cpu"),
            "argv": sys.argv,
            "disk_cache": False,
        },
        "config": run_config,
        "limitations": [
            "Hashes identify local files, not media rights or independently verified model license.",
            "Cold query latency and process RAM profiling may be NOT_MEASURED unless explicitly collected.",
        ],
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if not inputs_unchanged or not verify_hashes(output_hashes):
        raise RuntimeError("Benchmark provenance verification failed; retain output for inspection")
    if not gate["gate_passed"]:
        raise RuntimeError(
            f"Provenance leakage gate failed: eval_mode={gate['eval_mode']}, "
            f"overlap_with_eval={gate['overlap_with_eval']}, leakage_free={gate['leakage_free']}. "
            "Pass --allow-insample-eval to override non-heldout overlap."
        )
    if publish_reference and not gate["leakage_free"]:
        raise RuntimeError(
            f"Refusing --publish-reference when leakage_free=False (eval_mode={gate['eval_mode']})"
        )

    # E9: Save immutable provenance snapshot in work/experiments/<timestamp>_<split>_<git_sha>/
    exp_stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("work") / "experiments" / f"{exp_stamp}_all_splits_{git_sha}_{run_id[:8]}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    for src_f in out_dir.iterdir():
        if src_f.is_file():
            shutil.copy2(src_f, exp_dir / src_f.name)

    if publish_reference:
        ref_dir = Path("benchmark")
        if out_dir.resolve() != ref_dir.resolve():
            ref_dir.mkdir(parents=True, exist_ok=True)
            for src_f in out_dir.iterdir():
                if src_f.is_file():
                    tmp_dst = ref_dir / f"{src_f.name}.tmp.{os.getpid()}"
                    try:
                        shutil.copy2(src_f, tmp_dst)
                        os.replace(tmp_dst, ref_dir / src_f.name)
                    finally:
                        if tmp_dst.exists():
                            try:
                                tmp_dst.unlink()
                            except OSError:
                                pass

    runner.pipeline.close()

    ov = final_summary["metrics_overall"]
    print(f"\n{'='*95}")
    print("HOÀN TẤT BENCHMARK & KIỂM TOÁN TOÀN DIỆN HỆ THỐNG VIDENTRA (P0 -> P4)!")
    print(f"{'='*95}")
    print(f"- dev_150 Video Hit@1          : {ov['video_hit_at_1']*100:.2f}% (95% CI: [{ov['video_hit_at_1_ci95'][0]*100:.2f}%, {ov['video_hit_at_1_ci95'][1]*100:.2f}%])")
    print(f"- dev_150 MRR                  : {ov['mean_reciprocal_rank_mrr']:.4f} (95% CI: [{ov['mrr_ci95'][0]:.4f}, {ov['mrr_ci95'][1]:.4f}])")
    print(f"- dev_150 Video nDCG@10        : {ov['video_ndcg_at_10']:.4f}")
    print(f"- dev_150 Moment Hit@1 (0.5)   : {ov['moment_hit_at_1_iou05']*100:.2f}% | Moment nDCG@5: {ov['moment_ndcg_at_5']:.4f}")
    print(f"- Pure-ASR FP Top-1            : {ov['pure_asr_fp_top1_count']} ca")
    if "dev_full_859" in multi_split_summaries:
        ov859 = multi_split_summaries["dev_full_859"]["metrics_overall"]
        print(f"- dev_full_859 (859 queries)   : Hit@1 = {ov859['video_hit_at_1']*100:.2f}% | MRR = {ov859['mean_reciprocal_rank_mrr']:.4f} | nDCG@10 = {ov859['video_ndcg_at_10']:.4f}")
    if "heldout_test_201" in multi_split_summaries:
        ov201 = multi_split_summaries["heldout_test_201"]["metrics_overall"]
        print(f"- heldout_test_201 (201 q)     : Hit@1 = {ov201['video_hit_at_1']*100:.2f}% | MRR = {ov201['mean_reciprocal_rank_mrr']:.4f} | nDCG@10 = {ov201['video_ndcg_at_10']:.4f}")
    print(f"- Non-Neural Latency (p50/p95) : {ov['non_neural_latency_p50_ms']} ms / {ov['non_neural_latency_p95_ms']} ms")
    print(f"- Cold Query Latency (p50/p95) : {ov['cold_latency_p50_ms']} ms / {ov['cold_latency_p95_ms']} ms")
    print(f"- RAM Footprint (RSS / Peak)   : {ov['ram_rss_mb']} MB / {ov['ram_peak_mb']} MB")
    print(f"- Immutable Provenance Dir     : {exp_dir}")
    print(f"{'='*95}\n")


def generate_markdown_report(report_path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    ov = summary["metrics_overall"]
    verdicts = {m["metric"]: m["verdict"] for m in summary.get("acceptance_metrics", [])}
    pds = summary["per_dataset_metrics"]
    msplits = summary.get("multi_split_evaluation", {})
    ablation = summary.get("ablation_matrix_dev150", {})

    def fmt_opt(val: Any, unit: str = "") -> str:
        if val is None:
            return "NOT_MEASURED"
        return f"{val}{unit}"

    baseline_p3 = {
        "video_hit_at_1": 0.5133,
        "video_hit_at_5": 0.6867,
        "video_hit_at_10": 0.7533,
        "mean_reciprocal_rank_mrr": 0.5936,
        "video_ndcg_at_1": 0.5133,
        "video_ndcg_at_5": 0.5279,
        "video_ndcg_at_10": 0.5320,
        "moment_hit_at_1_iou03": 0.5909,
        "moment_hit_at_1_iou05": 0.4091,
        "moment_ndcg_at_5": 0.3620,
        "moment_ndcg_at_10": 0.3750,
    }

    baseline_pre_audit = {
        "video_hit_at_1": 0.5733,
        "video_hit_at_5": 0.6667,
        "video_hit_at_10": 0.7200,
        "mean_reciprocal_rank_mrr": 0.6251,
        "video_ndcg_at_1": 0.5733,
        "video_ndcg_at_5": 0.6259,
        "video_ndcg_at_10": 0.6423,
        "moment_hit_at_1_iou03": 0.7273,
        "moment_hit_at_1_iou05": 0.6818,
        "moment_ndcg_at_5": 0.5021,
        "moment_ndcg_at_10": 0.5021,
    }

    def fmt_delta_pct(cur: float, base: float) -> str:
        d = (cur - base) * 100.0
        sign = "+" if d >= 0 else ""
        return f"**{sign}{d:.2f}%**"

    def fmt_delta_val(cur: float, base: float) -> str:
        d = cur - base
        sign = "+" if d >= 0 else ""
        return f"**{sign}{d:.4f}**"

    attr_counts: dict[str, int] = defaultdict(int)
    for r in records:
        attr_counts[r["modality_attribution"]] += 1

    tax = ov.get("failure_taxonomy", {})
    br = ov.get("branch_recall", {})
    mom_n = ov.get("moment_labeled_n")
    if mom_n is None:
        mom_n = sum(1 for r in records if r.get("moment_hit_iou05") not in ("", None)) or ov.get("boundary_mae_n", 0)
    asr_top1_n = ov.get("pure_asr_top1_count")
    if asr_top1_n is None:
        asr_top1_n = sum(1 for r in records if r.get("modality_attribution") == "asr")
    asr_fp = ov.get("pure_asr_fp_top1_count", 0)
    total_q = summary.get("total_queries_evaluated", len(records))

    pval_h1 = ov.get("pvalue_vs_a0_hit1")
    pval_mrr = ov.get("pvalue_vs_a0_mrr")
    if pval_h1 is None or pval_mrr is None:
        pval_str = "NOT_MEASURED"
        pval_verdict = "NOT_MEASURED"
    else:
        pval_str = f"$p(\\text{{Hit@1}}) = {pval_h1:.5f}$, $p(\\text{{MRR}}) = {pval_mrr:.5f}$"
        pval_verdict = "PASS" if (float(pval_h1) < 0.05 and float(pval_mrr) < 0.05) else "FAIL"

    content = f"""# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** {summary['benchmark_timestamp']} | **Run ID:** `{summary.get('run_id', 'unknown')}` | **Git SHA:** `{summary.get('git_sha', 'HEAD')}` (Dirty: `{summary.get('git_worktree', {}).get('git_dirty', 'unknown')}`)  
**LTR Model SHA256:** `{summary.get('ltr_model_sha256', 'none')}` | **Eval Mode:** `{summary.get('eval_mode', 'unknown')}` | **Leakage Free:** `{summary.get('leakage_free', 'unknown')}`  
**Overlap With Eval:** `{json.dumps(summary.get('overlap_with_eval', {}), ensure_ascii=False)}` | **OOF Stats:** `{json.dumps(summary.get('oof_stats', {}), ensure_ascii=False)}`  
**Degraded Channels:** `{json.dumps(summary.get('degraded_channels', []), ensure_ascii=False)}` | **Disabled Channels:** `{json.dumps(summary.get('disabled_channels', []), ensure_ascii=False)}`  
**Channel Status:** `{json.dumps(summary.get('channel_status', {}), ensure_ascii=False)}`  
**Config:** `{json.dumps(summary.get('config', {}), ensure_ascii=False)}`  
**Hardware:** `{json.dumps(summary.get('hardware', {}), ensure_ascii=False)}`  
**Kiến trúc:** `UnifiedMultimodalSearchPipeline` (Single Source of Truth — SSOT 7-Stage Pipeline)  
**Giới hạn:** các số lịch sử bên dưới chưa được xác minh lại và không dùng để nghiệm thu.  

---

## 1. Bảng Nghiệm Thu Toàn Diện Các Chỉ Tiêu Kế Hoạch Kiểm Toán (P0 $\\to$ P4)

| Hạng Mục Nghiệm Thu (Acceptance Criterion) | Ngưỡng Mục Tiêu (Target) | Kết Quả Đạt Được (`dev_150`) | Khoảng Tin Cậy 95% CI | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: |
| **Video Hit@1 (`dev_150`)** | $\\ge 59.33\\%$ ($\\ge 89/150$) | **{ov['video_hit_at_1']*100:.2f}%** ({int(round(ov['video_hit_at_1']*total_q))}/{total_q}) | `[{ov['video_hit_at_1_ci95'][0]*100:.2f}%, {ov['video_hit_at_1_ci95'][1]*100:.2f}%]` | **{verdicts.get('video_hit_at_1', 'NOT_MEASURED')}** |
| **Mean Reciprocal Rank (MRR)** | $\\ge 0.6450$ | **{ov['mean_reciprocal_rank_mrr']:.4f}** (N={total_q}) | `[{ov['mrr_ci95'][0]:.4f}, {ov['mrr_ci95'][1]:.4f}]` | **{verdicts.get('mean_reciprocal_rank_mrr', 'NOT_MEASURED')}** |
| **Video nDCG@10** | $\\ge 0.6750$ | **{ov['video_ndcg_at_10']:.4f}** (N={total_q}) | `[{ov['video_ndcg_at_10_ci95'][0]:.4f}, {ov['video_ndcg_at_10_ci95'][1]:.4f}]` | **{verdicts.get('video_ndcg_at_10', 'NOT_MEASURED')}** |
| **Moment Hit@1 (tIoU $\\ge 0.5$)** | $\\ge 72.73\\%$ ($\\ge 16/22$) | **{ov['moment_hit_at_1_iou05']*100:.2f}%** (N={mom_n}) | `[{ov['moment_hit_at_1_iou05_ci95'][0]*100:.2f}%, {ov['moment_hit_at_1_iou05_ci95'][1]*100:.2f}%]` | **{verdicts.get('moment_hit_at_1_iou05', 'NOT_MEASURED')}** |
| **Moment nDCG@5 (Graded 4-Level)** | $\\ge 0.5300$ | **{ov['moment_ndcg_at_5']:.4f}** (N={mom_n}) | `[{ov['moment_ndcg_at_5_ci95'][0]:.4f}, {ov['moment_ndcg_at_5_ci95'][1]:.4f}]` | **{verdicts.get('moment_ndcg_at_5', 'NOT_MEASURED')}** |
| **Pure-ASR False Positive Top-1** | $\\le 1$ ca sai | **{asr_fp}/{asr_top1_n} pure-ASR** (N={total_q} total) | — | **{verdicts.get('pure_asr_fp_top1_count', 'NOT_MEASURED')}** |
| **Non-Neural Latency ($p_{{50}} / p_{{95}}$)** | $\\le 20\\text{{ ms}} / \\le 45\\text{{ ms}}$ | **{fmt_opt(ov.get('non_neural_latency_p50_ms'), ' ms')} / {fmt_opt(ov.get('non_neural_latency_p95_ms'), ' ms')}** | — | **{verdicts.get('non_neural_latency_p50_ms', 'NOT_MEASURED') + ' / ' + verdicts.get('non_neural_latency_p95_ms', 'NOT_MEASURED')}** |
| **End-to-End Latency ($p_{{50}} / p_{{95}}$)** | — | **{fmt_opt(ov.get('end_to_end_latency_p50_ms', ov.get('latency_p50_ms')), ' ms')} / {fmt_opt(ov.get('end_to_end_latency_p95_ms', ov.get('latency_p95_ms')), ' ms')}** | — | — |
| **Cold Query Latency ($p_{{50}} / p_{{95}}$)** | $\\le 220\\text{{ ms}} / \\le 420\\text{{ ms}}$ | **{fmt_opt(ov.get('cold_latency_p50_ms'), ' ms')} / {fmt_opt(ov.get('cold_latency_p95_ms'), ' ms')}** | — | **{verdicts.get('cold_latency_p50_ms', 'NOT_MEASURED') + ' / ' + verdicts.get('cold_latency_p95_ms', 'NOT_MEASURED')}** |
| **Peak RAM Footprint** | $< 4,000\\text{{ MB}}$ ($4.0\\text{{ GB}}$) | **{fmt_opt(ov.get('ram_peak_mb'), ' MB')}** (RSS: {fmt_opt(ov.get('ram_rss_mb'), ' MB')}) | — | **{verdicts.get('ram_peak_mb', 'NOT_MEASURED')}** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **{pval_str}** | Paired Sign Test | **{pval_verdict}** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\\Delta$ vs Phase 3 | $\\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **{ov['video_hit_at_1']*100:.2f}%** ({int(round(ov['video_hit_at_1']*summary['total_queries_evaluated']))}/{summary['total_queries_evaluated']}) | {fmt_delta_pct(ov['video_hit_at_1'], baseline_p3['video_hit_at_1'])} | {fmt_delta_pct(ov['video_hit_at_1'], baseline_pre_audit['video_hit_at_1'])} |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **{ov['video_hit_at_5']*100:.2f}%** ({int(round(ov['video_hit_at_5']*summary['total_queries_evaluated']))}/{summary['total_queries_evaluated']}) | {fmt_delta_pct(ov['video_hit_at_5'], baseline_p3['video_hit_at_5'])} | {fmt_delta_pct(ov['video_hit_at_5'], baseline_pre_audit['video_hit_at_5'])} |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **{ov['video_hit_at_10']*100:.2f}%** ({int(round(ov['video_hit_at_10']*summary['total_queries_evaluated']))}/{summary['total_queries_evaluated']}) | {fmt_delta_pct(ov['video_hit_at_10'], baseline_p3['video_hit_at_10'])} | {fmt_delta_pct(ov['video_hit_at_10'], baseline_pre_audit['video_hit_at_10'])} |
| **MRR** | 0.5936 | 0.6251 | **{ov['mean_reciprocal_rank_mrr']:.4f}** | {fmt_delta_val(ov['mean_reciprocal_rank_mrr'], baseline_p3['mean_reciprocal_rank_mrr'])} | {fmt_delta_val(ov['mean_reciprocal_rank_mrr'], baseline_pre_audit['mean_reciprocal_rank_mrr'])} |
| **Video nDCG@1** | 0.5133 | 0.5733 | **{ov['video_ndcg_at_1']:.4f}** | {fmt_delta_val(ov['video_ndcg_at_1'], baseline_p3['video_ndcg_at_1'])} | {fmt_delta_val(ov['video_ndcg_at_1'], baseline_pre_audit['video_ndcg_at_1'])} |
| **Video nDCG@5** | 0.5279 | 0.6259 | **{ov['video_ndcg_at_5']:.4f}** | {fmt_delta_val(ov['video_ndcg_at_5'], baseline_p3['video_ndcg_at_5'])} | {fmt_delta_val(ov['video_ndcg_at_5'], baseline_pre_audit['video_ndcg_at_5'])} |
| **Video nDCG@10** | 0.5320 | 0.6423 | **{ov['video_ndcg_at_10']:.4f}** | {fmt_delta_val(ov['video_ndcg_at_10'], baseline_p3['video_ndcg_at_10'])} | {fmt_delta_val(ov['video_ndcg_at_10'], baseline_pre_audit['video_ndcg_at_10'])} |
| **Moment Hit@1 (tIoU $\\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **{ov['moment_hit_at_1_iou03']*100:.2f}%** | {fmt_delta_pct(ov['moment_hit_at_1_iou03'], baseline_p3['moment_hit_at_1_iou03'])} | {fmt_delta_pct(ov['moment_hit_at_1_iou03'], baseline_pre_audit['moment_hit_at_1_iou03'])} |
| **Moment Hit@1 (tIoU $\\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **{ov['moment_hit_at_1_iou05']*100:.2f}%** | {fmt_delta_pct(ov['moment_hit_at_1_iou05'], baseline_p3['moment_hit_at_1_iou05'])} | {fmt_delta_pct(ov['moment_hit_at_1_iou05'], baseline_pre_audit['moment_hit_at_1_iou05'])} |
| **Moment Hit@1 (tIoU $\\ge 0.7$)** | — | — | **{ov['moment_hit_at_1_iou07']*100:.2f}%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **{ov['moment_ndcg_at_5']:.4f}** | {fmt_delta_val(ov['moment_ndcg_at_5'], baseline_p3['moment_ndcg_at_5'])} | {fmt_delta_val(ov['moment_ndcg_at_5'], baseline_pre_audit['moment_ndcg_at_5'])} |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **{ov['moment_map_at_05']:.4f} / {ov['moment_map_at_075']:.4f}** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **{ov['moment_map_range_05_095']:.4f}** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s_name, s_data in msplits.items():
        m_ov = s_data["metrics_overall"]
        c_h1 = m_ov.get("video_hit_at_1_ci95", [0.0, 0.0])
        c_mrr = m_ov.get("mrr_ci95", [0.0, 0.0])
        content += (
            f"| **`{s_name}`** | {s_data['total_queries_evaluated']} | "
            f"**{m_ov['video_hit_at_1']*100:.2f}%** (`[{c_h1[0]*100:.1f}%, {c_h1[1]*100:.1f}%]`) | "
            f"{m_ov['video_hit_at_5']*100:.2f}% | {m_ov['video_hit_at_10']*100:.2f}% | "
            f"**{m_ov['mean_reciprocal_rank_mrr']:.4f}** (`[{c_mrr[0]:.3f}, {c_mrr[1]:.3f}]`) | "
            f"**{m_ov['video_ndcg_at_10']:.4f}** | "
            f"{m_ov.get('within_dataset_hit_at_1', 0.0)*100:.2f}% | "
            f"{m_ov.get('within_dataset_mrr', 0.0):.4f} |\n"
        )

    content += """
---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    pool_sizes = {"qvhighlights": 24, "avc": 300, "msrvtt": 1000}
    for ds, m in pds.items():
        p_sz = pool_sizes.get(ds, 1324)
        content += (
            f"| **{ds.upper()}** | {m['queries']} | {p_sz} videos | "
            f"**{m['hit_at_1']*100:.2f}%** | **{m['within_dataset_hit_at_1']*100:.2f}%** | "
            f"{m['hit_at_5']*100:.2f}% | {m['hit_at_10']*100:.2f}% | "
            f"**{m['mrr']:.4f}** | **{m['within_dataset_mrr']:.4f}** | "
            f"{m['video_ndcg_at_10']:.4f} | {m['visual_recall_at_120']*100:.1f}% | {m['lexical_recall_at_30']*100:.1f}% |\n"
        )

    if ablation:
        content += """
---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        ab_labels = {
            "A0": "Raw 768d SigLIP 2 Visual Baseline",
            "A1": "+ Decoupled 864d Subspace (B5, B6, B8)",
            "A2": "+ Multi-Clause & Sequential Alignment (B4, B11)",
            "A3": "+ Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9)",
            "A4": "+ Anchor-Aware Contrast Boundary Expansion (B10)",
            "A5": "Full VIDENTRA Pipeline (+ Multi-Window Proposal)",
        }
        for cfg_k in ["A0", "A1", "A2", "A3", "A4", "A5"]:
            if cfg_k in ablation:
                a_m = ablation[cfg_k]
                content += (
                    f"| **`{cfg_k}`** | {ab_labels.get(cfg_k, cfg_k)} | "
                    f"**{a_m['video_hit_at_1']*100:.2f}%** | {a_m['video_hit_at_5']*100:.2f}% | {a_m['video_hit_at_10']*100:.2f}% | "
                    f"**{a_m['mrr']:.4f}** | **{a_m['video_ndcg_at_10']:.4f}** | "
                    f"**{a_m['moment_hit_at_1_iou05']*100:.2f}%** | **{a_m['moment_ndcg_at_5']:.4f}** | "
                    f"{a_m['moment_map_range_05_095']:.4f} |\n"
                )

    def fmt_pct_opt(val: Any) -> str:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return f"{val*100:.2f}%"
        return "NOT_MEASURED"

    content += f"""
---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `{summary.get('primary_split', 'dev_150')}`)
- **`Visual Recall@30`**: **{br.get('visual_recall_at_30', 0.0)*100:.2f}%**
- **`Visual Recall@120`**: **{br.get('visual_recall_at_120', 0.0)*100:.2f}%**
- **`Lexical Recall@30 (OCR + ASR)`**: **{br.get('lexical_recall_at_30', 0.0)*100:.2f}%**
- **`Visual Candidate Recall`**: **{fmt_pct_opt(br.get('visual_candidate_recall'))}** (mean pool: `{br.get('mean_visual_pool_size', 'NOT_MEASURED')}`)
- **`Union Recall (Visual + Lexical + Caption)`**: **{fmt_pct_opt(br.get('union_recall'))}** (mean pool: `{br.get('mean_union_pool_size', 'NOT_MEASURED')}`)
- **`Shortlist Recall (Pre-Rerank)`**: **{fmt_pct_opt(br.get('shortlist_recall'))}** | `@10`: **{fmt_pct_opt(br.get('shortlist_recall_at_10'))}** | `@20`: **{fmt_pct_opt(br.get('shortlist_recall_at_20'))}** | `@50`: **{fmt_pct_opt(br.get('shortlist_recall_at_50'))}** (mean pool: `{br.get('mean_pre_rerank_pool_size', 'NOT_MEASURED')}`)
- **`Reranker Input Recall`**: **{fmt_pct_opt(br.get('reranker_input_recall'))}** (mean pool: `{br.get('mean_reranker_input_pool_size', 'NOT_MEASURED')}`)
- **`Fused Recall@10`**: **{br.get('fused_recall_at_10', 0.0)*100:.2f}%**
- **`Target in Shortlist but Not Top-1`**: **{br.get('shortlist_not_top1_count', 0)}** ({br.get('shortlist_not_top1_rate', 0.0)*100:.2f}%)

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
"""
    for att, c in sorted(attr_counts.items(), key=lambda x: x[1], reverse=True):
        content += f"| `{att}` | {c} queries | {(c / max(1, len(records)))*100:.1f}% |\n"

    content += """
### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
"""
    tax_desc = {
        "PASS": "Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows)",
        "F1_VISUAL_CANDIDATE_MISS": "Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng",
        "F2_LEXICAL_DISTRACTOR_OVERTHROW": "Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác",
        "F3_SAME_SOURCE_CLIP_CONFUSION": "Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip)",
        "F4_FINE_GRAINED_VISUAL_CONFUSION": "Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự",
        "F5_BOUNDARY_LOCALIZATION_DRIFT": "Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5",
    }
    for code, desc in tax_desc.items():
        cnt = tax.get(code, 0)
        content += f"| `{code}` | {cnt} | {(cnt / max(1, len(records)))*100:.2f}% | {desc} |\n"

    st_lat = ov.get("stage_latencies_p50_ms", {})
    content += f"""
---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** {'NOT_MEASURED' if ov.get('cold_latency_p50_ms') is None else 'Measured'} (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **{fmt_opt(ov.get('cold_latency_mean_ms'), ' ms')}** | Trung vị (`p50`): **{fmt_opt(ov.get('cold_latency_p50_ms'), ' ms')}** | Ngưỡng `p95`: **{fmt_opt(ov.get('cold_latency_p95_ms'), ' ms')}**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **{fmt_opt(ov.get('end_to_end_latency_mean_ms', ov.get('latency_mean_ms')), ' ms')}** | Trung vị (`p50`): **{fmt_opt(ov.get('end_to_end_latency_p50_ms', ov.get('latency_p50_ms')), ' ms')}** | Ngưỡng `p95`: **{fmt_opt(ov.get('end_to_end_latency_p95_ms', ov.get('latency_p95_ms')), ' ms')}**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **{fmt_opt(ov.get('non_neural_latency_mean_ms'), ' ms')}** | Trung vị (`p50`): **{fmt_opt(ov.get('non_neural_latency_p50_ms'), ' ms')}** | Ngưỡng `p95`: **{fmt_opt(ov.get('non_neural_latency_p95_ms'), ' ms')}**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `{st_lat.get('faiss_ms', 0.0)} ms`, SQLite FTS5: `{st_lat.get('fts5_ms', 0.0)} ms`, Multimodal Fusion: `{st_lat.get('fusion_ms', 0.0)} ms`, Boundary Expansion: `{st_lat.get('boundary_ms', 0.0)} ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **{fmt_opt(ov.get('ram_rss_mb'), ' MB')}**, Peak Working Set = **{fmt_opt(ov.get('ram_peak_mb'), ' MB')}** (xem verdict ở bảng nghiệm thu).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_queries"))
    parser.add_argument("--out-dir", type=Path, default=Path("work/experiments") / ("run_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/siglip2"))
    parser.add_argument("--max-msrvtt", type=int, default=100)
    parser.add_argument("--single-split-only", action="store_true")
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated splits to evaluate (e.g. dev_150,dev_full_859)")
    parser.add_argument("--allow-insample-eval", action="store_true")
    parser.add_argument("--ltr-model-path", type=Path, default=Path("models/ranking/videntra_ltr_v1.json"))
    parser.add_argument("--no-ltr", action="store_true")
    parser.add_argument("--ltr-blend-weight", type=float, default=None)
    parser.add_argument("--a5-preset", choices=("a5_full", "a5_lite", "a5_no_consensus", "a5_no_ltr"), default=None)
    parser.add_argument("--enable-clip-consensus-boost", dest="enable_clip_consensus_boost", action="store_true")
    parser.add_argument("--disable-clip-consensus-boost", dest="enable_clip_consensus_boost", action="store_false")
    parser.set_defaults(enable_clip_consensus_boost=None)
    parser.add_argument("--publish-reference", action="store_true")
    parser.add_argument("--temporal-rerank-mode", type=str, default="attention", choices=["attention", "mean", "mean_top2", "mean_top3"])
    parser.add_argument("--temporal-rerank-lambda", type=float, default=0.0)
    parser.add_argument("--temporal-rerank-top-k", type=int, default=10)
    parser.add_argument("--disable-spatial-patches", action="store_true")
    parser.add_argument("--enable-vlm-caption-lexical", action="store_true")
    parser.add_argument("--enable-vlm-caption-dense", action="store_true")
    parser.add_argument("--disable-vlm-caption-lexical", action="store_true")
    parser.add_argument("--disable-vlm-caption-dense", action="store_true")
    parser.add_argument("--stage2-selection-mode", choices=("exact", "legacy_union"), default="exact")
    parser.add_argument("--video-proposal-mode", choices=("frame_topk", "video_max", "video_mean_top2_pts", "video_rrf_max_top2"), default="frame_topk")
    parser.add_argument("--video-proposal-top-n", type=int, default=60)
    parser.add_argument("--video-proposal-anchors-per-video", type=int, default=2)
    parser.add_argument("--video-proposal-min-pts-gap", type=float, default=1.0)
    parser.add_argument("--video-proposal-clip-reserve-ratio", type=float, default=0.0)
    parser.add_argument("--disable-pure-visual-lexical-bonus", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    run_benchmark(
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        db_path=args.db_path,
        faiss_path=args.faiss_path,
        model_dir=args.model_dir,
        max_msrvtt=args.max_msrvtt,
        run_all_splits=not args.single_split_only,
        allow_insample_eval=args.allow_insample_eval,
        ltr_model_path=None if args.no_ltr else args.ltr_model_path,
        splits=args.splits,
        temporal_rerank_mode=args.temporal_rerank_mode,
        temporal_rerank_lambda=args.temporal_rerank_lambda,
        temporal_rerank_top_k=args.temporal_rerank_top_k,
        enable_spatial_patches=not args.disable_spatial_patches,
        enable_vlm_caption_lexical=bool(args.enable_vlm_caption_lexical and not args.disable_vlm_caption_lexical),
        enable_vlm_caption_dense=bool(args.enable_vlm_caption_dense and not args.disable_vlm_caption_dense),
        enable_clip_consensus_boost=args.enable_clip_consensus_boost,
        ltr_blend_weight=args.ltr_blend_weight,
        a5_preset=args.a5_preset,
        publish_reference=args.publish_reference,
        stage2_selection_mode=args.stage2_selection_mode,
        video_proposal_mode=args.video_proposal_mode,
        video_proposal_top_n=args.video_proposal_top_n,
        video_proposal_anchors_per_video=args.video_proposal_anchors_per_video,
        video_proposal_min_pts_gap=args.video_proposal_min_pts_gap,
        video_proposal_clip_reserve_ratio=args.video_proposal_clip_reserve_ratio,
        disable_pure_visual_lexical_bonus=args.disable_pure_visual_lexical_bonus,
        device=args.device,
    )
