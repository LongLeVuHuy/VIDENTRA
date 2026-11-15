r"""Capture frozen A5 and SigLIP2 clip-branch candidate pools for offline VideoPrism replay.

Usage:
    .\.venv\Scripts\python.exe -B scripts/capture_msr_candidate_pools.py `
        --split dev_full_859 --max-k 20 --out-dir work/experiments/clip_vp_pools_dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from videntra.evaluation.provenance import (
    evaluate_provenance_gate,
    hash_paths,
    load_ltr_model_info,
    sha256_file,
)
from videntra.evaluation.split_lock import freeze_split, qvh_source_id, verify_split_lock
from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend
from videntra.retrieval.clip_branch import (
    build_clip_branch_index,
    build_clip_reserve_pool,
    score_clip_branch,
)
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_AVC_WINDOW_RE = re.compile(r"^(.+)__s\d+(?:\.\d+)?__e\d+(?:\.\d+)?$")
_MSRVTT_ID_RE = re.compile(r"^video\d+$")

FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "target_videos",
    "target_windows",
    "relevant_windows",
    "source_group",
    "source_group_status",
    "known_target_window_id",
})


def sha256_text(text: str) -> str:
    """Return SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_gallery_sha256(indexed_videos: Sequence[str] | set[str]) -> str:
    """Compute canonical SHA-256 over sorted indexed video IDs."""
    payload = json.dumps(sorted(str(v) for v in indexed_videos), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_git_commit() -> str | None:
    """Return full HEAD git commit hash if available, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _hash_directory_summary(root: Path, file_hashes: dict[str, str | None]) -> str | None:
    """Compute a canonical SHA-256 over relative paths and digests for a directory or file."""
    if not file_hashes:
        return None
    normalized = []
    for path_str, digest in file_hashes.items():
        p = Path(path_str)
        try:
            rel = str(p.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = p.name
        normalized.append([rel, digest])
    normalized.sort(key=lambda item: item[0])
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_gallery_fingerprint(
    *,
    gallery_sha256: str,
    indexed_video_count: int,
    db_sha256: str | None,
    faiss_sha256: str | None,
) -> str:
    """Compute canonical gallery snapshot fingerprint across video IDs and index files."""
    payload = json.dumps(
        {
            "gallery_sha256": gallery_sha256,
            "indexed_video_count": int(indexed_video_count),
            "db_sha256": db_sha256,
            "faiss_sha256": faiss_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_canonical_source_group(video_id: str, dataset: str | None = None) -> tuple[str, str]:
    """Resolve a clip/video ID to its canonical (source_group, status)."""
    cid = str(video_id).strip()
    if not cid:
        raise ValueError("video_id must be non-empty")
    avc_match = _AVC_WINDOW_RE.fullmatch(cid)
    if avc_match:
        return avc_match.group(1), "avc_window_stripped"
    try:
        return qvh_source_id(cid), "qvh_youtube_id"
    except ValueError:
        pass
    if dataset == "msrvtt" or _MSRVTT_ID_RE.fullmatch(cid):
        return cid, "msrvtt_video_id"
    return f"unknown:{cid}", "unknown_unverified"


def resolve_query_source_group(query: dict[str, Any]) -> tuple[str, str]:
    """Resolve canonical source_group for a query from its target_videos."""
    targets = [str(v) for v in (query.get("target_videos") or []) if str(v).strip()]
    if not targets:
        raise ValueError(f"Query {query.get('query_id')} has no target_videos")
    ds = query.get("dataset")
    resolved = [resolve_canonical_source_group(v, dataset=ds) for v in targets]
    groups = sorted({g for g, _ in resolved})
    statuses = sorted({s for _, s in resolved})
    if len(groups) == 1:
        return groups[0], statuses[0]
    return "|".join(groups), "+".join(statuses)


def _snapshot_ltr_state(ltr_ranker: Any) -> dict[str, Any]:
    return {
        "is_loaded": ltr_ranker.is_loaded,
        "model_type": ltr_ranker.model_type,
        "weights": ltr_ranker.weights.copy(),
        "bias": ltr_ranker.bias,
        "feature_mean": ltr_ranker.feature_mean.copy(),
        "feature_std": ltr_ranker.feature_std.copy(),
        "trees": list(ltr_ranker.trees),
    }


def _restore_ltr_state(ltr_ranker: Any, state: dict[str, Any]) -> None:
    ltr_ranker.is_loaded = state["is_loaded"]
    ltr_ranker.model_type = state["model_type"]
    ltr_ranker.weights = state["weights"].copy()
    ltr_ranker.bias = state["bias"]
    ltr_ranker.feature_mean = state["feature_mean"].copy()
    ltr_ranker.feature_std = state["feature_std"].copy()
    ltr_ranker.trees = list(state["trees"])


def _apply_ltr_fold_payload(ltr_ranker: Any, fold_payload: dict[str, Any]) -> None:
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


def _moment_anchor(m: Any) -> dict[str, Any]:
    if isinstance(m, dict):
        return {
            "moment_id": m.get("moment_id"),
            "start_sec": float(m.get("moment_start_seconds", m.get("start_sec", 0.0))),
            "end_sec": float(m.get("moment_end_seconds", m.get("end_sec", 0.0))),
            "representative_timestamp_seconds": float(m.get("representative_timestamp_seconds", 0.0)),
            "representative_original_frame_index": m.get("representative_original_frame_index"),
            "match_status": m.get("match_status"),
        }
    return {
        "moment_id": getattr(m, "moment_id", None),
        "start_sec": float(getattr(m, "moment_start_seconds", 0.0)),
        "end_sec": float(getattr(m, "moment_end_seconds", 0.0)),
        "representative_timestamp_seconds": float(getattr(m, "representative_timestamp_seconds", 0.0)),
        "representative_original_frame_index": getattr(m, "representative_original_frame_index", None),
        "match_status": getattr(m, "match_status", None),
    }


def build_a5_pool(
    ranked_videos: Sequence[Any],
    all_candidate_moments: Sequence[Any],
    max_k: int = 20,
) -> dict[str, Any]:
    """Build A5 pool: top-10 ranked_videos first, then remaining sorted candidate moments up to max_k."""
    if max_k < 1:
        raise ValueError(f"max_k must be >= 1, got {max_k}")

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    def _vid(item: Any) -> str:
        return str(item["video_id"] if isinstance(item, dict) else getattr(item, "video_id", item))

    def _score(item: Any) -> float:
        return float(item.get("ranking_score", 0.0) if isinstance(item, dict) else getattr(item, "ranking_score", 0.0))

    for item in list(ranked_videos)[:10]:
        vid = _vid(item)
        if not vid or vid in seen:
            continue
        seen.add(vid)
        candidates.append({
            "rank": len(candidates) + 1,
            "video_id": vid,
            "score": _score(item),
            "provenance": "a5_ranked_videos",
            "anchor": _moment_anchor(item),
        })
        if len(candidates) >= max_k:
            break

    if len(candidates) < max_k:
        for item in all_candidate_moments:
            vid = _vid(item)
            if not vid or vid in seen:
                continue
            seen.add(vid)
            candidates.append({
                "rank": len(candidates) + 1,
                "video_id": vid,
                "score": _score(item),
                "provenance": "a5_candidate_moments",
                "anchor": _moment_anchor(item),
            })
            if len(candidates) >= max_k:
                break

    return {
        "pool_video_ids": [c["video_id"] for c in candidates],
        "candidates": candidates,
        "pool_size": len(candidates),
    }


def apply_clip_reserve(
    a5_pool_ids: Sequence[str],
    branch_ranked_ids: Sequence[str],
    k: int,
) -> dict[str, Any]:
    """Apply Section 2.2 clip reserve policy via videntra.retrieval.clip_branch."""
    res = build_clip_reserve_pool(a5_pool_ids, branch_ranked_ids, k=k)
    return {
        "pool_video_ids": res.clip_pool,
        "added_video": res.added_video,
        "displaced_video": res.displaced_video,
        "duplicate_skipped": res.duplicate_skipped,
        "branch_rank": res.branch_rank,
        "fallback_reason": res.fallback_reason,
    }


class SigLIP2ClipBranchScorer:
    """In-memory SigLIP2 frame+clip branch scorer backed by videntra.retrieval.clip_branch."""

    def __init__(self, pipeline: UnifiedMultimodalSearchPipeline) -> None:
        self.pipeline = pipeline
        emb = getattr(pipeline, "all_embeddings", None)
        vid_fids = getattr(pipeline, "video_kf_fids", None) or {}
        vid_kfs = getattr(pipeline, "video_keyframes", None) or {}
        kf_map = getattr(pipeline, "keyframe_map", None) or {}
        if emb is None or len(emb) == 0 or (not vid_fids and not vid_kfs and not kf_map):
            self.index = None
            self.missing_feature_videos: list[str] = []
            return
        self.index = build_clip_branch_index(
            emb,
            video_keyframes=vid_kfs or None,
            video_kf_fids=vid_fids or None,
            keyframe_map=kf_map or None,
        )
        self.missing_feature_videos = sorted(self.index.invalid_videos.keys())

    def rank_query(self, query_text: str, top_n: int = 60) -> list[dict[str, Any]]:
        if self.index is None or not self.index.video_ids:
            return []
        q_vec = np.asarray(self.pipeline.encode_augmented_query(query_text)[0], dtype=np.float32)
        res = score_clip_branch(self.index, q_vec)
        return [
            {
                "video_id": vid,
                "branch_score": float(res.branch_scores[vid]),
                "frame_score": float(res.frame_scores[vid]),
                "clip_score": float(res.clip_scores[vid]),
                "rank_frame": int(res.rank_frame[vid]),
                "rank_clip": int(res.rank_clip[vid]),
            }
            for vid in res.branch_order[:top_n]
        ]


def _load_split_queries(
    split: str,
    val_dir: Path,
    test_dir: Path,
    indexed_videos: set[str],
) -> list[dict[str, Any]]:
    from scripts.run_validation_benchmark import load_validation_queries

    if split == "dev_150":
        return load_validation_queries(val_dir, indexed_videos, max_msrvtt=100)
    if split == "dev_full_859":
        return load_validation_queries(val_dir, indexed_videos, max_msrvtt=None)
    if split == "heldout_test_201":
        return load_validation_queries(test_dir, indexed_videos, max_msrvtt=None)
    raise ValueError(f"Unsupported split: {split}")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_no_gt_leakage(rows: Sequence[dict[str, Any]], label: str) -> None:
    for idx, row in enumerate(rows):
        leaked = FORBIDDEN_INFERENCE_FIELDS & set(row.keys())
        if leaked:
            raise AssertionError(f"Leakage detected in {label}[{idx}]: {sorted(leaked)}")


def capture_pools(
    *,
    split: str = "dev_full_859",
    max_k: int = 20,
    out_dir: Path = Path("work/experiments/clip_vp_pools_dev"),
    eval_dir: Path = Path("work/evaluation-only"),
    val_dir: Path = Path("validation"),
    test_dir: Path = Path("test_queries"),
    db_path: Path = Path("data/processed/metadata.db"),
    faiss_path: Path = Path("data/processed/faiss_index.bin"),
    model_dir: Path = Path("models/siglip2"),
    ltr_model_path: Path | None = Path("models/ranking/videntra_ltr_v1.json"),
    device: str = "cpu",
    seed: int = 20261115,
    temporal_rerank_lambda: float = 0.0,
    allow_insample_eval: bool = False,
    overwrite_eval_artifacts: bool = False,
    branch_top_n: int = 60,
    rules_only: bool = False,
    query_plan_cache: Path | None = None,
    query_plan_revision: str | None = None,
) -> dict[str, Any]:
    if max_k < 2:
        raise ValueError(f"max_k must be >= 2, got {max_k}")
    if (query_plan_cache is None) != (query_plan_revision is None):
        raise ValueError("--query-plan-cache and --query-plan-revision must be used together")
    if rules_only and query_plan_cache is not None:
        raise ValueError("--rules-only and --query-plan-cache are mutually exclusive")

    query_decomposer = None
    query_plan_source = "pipeline_default"
    if query_plan_cache is not None and query_plan_revision is not None:
        query_decomposer = AdaptiveQueryDecomposer.from_frozen_llm_cache(query_plan_cache, query_plan_revision)
        query_plan_source = "frozen_llm_cache"
    elif rules_only:
        query_decomposer = AdaptiveQueryDecomposer(
            llm_backend=LocalSmallLLMBackend(enabled=False),
            enable_disk_cache=False,
        )
        query_plan_source = "rules_only"
    ltr_info = load_ltr_model_info(ltr_model_path)
    pipeline = UnifiedMultimodalSearchPipeline(
        db_path=db_path,
        faiss_path=faiss_path,
        model_dir=model_dir,
        device=device,
        use_disk_cache=False,
        memory_cache_size=0,
        ltr_model_path=ltr_model_path,
        enable_spatial_patches=True,
        enable_vlm_caption_lexical=True,
        enable_vlm_caption_dense=True,
        enable_temporal_rerank=bool(float(temporal_rerank_lambda) > 0.0),
        temporal_rerank_lambda=float(temporal_rerank_lambda),
        query_decomposer=query_decomposer,
    )
    try:
        indexed_videos = set(pipeline.video_durations.keys()) if pipeline.video_durations else set()
        if not indexed_videos and pipeline.keyframe_map:
            indexed_videos = {k["video_id"] for k in pipeline.keyframe_map.values()}
        gallery_sha256 = compute_gallery_sha256(indexed_videos)

        queries = _load_split_queries(split, val_dir, test_dir, indexed_videos)
        if not queries:
            raise ValueError(f"No queries loaded for split={split}")

        # Verify provenance gate (OOF LTR for dev splits, zero overlap for heldout)
        split_query_ids = {split: [str(q["query_id"]) for q in queries]}
        used_oof = set()
        fold_map = ltr_info.get("fold_query_map") or {}
        if ltr_info.get("has_fold_weights"):
            trained_set = {str(q) for q in ltr_info.get("trained_query_ids", [])}
            overlapping = [str(q["query_id"]) for q in queries if not trained_set or str(q["query_id"]) in trained_set]
            if overlapping and all(qid in fold_map for qid in overlapping):
                used_oof.add(split)
        gate = evaluate_provenance_gate(
            ltr_info,
            split_query_ids,
            used_oof_splits=used_oof,
            allow_insample_eval=allow_insample_eval,
        )
        if not gate["gate_passed"]:
            raise RuntimeError(f"Provenance leakage gate failed: {gate}")

        ltr_ranker = getattr(pipeline, "ltr_ranker", None)
        orig_ltr_state = _snapshot_ltr_state(ltr_ranker) if (ltr_ranker is not None and split in used_oof) else None

        clip_scorer = SigLIP2ClipBranchScorer(pipeline)

        query_rows: list[dict[str, Any]] = []
        a5_rows: list[dict[str, Any]] = []
        clip_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        split_lock_records: list[dict[str, Any]] = []

        seen_qids: set[str] = set()
        oof_applied = 0
        oof_fallback = 0
        t_start = time.perf_counter()

        for q in queries:
            qid = str(q["query_id"])
            if qid in seen_qids:
                raise ValueError(f"Duplicate query_id in split {split}: {qid}")
            seen_qids.add(qid)
            q_text = str(q["query_text"])
            q_sha = sha256_text(q_text)
            ds = str(q["dataset"])

            if ltr_ranker is not None and split in used_oof:
                if qid in fold_map:
                    _apply_ltr_fold_payload(ltr_ranker, fold_map[qid])
                    oof_applied += 1
                else:
                    oof_fallback += 1
                    if orig_ltr_state is not None:
                        _restore_ltr_state(ltr_ranker, orig_ltr_state)

            try:
                response = pipeline.search_with_metadata(
                    snapshot_id="clip_vp_capture",
                    query=q_text,
                    top_k=10,
                    mode="accurate",
                    ranking_mode="diverse_video",
                    ablation_config="A5",
                )
            finally:
                if ltr_ranker is not None and orig_ltr_state is not None:
                    _restore_ltr_state(ltr_ranker, orig_ltr_state)

            diag = response.get("diagnostics") or {}
            ranked_videos = response.get("ranked_videos") or []
            all_cand_moments = diag.get("all_candidate_moments") or []

            a5_built = build_a5_pool(ranked_videos, all_cand_moments, max_k=max_k)
            query_plan = pipeline.last_routing_plan
            branch_ranked = clip_scorer.rank_query(q_text, top_n=max(branch_top_n, max_k + 20))
            branch_ids = [b["video_id"] for b in branch_ranked]
            reserve_max_k = apply_clip_reserve(a5_built["pool_video_ids"], branch_ids, k=max_k)

            query_rows.append({
                "query_id": qid,
                "query_text": q_text,
                "query_text_sha256": q_sha,
                "dataset": ds,
                "split": split,
            })
            a5_rows.append({
                "query_id": qid,
                "query_text_sha256": q_sha,
                "dataset": ds,
                "max_k": max_k,
                "pool_video_ids": a5_built["pool_video_ids"],
                "candidates": a5_built["candidates"],
                "pool_size": a5_built["pool_size"],
                "query_plan_mode": query_plan.routing_mode if query_plan is not None else None,
                "query_plan_cache_hit": query_plan.cache_hit if query_plan is not None else None,
            })
            clip_rows.append({
                "query_id": qid,
                "query_text_sha256": q_sha,
                "dataset": ds,
                "max_k": max_k,
                "pool_video_ids": reserve_max_k["pool_video_ids"],
                "branch_ranked_video_ids": branch_ids,
                "branch_candidates": branch_ranked,
                "reserve_trace_max_k": {
                    "k": max_k,
                    "added_video": reserve_max_k["added_video"],
                    "displaced_video": reserve_max_k["displaced_video"],
                    "duplicate_skipped": reserve_max_k["duplicate_skipped"],
                    "branch_rank": reserve_max_k["branch_rank"],
                    "fallback_reason": reserve_max_k["fallback_reason"],
                },
                "pool_size": len(reserve_max_k["pool_video_ids"]),
            })

            source_group, source_status = resolve_query_source_group(q)
            label_rows.append({
                "query_id": qid,
                "query_text_sha256": q_sha,
                "dataset": ds,
                "split": split,
                "target_videos": [str(v) for v in q["target_videos"]],
                "target_windows": [list(w) for w in (q.get("target_windows") or [])],
                "source_group": source_group,
                "source_group_status": source_status,
            })
            split_lock_records.append({
                "query_id": qid,
                "split": split,
                "source_group": source_group,
            })

        _assert_no_gt_leakage(query_rows, "queries.jsonl")
        _assert_no_gt_leakage(a5_rows, "a5_pool.jsonl")
        _assert_no_gt_leakage(clip_rows, "clip_pool.jsonl")

        out_dir.mkdir(parents=True, exist_ok=True)
        queries_path = out_dir / "queries.jsonl"
        a5_path = out_dir / "a5_pool.jsonl"
        clip_path = out_dir / "clip_pool.jsonl"
        _write_jsonl(queries_path, query_rows)
        _write_jsonl(a5_path, a5_rows)
        _write_jsonl(clip_path, clip_rows)

        eval_dir.mkdir(parents=True, exist_ok=True)
        labels_path = eval_dir / f"clip_vp_{split}_labels.jsonl"
        split_lock_path = eval_dir / f"clip_vp_{split}_split_lock.json"
        _write_jsonl(labels_path, label_rows)
        if split_lock_path.exists():
            if overwrite_eval_artifacts:
                split_lock_path.unlink()
            else:
                raise FileExistsError(f"Split lock already exists: {split_lock_path} (pass --overwrite-eval-artifacts to replace)")
        db_sha256 = sha256_file(db_path) if db_path.is_file() else None
        faiss_sha256 = sha256_file(faiss_path) if faiss_path.is_file() else None
        siglip2_files_sha256 = hash_paths([model_dir]) if model_dir.exists() else {}
        siglip2_weights_sha256 = _hash_directory_summary(model_dir, siglip2_files_sha256)
        gallery_fingerprint = compute_gallery_fingerprint(
            gallery_sha256=gallery_sha256,
            indexed_video_count=len(indexed_videos),
            db_sha256=db_sha256,
            faiss_sha256=faiss_sha256,
        )
        labels_sha256 = sha256_file(labels_path)
        freeze_split(
            split_lock_records,
            split_lock_path,
            metadata={
                "split": split,
                "query_count": len(query_rows),
                "gallery_sha256": gallery_sha256,
                "gallery_fingerprint": gallery_fingerprint,
                "labels_sha256": labels_sha256,
            },
        )
        if not verify_split_lock(split_lock_path):
            raise RuntimeError(f"Split lock verification failed: {split_lock_path}")
        split_lock_sha256 = sha256_file(split_lock_path)

        retrieval_config = {
            "ablation_config": "A5",
            "mode": "accurate",
            "ranking_mode": "diverse_video",
            "top_k": 10,
            "max_k": max_k,
            "branch_top_n": branch_top_n,
            "enable_spatial_patches": True,
            "enable_vlm_caption_lexical": True,
            "enable_vlm_caption_dense": True,
            "enable_temporal_rerank": bool(float(temporal_rerank_lambda) > 0.0),
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
        }
        git_commit = _get_git_commit()
        q_plan_cache_sha = sha256_file(query_plan_cache) if query_plan_cache else None
        run_fingerprint_payload = {
            "git_commit": git_commit,
            "split": split,
            "db_sha256": db_sha256,
            "faiss_sha256": faiss_sha256,
            "siglip2_weights_sha256": siglip2_weights_sha256,
            "ltr_model_sha256": gate.get("ltr_model_sha256"),
            "gallery_fingerprint": gallery_fingerprint,
            "split_lock_sha256": split_lock_sha256,
            "retrieval_config": retrieval_config,
            "seed": int(seed),
            "device": device,
            "query_plan_mode": query_plan_source,
            "query_plan_revision": query_plan_revision,
            "query_plan_cache_sha256": q_plan_cache_sha,
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(run_fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        receipt = {
            "schema_version": 1,
            "git_commit": git_commit,
            "split": split,
            "max_k": max_k,
            "query_count": len(query_rows),
            "indexed_video_count": len(indexed_videos),
            "gallery_sha256": gallery_sha256,
            "gallery_fingerprint": gallery_fingerprint,
            "db_path": str(db_path),
            "db_sha256": db_sha256,
            "faiss_path": str(faiss_path),
            "faiss_sha256": faiss_sha256,
            "model_dir": str(model_dir),
            "siglip2_weights_sha256": siglip2_weights_sha256,
            "siglip2_files_sha256": siglip2_files_sha256,
            "ltr_model_path": str(ltr_model_path) if ltr_model_path else None,
            "ltr_model_sha256": gate.get("ltr_model_sha256"),
            "eval_mode": gate.get("eval_mode"),
            "oof_applied_count": oof_applied,
            "oof_fallback_count": oof_fallback,
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
            "retrieval_config": retrieval_config,
            "seed": int(seed),
            "device": device,
            "query_plan_source": query_plan_source,
            "query_plan_mode": query_plan_source,
            "query_plan_revision": query_plan_revision,
            "query_plan_cache_sha256": q_plan_cache_sha,
            "query_decomposer_stats": query_decomposer.get_stats() if hasattr(query_decomposer, "get_stats") else None,
            "run_fingerprint": run_fingerprint,
            "elapsed_seconds": round(time.perf_counter() - t_start, 2),
            "artifact_sha256": hash_paths([queries_path, a5_path, clip_path]),
            "labels_path": str(labels_path),
            "labels_sha256": labels_sha256,
            "split_lock_path": str(split_lock_path),
            "split_lock_sha256": split_lock_sha256,
            "clip_branch_missing_feature_videos": len(clip_scorer.missing_feature_videos),
        }
        receipt_path = out_dir / "capture_receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt
    finally:
        pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture frozen A5 and clip-branch candidate pools.")
    parser.add_argument("--split", default="dev_full_859", choices=("dev_150", "dev_full_859", "heldout_test_201"))
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("work/experiments/clip_vp_pools_dev"))
    parser.add_argument("--eval-dir", type=Path, default=Path("work/evaluation-only"))
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_queries"))
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/siglip2"))
    parser.add_argument("--ltr-model-path", type=Path, default=Path("models/ranking/videntra_ltr_v1.json"))
    parser.add_argument("--no-ltr", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--temporal-rerank-lambda", type=float, default=0.0)
    parser.add_argument("--allow-insample-eval", action="store_true")
    parser.add_argument("--overwrite-eval-artifacts", action="store_true")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--query-plan-cache", type=Path)
    parser.add_argument("--query-plan-revision")
    args = parser.parse_args()

    receipt = capture_pools(
        split=args.split,
        max_k=args.max_k,
        out_dir=args.out_dir,
        eval_dir=args.eval_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        db_path=args.db_path,
        faiss_path=args.faiss_path,
        model_dir=args.model_dir,
        ltr_model_path=None if args.no_ltr else args.ltr_model_path,
        device=args.device,
        seed=args.seed,
        temporal_rerank_lambda=args.temporal_rerank_lambda,
        allow_insample_eval=args.allow_insample_eval,
        overwrite_eval_artifacts=args.overwrite_eval_artifacts,
        rules_only=args.rules_only,
        query_plan_cache=args.query_plan_cache,
        query_plan_revision=args.query_plan_revision,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
