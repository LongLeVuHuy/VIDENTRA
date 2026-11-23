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
    compute_code_sha256,
    compute_effective_config_sha256,
    evaluate_provenance_gate,
    get_cpu_thread_settings,
    hash_paths,
    load_ltr_model_info,
    resolve_a5_preset,
    resolve_canonical_source_group,
    resolve_oof_fold_for_query,
    resolve_query_source_groups,
    sha256_file,
)
from videntra.evaluation.split_lock import freeze_split, verify_split_lock
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


def resolve_query_source_group(query: dict[str, Any]) -> tuple[str, str]:
    """Resolve canonical source_group for a query from its target_videos."""
    _groups, group_key, status_key = resolve_query_source_groups(query)
    return group_key, status_key


def _snapshot_ltr_state(ltr_ranker: Any) -> dict[str, Any]:
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


def _restore_ltr_state(ltr_ranker: Any, state: dict[str, Any]) -> None:
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
        cached_idx = (
            pipeline._ensure_clip_branch_index()
            if hasattr(pipeline, "_ensure_clip_branch_index")
            else None
        )
        if cached_idx is not None:
            self.index = cached_idx
            self.missing_feature_videos = sorted(cached_idx.invalid_videos.keys())
            return

        emb = getattr(pipeline, "all_embeddings", None)
        vid_fids = getattr(pipeline, "video_kf_fids", None) or {}
        vid_kfs = getattr(pipeline, "video_keyframes", None) or {}
        kf_map = getattr(pipeline, "keyframe_map", None) or {}
        if emb is None or len(emb) == 0 or (not vid_fids and not vid_kfs and not kf_map):
            self.index = None
            self.missing_feature_videos = []
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
        res = getattr(self.pipeline, "_last_clip_score_res", None)
        if res is None or not getattr(res, "branch_scores", None):
            cached_q = getattr(self.pipeline, "_last_primary_q_768", None)
            if cached_q is not None:
                q_vec = np.asarray(cached_q, dtype=np.float32).reshape(-1)
            else:
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


def _verify_query_plan_cache_and_receipt(
    *,
    query_plan_cache: Path | None,
    query_plan_revision: str | None,
    query_plan_prompt_version: str,
    query_plan_receipt: Path | None,
    allow_dry_run_cache: bool,
) -> dict[str, Any]:
    """Verify frozen query plan cache SHA-256 and optional receipt before capture."""
    if query_plan_receipt is not None and query_plan_cache is None:
        raise ValueError("--query-plan-receipt requires --query-plan-cache")
    if query_plan_cache is None:
        return {
            "cache_sha256": None,
            "receipt_sha256": None,
            "dry_run": None,
            "adapter_revision": None,
            "validator_commit": None,
        }
    if not query_plan_cache.is_file():
        raise FileNotFoundError(f"Frozen query plan cache not found: {query_plan_cache}")
    cache_sha256 = sha256_file(query_plan_cache)

    receipt_sha256: str | None = None
    dry_run_flag: bool | None = None
    adapter_revision: str | None = None
    validator_commit: str | None = None
    if query_plan_receipt is not None:
        if not query_plan_receipt.is_file():
            raise FileNotFoundError(f"Query plan receipt not found: {query_plan_receipt}")
        receipt_sha256 = sha256_file(query_plan_receipt)
        receipt_data = json.loads(query_plan_receipt.read_text(encoding="utf-8"))
        if not isinstance(receipt_data, dict):
            raise ValueError(f"Invalid query plan receipt JSON object: {query_plan_receipt}")

        dry_run_val = receipt_data.get("dry_run")
        if dry_run_val is not None:
            dry_run_flag = bool(dry_run_val)
            if dry_run_flag and not allow_dry_run_cache:
                raise ValueError(
                    "Refusing dry_run=true query plan receipt for benchmark capture "
                    "(pass --allow-dry-run-cache for smoke-only runs)"
                )

        rec_pv = receipt_data.get("prompt_version")
        if rec_pv is not None and str(rec_pv) != str(query_plan_prompt_version):
            raise ValueError(
                f"Query plan receipt prompt_version mismatch: expected {query_plan_prompt_version!r}, got {rec_pv!r}"
            )

        rec_rev = receipt_data.get("model_revision")
        if rec_rev is not None and str(rec_rev) != str(query_plan_revision):
            raise ValueError(
                f"Query plan receipt model_revision mismatch: expected {query_plan_revision!r}, got {rec_rev!r}"
            )

        cache_export = receipt_data.get("cache_export")
        expected_cache_sha = receipt_data.get("cache_sha256")
        if expected_cache_sha is None and isinstance(cache_export, dict):
            expected_cache_sha = cache_export.get("cache_sha256")
        if not expected_cache_sha:
            raise ValueError(f"Query plan receipt {query_plan_receipt} is missing cache_sha256")
        if str(expected_cache_sha) != str(cache_sha256):
            raise ValueError(
                f"Query plan cache SHA-256 mismatch: expected {expected_cache_sha}, got {cache_sha256}"
            )

        if receipt_data.get("adapter_revision") is not None:
            adapter_revision = str(receipt_data["adapter_revision"])
        elif receipt_data.get("lora_config") is not None and rec_rev is not None:
            adapter_revision = str(rec_rev)

        if receipt_data.get("validator_commit") is not None:
            validator_commit = str(receipt_data["validator_commit"])

    return {
        "cache_sha256": cache_sha256,
        "receipt_sha256": receipt_sha256,
        "dry_run": dry_run_flag,
        "adapter_revision": adapter_revision,
        "validator_commit": validator_commit,
    }


def capture_pools(
    *,
    split: str = "dev_full_859",
    max_k: int = 20,
    out_dir: Path = Path("work/experiments/clip_vp_pools_dev"),
    eval_dir: Path = Path("work/evaluation-only"),
    val_dir: Path = Path("validation"),
    test_dir: Path = Path("test_queries"),
    db_path: Path = Path("data/processed/catalog/corpus_metadata.db"),
    faiss_path: Path = Path("data/processed/visual_keyframes/siglip2_keyframes_768d.faiss"),
    model_dir: Path = Path("models/siglip2"),
    ltr_model_path: Path | None = Path("models/ranking/videntra_ltr.json"),
    device: str = "cpu",
    seed: int = 20261115,
    temporal_rerank_lambda: float = 0.0,
    allow_insample_eval: bool = False,
    overwrite_eval_artifacts: bool = False,
    branch_top_n: int = 60,
    rules_only: bool = False,
    query_plan_cache: Path | None = None,
    query_plan_revision: str | None = None,
    query_plan_prompt_version: str = "compact-v1",
    query_plan_receipt: Path | None = None,
    allow_dry_run_cache: bool = False,
    enable_clip_consensus_boost: bool | None = None,
    ltr_blend_weight: float | None = None,
    a5_preset: str | None = None,
    stage2_selection_mode: str = "exact",
    video_proposal_mode: str = "frame_topk",
    video_proposal_top_n: int = 60,
    video_proposal_anchors_per_video: int = 2,
    video_proposal_min_pts_gap: float = 1.0,
    video_proposal_clip_reserve_ratio: float = 0.0,
    disable_pure_visual_lexical_bonus: bool = False,
) -> dict[str, Any]:
    if max_k < 2:
        raise ValueError(f"max_k must be >= 2, got {max_k}")
    if (query_plan_cache is None) != (query_plan_revision is None):
        raise ValueError("--query-plan-cache and --query-plan-revision must be used together")
    if rules_only and query_plan_cache is not None:
        raise ValueError("--rules-only and --query-plan-cache are mutually exclusive")

    resolved_consensus_boost, resolved_ltr_weight = resolve_a5_preset(
        a5_preset,
        enable_clip_consensus_boost=enable_clip_consensus_boost,
        ltr_blend_weight=ltr_blend_weight,
    )

    q_plan_meta = _verify_query_plan_cache_and_receipt(
        query_plan_cache=query_plan_cache,
        query_plan_revision=query_plan_revision,
        query_plan_prompt_version=query_plan_prompt_version,
        query_plan_receipt=query_plan_receipt,
        allow_dry_run_cache=allow_dry_run_cache,
    )

    if query_plan_cache is not None and query_plan_revision is not None:
        query_plan_source = "frozen_llm_cache"
    elif rules_only:
        query_plan_source = "rules_only"
    else:
        query_plan_source = "pipeline_default"
    ltr_info = load_ltr_model_info(ltr_model_path)
    pipeline = UnifiedMultimodalSearchPipeline(
        db_path=db_path,
        faiss_path=faiss_path,
        model_dir=model_dir,
        device=device,
        use_disk_cache=False,
        memory_cache_size=0,
        ltr_model_path=ltr_model_path,
        ltr_blend_weight=resolved_ltr_weight,
        enable_spatial_patches=True,
        enable_vlm_caption_lexical=True,
        enable_vlm_caption_dense=True,
        enable_temporal_rerank=bool(float(temporal_rerank_lambda) > 0.0),
        temporal_rerank_lambda=float(temporal_rerank_lambda),
        enable_clip_consensus_boost=resolved_consensus_boost,
        rules_only=rules_only,
        query_plan_cache=query_plan_cache,
        query_plan_revision=query_plan_revision,
        query_plan_prompt_version=query_plan_prompt_version,
        stage2_selection_mode=stage2_selection_mode,
        video_proposal_mode=video_proposal_mode,
        video_proposal_top_n=video_proposal_top_n,
        video_proposal_anchors_per_video=video_proposal_anchors_per_video,
        video_proposal_min_pts_gap=video_proposal_min_pts_gap,
        video_proposal_clip_reserve_ratio=video_proposal_clip_reserve_ratio,
        disable_pure_visual_lexical_bonus=disable_pure_visual_lexical_bonus,
    )
    try:
        indexed_videos = set(pipeline.video_durations.keys()) if pipeline.video_durations else set()
        if not indexed_videos and pipeline.keyframe_map:
            indexed_videos = {k["video_id"] for k in pipeline.keyframe_map.values()}
        gallery_sha256 = compute_gallery_sha256(indexed_videos)

        queries = _load_split_queries(split, val_dir, test_dir, indexed_videos)
        if not queries:
            raise ValueError(f"No queries loaded for split={split}")

        q_groups_map: dict[str, list[str]] = {}
        q_group_meta: dict[str, tuple[str, str]] = {}
        for q in queries:
            qid_str = str(q["query_id"])
            groups_list, group_key, status_key = resolve_query_source_groups(q)
            q_groups_map[qid_str] = groups_list
            q_group_meta[qid_str] = (group_key, status_key)

        # Verify provenance gate (OOF LTR by query_id/source_group for dev splits, zero leakage for heldout)
        split_query_ids = {split: [str(q["query_id"]) for q in queries]}
        used_oof = set()
        trained_set = {str(q) for q in ltr_info.get("trained_query_ids", [])}
        trained_groups = {str(g) for g in ltr_info.get("trained_group_ids", [])}
        if ltr_info.get("has_fold_weights"):
            relevant_qids = [
                str(q["query_id"])
                for q in queries
                if (not trained_set and not trained_groups)
                or str(q["query_id"]) in trained_set
                or bool(set(q_groups_map[str(q["query_id"])]) & trained_groups)
            ]
            if relevant_qids and all(
                resolve_oof_fold_for_query(qid_item, q_groups_map[qid_item], ltr_info)[0] is not None
                for qid_item in relevant_qids
            ):
                used_oof.add(split)
        gate = evaluate_provenance_gate(
            ltr_info,
            split_query_ids,
            used_oof_splits=used_oof,
            allow_insample_eval=allow_insample_eval,
            split_queries={split: queries},
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
        consensus_applied_count = 0
        consensus_top1_changed_count = 0
        ltr_applied_count = 0
        ltr_top1_flipped_count = 0
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
                fold_payload, _oof_status = resolve_oof_fold_for_query(
                    qid, q_groups_map.get(qid, ()), ltr_info
                )
                if fold_payload is not None:
                    _apply_ltr_fold_payload(ltr_ranker, fold_payload)
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
            cons_st = diag.get("clip_consensus_stats") or {}
            ltr_st = diag.get("ltr_stats") or {}
            if cons_st.get("applied"):
                consensus_applied_count += 1
            if cons_st.get("top1_changed"):
                consensus_top1_changed_count += 1
            if ltr_st.get("applied"):
                ltr_applied_count += 1
            if ltr_st.get("top1_flipped"):
                ltr_top1_flipped_count += 1

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
                "modality_intent": diag.get("modality_intent"),
                "stage2_retrieval_path": diag.get("stage2_retrieval_path"),
                "video_proposal_stats": diag.get("video_proposal_stats"),
                "stage_traces": diag.get("stage_traces"),
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

            source_group, source_status = q_group_meta[qid]
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
            "a5_preset": a5_preset,
            "stage2_selection_mode": str(stage2_selection_mode),
            "video_proposal_mode": str(video_proposal_mode),
            "video_proposal_top_n": int(video_proposal_top_n),
            "video_proposal_anchors_per_video": int(video_proposal_anchors_per_video),
            "video_proposal_min_pts_gap": float(video_proposal_min_pts_gap),
            "video_proposal_clip_reserve_ratio": float(video_proposal_clip_reserve_ratio),
            "disable_pure_visual_lexical_bonus": bool(disable_pure_visual_lexical_bonus),
            "enable_spatial_patches": True,
            "enable_vlm_caption_lexical": True,
            "enable_vlm_caption_dense": True,
            "enable_temporal_rerank": bool(float(temporal_rerank_lambda) > 0.0),
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
            "enable_clip_consensus_boost": resolved_consensus_boost,
            "ltr_enabled": bool(ltr_model_path is not None and resolved_ltr_weight > 0.0),
            "ltr_blend_weight": float(resolved_ltr_weight),
        }
        effective_config_sha256 = compute_effective_config_sha256(retrieval_config)
        code_sha256 = compute_code_sha256([Path(__file__)])
        cpu_thread_settings = get_cpu_thread_settings()
        valid_branch_vids = len(clip_scorer.index.video_ids) if clip_scorer.index is not None else 0
        gallery_coverage_rate = round(valid_branch_vids / max(1, len(indexed_videos)), 6)

        git_commit = _get_git_commit()
        q_plan_cache_sha = q_plan_meta["cache_sha256"]
        effective_prompt_version = (
            query_plan_prompt_version
            if query_plan_cache is not None
            else getattr(getattr(pipeline, "query_decomposer", None), "prompt_version", None)
        )
        run_fingerprint_payload = {
            "git_commit": git_commit,
            "code_sha256": code_sha256,
            "split": split,
            "db_sha256": db_sha256,
            "faiss_sha256": faiss_sha256,
            "siglip2_weights_sha256": siglip2_weights_sha256,
            "ltr_model_sha256": gate.get("ltr_model_sha256"),
            "gallery_fingerprint": gallery_fingerprint,
            "split_lock_sha256": split_lock_sha256,
            "retrieval_config": retrieval_config,
            "effective_config_sha256": effective_config_sha256,
            "seed": int(seed),
            "device": device,
            "query_plan_mode": query_plan_source,
            "query_plan_revision": query_plan_revision,
            "query_plan_prompt_version": effective_prompt_version,
            "query_plan_cache_sha256": q_plan_cache_sha,
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(run_fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        decomposer_obj = getattr(pipeline, "query_decomposer", None)
        receipt = {
            "schema_version": 2,
            "git_commit": git_commit,
            "code_sha256": code_sha256,
            "split": split,
            "max_k": max_k,
            "query_count": len(query_rows),
            "indexed_video_count": len(indexed_videos),
            "gallery_coverage_rate": gallery_coverage_rate,
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
            "a5_preset": a5_preset,
            "enable_clip_consensus_boost": resolved_consensus_boost,
            "ltr_blend_weight": float(resolved_ltr_weight),
            "consensus_stats": {
                "applied_count": consensus_applied_count,
                "top1_changed_count": consensus_top1_changed_count,
            },
            "ltr_stats": {
                "applied_count": ltr_applied_count,
                "top1_flipped_count": ltr_top1_flipped_count,
            },
            "temporal_rerank_lambda": float(temporal_rerank_lambda),
            "retrieval_config": retrieval_config,
            "effective_config_sha256": effective_config_sha256,
            "cpu_thread_settings": cpu_thread_settings,
            "seed": int(seed),
            "device": device,
            "query_plan_source": query_plan_source,
            "query_plan_mode": query_plan_source,
            "query_plan_revision": query_plan_revision,
            "query_plan_model_revision": query_plan_revision,
            "query_plan_adapter_revision": q_plan_meta["adapter_revision"],
            "query_plan_prompt_version": effective_prompt_version,
            "query_plan_cache_sha256": q_plan_cache_sha,
            "query_plan_receipt_path": str(query_plan_receipt) if query_plan_receipt else None,
            "query_plan_receipt_sha256": q_plan_meta["receipt_sha256"],
            "query_plan_dry_run": q_plan_meta["dry_run"],
            "validator_commit": git_commit,
            "query_plan_validator_commit": q_plan_meta["validator_commit"],
            "query_decomposer_stats": decomposer_obj.get_stats() if hasattr(decomposer_obj, "get_stats") else None,
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
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/catalog/corpus_metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("data/processed/visual_keyframes/siglip2_keyframes_768d.faiss"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/siglip2"))
    parser.add_argument("--ltr-model-path", type=Path, default=Path("models/ranking/videntra_ltr.json"))
    parser.add_argument("--no-ltr", action="store_true")
    parser.add_argument("--ltr-blend-weight", type=float, default=None)
    parser.add_argument("--a5-preset", choices=("a5_full", "a5_lite", "a5_no_consensus", "a5_no_ltr"), default=None)
    parser.add_argument("--enable-clip-consensus-boost", dest="enable_clip_consensus_boost", action="store_true")
    parser.add_argument("--disable-clip-consensus-boost", dest="enable_clip_consensus_boost", action="store_false")
    parser.set_defaults(enable_clip_consensus_boost=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--temporal-rerank-lambda", type=float, default=0.0)
    parser.add_argument("--allow-insample-eval", action="store_true")
    parser.add_argument("--overwrite-eval-artifacts", action="store_true")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--query-plan-cache", type=Path)
    parser.add_argument("--query-plan-revision")
    parser.add_argument("--query-plan-prompt-version", default="compact-v1")
    parser.add_argument("--query-plan-receipt", type=Path)
    parser.add_argument("--allow-dry-run-cache", action="store_true")
    parser.add_argument("--stage2-selection-mode", choices=("exact", "legacy_union"), default="exact")
    parser.add_argument("--video-proposal-mode", choices=("frame_topk", "video_max", "video_mean_top2_pts", "video_rrf_max_top2"), default="frame_topk")
    parser.add_argument("--video-proposal-top-n", type=int, default=60)
    parser.add_argument("--video-proposal-anchors-per-video", type=int, default=2)
    parser.add_argument("--video-proposal-min-pts-gap", type=float, default=1.0)
    parser.add_argument("--video-proposal-clip-reserve-ratio", type=float, default=0.0)
    parser.add_argument("--disable-pure-visual-lexical-bonus", action="store_true")
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
        query_plan_prompt_version=args.query_plan_prompt_version,
        query_plan_receipt=args.query_plan_receipt,
        allow_dry_run_cache=args.allow_dry_run_cache,
        enable_clip_consensus_boost=args.enable_clip_consensus_boost,
        ltr_blend_weight=args.ltr_blend_weight,
        a5_preset=args.a5_preset,
        stage2_selection_mode=args.stage2_selection_mode,
        video_proposal_mode=args.video_proposal_mode,
        video_proposal_top_n=args.video_proposal_top_n,
        video_proposal_anchors_per_video=args.video_proposal_anchors_per_video,
        video_proposal_min_pts_gap=args.video_proposal_min_pts_gap,
        video_proposal_clip_reserve_ratio=args.video_proposal_clip_reserve_ratio,
        disable_pure_visual_lexical_bonus=args.disable_pure_visual_lexical_bonus,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
