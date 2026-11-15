r"""Paired evaluation and source-group bootstrap comparison for Clip/VideoPrism replay runs.

Usage:
    .\.venv\Scripts\python.exe -B scripts/compare_msr_runs.py `
        --baseline work/experiments/clip_vp_k20_baseline `
        --challenger work/experiments/clip_vp_k20_vp_a5 `
        --labels work/evaluation-only/clip_vp_dev_full_859_labels.jsonl `
        --split-lock work/evaluation-only/clip_vp_dev_full_859_split_lock.json `
        --out-dir work/experiments/clip_vp_k20_compare_vp
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import verify_split_lock
from videntra.retrieval.visual import temporal_iou

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _read_jsonl_by_id(path: Path, label: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                raise ValueError(f"Missing query_id in {label} line {line_no}")
            if qid in out:
                raise ValueError(f"Duplicate query_id '{qid}' in {label}")
            out[qid] = row
    return out


def _first_target_rank(ranked_ids: Sequence[str], target_set: set[str]) -> int:
    for rank, vid in enumerate(ranked_ids, 1):
        if vid in target_set:
            return rank
    return 0


def _compute_side_metrics(ranked_ids: Sequence[str], target_set: set[str], k: int) -> dict[str, Any]:
    pool = list(ranked_ids[:k])
    h1 = 1 if any(v in target_set for v in pool[:1]) else 0
    h5 = 1 if any(v in target_set for v in pool[:5]) else 0
    h10 = 1 if any(v in target_set for v in pool[:10]) else 0
    in_pool = 1 if any(v in target_set for v in pool) else 0
    rr10 = 0.0
    for rank, vid in enumerate(pool[:10], 1):
        if vid in target_set:
            rr10 = 1.0 / rank
            break
    return {
        "hit1": h1,
        "hit5": h5,
        "hit10": h10,
        "mrr10": rr10,
        "in_pool_hit": in_pool,
        "target_rank": _first_target_rank(pool, target_set),
    }


def paired_source_group_bootstrap_ci(
    paired_rows: Sequence[dict[str, Any]],
    *,
    n_resamples: int = 10000,
    seed: int = 20261115,
) -> dict[str, Any]:
    """Compute paired source-group bootstrap 95% CI for Hit@1 delta_pp."""
    if not paired_rows:
        return {
            "n_queries": 0,
            "n_source_groups": 0,
            "n_resamples": n_resamples,
            "seed": seed,
            "delta_pp": 0.0,
            "ci95_pp": [0.0, 0.0],
        }

    group_diffs_map: dict[str, int] = defaultdict(int)
    group_counts_map: dict[str, int] = defaultdict(int)
    for row in paired_rows:
        grp = str(row["source_group"])
        group_diffs_map[grp] += int(row["new_hit"]) - int(row["base_hit"])
        group_counts_map[grp] += 1

    groups = sorted(group_counts_map.keys())
    g_diffs = np.asarray([group_diffs_map[g] for g in groups], dtype=np.float64)
    g_counts = np.asarray([group_counts_map[g] for g in groups], dtype=np.float64)
    n_groups = len(groups)
    n_queries = int(g_counts.sum())
    point_delta_pp = float(100.0 * g_diffs.sum() / max(1, n_queries))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    boot_diffs = g_diffs[idx].sum(axis=1)
    boot_counts = g_counts[idx].sum(axis=1)
    boot_deltas = 100.0 * boot_diffs / np.maximum(1.0, boot_counts)
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

    return {
        "n_queries": n_queries,
        "n_source_groups": n_groups,
        "n_resamples": n_resamples,
        "seed": seed,
        "delta_pp": round(point_delta_pp, 4),
        "ci95_pp": [round(float(ci_low), 4), round(float(ci_high), 4)],
    }


def _summarize_slice(
    rows: Sequence[dict[str, Any]],
    *,
    n_resamples: int = 10000,
    seed: int = 20261115,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_queries": 0,
            "n_source_groups": 0,
            "baseline": {},
            "challenger": {},
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "net_gain": 0,
            "delta_pp": 0.0,
            "ci95_pp": [0.0, 0.0],
            "candidate_gains": 0,
            "candidate_losses": 0,
            "recall_net_gain": 0,
            "recall_delta_pp": 0.0,
        }

    def _agg(prefix: str) -> dict[str, Any]:
        h1 = sum(r[f"{prefix}_hit1"] for r in rows)
        h5 = sum(r[f"{prefix}_hit5"] for r in rows)
        h10 = sum(r[f"{prefix}_hit10"] for r in rows)
        rec = sum(r[f"{prefix}_in_pool_hit"] for r in rows)
        mrr = sum(r[f"{prefix}_mrr10"] for r in rows)
        return {
            "hit_at_1": round(h1 / n, 6),
            "hit_at_5": round(h5 / n, 6),
            "hit_at_10": round(h10 / n, 6),
            "mrr_at_10": round(mrr / n, 6),
            "recall_at_k": round(rec / n, 6),
            "hit_at_1_count": int(h1),
            "hit_at_5_count": int(h5),
            "hit_at_10_count": int(h10),
            "recall_at_k_count": int(rec),
        }

    wins = sum(1 for r in rows if r["base_hit"] == 0 and r["new_hit"] == 1)
    losses = sum(1 for r in rows if r["base_hit"] == 1 and r["new_hit"] == 0)
    ties = n - wins - losses
    cand_gains = sum(1 for r in rows if r["base_in_pool_hit"] == 0 and r["in_pool_hit"] == 1)
    cand_losses = sum(1 for r in rows if r["base_in_pool_hit"] == 1 and r["in_pool_hit"] == 0)

    boot = paired_source_group_bootstrap_ci(rows, n_resamples=n_resamples, seed=seed)
    return {
        "n_queries": n,
        "n_source_groups": boot["n_source_groups"],
        "baseline": _agg("base"),
        "challenger": _agg("new"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_gain": wins - losses,
        "delta_pp": boot["delta_pp"],
        "ci95_pp": boot["ci95_pp"],
        "candidate_gains": cand_gains,
        "candidate_losses": cand_losses,
        "recall_net_gain": cand_gains - cand_losses,
        "recall_delta_pp": round(100.0 * (cand_gains - cand_losses) / n, 4),
    }


def _check_moment_hit_iou05(top1_vid: str | None, anchor_map: dict[str, Any] | None, target_vids: set[str], target_wins: Sequence[Any]) -> int | None:
    if not top1_vid or not anchor_map or top1_vid not in anchor_map or not anchor_map[top1_vid]:
        return None
    if top1_vid not in target_vids:
        return 0
    anchor = anchor_map[top1_vid]
    s = float(anchor.get("start_sec", 0.0))
    e = float(anchor.get("end_sec", 0.0))
    for w in target_wins:
        if temporal_iou(s, e, float(w[0]), float(w[1])) >= 0.5:
            return 1
    return 0


def _verify_replay_output_hash(run_dir: Path, receipt: dict[str, Any], label: str) -> None:
    """Verify per_query_evaluation.jsonl SHA-256 against receipt.output_sha256 if recorded."""
    out_hashes = receipt.get("output_sha256")
    if not isinstance(out_hashes, dict) or not out_hashes:
        return
    eval_path = run_dir / "per_query_evaluation.jsonl"
    actual_sha = sha256_file(eval_path)
    expected_sha: str | None = None
    for key, digest in out_hashes.items():
        if Path(key).name == "per_query_evaluation.jsonl":
            expected_sha = str(digest) if digest is not None else None
            break
    if expected_sha is None and len(out_hashes) == 1:
        val = next(iter(out_hashes.values()))
        expected_sha = str(val) if val is not None else None
    if expected_sha is None or actual_sha != expected_sha:
        raise ValueError(
            f"{label} output_sha256 mismatch for per_query_evaluation.jsonl: "
            f"expected={expected_sha}, got={actual_sha}"
        )


def _load_filter_manifest_qids(manifest_path: Path, filter_split: str | None = None) -> set[str]:
    """Load allowed query_ids from a tune/confirm filter manifest (JSON, JSONL, or text)."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing filter manifest: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty filter manifest: {manifest_path}")

    qids: set[str] = set()
    if manifest_path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if filter_split is not None:
                row_split = row.get("split") or row.get("subset") or row.get("partition")
                if row_split is not None and str(row_split) != filter_split:
                    continue
            qid = str(row.get("query_id") or row.get("qid") or "").strip()
            if qid:
                qids.add(qid)
    elif manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            qids = {str(x).strip() for x in payload if str(x).strip()}
        elif isinstance(payload, dict):
            if filter_split is not None:
                for key in (f"{filter_split}_query_ids", filter_split, f"{filter_split}_ids"):
                    if isinstance(payload.get(key), list):
                        qids = {str(x).strip() for x in payload[key] if str(x).strip()}
                        break
            if not qids and isinstance(payload.get("query_ids"), list):
                qids = {str(x).strip() for x in payload["query_ids"] if str(x).strip()}
            elif not qids and isinstance(payload.get("records"), list):
                for row in payload["records"]:
                    if not isinstance(row, dict):
                        continue
                    if filter_split is not None:
                        row_split = row.get("split") or row.get("subset") or row.get("partition")
                        if row_split is not None and str(row_split) != filter_split:
                            continue
                    qid = str(row.get("query_id") or row.get("qid") or "").strip()
                    if qid:
                        qids.add(qid)
        else:
            raise ValueError(f"Unsupported JSON manifest root in {manifest_path}")
    else:
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                qids.add(s)

    if not qids:
        raise ValueError(f"No query_ids resolved from filter manifest {manifest_path} (split={filter_split})")
    return qids


def compare_runs(
    *,
    baseline_dir: Path,
    challenger_dir: Path,
    labels_path: Path,
    split_lock_path: Path,
    out_dir: Path,
    n_resamples: int = 10000,
    seed: int = 20261115,
    require_same_base_pool: bool | None = None,
    filter_manifest: Path | None = None,
    filter_split: str | None = None,
) -> dict[str, Any]:
    if not verify_split_lock(split_lock_path):
        raise ValueError(f"Invalid or tampered split_lock: {split_lock_path}")

    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    lock_by_qid = {str(r["query_id"]): r for r in split_lock["records"]}

    base_rows = _read_jsonl_by_id(baseline_dir / "per_query_evaluation.jsonl", "baseline/per_query_evaluation.jsonl")
    chal_rows = _read_jsonl_by_id(challenger_dir / "per_query_evaluation.jsonl", "challenger/per_query_evaluation.jsonl")
    label_rows = _read_jsonl_by_id(labels_path, "labels.jsonl")

    base_receipt = json.loads((baseline_dir / "run_receipt.json").read_text(encoding="utf-8"))
    chal_receipt = json.loads((challenger_dir / "run_receipt.json").read_text(encoding="utf-8"))

    # Parity & integrity checks
    if set(base_rows.keys()) != set(chal_rows.keys()):
        raise ValueError("query_id set mismatch between baseline and challenger")
    if set(base_rows.keys()) != set(label_rows.keys()):
        raise ValueError("query_id set mismatch between runs and labels")
    if not set(base_rows.keys()).issubset(set(lock_by_qid.keys())):
        missing_lock = sorted(set(base_rows.keys()) - set(lock_by_qid.keys()))
        raise ValueError(f"query_ids missing from split_lock: {missing_lock[:5]}")

    if filter_manifest is not None:
        allowed_qids = _load_filter_manifest_qids(filter_manifest, filter_split=filter_split)
        unknown_qids = sorted(allowed_qids - set(base_rows.keys()))
        if unknown_qids:
            raise ValueError(f"filter_manifest contains query_ids absent from runs: {unknown_qids[:5]}")
        base_rows = {qid: base_rows[qid] for qid in allowed_qids}
        chal_rows = {qid: chal_rows[qid] for qid in allowed_qids}
        label_rows = {qid: label_rows[qid] for qid in allowed_qids}

    if base_receipt.get("k") != chal_receipt.get("k"):
        raise ValueError(f"Config k mismatch: baseline={base_receipt.get('k')}, challenger={chal_receipt.get('k')}")

    base_fp = base_receipt.get("gallery_fingerprint") or base_receipt.get("gallery_sha256")
    chal_fp = chal_receipt.get("gallery_fingerprint") or chal_receipt.get("gallery_sha256")
    if not base_fp or not chal_fp:
        raise ValueError("Missing gallery_fingerprint / gallery_sha256 in baseline or challenger run_receipt.json")
    if base_fp != chal_fp:
        raise ValueError(f"gallery_fingerprint mismatch between baseline ({base_fp}) and challenger ({chal_fp})")
    if (
        base_receipt.get("gallery_sha256") is not None
        and chal_receipt.get("gallery_sha256") is not None
        and base_receipt.get("gallery_sha256") != chal_receipt.get("gallery_sha256")
    ):
        raise ValueError("gallery_sha256 mismatch between baseline and challenger")
    if (
        base_receipt.get("queries_sha256") is not None
        and chal_receipt.get("queries_sha256") is not None
        and base_receipt.get("queries_sha256") != chal_receipt.get("queries_sha256")
    ):
        raise ValueError("queries_sha256 mismatch between baseline and challenger")

    if require_same_base_pool is None:
        check_same_base_pool = (
            base_receipt.get("pool") == chal_receipt.get("pool")
            and base_receipt.get("query_plan_source") == chal_receipt.get("query_plan_source")
            and base_receipt.get("query_plan_revision") == chal_receipt.get("query_plan_revision")
            and base_receipt.get("query_plan_cache_sha256") == chal_receipt.get("query_plan_cache_sha256")
        )
    else:
        check_same_base_pool = bool(require_same_base_pool)

    k = int(base_receipt["k"])
    paired_records: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []

    mom_base_hits: list[int] = []
    mom_new_hits: list[int] = []
    mom_labeled_n = 0

    for qid in sorted(base_rows.keys()):
        b_row = base_rows[qid]
        c_row = chal_rows[qid]
        l_row = label_rows[qid]
        lock_row = lock_by_qid[qid]

        b_sha = str(b_row.get("query_text_sha256", ""))
        c_sha = str(c_row.get("query_text_sha256", ""))
        l_sha = str(l_row.get("query_text_sha256", ""))
        if not (b_sha and b_sha == c_sha == l_sha):
            raise ValueError(f"query_text_sha256 mismatch for {qid}: base={b_sha}, chal={c_sha}, label={l_sha}")

        if check_same_base_pool:
            b_base_pool = [str(v) for v in b_row.get("base_video_ids", [])]
            c_base_pool = [str(v) for v in c_row.get("base_video_ids", [])]
            if b_base_pool != c_base_pool:
                raise ValueError(f"base_video_ids mismatch for {qid} in same-pool comparison")

        target_videos = [str(v) for v in (l_row.get("target_videos") or []) if str(v).strip()]
        if not target_videos:
            raise ValueError(f"Empty target_videos for {qid}")
        target_set = set(target_videos)
        target_windows = l_row.get("target_windows") or []
        dataset = str(l_row.get("dataset") or b_row.get("dataset") or "unknown")
        source_group = str(lock_row["source_group"])

        b_ranked = [str(v) for v in b_row.get("ranked_video_ids", [])]
        c_ranked = [str(v) for v in c_row.get("ranked_video_ids", [])]
        b_metrics = _compute_side_metrics(b_ranked, target_set, k=k)
        c_metrics = _compute_side_metrics(c_ranked, target_set, k=k)

        if target_windows:
            mom_labeled_n += 1
            mb = _check_moment_hit_iou05(b_ranked[0] if b_ranked else None, b_row.get("anchor_by_video_id"), target_set, target_windows)
            mc = _check_moment_hit_iou05(c_ranked[0] if c_ranked else None, c_row.get("anchor_by_video_id"), target_set, target_windows)
            if mb is not None and mc is not None:
                mom_base_hits.append(mb)
                mom_new_hits.append(mc)

        fallback_reason = c_row.get("fallback_reason") or b_row.get("fallback_reason")
        vp_cov_ok = bool(c_row.get("vp_coverage_ok", False)) and fallback_reason is None

        top1_changed = bool(b_ranked[:1] != c_ranked[:1])
        top1_eligible = bool(c_row.get("top1_change_eligible", False))

        internal = {
            "query_id": qid,
            "dataset": dataset,
            "source_group": source_group,
            "base_top1": b_ranked[0] if b_ranked else None,
            "new_top1": c_ranked[0] if c_ranked else None,
            "top1_change_eligible": top1_eligible,
            "top1_changed": top1_changed,
            "target_rank_before": b_metrics["target_rank"],
            "target_rank_after": c_metrics["target_rank"],
            "base_hit": b_metrics["hit1"],
            "new_hit": c_metrics["hit1"],
            "base_hit1": b_metrics["hit1"],
            "base_hit5": b_metrics["hit5"],
            "base_hit10": b_metrics["hit10"],
            "base_mrr10": b_metrics["mrr10"],
            "base_in_pool_hit": b_metrics["in_pool_hit"],
            "new_hit1": c_metrics["hit1"],
            "new_hit5": c_metrics["hit5"],
            "new_hit10": c_metrics["hit10"],
            "new_mrr10": c_metrics["mrr10"],
            "in_pool_hit": c_metrics["in_pool_hit"],
            "new_in_pool_hit": c_metrics["in_pool_hit"],
            "fallback_reason": fallback_reason,
            "vp_coverage_ok": vp_cov_ok,
            "vp_partial_applied": bool(c_row.get("vp_partial_applied", False)),
            "query_plan_mode": c_row.get("query_plan_mode"),
            "query_plan_cache_hit": c_row.get("query_plan_cache_hit"),
        }
        internal_rows.append(internal)
        paired_records.append({
            "query_id": qid,
            "dataset": dataset,
            "source_group": source_group,
            "base_top1": internal["base_top1"],
            "new_top1": internal["new_top1"],
            "target_rank_before": internal["target_rank_before"],
            "target_rank_after": internal["target_rank_after"],
            "base_hit": internal["base_hit"],
            "new_hit": internal["new_hit"],
            "base_in_pool_hit": internal["base_in_pool_hit"],
            "in_pool_hit": internal["in_pool_hit"],
            "fallback_reason": fallback_reason,
            "vp_coverage_ok": vp_cov_ok,
            "vp_partial_applied": internal["vp_partial_applied"],
            "query_plan_mode": internal["query_plan_mode"],
            "query_plan_cache_hit": internal["query_plan_cache_hit"],
        })

    _verify_replay_output_hash(baseline_dir, base_receipt, "baseline")
    _verify_replay_output_hash(challenger_dir, chal_receipt, "challenger")

    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in internal_rows:
        by_ds[r["dataset"]].append(r)

    cov_ok_rows = [r for r in internal_rows if r["vp_coverage_ok"]]
    cov_partial_rows = [r for r in internal_rows if r["vp_partial_applied"]]
    cov_fb_rows = [r for r in internal_rows if not r["vp_coverage_ok"] and not r["vp_partial_applied"]]
    cached_llm_rows = [
        r for r in internal_rows
        if r["query_plan_mode"] == "llm_compact" and r["query_plan_cache_hit"]
    ]

    moment_guardrail: dict[str, Any]
    if mom_labeled_n == 0:
        moment_guardrail = {"moment_labeled_n": 0, "status": "NO_MOMENT_LABELS"}
    elif len(mom_base_hits) == mom_labeled_n:
        moment_guardrail = {
            "moment_labeled_n": mom_labeled_n,
            "status": "MEASURED_TOP1_ANCHOR",
            "baseline_moment_hit_at_1_iou05": round(sum(mom_base_hits) / mom_labeled_n, 6),
            "challenger_moment_hit_at_1_iou05": round(sum(mom_new_hits) / mom_labeled_n, 6),
        }
    else:
        moment_guardrail = {"moment_labeled_n": mom_labeled_n, "status": "NOT_MEASURED_VIDEO_REPLAY"}

    top1_eligible_count = sum(1 for r in internal_rows if r["top1_change_eligible"])
    top1_changed_count = sum(1 for r in internal_rows if r["top1_changed"])

    summary = {
        "schema_version": 1,
        "k": k,
        "filter_manifest": str(filter_manifest) if filter_manifest is not None else None,
        "filter_split": filter_split,
        "require_same_base_pool": check_same_base_pool,
        "baseline": {
            "dir": str(baseline_dir),
            "pool": base_receipt.get("pool"),
            "alpha": base_receipt.get("alpha"),
            "gallery_sha256": base_receipt.get("gallery_sha256"),
            "gallery_fingerprint": base_fp,
            "query_plan_source": base_receipt.get("query_plan_source"),
        },
        "challenger": {
            "dir": str(challenger_dir),
            "pool": chal_receipt.get("pool"),
            "alpha": chal_receipt.get("alpha"),
            "gallery_sha256": chal_receipt.get("gallery_sha256"),
            "gallery_fingerprint": chal_fp,
            "vp_dry_run": chal_receipt.get("vp_dry_run"),
            "vp_receipt_sha256": chal_receipt.get("vp_receipt_sha256"),
            "vp_coverage_policy": chal_receipt.get("vp_coverage_policy"),
            "query_plan_source": chal_receipt.get("query_plan_source"),
            "query_plan_revision": chal_receipt.get("query_plan_revision"),
            "query_plan_cache_sha256": chal_receipt.get("query_plan_cache_sha256"),
        },
        "coverage_counts": {
            "vp_full_coverage_count": len(cov_ok_rows),
            "vp_partial_coverage_count": len(cov_partial_rows),
            "fallback_count": len(cov_fb_rows),
            "top1_eligible_count": top1_eligible_count,
            "top1_changed_count": top1_changed_count,
        },
        "overall": _summarize_slice(internal_rows, n_resamples=n_resamples, seed=seed),
        "per_dataset": {
            ds: _summarize_slice(ds_rows, n_resamples=n_resamples, seed=seed)
            for ds, ds_rows in sorted(by_ds.items())
        },
        "coverage_breakdown": {
            "vp_coverage_intersection": _summarize_slice(cov_ok_rows, n_resamples=n_resamples, seed=seed),
            "vp_partial_coverage": _summarize_slice(cov_partial_rows, n_resamples=n_resamples, seed=seed),
            "fallback_subset": _summarize_slice(cov_fb_rows, n_resamples=n_resamples, seed=seed),
        },
        "cached_llm_subset": _summarize_slice(cached_llm_rows, n_resamples=n_resamples, seed=seed),
        "guardrails": {
            "qvhighlights": _summarize_slice(by_ds.get("qvhighlights", []), n_resamples=n_resamples, seed=seed),
            "avc": _summarize_slice(by_ds.get("avc", []), n_resamples=n_resamples, seed=seed),
            "moment": moment_guardrail,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_query_evaluation.jsonl").open("w", encoding="utf-8") as stream:
        for rec in paired_records:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_markdown_report(out_dir / "benchmark_report.md", summary)
    return summary


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    ov = summary["overall"]
    msr = summary["per_dataset"].get("msrvtt")
    k = summary["k"]

    def _fmt_row(name: str, s: dict[str, Any]) -> str:
        if not s or s.get("n_queries", 0) == 0:
            return f"| {name} | 0 | - | - | - | - | - | - |\n"
        b = s["baseline"]
        c = s["challenger"]
        ci = s["ci95_pp"]
        return (
            f"| {name} | {s['n_queries']} ({s['n_source_groups']} grp) | "
            f"{b['hit_at_1']*100:.2f}% ({b['hit_at_1_count']}/{s['n_queries']}) -> "
            f"{c['hit_at_1']*100:.2f}% ({c['hit_at_1_count']}/{s['n_queries']}) | "
            f"{b['recall_at_k']*100:.2f}% ({b['recall_at_k_count']}) -> "
            f"{c['recall_at_k']*100:.2f}% ({c['recall_at_k_count']}) | "
            f"{b['mrr_at_10']:.4f} -> {c['mrr_at_10']:.4f} | "
            f"{s['wins']}/{s['losses']} | "
            f"{s['delta_pp']:+.2f} pp | [{ci[0]:+.2f}, {ci[1]:+.2f}] |\n"
        )

    lines = [
        f"# Paired Clip / VideoPrism Replay Comparison (K={k})\n\n",
        f"- **Baseline**: `{summary['baseline']['pool']}` (alpha={summary['baseline']['alpha']})\n",
        f"- **Challenger**: `{summary['challenger']['pool']}` (alpha={summary['challenger']['alpha']})\n\n",
        "| Slice | N (Groups) | Hit@1 (Base -> Chal) | Recall@K (Base -> Chal) | MRR@10 | Wins/Losses | Delta (pp) | 95% Paired Group Bootstrap CI |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        _fmt_row("Overall", ov),
    ]
    if msr:
        lines.append(_fmt_row("dataset=msrvtt", msr))
    for ds_name in ("qvhighlights", "avc"):
        if ds_name in summary["per_dataset"]:
            lines.append(_fmt_row(f"dataset={ds_name}", summary["per_dataset"][ds_name]))
    lines.append("\n## Coverage Breakdown\n\n")
    cov_counts = summary.get("coverage_counts") or {}
    lines.append(
        f"- Full VP coverage: {cov_counts.get('vp_full_coverage_count', 0)}, "
        f"Partial VP coverage: {cov_counts.get('vp_partial_coverage_count', 0)}, "
        f"Fallback: {cov_counts.get('fallback_count', 0)}, "
        f"Top-1 eligible: {cov_counts.get('top1_eligible_count', 0)}, "
        f"Top-1 changed: {cov_counts.get('top1_changed_count', 0)}\n\n"
    )
    lines.append("| Coverage Slice | N (Groups) | Hit@1 (Base -> Chal) | Recall@K (Base -> Chal) | MRR@10 | Wins/Losses | Delta (pp) | 95% Paired Group Bootstrap CI |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    lines.append(_fmt_row("VP Coverage Intersection", summary["coverage_breakdown"]["vp_coverage_intersection"]))
    if summary["challenger"]["vp_coverage_policy"] == "preserve-unscored":
        lines.append(_fmt_row("VP Partial Coverage", summary["coverage_breakdown"]["vp_partial_coverage"]))
    lines.append(_fmt_row("Fallback Subset", summary["coverage_breakdown"]["fallback_subset"]))
    if summary["challenger"]["query_plan_source"] == "frozen_llm_cache":
        lines.append(_fmt_row("Validated LLM cache hits", summary["cached_llm_subset"]))
    if summary["challenger"]["vp_dry_run"] is True:
        lines.append("\nVideoPrism shard used synthetic dry-run vectors.\n")
    lines.append(f"\n## Moment Guardrail\n\n- Status: `{summary['guardrails']['moment'].get('status')}` (n={summary['guardrails']['moment'].get('moment_labeled_n', 0)})\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison between baseline and challenger replay runs.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--filter-manifest", type=Path, default=None)
    parser.add_argument("--filter-split", default=None)
    parser.add_argument("--require-same-base-pool", dest="require_same_base_pool", action="store_true", default=None)
    parser.add_argument("--allow-different-base-pool", dest="require_same_base_pool", action="store_false")
    args = parser.parse_args()

    summary = compare_runs(
        baseline_dir=args.baseline,
        challenger_dir=args.challenger,
        labels_path=args.labels,
        split_lock_path=args.split_lock,
        out_dir=args.out_dir,
        n_resamples=args.n_resamples,
        seed=args.seed,
        require_same_base_pool=args.require_same_base_pool,
        filter_manifest=args.filter_manifest,
        filter_split=args.filter_split,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
