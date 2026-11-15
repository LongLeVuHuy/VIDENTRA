r"""Offline CPU replay for A5 / Clip-Reserve pools and VideoPrism-LvT-B RRF reranking.

Usage:
    .\.venv\Scripts\python.exe -B scripts/run_clip_vp_replay.py `
        --pools-dir work/experiments/clip_vp_pools_dev --pool clip --k 20 --alpha 0.25 `
        --vp-dir work/kaggle/videoprism_dev --out-dir work/experiments/clip_vp_k20_vp_clip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.capture_msr_candidate_pools import apply_clip_reserve, sha256_text
from videntra.evaluation.provenance import hash_paths, sha256_file

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _read_jsonl_unique(path: Path, id_key: str, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get(id_key, "")).strip()
            if not key:
                raise ValueError(f"Missing {id_key} in {label} line {line_no}")
            if key in seen:
                raise ValueError(f"Duplicate {id_key} '{key}' in {label}")
            seen.add(key)
            rows.append(row)
    return rows


def _dedup_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _verify_vp_receipt_hashes(vp_dir: Path, receipt: dict[str, Any]) -> None:
    """Verify SHA-256 hashes recorded in VideoPrism receipt.json if present."""
    hash_maps: list[dict[str, str]] = []
    for key in ("files", "artifact_sha256", "sha256", "hashes"):
        val = receipt.get(key)
        if isinstance(val, dict):
            hash_maps.append({str(k): str(v) for k, v in val.items() if v is not None})

    for fname in ("video_vectors.npy", "video_rows.jsonl", "query_vectors.npy", "query_rows.jsonl"):
        stem = fname.replace(".", "_")
        for candidate_key in (f"{fname}_sha256", f"{stem}_sha256"):
            if candidate_key in receipt and receipt[candidate_key]:
                hash_maps.append({fname: str(receipt[candidate_key])})

    for hmap in hash_maps:
        for rel_name, expected_sha in hmap.items():
            target = vp_dir / Path(rel_name).name
            if not target.is_file():
                target = vp_dir / rel_name
            if not target.is_file():
                raise FileNotFoundError(f"VP receipt references missing file: {rel_name}")
            actual_sha = sha256_file(target)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"VP receipt SHA-256 mismatch for {target.name}: expected {expected_sha}, got {actual_sha}"
                )


def _load_vp_artifacts(
    vp_dir: Path,
    queries_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate VideoPrism vectors/rows/receipt. Returns (bundle, missing_artifact_reason)."""
    required = ("receipt.json", "video_vectors.npy", "video_rows.jsonl", "query_vectors.npy", "query_rows.jsonl")
    if not vp_dir.is_dir():
        return None, "missing_vp_dir"
    missing = [name for name in required if not (vp_dir / name).is_file()]
    if missing:
        return None, f"missing_vp_artifact:{','.join(missing)}"

    receipt = json.loads((vp_dir / "receipt.json").read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("VP receipt.json must be a JSON object")
    _verify_vp_receipt_hashes(vp_dir, receipt)

    video_vecs = np.load(vp_dir / "video_vectors.npy", mmap_mode="r")
    query_vecs = np.load(vp_dir / "query_vectors.npy", mmap_mode="r")
    if video_vecs.ndim != 2 or query_vecs.ndim != 2:
        raise ValueError(f"VP vectors must be 2D, got video={video_vecs.shape}, query={query_vecs.shape}")
    if video_vecs.shape[1] != query_vecs.shape[1] or video_vecs.shape[1] <= 0:
        raise ValueError(f"VP dim mismatch: video={video_vecs.shape}, query={query_vecs.shape}")

    video_arr = np.asarray(video_vecs, dtype=np.float32)
    query_arr = np.asarray(query_vecs, dtype=np.float32)
    if not np.isfinite(video_arr).all():
        raise ValueError("Non-finite values in VP video_vectors.npy")
    if not np.isfinite(query_arr).all():
        raise ValueError("Non-finite values in VP query_vectors.npy")

    v_norms = np.linalg.norm(video_arr, axis=1, keepdims=True)
    q_norms = np.linalg.norm(query_arr, axis=1, keepdims=True)
    if video_arr.shape[0] > 0 and np.any(v_norms <= 1e-8):
        raise ValueError("Zero-norm row in VP video_vectors.npy")
    if query_arr.shape[0] > 0 and np.any(q_norms <= 1e-8):
        raise ValueError("Zero-norm row in VP query_vectors.npy")
    video_arr = video_arr / np.maximum(v_norms, 1e-8)
    query_arr = query_arr / np.maximum(q_norms, 1e-8)

    video_rows = _read_jsonl_unique(vp_dir / "video_rows.jsonl", "video_id", "video_rows.jsonl")
    query_rows = _read_jsonl_unique(vp_dir / "query_rows.jsonl", "query_id", "query_rows.jsonl")

    video_map: dict[str, np.ndarray] = {}
    for idx, row in enumerate(video_rows):
        row_idx = int(row.get("row_id", idx))
        if row_idx < 0 or row_idx >= video_arr.shape[0]:
            raise ValueError(f"VP video row_id {row_idx} out of bounds [0, {video_arr.shape[0]})")
        video_map[str(row["video_id"])] = video_arr[row_idx]

    query_map: dict[str, np.ndarray] = {}
    for idx, row in enumerate(query_rows):
        qid = str(row["query_id"])
        row_idx = int(row.get("row_id", idx))
        if row_idx < 0 or row_idx >= query_arr.shape[0]:
            raise ValueError(f"VP query row_id {row_idx} out of bounds [0, {query_arr.shape[0]})")
        if qid in queries_by_id and row.get("query_text_sha256") is not None:
            expected_sha = queries_by_id[qid]["query_text_sha256"]
            if str(row["query_text_sha256"]) != expected_sha:
                raise ValueError(
                    f"VP query_text_sha256 mismatch for {qid}: expected {expected_sha}, got {row['query_text_sha256']}"
                )
        query_map[qid] = query_arr[row_idx]

    return {
        "receipt": receipt,
        "receipt_sha256": sha256_file(vp_dir / "receipt.json"),
        "video_map": video_map,
        "query_map": query_map,
    }, None


def rerank_pool_with_vp(
    base_pool: Sequence[str],
    q_vec: np.ndarray,
    video_map: dict[str, np.ndarray],
    alpha: float,
) -> tuple[list[str], dict[str, float]]:
    """Compute RRF blend score = (1-alpha)/(60+r_base) + alpha/(60+r_vp) with deterministic tie-break."""
    pool = _dedup_preserve_order(base_pool)
    r_base = {vid: idx for idx, vid in enumerate(pool, 1)}
    vp_sims = {vid: float(np.dot(q_vec, video_map[vid])) for vid in pool}
    vp_order = sorted(pool, key=lambda vid: (-vp_sims[vid], r_base[vid], vid))
    r_vp = {vid: idx for idx, vid in enumerate(vp_order, 1)}

    scores = {
        vid: (1.0 - alpha) / (60.0 + r_base[vid]) + alpha / (60.0 + r_vp[vid])
        for vid in pool
    }
    ranked = sorted(pool, key=lambda vid: (-scores[vid], r_base[vid], vid))
    return ranked, scores


def run_replay(
    *,
    pools_dir: Path,
    pool: str = "a5",
    k: int = 20,
    alpha: float = 0.0,
    vp_dir: Path | None = None,
    out_dir: Path = Path("work/experiments/clip_vp_replay"),
) -> dict[str, Any]:
    if pool not in ("a5", "clip"):
        raise ValueError(f"pool must be 'a5' or 'clip', got {pool}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if pool == "clip" and k < 2:
        raise ValueError(f"clip pool requires k >= 2, got {k}")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    alpha = float(alpha)

    t0 = time.perf_counter()
    queries_path = pools_dir / "queries.jsonl"
    a5_path = pools_dir / "a5_pool.jsonl"
    clip_path = pools_dir / "clip_pool.jsonl"
    capture_receipt_path = pools_dir / "capture_receipt.json"

    queries = _read_jsonl_unique(queries_path, "query_id", "queries.jsonl")
    a5_rows = _read_jsonl_unique(a5_path, "query_id", "a5_pool.jsonl")
    clip_rows = _read_jsonl_unique(clip_path, "query_id", "clip_pool.jsonl") if pool == "clip" else []

    queries_by_id: dict[str, dict[str, Any]] = {}
    for q in queries:
        qid = str(q["query_id"])
        q_sha = q.get("query_text_sha256")
        if "query_text" in q and q_sha is not None and sha256_text(str(q["query_text"])) != str(q_sha):
            raise ValueError(f"query_text_sha256 mismatch inside queries.jsonl for {qid}")
        if q_sha is None:
            q["query_text_sha256"] = sha256_text(str(q.get("query_text", "")))
        queries_by_id[qid] = q

    a5_by_id = {str(r["query_id"]): r for r in a5_rows}
    if set(a5_by_id.keys()) != set(queries_by_id.keys()):
        raise ValueError("query_id set mismatch between queries.jsonl and a5_pool.jsonl")

    clip_by_id: dict[str, dict[str, Any]] = {}
    if pool == "clip":
        clip_by_id = {str(r["query_id"]): r for r in clip_rows}
        if set(clip_by_id.keys()) != set(queries_by_id.keys()):
            raise ValueError("query_id set mismatch between queries.jsonl and clip_pool.jsonl")

    for qid, q in queries_by_id.items():
        q_sha = q["query_text_sha256"]
        if a5_by_id[qid].get("query_text_sha256") is not None and str(a5_by_id[qid]["query_text_sha256"]) != q_sha:
            raise ValueError(f"query_text_sha256 mismatch in a5_pool.jsonl for {qid}")
        if pool == "clip" and clip_by_id[qid].get("query_text_sha256") is not None:
            if str(clip_by_id[qid]["query_text_sha256"]) != q_sha:
                raise ValueError(f"query_text_sha256 mismatch in clip_pool.jsonl for {qid}")

    vp_bundle: dict[str, Any] | None = None
    vp_global_fallback: str | None = None
    if alpha > 0.0:
        if vp_dir is None:
            vp_global_fallback = "missing_vp_dir"
        else:
            vp_bundle, vp_global_fallback = _load_vp_artifacts(vp_dir, queries_by_id)
    elif vp_dir is not None and vp_dir.exists():
        # Even at alpha=0, if vp_dir is explicitly passed and exists, validate its integrity
        vp_bundle, vp_global_fallback = _load_vp_artifacts(vp_dir, queries_by_id)

    capture_receipt = (
        json.loads(capture_receipt_path.read_text(encoding="utf-8"))
        if capture_receipt_path.is_file()
        else {}
    )

    per_query_records: list[dict[str, Any]] = []
    fallback_count = 0
    vp_coverage_count = 0

    for q in queries:
        qid = str(q["query_id"])
        q_sha = str(q["query_text_sha256"])
        ds = str(q.get("dataset", "unknown"))
        a5_row = a5_by_id[qid]
        a5_full = _dedup_preserve_order(a5_row.get("pool_video_ids", []))

        reserve_info: dict[str, Any] | None = None
        if pool == "a5":
            base_pool = a5_full[:k]
        else:
            c_row = clip_by_id[qid]
            branch_ids = c_row.get("branch_ranked_video_ids")
            if isinstance(branch_ids, list) and branch_ids:
                reserve_info = apply_clip_reserve(a5_full, branch_ids, k=k)
                base_pool = reserve_info["pool_video_ids"]
            else:
                base_pool = _dedup_preserve_order(c_row.get("pool_video_ids", []))[:k]

        anchor_by_vid = {
            str(c["video_id"]): c.get("anchor")
            for c in (a5_row.get("candidates") or [])
            if isinstance(c, dict) and c.get("video_id")
        }

        fallback_reason: str | None = None
        vp_coverage_ok = False
        scores: dict[str, float] = {}

        if alpha == 0.0:
            ranked_video_ids = list(base_pool)
            scores = {vid: 1.0 / (60.0 + idx) for idx, vid in enumerate(ranked_video_ids, 1)}
            if vp_bundle is not None:
                vp_coverage_ok = (
                    qid in vp_bundle["query_map"]
                    and all(vid in vp_bundle["video_map"] for vid in base_pool)
                )
        elif vp_global_fallback is not None:
            ranked_video_ids = list(base_pool)
            fallback_reason = vp_global_fallback
            scores = {vid: 1.0 / (60.0 + idx) for idx, vid in enumerate(ranked_video_ids, 1)}
        else:
            assert vp_bundle is not None
            q_vec = vp_bundle["query_map"].get(qid)
            if q_vec is None:
                ranked_video_ids = list(base_pool)
                fallback_reason = "missing_vp_query"
                scores = {vid: 1.0 / (60.0 + idx) for idx, vid in enumerate(ranked_video_ids, 1)}
            else:
                missing_vids = [vid for vid in base_pool if vid not in vp_bundle["video_map"]]
                if missing_vids:
                    ranked_video_ids = list(base_pool)
                    fallback_reason = f"missing_vp_videos:{len(missing_vids)}"
                    scores = {vid: 1.0 / (60.0 + idx) for idx, vid in enumerate(ranked_video_ids, 1)}
                else:
                    ranked_video_ids, scores = rerank_pool_with_vp(
                        base_pool, q_vec, vp_bundle["video_map"], alpha=alpha
                    )
                    vp_coverage_ok = True

        if fallback_reason is not None:
            fallback_count += 1
        if vp_coverage_ok:
            vp_coverage_count += 1

        record = {
            "query_id": qid,
            "query_text_sha256": q_sha,
            "dataset": ds,
            "pool": pool,
            "k": k,
            "alpha": alpha,
            "base_video_ids": base_pool,
            "ranked_video_ids": ranked_video_ids,
            "base_top1": base_pool[0] if base_pool else None,
            "new_top1": ranked_video_ids[0] if ranked_video_ids else None,
            "scores": {vid: round(scores[vid], 8) for vid in ranked_video_ids},
            "fallback_reason": fallback_reason,
            "vp_coverage_ok": vp_coverage_ok,
            "anchor_by_video_id": anchor_by_vid,
        }
        if reserve_info is not None:
            record["reserve_trace"] = {
                "added_video": reserve_info["added_video"],
                "displaced_video": reserve_info["displaced_video"],
                "duplicate_skipped": reserve_info["duplicate_skipped"],
                "branch_rank": reserve_info["branch_rank"],
                "reserve_fallback_reason": reserve_info["fallback_reason"],
            }
        per_query_records.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "per_query_evaluation.jsonl"
    with eval_path.open("w", encoding="utf-8") as stream:
        for rec in per_query_records:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

    receipt = {
        "schema_version": 1,
        "pool": pool,
        "k": k,
        "alpha": alpha,
        "pools_dir": str(pools_dir),
        "vp_dir": str(vp_dir) if vp_dir is not None else None,
        "query_count": len(per_query_records),
        "fallback_count": fallback_count,
        "vp_coverage_count": vp_coverage_count,
        "gallery_sha256": capture_receipt.get("gallery_sha256"),
        "queries_sha256": sha256_file(queries_path),
        "pool_sha256": sha256_file(a5_path if pool == "a5" else clip_path),
        "vp_receipt_sha256": vp_bundle["receipt_sha256"] if vp_bundle else None,
        "elapsed_seconds": round(time.perf_counter() - t0, 4),
        "output_sha256": hash_paths([eval_path]),
    }
    receipt_path = out_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Clip/VideoPrism replay on frozen candidate pools.")
    parser.add_argument("--pools-dir", required=True, type=Path)
    parser.add_argument("--pool", required=True, choices=("a5", "clip"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--vp-dir", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    receipt = run_replay(
        pools_dir=args.pools_dir,
        pool=args.pool,
        k=args.k,
        alpha=args.alpha,
        vp_dir=args.vp_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
