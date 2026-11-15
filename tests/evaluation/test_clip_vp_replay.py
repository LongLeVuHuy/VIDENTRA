"""Unit and integration tests for Clip/VideoPrism replay & paired evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.capture_msr_candidate_pools import (
    _assert_no_gt_leakage,
    apply_clip_reserve,
    build_a5_pool,
    compute_gallery_sha256,
    sha256_text,
)
from scripts.compare_msr_runs import (
    compare_runs,
    paired_source_group_bootstrap_ci,
)
from scripts.run_clip_vp_replay import run_replay
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import freeze_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_synthetic_pools(tmp_path: Path, n_queries: int = 4) -> tuple[Path, Path, Path, list[dict]]:
    pools_dir = tmp_path / "pools"
    eval_dir = tmp_path / "eval_only"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_vids = [f"video{i:04d}" for i in range(1, 31)]
    gallery_sha = compute_gallery_sha256(all_vids)

    queries_data = [
        {"query_id": "msr_q1", "query_text": "a dog running on grass", "dataset": "msrvtt", "target": "video0004", "grp": "video0004"},
        {"query_id": "msr_q2", "query_text": "a person cooking pasta", "dataset": "msrvtt", "target": "video0001", "grp": "video0001"},
        {"query_id": "qvh_q3", "query_text": "a sunset over the ocean", "dataset": "qvhighlights", "target": "qvh_yt11111_0.0_10.0", "grp": "qvh_yt11111"},
        {"query_id": "avc_q4", "query_text": "a car turning left", "dataset": "avc", "target": "avc_vid9__s0__e10", "grp": "avc_vid9"},
    ][:n_queries]

    q_rows, a5_rows, clip_rows, label_rows, lock_rows = [], [], [], [], []
    base_pool_20 = [f"video{i:04d}" for i in range(1, 21)]

    for idx, q in enumerate(queries_data):
        q_sha = sha256_text(q["query_text"])
        a5_ids = list(base_pool_20)
        if q["dataset"] == "qvhighlights":
            a5_ids[2] = q["target"]
        elif q["dataset"] == "avc":
            a5_ids[0] = q["target"]

        branch_ids = [q["target"]] + [f"video{i:04d}" for i in range(21, 26)] + a5_ids[:5]
        reserve_20 = apply_clip_reserve(a5_ids, branch_ids, k=20)

        q_rows.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "query_text_sha256": q_sha,
            "dataset": q["dataset"],
            "split": "dev_full_859",
        })
        a5_rows.append({
            "query_id": q["query_id"],
            "query_text_sha256": q_sha,
            "dataset": q["dataset"],
            "max_k": 20,
            "pool_video_ids": a5_ids,
            "candidates": [
                {"rank": r, "video_id": vid, "score": 1.0 - r * 0.01, "provenance": "a5_ranked_videos" if r <= 10 else "a5_candidate_moments", "anchor": {"start_sec": 0.0, "end_sec": 5.0}}
                for r, vid in enumerate(a5_ids, 1)
            ],
            "pool_size": len(a5_ids),
        })
        clip_rows.append({
            "query_id": q["query_id"],
            "query_text_sha256": q_sha,
            "dataset": q["dataset"],
            "max_k": 20,
            "pool_video_ids": reserve_20["pool_video_ids"],
            "branch_ranked_video_ids": branch_ids,
            "pool_size": len(reserve_20["pool_video_ids"]),
        })
        label_rows.append({
            "query_id": q["query_id"],
            "query_text_sha256": q_sha,
            "dataset": q["dataset"],
            "split": "dev_full_859",
            "target_videos": [q["target"]],
            "target_windows": [[0.0, 5.0]] if q["dataset"] == "qvhighlights" else [],
            "source_group": q["grp"],
        })
        lock_rows.append({
            "query_id": q["query_id"],
            "split": "dev_full_859",
            "source_group": q["grp"],
        })

    _write_jsonl(pools_dir / "queries.jsonl", q_rows)
    _write_jsonl(pools_dir / "a5_pool.jsonl", a5_rows)
    _write_jsonl(pools_dir / "clip_pool.jsonl", clip_rows)
    (pools_dir / "capture_receipt.json").write_text(
        json.dumps({"gallery_sha256": gallery_sha, "split": "dev_full_859", "max_k": 20}),
        encoding="utf-8",
    )

    labels_path = eval_dir / "labels.jsonl"
    split_lock_path = eval_dir / "split_lock.json"
    _write_jsonl(labels_path, label_rows)
    freeze_split(lock_rows, split_lock_path, metadata={"split": "dev_full_859"})
    return pools_dir, labels_path, split_lock_path, queries_data


def _make_vp_dir(
    vp_dir: Path,
    pools_dir: Path,
    *,
    promote_map: dict[str, str] | None = None,
    omit_queries: set[str] | None = None,
    omit_videos: set[str] | None = None,
    corrupt_hash: bool = False,
    non_finite_vec: bool = False,
) -> Path:
    vp_dir.mkdir(parents=True, exist_ok=True)
    q_rows_src = [json.loads(l) for l in (pools_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    a5_rows_src = [json.loads(l) for l in (pools_dir / "a5_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    clip_rows_src = [json.loads(l) for l in (pools_dir / "clip_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    all_vids = sorted({
        vid
        for r in a5_rows_src + clip_rows_src
        for vid in (r.get("pool_video_ids", []) + r.get("branch_ranked_video_ids", []))
    } - (omit_videos or set()))

    dim = 16
    rng = np.random.default_rng(123)
    vid_vecs = {}
    for vid in all_vids:
        v = rng.normal(size=(dim,)).astype(np.float32)
        v /= np.linalg.norm(v)
        vid_vecs[vid] = v

    # Orthogonalize query vectors away from default videos, and align with promoted target
    q_vecs_list = []
    vp_q_rows = []
    for q in q_rows_src:
        qid = q["query_id"]
        if omit_queries and qid in omit_queries:
            continue
        if promote_map and qid in promote_map and promote_map[qid] in vid_vecs:
            target_v = vid_vecs[promote_map[qid]]
            for v_id in ("video0001", "video0002", "video0003"):
                if v_id in vid_vecs and promote_map[qid] != v_id:
                    vid_vecs[v_id] = -target_v.copy()
            qv = target_v.copy()
        else:
            qv = rng.normal(size=(dim,)).astype(np.float32)
            qv /= np.linalg.norm(qv)
        vp_q_rows.append({
            "row_id": len(vp_q_rows),
            "query_id": qid,
            "query_text_sha256": q["query_text_sha256"],
        })
        q_vecs_list.append(qv)

    vp_v_rows = [{"row_id": idx, "video_id": vid} for idx, vid in enumerate(all_vids)]
    v_mat = np.vstack([vid_vecs[v] for v in all_vids]).astype(np.float32)
    q_mat = np.vstack(q_vecs_list).astype(np.float32) if q_vecs_list else np.zeros((0, dim), dtype=np.float32)

    if non_finite_vec and v_mat.shape[0] > 0:
        v_mat[0, 0] = np.nan

    np.save(vp_dir / "video_vectors.npy", v_mat)
    np.save(vp_dir / "query_vectors.npy", q_mat)
    _write_jsonl(vp_dir / "video_rows.jsonl", vp_v_rows)
    _write_jsonl(vp_dir / "query_rows.jsonl", vp_q_rows)

    files_sha = {
        "video_vectors.npy": sha256_file(vp_dir / "video_vectors.npy"),
        "query_vectors.npy": sha256_file(vp_dir / "query_vectors.npy"),
        "video_rows.jsonl": sha256_file(vp_dir / "video_rows.jsonl"),
        "query_rows.jsonl": sha256_file(vp_dir / "query_rows.jsonl"),
    }
    if corrupt_hash:
        files_sha["video_vectors.npy"] = "0" * 64

    (vp_dir / "receipt.json").write_text(
        json.dumps({"schema_version": 1, "files": files_sha}),
        encoding="utf-8",
    )
    return vp_dir


def test_build_a5_pool_and_clip_reserve_k10_k20_uniqueness() -> None:
    ranked_videos = [{"video_id": f"v{i}", "ranking_score": 1.0 - i * 0.01} for i in range(1, 11)]
    cand_moments = [{"video_id": f"v{i}", "ranking_score": 0.5 - i * 0.001} for i in range(5, 25)]
    a5 = build_a5_pool(ranked_videos, cand_moments, max_k=20)
    assert len(a5["pool_video_ids"]) == 20
    assert len(set(a5["pool_video_ids"])) == 20
    assert a5["pool_video_ids"][:10] == [f"v{i}" for i in range(1, 11)]

    branch = ["v1", "v2", "v_new_clip", "v25"]
    res10 = apply_clip_reserve(a5["pool_video_ids"], branch, k=10)
    assert len(res10["pool_video_ids"]) == 10
    assert len(set(res10["pool_video_ids"])) == 10
    assert res10["pool_video_ids"][:9] == a5["pool_video_ids"][:9]
    assert res10["pool_video_ids"][9] == "v_new_clip"
    assert res10["added_video"] == "v_new_clip"
    assert res10["displaced_video"] == "v10"
    assert res10["duplicate_skipped"] == 2

    res20 = apply_clip_reserve(a5["pool_video_ids"], branch, k=20)
    assert len(res20["pool_video_ids"]) == 20
    assert len(set(res20["pool_video_ids"])) == 20
    assert res20["pool_video_ids"][:19] == a5["pool_video_ids"][:19]
    assert res20["pool_video_ids"][19] == "v_new_clip"

    # Leakage guard
    with pytest.raises(AssertionError, match="Leakage"):
        _assert_no_gt_leakage([{"query_id": "q1", "target_videos": ["v1"]}], "queries.jsonl")


def test_alpha_zero_exact_parity_and_k_uniqueness(tmp_path: Path) -> None:
    pools_dir, _, _, _ = _make_synthetic_pools(tmp_path)

    out_a5_k10 = tmp_path / "out_a5_k10"
    out_clip_k10 = tmp_path / "out_clip_k10"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=out_a5_k10)
    run_replay(pools_dir=pools_dir, pool="clip", k=10, alpha=0.0, out_dir=out_clip_k10)

    a5_recs = [json.loads(l) for l in (out_a5_k10 / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    clip_recs = [json.loads(l) for l in (out_clip_k10 / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines()]

    for a_r, c_r in zip(a5_recs, clip_recs):
        assert a_r["ranked_video_ids"] == a_r["base_video_ids"]
        assert len(a_r["ranked_video_ids"]) == 10
        assert len(set(a_r["ranked_video_ids"])) == 10
        assert c_r["ranked_video_ids"] == c_r["base_video_ids"]
        assert len(c_r["ranked_video_ids"]) == 10
        assert len(set(c_r["ranked_video_ids"])) == 10
        # Reserve preserves Top-1 and first K-1 head
        assert c_r["ranked_video_ids"][:9] == a_r["ranked_video_ids"][:9]
        assert c_r["new_top1"] == a_r["new_top1"]


def test_vp_rerank_promotion_and_paired_comparison(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)
    # In msr_q1, video0004 is at rank 4 in a5_pool; with alpha=0.25 and VP rank 1 it promotes to top-1
    vp_dir = _make_vp_dir(tmp_path / "vp", pools_dir, promote_map={"msr_q1": "video0004"})

    base_dir = tmp_path / "base_k20"
    chal_dir = tmp_path / "chal_k20"
    run_replay(pools_dir=pools_dir, pool="a5", k=20, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="a5", k=20, alpha=0.25, vp_dir=vp_dir, out_dir=chal_dir)

    chal_recs = {
        r["query_id"]: r
        for r in (json.loads(l) for l in (chal_dir / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert chal_recs["msr_q1"]["new_top1"] == "video0004"
    assert chal_recs["msr_q1"]["vp_coverage_ok"] is True
    assert chal_recs["msr_q1"]["fallback_reason"] is None

    cmp_dir = tmp_path / "compare"
    summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=cmp_dir,
        n_resamples=1000,
        seed=20261115,
    )
    assert summary["overall"]["wins"] >= 1
    assert summary["overall"]["delta_pp"] > 0.0
    assert (cmp_dir / "benchmark_summary.json").is_file()
    assert (cmp_dir / "benchmark_report.md").is_file()
    assert (cmp_dir / "per_query_evaluation.jsonl").is_file()
    assert "msrvtt" in summary["per_dataset"]
    assert "qvhighlights" in summary["guardrails"]
    assert "avc" in summary["guardrails"]


def test_missing_artifact_and_missing_vector_fallback(tmp_path: Path) -> None:
    pools_dir, _, _, _ = _make_synthetic_pools(tmp_path)

    # 1. Missing vp_dir altogether
    out_missing_dir = tmp_path / "out_missing_dir"
    receipt = run_replay(
        pools_dir=pools_dir,
        pool="a5",
        k=10,
        alpha=0.25,
        vp_dir=tmp_path / "nonexistent_vp",
        out_dir=out_missing_dir,
    )
    assert receipt["query_count"] == 4
    assert receipt["fallback_count"] == 4
    recs = [json.loads(l) for l in (out_missing_dir / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["ranked_video_ids"] == r["base_video_ids"] for r in recs)
    assert all(r["fallback_reason"] == "missing_vp_dir" for r in recs)

    # 2. Missing single query vector & missing single video vector (video0003 is in msr_q1, msr_q2, avc_q4; not qvh_q3)
    vp_partial = _make_vp_dir(
        tmp_path / "vp_partial",
        pools_dir,
        omit_queries={"msr_q2"},
        omit_videos={"video0003"},
    )
    out_partial = tmp_path / "out_partial"
    receipt_p = run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=vp_partial, out_dir=out_partial)
    assert receipt_p["query_count"] == 4
    assert receipt_p["fallback_count"] == 3
    assert receipt_p["vp_coverage_count"] == 1
    recs_p = {
        r["query_id"]: r
        for r in (json.loads(l) for l in (out_partial / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert recs_p["msr_q2"]["fallback_reason"] == "missing_vp_query"
    assert recs_p["msr_q1"]["fallback_reason"].startswith("missing_vp_videos:")
    assert recs_p["qvh_q3"]["vp_coverage_ok"] is True


def test_duplicate_query_and_hash_mismatch_rejection(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)

    # 1. Corrupt VP receipt hash -> ValueError
    vp_bad_hash = _make_vp_dir(tmp_path / "vp_bad_hash", pools_dir, corrupt_hash=True)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=vp_bad_hash, out_dir=tmp_path / "out_bh")

    # 2. Non-finite VP vector -> ValueError
    vp_nan = _make_vp_dir(tmp_path / "vp_nan", pools_dir, non_finite_vec=True)
    with pytest.raises(ValueError, match="Non-finite"):
        run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=vp_nan, out_dir=tmp_path / "out_nan")

    # 3. Duplicate query_id in queries.jsonl -> ValueError
    dup_pools = tmp_path / "dup_pools"
    dup_pools.mkdir()
    q_lines = (pools_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    (dup_pools / "queries.jsonl").write_text("\n".join([q_lines[0], q_lines[0]]) + "\n", encoding="utf-8")
    (dup_pools / "a5_pool.jsonl").write_text((pools_dir / "a5_pool.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate query_id"):
        run_replay(pools_dir=dup_pools, pool="a5", k=10, alpha=0.0, out_dir=tmp_path / "out_dup")

    # 4. Query hash mismatch in compare_msr_runs -> ValueError
    base_dir = tmp_path / "base_ok"
    chal_dir = tmp_path / "chal_bad"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=chal_dir)
    chal_eval = chal_dir / "per_query_evaluation.jsonl"
    rows = [json.loads(l) for l in chal_eval.read_text(encoding="utf-8").splitlines()]
    rows[0]["query_text_sha256"] = "f" * 64
    _write_jsonl(chal_eval, rows)
    with pytest.raises(ValueError, match="query_text_sha256 mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_bad",
        )


def test_paired_source_group_bootstrap_and_wins_losses() -> None:
    paired = [
        {"source_group": "g1", "base_hit": 0, "new_hit": 1},
        {"source_group": "g1", "base_hit": 0, "new_hit": 1},
        {"source_group": "g2", "base_hit": 1, "new_hit": 0},
        {"source_group": "g3", "base_hit": 1, "new_hit": 1},
    ]
    res1 = paired_source_group_bootstrap_ci(paired, n_resamples=2000, seed=20261115)
    res2 = paired_source_group_bootstrap_ci(paired, n_resamples=2000, seed=20261115)
    assert res1 == res2
    assert res1["n_queries"] == 4
    assert res1["n_source_groups"] == 3
    assert res1["delta_pp"] == pytest.approx(25.0)
    assert res1["ci95_pp"][0] <= res1["delta_pp"] <= res1["ci95_pp"][1]
