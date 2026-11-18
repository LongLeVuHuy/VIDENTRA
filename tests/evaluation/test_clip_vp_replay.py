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
        json.dumps({
            "status": "SUCCESS",
            "dry_run": False,
            "files": {
                name: {"sha256": digest, "bytes": (vp_dir / name).stat().st_size}
                for name, digest in files_sha.items()
            },
        }),
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

    # 1. Missing vp_dir altogether: strict alpha>0 blocks by default; strict_vp=False falls back
    out_missing_dir = tmp_path / "out_missing_dir"
    with pytest.raises(FileNotFoundError, match="missing_vp_dir"):
        run_replay(
            pools_dir=pools_dir,
            pool="a5",
            k=10,
            alpha=0.25,
            vp_dir=tmp_path / "nonexistent_vp",
            out_dir=out_missing_dir,
        )
    receipt = run_replay(
        pools_dir=pools_dir,
        pool="a5",
        k=10,
        alpha=0.25,
        vp_dir=tmp_path / "nonexistent_vp",
        strict_vp=False,
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

    partial_dir = tmp_path / "preserved_partial"
    partial = run_replay(
        pools_dir=pools_dir, k=10, alpha=1.0, vp_dir=vp_partial, out_dir=partial_dir,
        vp_coverage_policy="preserve-unscored",
    )
    assert partial["vp_partial_count"] == 2
    partial_rows = [
        json.loads(line)
        for line in (partial_dir / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for record in partial_rows:
        if record["vp_partial_applied"]:
            slot = record["base_video_ids"].index("video0003")
            assert record["ranked_video_ids"][slot] == "video0003"
            assert set(record["base_video_ids"]) == set(record["ranked_video_ids"])
            assert record["fallback_reason"] is None


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

    vp_dry = _make_vp_dir(tmp_path / "vp_dry", pools_dir)
    receipt_path = vp_dry / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dry_run"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic VideoPrism"):
        run_replay(pools_dir=pools_dir, vp_dir=vp_dry, alpha=0.25, out_dir=tmp_path / "out_dry")
    replay = run_replay(
        pools_dir=pools_dir, vp_dir=vp_dry, alpha=0.25, out_dir=tmp_path / "out_dry", allow_synthetic=True
    )
    assert replay["vp_dry_run"] is True

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


def test_strict_alpha_positive_blocks_zero_scored_and_missing_shard(tmp_path: Path) -> None:
    pools_dir, _, _, _ = _make_synthetic_pools(tmp_path)

    # 1. vp_dir=None with alpha > 0 -> ValueError
    with pytest.raises(ValueError, match="requires --vp-dir"):
        run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=None, out_dir=tmp_path / "out_none")

    # 2. Missing single shard file -> FileNotFoundError
    vp_incomplete = _make_vp_dir(tmp_path / "vp_incomplete", pools_dir)
    (vp_incomplete / "query_vectors.npy").unlink()
    with pytest.raises(FileNotFoundError, match="missing_vp_artifact"):
        run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=vp_incomplete, out_dir=tmp_path / "out_inc")

    # 3. Valid shard structure with 0 scored queries (all queries omitted) -> ValueError
    all_qids = {"msr_q1", "msr_q2", "qvh_q3", "avc_q4"}
    vp_zero = _make_vp_dir(tmp_path / "vp_zero", pools_dir, omit_queries=all_qids)
    with pytest.raises(ValueError, match="scored 0 queries"):
        run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.25, vp_dir=vp_zero, out_dir=tmp_path / "out_zero")


def test_preserve_unscored_top1_change_eligible_and_counts(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)
    # video0001 is rank-1 (A5 top-1) for msr_q1, msr_q2, qvh_q3; avc_vid9__s0__e10 is rank-1 for avc_q4.
    # Omitting video0001 means msr_q1, msr_q2, qvh_q3 lack top-1 vector -> kept at #1, top1_change_eligible=False.
    # avc_q4 has top-1 vector (avc_vid9__s0__e10) and >=2 scored candidates -> top1_change_eligible=True.
    vp_partial = _make_vp_dir(
        tmp_path / "vp_no_top1",
        pools_dir,
        omit_videos={"video0001"},
        promote_map={"avc_q4": "video0004"},
    )
    base_dir = tmp_path / "base_k10"
    chal_dir = tmp_path / "chal_k10"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    chal_receipt = run_replay(
        pools_dir=pools_dir,
        pool="a5",
        k=10,
        alpha=1.0,
        vp_dir=vp_partial,
        vp_coverage_policy="preserve-unscored",
        out_dir=chal_dir,
    )
    assert chal_receipt["vp_dry_run"] is False
    assert chal_receipt["vp_receipt_sha256"] == sha256_file(vp_partial / "receipt.json")
    assert chal_receipt["top1_eligible_count"] == 1
    assert chal_receipt["top1_changed_count"] == 1

    recs = {
        r["query_id"]: r
        for r in (json.loads(l) for l in (chal_dir / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines())
    }
    for qid in ("msr_q1", "msr_q2", "qvh_q3"):
        assert recs[qid]["new_top1"] == recs[qid]["base_top1"] == "video0001"
        assert recs[qid]["top1_change_eligible"] is False
    assert recs["avc_q4"]["top1_change_eligible"] is True
    assert recs["avc_q4"]["new_top1"] == "video0004"

    summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_preserve",
        n_resamples=500,
    )
    assert summary["coverage_counts"]["vp_full_coverage_count"] == 1
    assert summary["coverage_counts"]["vp_partial_coverage_count"] == 3
    assert summary["coverage_counts"]["top1_eligible_count"] == 1
    assert summary["coverage_counts"]["top1_changed_count"] == 1


def test_compare_rejects_gallery_fingerprint_base_pool_and_output_hash_mismatch(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)
    base_dir = tmp_path / "base"
    chal_dir = tmp_path / "chal"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=chal_dir)

    # 1. gallery_fingerprint mismatch -> ValueError
    chal_receipt_path = chal_dir / "run_receipt.json"
    orig_receipt = json.loads(chal_receipt_path.read_text(encoding="utf-8"))
    bad_fp_receipt = dict(orig_receipt, gallery_fingerprint="deadbeef" * 8)
    chal_receipt_path.write_text(json.dumps(bad_fp_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="gallery_fingerprint mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_bad_fp",
            n_resamples=200,
        )
    chal_receipt_path.write_text(json.dumps(orig_receipt), encoding="utf-8")

    # 2. base_video_ids mismatch in same-pool comparison -> ValueError
    chal_eval_path = chal_dir / "per_query_evaluation.jsonl"
    orig_eval_text = chal_eval_path.read_text(encoding="utf-8")
    rows = [json.loads(l) for l in orig_eval_text.splitlines() if l.strip()]
    rows[0]["base_video_ids"] = list(reversed(rows[0]["base_video_ids"]))
    _write_jsonl(chal_eval_path, rows)
    with pytest.raises(ValueError, match="base_video_ids mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_bad_base",
            n_resamples=200,
        )

    # 3. output_sha256 mismatch when ranked_video_ids tampered -> ValueError
    rows = [json.loads(l) for l in orig_eval_text.splitlines() if l.strip()]
    rows[0]["ranked_video_ids"] = list(reversed(rows[0]["ranked_video_ids"]))
    _write_jsonl(chal_eval_path, rows)
    with pytest.raises(ValueError, match="output_sha256 mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_bad_hash",
            n_resamples=200,
        )
    chal_eval_path.write_text(orig_eval_text, encoding="utf-8")

    # 4. Different pool (a5 vs clip) still succeeds automatically without --allow-different-base-pool
    clip_dir = tmp_path / "clip"
    run_replay(pools_dir=pools_dir, pool="clip", k=10, alpha=0.0, out_dir=clip_dir)
    ok_summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=clip_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_a5_clip",
        n_resamples=200,
    )
    assert ok_summary["require_same_base_pool"] is False
    assert ok_summary["overall"]["n_queries"] == 4


def test_compare_filter_manifest_tune_confirm(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)
    base_dir = tmp_path / "base"
    chal_dir = tmp_path / "chal"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="clip", k=10, alpha=0.0, out_dir=chal_dir)

    manifest_path = tmp_path / "tune_confirm_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "tune_query_ids": ["msr_q1", "msr_q2", "qvh_q3"],
            "confirm_query_ids": ["avc_q4"],
        }),
        encoding="utf-8",
    )

    tune_summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_tune",
        n_resamples=200,
        filter_manifest=manifest_path,
        filter_split="tune",
    )
    assert tune_summary["overall"]["n_queries"] == 3
    assert tune_summary["filter_split"] == "tune"

    confirm_summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_confirm",
        n_resamples=200,
        filter_manifest=manifest_path,
        filter_split="confirm",
    )
    assert confirm_summary["overall"]["n_queries"] == 1
    assert confirm_summary["filter_split"] == "confirm"


def test_compare_rejects_tampered_labels_sha256(tmp_path: Path) -> None:
    pools_dir, labels_path, _, _ = _make_synthetic_pools(tmp_path)
    base_dir = tmp_path / "base"
    chal_dir = tmp_path / "chal"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="clip", k=10, alpha=0.0, out_dir=chal_dir)

    lock_rows = [
        {
            "query_id": row["query_id"],
            "split": row["split"],
            "source_group": row["source_group"],
        }
        for row in (json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip())
    ]
    hashed_lock_path = tmp_path / "eval_only" / "split_lock_with_hash.json"
    freeze_split(
        lock_rows,
        hashed_lock_path,
        metadata={"split": "dev_full_859", "labels_sha256": sha256_file(labels_path)},
    )

    ok_summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=hashed_lock_path,
        out_dir=tmp_path / "cmp_ok_hash",
        n_resamples=200,
    )
    assert ok_summary["overall"]["n_queries"] == 4

    # Modify one target_videos entry on a copy of labels while keeping split_lock & query_id unchanged
    tampered_labels_path = tmp_path / "eval_only" / "labels_tampered.jsonl"
    label_rows = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    label_rows[0]["target_videos"] = ["video0099"]
    _write_jsonl(tampered_labels_path, label_rows)

    with pytest.raises(ValueError, match="labels_sha256 mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=tampered_labels_path,
            split_lock_path=hashed_lock_path,
            out_dir=tmp_path / "cmp_tampered_labels",
            n_resamples=200,
        )


def test_compare_filter_manifest_jsonl_hardening(tmp_path: Path) -> None:
    pools_dir, labels_path, split_lock_path, _ = _make_synthetic_pools(tmp_path)
    base_dir = tmp_path / "base"
    chal_dir = tmp_path / "chal"
    run_replay(pools_dir=pools_dir, pool="a5", k=10, alpha=0.0, out_dir=base_dir)
    run_replay(pools_dir=pools_dir, pool="clip", k=10, alpha=0.0, out_dir=chal_dir)

    # Valid JSONL with explicit tune/confirm splits and matching source_group succeeds
    valid_jsonl = tmp_path / "valid_manifest.jsonl"
    _write_jsonl(valid_jsonl, [
        {"query_id": "msr_q1", "split": "tune", "source_group": "video0004"},
        {"query_id": "msr_q2", "split": "tune", "source_group": "video0001"},
        {"query_id": "qvh_q3", "split": "confirm", "source_group": "qvh_yt11111"},
        {"query_id": "avc_q4", "split": "confirm", "source_group": "avc_vid9"},
    ])
    summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_valid_jsonl",
        n_resamples=200,
        filter_manifest=valid_jsonl,
        filter_split="tune",
    )
    assert summary["overall"]["n_queries"] == 2

    # 1. Missing split
    missing_split_jsonl = tmp_path / "missing_split.jsonl"
    _write_jsonl(missing_split_jsonl, [
        {"query_id": "msr_q1"},
        {"query_id": "avc_q4", "split": "confirm"},
    ])
    with pytest.raises(ValueError, match="missing split"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_missing_split",
            n_resamples=200,
            filter_manifest=missing_split_jsonl,
            filter_split="tune",
        )

    # 2. Unknown partition
    unknown_part_jsonl = tmp_path / "unknown_partition.jsonl"
    _write_jsonl(unknown_part_jsonl, [
        {"query_id": "msr_q1", "split": "heldout_test"},
        {"query_id": "avc_q4", "split": "confirm"},
    ])
    with pytest.raises(ValueError, match="Unknown partition"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_unknown_part",
            n_resamples=200,
            filter_manifest=unknown_part_jsonl,
            filter_split="tune",
        )

    # 3. source_group mismatch vs split_lock
    bad_group_jsonl = tmp_path / "bad_group.jsonl"
    _write_jsonl(bad_group_jsonl, [
        {"query_id": "msr_q1", "split": "tune", "source_group": "wrong_group"},
        {"query_id": "avc_q4", "split": "confirm", "source_group": "avc_vid9"},
    ])
    with pytest.raises(ValueError, match="source_group mismatch"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_bad_group",
            n_resamples=200,
            filter_manifest=bad_group_jsonl,
            filter_split="tune",
        )

    # 4. query_id missing from split_lock
    missing_lock_jsonl = tmp_path / "missing_lock.jsonl"
    _write_jsonl(missing_lock_jsonl, [
        {"query_id": "unknown_q99", "split": "tune"},
        {"query_id": "avc_q4", "split": "confirm"},
    ])
    with pytest.raises(ValueError, match="missing from split_lock"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_missing_lock",
            n_resamples=200,
            filter_manifest=missing_lock_jsonl,
            filter_split="tune",
        )

    # 5. source_group appearing in both tune and confirm
    shared_lock_path = tmp_path / "eval_only" / "split_lock_shared_group.json"
    freeze_split(
        [
            {"query_id": "msr_q1", "split": "dev_full_859", "source_group": "shared_grp"},
            {"query_id": "msr_q2", "split": "dev_full_859", "source_group": "shared_grp"},
            {"query_id": "qvh_q3", "split": "dev_full_859", "source_group": "qvh_yt11111"},
            {"query_id": "avc_q4", "split": "dev_full_859", "source_group": "avc_vid9"},
        ],
        shared_lock_path,
        metadata={"split": "dev_full_859"},
    )
    overlap_group_jsonl = tmp_path / "overlap_group.jsonl"
    _write_jsonl(overlap_group_jsonl, [
        {"query_id": "msr_q1", "split": "tune", "source_group": "shared_grp"},
        {"query_id": "msr_q2", "split": "confirm", "source_group": "shared_grp"},
    ])
    with pytest.raises(ValueError, match="Overlapping source_group"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=shared_lock_path,
            out_dir=tmp_path / "cmp_overlap_group",
            n_resamples=200,
            filter_manifest=overlap_group_jsonl,
            filter_split="tune",
        )

    # 6. Overlapping query_id between tune and confirm
    overlap_qid_jsonl = tmp_path / "overlap_qid.jsonl"
    _write_jsonl(overlap_qid_jsonl, [
        {"query_id": "msr_q1", "split": "tune"},
        {"query_id": "msr_q1", "split": "confirm"},
    ])
    with pytest.raises(ValueError, match="Overlapping query_id"):
        compare_runs(
            baseline_dir=base_dir,
            challenger_dir=chal_dir,
            labels_path=labels_path,
            split_lock_path=split_lock_path,
            out_dir=tmp_path / "cmp_overlap_qid",
            n_resamples=200,
            filter_manifest=overlap_qid_jsonl,
            filter_split="tune",
        )


def test_c3_vl_reranker_cache_contract_order_invariance_cascade_and_flip_precision(tmp_path: Path):
    from videntra.ranking.vl_reranker_cache import (
        load_vl_reranker_cache,
        rerank_top_m_with_vl,
        verify_order_invariance,
    )

    pools_dir, labels_path, split_lock_path, queries_data = _make_synthetic_pools(tmp_path)
    q1 = queries_data[0]  # target video0004 (at rank 4 in base A5 pool), msr_q1 base_top1=video0001 (miss)
    q2 = queries_data[1]  # target video0001 (at rank 1 in base A5 pool), msr_q2 base_top1=video0001 (hit)
    q1_sha = sha256_text(q1["query_text"])
    q2_sha = sha256_text(q2["query_text"])

    def _make_vl_row(q_id: str, q_sha: str, vid: str, score: float | None, status: str = "ok"):
        return {
            "query_id": q_id,
            "query_text_sha256": q_sha,
            "video_id": vid,
            "media_sha256": "deadbeef",
            "snapshot_sha256": "cafebabe",
            "frame_ordinals": [0, 4, 8, 12],
            "frame_pts_time": [0.0, 1.0, 2.0, 3.0],
            "sampling_recipe": "uniform_4f",
            "model_revision": "Qwen/Qwen3-VL-Reranker-2B@rev1",
            "processor_revision": "proc@rev1",
            "code_sha256": "code123",
            "prompt_sha256": "prompt123",
            "score": score,
            "status": status,
        }

    # 1. Leakage/missing contract check
    bad_cache = tmp_path / "bad_vl.jsonl"
    _write_jsonl(bad_cache, [{**_make_vl_row("msr_q1", q1_sha, "video0001", 0.5), "a5_rank": 1}])
    with pytest.raises(ValueError, match="forbidden"):
        load_vl_reranker_cache(bad_cache, strict_contract=True)

    # 2. Valid cache: rescues msr_q1 (moves video0004 -> top1); protects msr_q2 because top1 is unscored!
    vl_cache = tmp_path / "vl_cache.jsonl"
    vl_rows = [
        _make_vl_row("msr_q1", q1_sha, "video0001", 0.25),
        _make_vl_row("msr_q1", q1_sha, "video0002", None, status="oom"),
        _make_vl_row("msr_q1", q1_sha, "video0003", 0.40),
        _make_vl_row("msr_q1", q1_sha, "video0004", 0.95),
        # msr_q2: baseline top1 video0001 is unscored -> baseline top1 protected!
        _make_vl_row("msr_q2", q2_sha, "video0002", 0.99),
        _make_vl_row("msr_q2", q2_sha, "video0003", 0.80),
        # qvh_q3: wrong-to-wrong swap (base_top1=video0001, vl elevates video0005 to top1 while target is qvh_yt11111_0.0_10.0)
        _make_vl_row("qvh_q3", sha256_text(queries_data[2]["query_text"]), "video0001", 0.10),
        _make_vl_row("qvh_q3", sha256_text(queries_data[2]["query_text"]), "video0005", 0.90),
    ]
    _write_jsonl(vl_cache, vl_rows)
    verify_order_invariance(vl_rows, list(reversed(vl_rows)))

    # Unit check on top1_unscored & unscored slot preservation
    reranked_q1, st_q1 = rerank_top_m_with_vl(
        ["video0001", "video0002", "video0003", "video0004", "video0005", "video0006"],
        load_vl_reranker_cache(vl_cache)["by_qsha"][q1_sha],
        top_m=5,
        mode="pure",
    )
    assert reranked_q1[0] == "video0004"
    assert reranked_q1[1] == "video0002"  # unscored slot 1 preserved!
    assert reranked_q1[5] == "video0006"  # tail beyond top_m untouched!
    assert st_q1["applied"] is True

    reranked_q2, st_q2 = rerank_top_m_with_vl(
        ["video0001", "video0002", "video0003", "video0004"],
        load_vl_reranker_cache(vl_cache)["by_qsha"][q2_sha],
        top_m=5,
        mode="pure",
    )
    assert reranked_q2[0] == "video0001"  # baseline top-1 returned
    assert st_q2["fallback_reason"] == "top1_unscored"

    # 3. End-to-end replay & flip_precision in compare_runs
    base_dir = tmp_path / "base_a5"
    chal_vl_dir = tmp_path / "chal_vl"
    run_replay(pools_dir=pools_dir, pool="a5", k=20, alpha=0.0, out_dir=base_dir)
    vl_rec = run_replay(
        pools_dir=pools_dir,
        pool="a5",
        k=20,
        alpha=0.0,
        vl_cache=vl_cache,
        vl_top_m=5,
        vl_mode="pure",
        out_dir=chal_vl_dir,
    )
    assert vl_rec["vl_cache_sha256"] == sha256_file(vl_cache)
    assert vl_rec["vl_applied_count"] == 2  # msr_q1 and qvh_q3

    cmp_summary = compare_runs(
        baseline_dir=base_dir,
        challenger_dir=chal_vl_dir,
        labels_path=labels_path,
        split_lock_path=split_lock_path,
        out_dir=tmp_path / "cmp_vl",
        require_same_base_pool=True,
        n_resamples=200,
    )
    ov = cmp_summary["overall"]
    assert ov["rescued_hits"] == 1
    assert ov["lost_hits"] == 0
    assert ov["both_wrong_swaps"] == 1
    assert ov["top1_changed_count"] == 2
    assert ov["flip_precision"] == pytest.approx(0.5, abs=1e-6)
    assert "Flip precision:" in (tmp_path / "cmp_vl" / "benchmark_report.md").read_text(encoding="utf-8")


def test_vl_cache_strict_qsha_join_no_qid_leak_and_provenance(tmp_path: Path) -> None:
    from videntra.ranking.vl_reranker_cache import load_vl_reranker_cache

    pools_dir, _, _, queries_data = _make_synthetic_pools(tmp_path)
    q1 = queries_data[0]
    stale_q_sha = sha256_text("completely different old query sharing the same query_id")

    def _row(qid: str, q_sha: str, vid: str, score: float, **overrides):
        base = {
            "query_id": qid,
            "query_text_sha256": q_sha,
            "video_id": vid,
            "media_sha256": "deadbeef",
            "snapshot_sha256": "cafebabe",
            "frame_ordinals": [0, 4, 8],
            "frame_pts_time": [0.0, 1.0, 2.0],
            "sampling_recipe": "uniform_3f",
            "model_revision": "Qwen/Qwen3-VL-Reranker-2B@rev1",
            "processor_revision": "proc@rev1",
            "code_sha256": "code123",
            "prompt_sha256": "prompt123",
            "score": score,
            "status": "ok",
        }
        base.update(overrides)
        return base

    # 1. Cross-query query_id collision must NOT inject stale VL scores into run_replay
    stale_cache = tmp_path / "stale_vl.jsonl"
    _write_jsonl(stale_cache, [
        _row(q1["query_id"], stale_q_sha, "video0001", 0.05),
        _row(q1["query_id"], stale_q_sha, "video0004", 0.99),
    ])
    out_replay = tmp_path / "replay_collision"
    run_replay(
        pools_dir=pools_dir,
        pool="a5",
        k=20,
        alpha=0.0,
        vl_cache=stale_cache,
        vl_top_m=5,
        vl_mode="pure",
        out_dir=out_replay,
    )
    recs = {
        r["query_id"]: r
        for r in (json.loads(l) for l in (out_replay / "per_query_evaluation.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert recs[q1["query_id"]]["vl_applied"] is False
    assert recs[q1["query_id"]]["vl_fallback_reason"] == "missing_vl_scores"
    assert recs[q1["query_id"]]["new_top1"] == recs[q1["query_id"]]["base_top1"]

    # 2. Duplicate (query_text_sha256, video_id) must raise ValueError in strict mode
    dup_cache = tmp_path / "dup_vl.jsonl"
    _write_jsonl(dup_cache, [
        _row(q1["query_id"], stale_q_sha, "video0001", 0.1),
        _row(q1["query_id"], stale_q_sha, "video0001", 0.9),
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        load_vl_reranker_cache(dup_cache, strict=True)

    # 3. Non-monotonic frame_ordinals & mismatched expected_provenance must raise ValueError
    nonmono_cache = tmp_path / "nonmono_vl.jsonl"
    _write_jsonl(nonmono_cache, [
        _row(q1["query_id"], stale_q_sha, "video0001", 0.8, frame_ordinals=[0, 8, 4]),
    ])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_vl_reranker_cache(nonmono_cache, strict=True)

    with pytest.raises(ValueError, match="snapshot_sha256 mismatch"):
        load_vl_reranker_cache(
            stale_cache,
            strict=True,
            expected_provenance={"snapshot_sha256": "other_snapshot"},
        )
    with pytest.raises(ValueError, match="model_revision mismatch"):
        load_vl_reranker_cache(
            stale_cache,
            strict=True,
            expected_provenance={"model_revision": "other_rev"},
        )


