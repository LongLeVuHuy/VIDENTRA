"""Unit tests for VideoPrismClipRetriever (P1 retrieval + Step H Top-20 reranking)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.retrieval.videoprism_branch import VIDEOPRISM_DIM, VideoPrismClipRetriever


def _unit_vec(seed: int, dim: int = VIDEOPRISM_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_videoprism_missing_artifacts_graceful_fallback(tmp_path: Path) -> None:
    retriever = VideoPrismClipRetriever(processed_dir=tmp_path / "missing", cache_path=tmp_path / "missing.npz")
    assert retriever.available is False
    assert retriever.status == "not_built"
    res = retriever.retrieve_parallel("a person cooking", top_n=10)
    assert res.status == "not_built"
    assert res.ranked_videos == []
    assert retriever.rerank_top_candidates("a person cooking", [], ["v1", "v2"]) == {}


def test_videoprism_disk_roundtrip_and_revision_mismatch_guard(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    vids = ["vid_a", "vid_b", "vid_c"]
    q_vec = _unit_vec(101)
    v_vecs = np.vstack([
        0.92 * q_vec + 0.08 * _unit_vec(1),
        0.15 * q_vec + 0.85 * _unit_vec(2),
        -0.3 * q_vec + 0.7 * _unit_vec(3),
    ]).astype(np.float32)
    v_vecs /= np.linalg.norm(v_vecs, axis=1, keepdims=True)

    idx = faiss.IndexFlatIP(VIDEOPRISM_DIM)
    idx.add(v_vecs)
    faiss.write_index(idx, str(proc / "faiss_videoprism_clip_768.bin"))
    (proc / "videoprism_clip_manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "video_ids": vids,
            "model": {"hf_revision": "rev_abc"},
            "source_receipt_sha256": "deadbeef",
        }),
        encoding="utf-8",
    )

    s_vecs = np.vstack([0.96 * q_vec + 0.04 * _unit_vec(4)]).astype(np.float32)
    s_vecs /= np.linalg.norm(s_vecs, axis=1, keepdims=True)
    np.save(proc / "videoprism_subclips_768.npy", s_vecs)
    (proc / "videoprism_subclips_map.jsonl").write_text(
        json.dumps({"subclip_id": "vid_b__w000", "video_id": "vid_b", "clip_start_sec": 8.0, "clip_end_sec": 24.0}) + "\n",
        encoding="utf-8",
    )

    q_sha = hashlib.sha256("a cat jumps".encode("utf-8")).hexdigest()
    cache_path = tmp_path / "vp_cache.npz"
    np.savez(
        cache_path,
        keys=np.array([q_sha]),
        vectors=q_vec[None, :],
        model_revision=np.array(["rev_abc"]),
    )

    retriever = VideoPrismClipRetriever(processed_dir=proc, cache_path=cache_path)
    assert retriever.available is True
    assert retriever.status == "ready"
    res = retriever.retrieve_parallel("a cat jumps", top_n=3)
    assert res.status == "ok"
    assert res.ranked_videos[0][0] == "vid_a"
    assert res.best_subclip_windows["vid_b"][:2] == (8.0, 24.0)
    assert res.subclip_max_scores["vid_b"] > res.whole_scores["vid_b"]

    # Revision mismatch blocks poisoned query cache
    cache_bad = tmp_path / "vp_cache_bad.npz"
    np.savez(cache_bad, keys=np.array([q_sha]), vectors=q_vec[None, :], model_revision=np.array(["wrong_rev"]))
    ret_bad = VideoPrismClipRetriever(processed_dir=proc, cache_path=cache_bad)
    assert ret_bad.status == "cache_revision_mismatch"
    assert ret_bad.encode_text("a cat jumps")[1] == "cache_miss"


def test_videoprism_top20_rerank_chronological_clause_order_and_curve() -> None:
    base1 = _unit_vec(11)
    base2 = _unit_vec(22)
    q_vec = ((base1 + base2) / np.linalg.norm(base1 + base2)).astype(np.float32)

    # vid_order has clause1 at [0..16] then clause2 at [16..32]
    # vid_reverse has clause2 at [0..16] then clause1 at [16..32]
    v_vecs = np.vstack([q_vec, q_vec, _unit_vec(33)]).astype(np.float32)
    sub_vecs = np.vstack([
        base1, base2,   # vid_order
        base2, base1,   # vid_reverse
    ]).astype(np.float32)
    sub_rows = [
        {"subclip_id": "v1_w0", "video_id": "vid_order", "clip_start_sec": 0.0, "clip_end_sec": 16.0},
        {"subclip_id": "v1_w1", "video_id": "vid_order", "clip_start_sec": 16.0, "clip_end_sec": 32.0},
        {"subclip_id": "v2_w0", "video_id": "vid_reverse", "clip_start_sec": 0.0, "clip_end_sec": 16.0},
        {"subclip_id": "v2_w1", "video_id": "vid_reverse", "clip_start_sec": 16.0, "clip_end_sec": 32.0},
    ]

    retriever = VideoPrismClipRetriever(
        video_vectors=v_vecs,
        video_ids=["vid_order", "vid_reverse", "vid_distractor"],
        subclip_vectors=sub_vecs,
        subclip_rows=sub_rows,
        query_cache_map={
            "clause1 then clause2": q_vec,
            "clause1": base1,
            "clause2": base2,
        },
    )
    sigs = retriever.rerank_top_candidates(
        "clause1 then clause2",
        ["clause1", "clause2"],
        ["vid_reverse", "vid_order", "vid_distractor"],
    )
    assert sigs["vid_order"].clause_order_score == 1.0
    assert sigs["vid_reverse"].clause_order_score == 0.0
    assert sigs["vid_order"].rerank_score > sigs["vid_reverse"].rerank_score

    curve = retriever.get_subclip_curve("vid_order", "clause1 then clause2", ["clause1", "clause2"])
    assert curve is not None
    centers, scores, clause_mat = curve
    assert centers.tolist() == [8.0, 24.0]
    assert clause_mat is not None and clause_mat.shape == (2, 2)
