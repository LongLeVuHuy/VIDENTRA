"""Unit and integration tests for SigLIP 2 frame+clip branch and clip reserve pool."""
from collections import defaultdict
import numpy as np
import pytest

from videntra.retrieval.clip_branch import (
    SnapshotMismatchError,
    build_clip_branch_index,
    build_clip_reserve_pool,
    build_dual_prompt_query_vector,
    score_clip_branch,
    score_presentation_window_max,
)
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _pad_768(prefix: list[float]) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    v[: len(prefix)] = prefix
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class _DummyTokenizer:
    pass


class _DummyTextModel:
    pass


def _make_synthetic_pipeline(
    *,
    embeddings_768: np.ndarray,
    video_keyframes: dict[str, list[dict]],
    video_durations: dict[str, float],
    query_vec_768: np.ndarray,
    enable_clip_branch: bool = False,
    clip_reserve_k: int = 10,
    clip_window_size: int | None = None,
    enable_temporal_rerank: bool = False,
    temporal_rerank_mode: str = "mean",
    temporal_rerank_lambda: float = 0.0,
    temporal_rerank_top_k: int = 10,
    search_depth: int = 120,
) -> UnifiedMultimodalSearchPipeline:
    n_rows = embeddings_768.shape[0]
    aug_864 = np.zeros((n_rows, 864), dtype=np.float32)
    aug_864[:, :768] = embeddings_768
    keyframe_map = {}
    for vid, kfs in video_keyframes.items():
        for kf in kfs:
            keyframe_map[int(kf["faiss_id"])] = {
                "video_id": vid,
                "pts_time": float(kf["pts_time"]),
                "source_frame_index": int(kf.get("source_frame_index", kf.get("original_frame_index", 0))),
                "original_frame_index": int(kf["original_frame_index"]),
            }

    import faiss

    idx = faiss.IndexFlatIP(768)
    idx.add(np.ascontiguousarray(embeddings_768, dtype=np.float32))

    pipe = UnifiedMultimodalSearchPipeline(
        alpha_obj=0.25,
        search_depth=search_depth,
        use_disk_cache=False,
        lazy_load=True,
        ltr_model_path=None,
        enable_spatial_patches=False,
        enable_vlm_caption_lexical=False,
        enable_vlm_caption_dense=False,
        enable_temporal_rerank=enable_temporal_rerank,
        temporal_rerank_mode=temporal_rerank_mode,
        temporal_rerank_lambda=temporal_rerank_lambda,
        temporal_rerank_top_k=temporal_rerank_top_k,
        enable_clip_branch=enable_clip_branch,
        clip_reserve_k=clip_reserve_k,
        clip_window_size=clip_window_size,
        tokenizer=_DummyTokenizer(),
        text_model=_DummyTextModel(),
        index=idx,
        augmented_embeddings=aug_864,
        coco_idf_map={"_dummy": 1.0},
        keyframe_map=keyframe_map,
        video_keyframes=defaultdict(list, video_keyframes),
        video_durations=video_durations,
        video_objects=defaultdict(list),
    )
    pipe.all_embeddings = np.ascontiguousarray(embeddings_768, dtype=np.float32)
    pipe.aug_index = faiss.IndexFlatIP(864)
    pipe.aug_index.add(aug_864)
    for vid, kfs in video_keyframes.items():
        fids = [int(k["faiss_id"]) for k in kfs]
        pts = [float(k["pts_time"]) for k in kfs]
        pipe.video_kf_fids[vid] = fids
        pipe.video_kf_pts[vid] = pts
        pipe.video_kf_matrix[vid] = np.ascontiguousarray(aug_864[fids])

    q_norm = query_vec_768.astype(np.float32) / np.linalg.norm(query_vec_768)
    pipe.encode_texts = lambda texts: [q_norm.copy() for _ in texts]  # type: ignore[method-assign]
    return pipe


class TestClipBranchUnit:
    def test_dual_prompt_query_vector_and_guards(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        q = build_dual_prompt_query_vector(v1, v2)
        expected = np.array([0.75, 0.25, 0.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(q, expected, atol=1e-6)

        with pytest.raises(ValueError, match="non-finite"):
            build_dual_prompt_query_vector(np.array([np.nan, 1.0]), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="near-zero norm"):
            build_dual_prompt_query_vector(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="dimension mismatch"):
            build_dual_prompt_query_vector(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_index_build_guards_and_snapshot_mismatch(self):
        embs = np.array(
            [
                [3.0, 0.0, 0.0],       # vid_a f0 (unnormalized -> normalized to [1,0,0])
                [0.0, 4.0, 0.0],       # vid_a f1
                [np.nan, 1.0, 0.0],    # vid_bad f2 (non-finite)
                [0.0, 0.0, 0.0],       # vid_zero f3 (zero norm)
                [0.0, 0.0, 2.0],       # vid_b f4
            ],
            dtype=np.float32,
        )
        kfs = {
            "vid_a": [{"faiss_id": 0, "pts_time": 1.0}, {"faiss_id": 1, "pts_time": 2.0}],
            "vid_bad": [{"faiss_id": 2, "pts_time": 0.5}],
            "vid_zero": [{"faiss_id": 3, "pts_time": 0.5}],
            "vid_empty": [],
            "vid_b": [{"faiss_id": 4, "pts_time": 3.0}],
        }
        idx = build_clip_branch_index(embs, video_keyframes=kfs)
        assert idx.video_ids == ["vid_a", "vid_b"]
        assert set(idx.invalid_videos.keys()) == {"vid_bad", "vid_zero", "vid_empty"}
        np.testing.assert_allclose(
            idx.clip_mean_matrix[idx.video_to_pos["vid_a"]],
            _unit([1.0, 1.0, 0.0]),
            atol=1e-6,
        )
        idx.validate_snapshot(embs)
        with pytest.raises(SnapshotMismatchError):
            idx.validate_snapshot(embs[:4])
        with pytest.raises(SnapshotMismatchError):
            idx.validate_snapshot(embs, video_kf_fids={"vid_a": [0], "vid_b": [4]})

    def test_scoring_ties_and_dataset_filter(self):
        embs = np.array(
            [
                [1.0, 0.0, 0.0],   # msrvtt_b
                [1.0, 0.0, 0.0],   # msrvtt_a (exact tie with msrvtt_b -> msrvtt_a wins)
                [0.0, 1.0, 0.0],   # qvh_c
            ],
            dtype=np.float32,
        )
        kfs = {
            "msrvtt_b": [{"faiss_id": 0, "pts_time": 2.0}],
            "msrvtt_a": [{"faiss_id": 1, "pts_time": 1.0}],
            "qvh_c": [{"faiss_id": 2, "pts_time": 4.0}],
        }
        idx = build_clip_branch_index(embs, video_keyframes=kfs)
        q = np.array([1.0, 0.1, 0.0], dtype=np.float32)

        res_all = score_clip_branch(idx, q)
        assert res_all.branch_order == ["msrvtt_a", "msrvtt_b", "qvh_c"]
        assert res_all.rank_frame["msrvtt_a"] == 1
        assert res_all.rank_frame["msrvtt_b"] == 2
        assert res_all.rank_clip["msrvtt_a"] == 1
        assert pytest.approx(res_all.branch_scores["msrvtt_a"], rel=1e-6) == 2.0 / 61.0

        res_filtered = score_clip_branch(idx, q, valid_video_ids={"qvh_c", "msrvtt_b"})
        assert res_filtered.branch_order == ["msrvtt_b", "qvh_c"]
        assert res_filtered.rank_frame == {"msrvtt_b": 1, "qvh_c": 2}
        assert res_filtered.valid_gallery_count == 2

    def test_presentation_window_3_keyframe_scoring(self):
        frames = np.array(
            [
                _unit([0.0, 1.0, 0.0]),
                _unit([0.8, 0.6, 0.0]),
                _unit([1.0, 0.0, 0.0]),
                _unit([0.9, 0.1, 0.0]),
                _unit([0.0, 0.0, 1.0]),
            ],
            dtype=np.float32,
        )
        q = _unit([1.0, 0.0, 0.0])
        score, anchor_pos = score_presentation_window_max(frames, q, window_size=3)
        assert score > 0.9
        assert anchor_pos == 2  # frame with [1, 0, 0] inside winning [1,2,3] window

    def test_clip_reserve_pool_cases(self):
        with pytest.raises(ValueError, match="k >= 2"):
            build_clip_reserve_pool(["v1", "v2"], ["v3"], k=1)

        # Normal replacement at K-th slot
        res = build_clip_reserve_pool(
            a5_pool=["v1", "v2", "v3", "v4"],
            branch_order=["v1", "v2", "v5", "v4"],
            k=4,
        )
        assert res.clip_pool == ["v1", "v2", "v3", "v5"]
        assert res.protected == ["v1", "v2", "v3"]
        assert res.added_video == "v5"
        assert res.displaced_video == "v4"
        assert res.duplicate_skipped == 2
        assert res.branch_rank == 3
        assert res.fallback_reason is None
        assert len(res.clip_pool) == len(set(res.clip_pool)) <= 4

        # Tail matches existing K-th item
        same = build_clip_reserve_pool(
            a5_pool=["v1", "v2", "v3"],
            branch_order=["v1", "v3", "v4"],
            k=3,
        )
        assert same.clip_pool == ["v1", "v2", "v3"]
        assert same.added_video is None
        assert same.displaced_video is None
        assert same.fallback_reason == "tail_matches_a5_kth"

        # No eligible tail (all branch videos already in protected)
        missing = build_clip_reserve_pool(
            a5_pool=["v1", "v2", "v3"],
            branch_order=["v1", "v2"],
            k=3,
        )
        assert missing.clip_pool == ["v1", "v2", "v3"]
        assert missing.added_video is None
        assert missing.fallback_reason == "no_eligible_tail_candidate"


class TestClipBranchPipelineIntegration:
    def test_branch_off_matches_baseline_bit_for_bit(self):
        embs = np.vstack([
            _pad_768([0.95, 0.1]),
            _pad_768([0.85, 0.2]),
            _pad_768([0.75, 0.3]),
            _pad_768([0.65, 0.4]),
        ])
        kfs = {
            "msrvtt_v1": [{"faiss_id": 0, "pts_time": 1.0, "original_frame_index": 10}],
            "msrvtt_v2": [{"faiss_id": 1, "pts_time": 2.0, "original_frame_index": 20}],
            "msrvtt_v3": [{"faiss_id": 2, "pts_time": 3.0, "original_frame_index": 30}],
            "msrvtt_v4": [{"faiss_id": 3, "pts_time": 4.0, "original_frame_index": 40}],
        }
        durs = {vid: 10.0 for vid in kfs}
        q = _pad_768([1.0, 0.0])

        pipe_base = _make_synthetic_pipeline(
            embeddings_768=embs, video_keyframes=kfs, video_durations=durs, query_vec_768=q,
            enable_clip_branch=False,
        )
        out_base = pipe_base.search_with_metadata(snapshot_id="s1", query="dog running", top_k=3)

        pipe_default = _make_synthetic_pipeline(
            embeddings_768=embs, video_keyframes=kfs, video_durations=durs, query_vec_768=q,
        )
        out_default = pipe_default.search_with_metadata(snapshot_id="s1", query="dog running", top_k=3)

        assert out_base["ranked_moments"] == out_default["ranked_moments"]
        assert out_base["ranked_videos"] == out_default["ranked_videos"]
        assert out_default["diagnostics"]["clip_pool_ids"] == []

    def test_reserve_preserves_head_and_reranked_clip_only_wins_rank1(self):
        # 5 videos: v1..v4 are in A5 top 4; v_clip is ranked #1 by clip branch (both frame & clip mean)
        # while v1..v4 have lower cosine on 3rd dim or v_clip is outside A5 top_k=3 when search_depth is small.
        # Construct 6 videos where v1, v2, v3 have highest single-frame dot with q,
        # but v_clip has 3 consistent frames so clip_mean + 3-frame window score is highest,
        # while its single-frame score is slightly below v1..v3 so A5 puts v1, v2, v3 at Top-3.
        embs = np.vstack([
            _pad_768([0.92, 0.39, 0.0]),   # 0: msrvtt_v1 f0 (high single frame)
            _pad_768([0.0, 1.0, 0.0]),     # 1: msrvtt_v1 f1 (noise -> hurts clip mean & temporal pool)
            _pad_768([0.90, 0.43, 0.0]),   # 2: msrvtt_v2 f0
            _pad_768([0.0, 1.0, 0.0]),     # 3: msrvtt_v2 f1
            _pad_768([0.88, 0.47, 0.0]),   # 4: msrvtt_v3 f0
            _pad_768([0.0, 1.0, 0.0]),     # 5: msrvtt_v3 f1
            _pad_768([0.85, 0.0, 0.52]),   # 6: msrvtt_vclip f0 (all 3 frames strongly aligned!)
            _pad_768([0.86, 0.0, 0.51]),   # 7: msrvtt_vclip f1
            _pad_768([0.87, 0.0, 0.49]),   # 8: msrvtt_vclip f2
            _pad_768([0.99, 0.0, 0.0]),    # 9: qvh_other (filtered out by dataset_prefix)
        ])
        kfs = {
            "msrvtt_v1": [
                {"faiss_id": 0, "pts_time": 1.0, "source_frame_index": 10, "original_frame_index": 10},
                {"faiss_id": 1, "pts_time": 2.0, "source_frame_index": 20, "original_frame_index": 20},
            ],
            "msrvtt_v2": [
                {"faiss_id": 2, "pts_time": 1.0, "source_frame_index": 12, "original_frame_index": 12},
                {"faiss_id": 3, "pts_time": 2.0, "source_frame_index": 22, "original_frame_index": 22},
            ],
            "msrvtt_v3": [
                {"faiss_id": 4, "pts_time": 1.0, "source_frame_index": 14, "original_frame_index": 14},
                {"faiss_id": 5, "pts_time": 2.0, "source_frame_index": 24, "original_frame_index": 24},
            ],
            "msrvtt_vclip": [
                {"faiss_id": 6, "pts_time": 2.0, "source_frame_index": 20, "original_frame_index": 20},
                {"faiss_id": 7, "pts_time": 4.0, "source_frame_index": 40, "original_frame_index": 40},
                {"faiss_id": 8, "pts_time": 6.0, "source_frame_index": 60, "original_frame_index": 60},
            ],
            "qvh_other": [
                {"faiss_id": 9, "pts_time": 1.0, "source_frame_index": 5, "original_frame_index": 5},
            ],
        }
        durs = {vid: 10.0 for vid in kfs}
        q = _pad_768([1.0, 0.0, 0.0])

        # 1) Reserve-only: K=3 preserves A5 Top-1 & Top-2 (msrvtt_v1, msrvtt_v2), replaces #3 with msrvtt_vclip
        pipe_reserve = _make_synthetic_pipeline(
            embeddings_768=embs, video_keyframes=kfs, video_durations=durs, query_vec_768=q,
            enable_clip_branch=True, clip_reserve_k=3, clip_window_size=3,
            search_depth=4,
        )
        out_res = pipe_reserve.search_with_metadata(
            snapshot_id="s1",
            query="person playing guitar",
            top_k=3,
            filters={"dataset_prefix": "msrvtt_"},
        )
        diag_res = out_res["diagnostics"]
        assert diag_res["a5_pool_ids"] == ["msrvtt_v1", "msrvtt_v2", "msrvtt_v3"]
        assert diag_res["clip_pool_ids"] == ["msrvtt_v1", "msrvtt_v2", "msrvtt_vclip"]
        assert diag_res["reranker_input_ids"] == ["msrvtt_v1", "msrvtt_v2", "msrvtt_vclip"]
        assert diag_res["ranked_video_ids"] == ["msrvtt_v1", "msrvtt_v2", "msrvtt_vclip"]
        assert [m.video_id for m in out_res["ranked_videos"]] == ["msrvtt_v1", "msrvtt_v2", "msrvtt_vclip"]
        assert "qvh_other" not in diag_res["clip_pool_ids"]

        # 2) Temporal reranker enabled on clip_pool: msrvtt_vclip has highest clip mean & temporal pool -> wins rank 1!
        pipe_rerank = _make_synthetic_pipeline(
            embeddings_768=embs, video_keyframes=kfs, video_durations=durs, query_vec_768=q,
            enable_clip_branch=True, clip_reserve_k=3, clip_window_size=3,
            enable_temporal_rerank=True, temporal_rerank_mode="mean",
            temporal_rerank_lambda=2.0, temporal_rerank_top_k=3,
            search_depth=4,
        )
        out_rr = pipe_rerank.search_with_metadata(
            snapshot_id="s1",
            query="person playing guitar",
            top_k=3,
            filters={"dataset_prefix": "msrvtt_"},
        )
        top_moment = out_rr["ranked_moments"][0]
        top_video = out_rr["ranked_videos"][0]
        assert top_moment.video_id == "msrvtt_vclip"
        assert top_video.video_id == "msrvtt_vclip"
        assert out_rr["diagnostics"]["ranked_video_ids"][0] == "msrvtt_vclip"
        assert 0.0 <= top_moment.moment_start_seconds < top_moment.moment_end_seconds <= 10.0
        assert top_moment.representative_original_frame_index in (20, 40, 60)
        assert len(top_moment.evidence_frame_indices) >= 1


def test_clip_branch_snapshot_mutation_detection_and_invalidation():
    from videntra.retrieval.clip_branch import SnapshotMismatchError, build_clip_branch_index
    from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline

    embs = np.zeros((10, 4), dtype=np.float32)
    embs[:, 0] = 1.0
    kfs = {
        f"v{i}": [
            {"faiss_id": 2 * i, "pts_time": 1.0, "source_frame_index": 10},
            {"faiss_id": 2 * i + 1, "pts_time": 2.0, "source_frame_index": 20},
        ]
        for i in range(5)
    }
    idx = build_clip_branch_index(embs, video_keyframes=kfs)
    idx.validate_snapshot(embs, video_keyframes=kfs)

    # Tail embedding mutation (row >= 3) must be detected by full-bytes SHA-256
    mutated = embs.copy()
    mutated[9, 1] = 0.5
    with pytest.raises(SnapshotMismatchError, match="Snapshot vector content mismatch"):
        idx.validate_snapshot(mutated, video_keyframes=kfs)
    assert idx.matches(mutated) is False

    # PTS mutation must also be detected
    kfs_pts = {k: [dict(x) for x in v] for k, v in kfs.items()}
    kfs_pts["v4"][1]["pts_time"] = 99.0
    with pytest.raises(SnapshotMismatchError, match="Snapshot mapping/PTS mismatch"):
        idx.validate_snapshot(embs, video_keyframes=kfs_pts)

    # In-memory cache & explicit invalidation on pipeline
    pipe = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipe.all_embeddings = embs.copy()
    pipe.video_keyframes = kfs
    first = pipe._ensure_clip_branch_index()
    assert first is not None
    assert pipe._ensure_clip_branch_index() is first
    pipe.invalidate_clip_branch_index()
    second = pipe._ensure_clip_branch_index()
    assert second is not None and second is not first
