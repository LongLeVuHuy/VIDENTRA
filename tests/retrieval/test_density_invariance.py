"""Unit tests verifying density-invariant Stage 2/4/6 retrieval, VideoPrism union rerank, and boundary expansion."""
from __future__ import annotations

import faiss
import numpy as np
import pytest

from videntra.indexing.augmented_vector import build_augmented_864d_embeddings, build_query_864d_vector
from videntra.localization.refine import expand_boundary_contrast
from videntra.retrieval.alignment import align_sequential_keyframe_clauses
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
from videntra.retrieval.videoprism_branch import (
    VideoPrismCandidateSignals,
    VideoPrismRetrievalResult,
)
from videntra.retrieval.visual import RankedMoment


def _make_dense_pipeline(
    raw_768: np.ndarray,
    aug_864: np.ndarray,
    kf_map: dict[int, dict],
    vid_kfs: dict[str, list[dict]],
    vid_durations: dict[str, float],
    **pipeline_kwargs,
) -> UnifiedMultimodalSearchPipeline:
    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False, **pipeline_kwargs)
    pipeline.all_embeddings = np.ascontiguousarray(raw_768, dtype=np.float32)
    pipeline.augmented_embeddings = np.ascontiguousarray(aug_864, dtype=np.float32)
    pipeline.keyframe_map = kf_map
    pipeline.video_keyframes = vid_kfs
    pipeline.video_durations = vid_durations
    pipeline.aug_index = faiss.IndexFlatIP(864)
    pipeline.aug_index.add(pipeline.augmented_embeddings)
    pipeline.index = faiss.IndexFlatIP(768)
    pipeline.index.add(pipeline.all_embeddings)
    return pipeline


def test_burst_dedup_preserves_distinct_anchors_and_backfills() -> None:
    kf_map = {
        0: {"video_id": "vid_burst", "pts_time": 10.00},
        1: {"video_id": "vid_burst", "pts_time": 10.20},
        2: {"video_id": "vid_burst", "pts_time": 10.40},
        3: {"video_id": "vid_burst", "pts_time": 10.50},
        4: {"video_id": "vid_burst", "pts_time": 11.20},
        5: {"video_id": "vid_other", "pts_time": 4.00},
        6: {"video_id": "vid_third", "pts_time": 8.00},
    }
    hits = [(i, 0.90 - 0.01 * i) for i in range(7)]

    deduped = UnifiedMultimodalSearchPipeline._deduplicate_burst_visual_hits(
        hits, kf_map, target_depth=4, min_pts_gap=0.55
    )
    assert [fid for fid, _ in deduped] == [0, 4, 5, 6]

    # When fewer distinct-gap hits exist than target_depth, skipped hits backfill in score order
    backfilled = UnifiedMultimodalSearchPipeline._deduplicate_burst_visual_hits(
        hits[:5], kf_map, target_depth=4, min_pts_gap=0.55
    )
    assert len(backfilled) == 4
    assert [fid for fid, _ in backfilled] == [0, 4, 1, 2]


def test_stage2_density_scale_and_dedup_keep_target_in_pool() -> None:
    # 50 keyframes across 2 videos (avg 25 kf/video -> density_scale = 2.5x, so search_depth=3 scales to 8)
    # First 5 frames are a tight 0.1s burst from vid_distractor outscoring vid_target; without
    # density scaling + burst dedup, search_depth=3 would miss vid_target completely.
    n = 50
    raw_768 = np.zeros((n, 768), dtype=np.float32)
    aug_864 = np.zeros((n, 864), dtype=np.float32)
    kf_map: dict[int, dict] = {}
    vid_kfs: dict[str, list[dict]] = {"vid_distractor": [], "vid_target": []}

    for i in range(45):
        score = 0.85 - 0.002 * i if i < 5 else 0.10
        raw_768[i, 0] = score
        aug_864[i, 0] = score
        t = 2.0 + 0.10 * i if i < 5 else 20.0 + float(i)
        kf_map[i] = {"video_id": "vid_distractor", "pts_time": t, "source_frame_index": i}
        vid_kfs["vid_distractor"].append({"faiss_id": i, "pts_time": t})

    for j in range(5):
        fid = 45 + j
        raw_768[fid, 0] = 0.70 - 0.01 * j
        aug_864[fid, 0] = 0.70 - 0.01 * j
        t = 5.0 + 2.0 * j
        kf_map[fid] = {"video_id": "vid_target", "pts_time": t, "source_frame_index": fid}
        vid_kfs["vid_target"].append({"faiss_id": fid, "pts_time": t})

    pipe = _make_dense_pipeline(
        raw_768,
        aug_864,
        kf_map,
        vid_kfs,
        {"vid_distractor": 150.0, "vid_target": 150.0},
        search_depth=3,
    )
    pipe.encode_texts = lambda texts: np.tile(np.eye(1, 768, dtype=np.float32), (len(texts), 1))  # type: ignore[method-assign]

    res = pipe.search_with_metadata(snapshot_id="test", query="person running", top_k=5)
    assert "vid_target" in res["diagnostics"]["visual_candidate_ids"]
    pipe.close()


def test_duration_normalized_eff_n_kf_avoids_dense_penalty() -> None:
    # A 200-keyframe video (n_kf=200) of real duration 150s with m_count=10 hits at s_vis=0.17 (< 0.225):
    # - With v_dur=150s: eff_n_kf = min(200, 150/2) = 75 -> m_count/eff_n_kf = 10/75 = 0.133 >= 0.12 (no penalty)
    # - With v_dur=400s: eff_n_kf = min(200, 400/2) = 200 -> m_count/eff_n_kf = 10/200 = 0.05 < 0.12 (0.5x penalty)
    n_kf = 200
    raw_768 = np.zeros((n_kf, 768), dtype=np.float32)
    kf_map: dict[int, dict] = {}
    vid_kfs: dict[str, list[dict]] = {"vid_dense": []}
    for k in range(n_kf):
        if k < 10:
            raw_768[k, 0] = 0.17
            t = 10.0 + 0.8 * k
        else:
            raw_768[k, 1] = 0.50
            t = 30.0 + 0.5 * (k - 10)
        kf_map[k] = {"video_id": "vid_dense", "pts_time": t, "source_frame_index": k}
        vid_kfs["vid_dense"].append({"faiss_id": k, "pts_time": t})
    aug_864 = np.hstack([raw_768, np.zeros((n_kf, 96), dtype=np.float32)])

    def _run_score(v_dur: float) -> float:
        pipe = _make_dense_pipeline(
            raw_768,
            aug_864,
            kf_map,
            vid_kfs,
            {"vid_dense": v_dur},
            search_depth=10,
            enable_videoprism=False,
        )
        pipe.encode_texts = lambda texts: np.tile(np.eye(1, 768, dtype=np.float32), (len(texts), 1))  # type: ignore[method-assign]
        res = pipe.search_with_metadata(snapshot_id="dens_bonus", query="solo scene", top_k=1)
        sc = float(res["ranked_videos"][0].ranking_score)
        pipe.close()
        return sc

    score_150s = _run_score(150.0)
    score_400s = _run_score(400.0)
    # 150s duration avoids the 0.5x density_bonus penalty that triggers at v_dur=400s
    assert score_150s > score_400s + 0.01


def test_videoprism_rerank_includes_union_candidates_outside_prerank_top20() -> None:
    raw_768 = np.zeros((3, 768), dtype=np.float32)
    raw_768[0, 0] = 0.50
    raw_768[1, 0] = 0.45
    raw_768[2, 0] = 0.30
    aug_864 = np.hstack([raw_768, np.zeros((3, 96), dtype=np.float32)])
    kf_map = {
        0: {"video_id": "vid_a", "pts_time": 5.0, "source_frame_index": 0},
        1: {"video_id": "vid_b", "pts_time": 5.0, "source_frame_index": 0},
        2: {"video_id": "vid_vp_star", "pts_time": 12.0, "source_frame_index": 0},
    }
    vid_kfs = {
        "vid_a": [{"faiss_id": 0, "pts_time": 5.0}],
        "vid_b": [{"faiss_id": 1, "pts_time": 5.0}],
        "vid_vp_star": [{"faiss_id": 2, "pts_time": 12.0}],
    }

    seen_rerank_ids: list[str] = []

    class _StubVPRetriever:
        available = True
        status = "ready"
        video_ids = ("vid_a", "vid_b", "vid_vp_star")

        def retrieve_parallel(self, *, query, clauses, top_n=80, allowed_video_ids=None):
            return VideoPrismRetrievalResult(
                status="ok",
                query_source="stub",
                ranked_videos=[("vid_vp_star", 0.32), ("vid_a", 0.20), ("vid_b", 0.18)],
                combined_scores={"vid_vp_star": 0.32, "vid_a": 0.20, "vid_b": 0.18},
                best_subclip_windows={"vid_vp_star": (8.0, 16.0, 0.32)},
            )

        def rerank_top_candidates(self, *, query, clauses, candidate_video_ids, retrieval_result=None):
            seen_rerank_ids.extend(candidate_video_ids)
            return {
                "vid_a": VideoPrismCandidateSignals(whole_cos=0.20, subclip_max_cos=0.20, vp_z=-0.5),
                "vid_b": VideoPrismCandidateSignals(whole_cos=0.18, subclip_max_cos=0.18, vp_z=-0.7),
                "vid_vp_star": VideoPrismCandidateSignals(whole_cos=0.35, subclip_max_cos=0.36, vp_z=3.0),
            }

        def get_subclip_curve(self, vid, query, sub_clauses):
            return None

    pipe = _make_dense_pipeline(
        raw_768,
        aug_864,
        kf_map,
        vid_kfs,
        {"vid_a": 30.0, "vid_b": 30.0, "vid_vp_star": 30.0},
        enable_videoprism=True,
        videoprism_retriever=_StubVPRetriever(),
        videoprism_union_k=10,
        videoprism_rerank_top_k=2,
        ltr_model_path=None,
    )
    pipe.encode_texts = lambda texts: np.tile(np.eye(1, 768, dtype=np.float32), (len(texts), 1))  # type: ignore[method-assign]

    res = pipe.search_with_metadata(snapshot_id="vp_union", query="person waving", top_k=3)
    assert "vid_vp_star" in seen_rerank_ids
    feats = res["diagnostics"]["candidate_ltr_features"]["vid_vp_star"]
    assert feats["vp_z"] == 3.0
    assert feats["vp_whole_cos"] == 0.35
    pipe.close()


def test_expand_boundary_contrast_dense_vs_sparse_bridge() -> None:
    # Dense 150s curve (n=200, step=0.75s) with a 3.0s (5 frames) low dip inside a [30s..60s] event:
    # must bridge across the < 4.0s dip instead of cutting off after 4 dense frames.
    n_dense = 200
    ts_dense = [0.75 * i for i in range(n_dense)]
    curve_dense = [0.10] * n_dense
    for i, t in enumerate(ts_dense):
        if 30.0 <= t <= 60.0:
            curve_dense[i] = 0.85
        if 48.0 <= t <= 51.0:
            curve_dense[i] = 0.20  # 5 consecutive dense frames (< 4.0s total)
    anchor_dense = min(range(n_dense), key=lambda i: abs(ts_dense[i] - 38.0))
    st_d, et_d = expand_boundary_contrast(ts_dense, curve_dense, v_dur=150.0, anchor_idx=anchor_dense)
    assert st_d <= 31.0 and et_d >= 59.0

    # Sparse 150s curve (n=30, step=5.0s -> max_consec=4): time gap > max_gap_sec stops expansion.
    ts_sparse = [5.0 * i for i in range(30)]
    curve_sparse = [0.10] * 30
    for i in range(4, 9):
        curve_sparse[i] = 0.85
    for i in range(15, 20):
        curve_sparse[i] = 0.85
    st_s, et_s = expand_boundary_contrast(ts_sparse, curve_sparse, v_dur=150.0, anchor_idx=6)
    assert et_s < 55.0


def test_spaced_adjacent_pairs_collapses_subsecond_bursts() -> None:
    burst_items = [
        (0.50, 9.0 + 0.3 * i, 11.0 + 0.3 * i, 10.0 + 0.3 * i, "visual")
        for i in range(4)
    ]
    assert UnifiedMultimodalSearchPipeline._count_spaced_adjacent_pairs(burst_items, min_spacing_sec=1.5) == 0

    spaced_items = [
        (0.50, 9.0 + 2.0 * i, 11.0 + 2.0 * i, 10.0 + 2.0 * i, "visual")
        for i in range(4)
    ]
    assert UnifiedMultimodalSearchPipeline._count_spaced_adjacent_pairs(spaced_items, min_spacing_sec=1.5) == 3


def test_clip_consensus_duration_mismatch_guard() -> None:
    emb_768 = np.zeros((10, 768), dtype=np.float32)
    emb_768[2:4, 0], emb_768[2:4, 1] = 0.90, 0.43589
    emb_768[0, 0], emb_768[0, 2] = 0.85, 0.52678
    emb_768[1, 0], emb_768[1, 2] = 0.70, 0.71414
    emb_768[4:, 0], emb_768[4:, 3] = 0.20, 0.97980

    p = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    p.all_embeddings = emb_768
    p.augmented_embeddings = np.pad(emb_768, ((0, 0), (0, 96)))
    p.video_keyframes = {
        vid: [{"faiss_id": idx * 2, "pts_time": 1.0}, {"faiss_id": idx * 2 + 1, "pts_time": 2.0}]
        for idx, vid in enumerate(["v_a5", "v_win", "v_2", "v_3", "v_4"])
    }
    p.video_kf_fids = {vid: [k["faiss_id"] for k in kfs] for vid, kfs in p.video_keyframes.items()}
    p.video_durations = {"v_a5": 150.0, "v_win": 10.0, "v_2": 10.0, "v_3": 10.0, "v_4": 10.0}

    q_768 = np.zeros((768,), dtype=np.float32)
    q_768[0] = 1.0
    cands = [
        RankedMoment("m_a5", "v_a5", 0.0, 2.0, 0, 0, 0, 1.0, ranking_score=0.520),
        RankedMoment("m_win", "v_win", 0.0, 2.0, 2, 2, 2, 1.0, ranking_score=0.500),
        RankedMoment("m_2", "v_2", 0.0, 2.0, 4, 4, 4, 1.0, ranking_score=0.200),
    ]
    _, _, stats_blocked = p._apply_clip_consensus_boost(cands, q_768, filters=None)
    assert stats_blocked["applied"] is False
    assert stats_blocked["reason"] == "duration_mismatch_guard"

    p.video_durations["v_a5"] = 15.0
    p.video_durations["v_win"] = 15.0
    _, _, stats_ok = p._apply_clip_consensus_boost(cands, q_768, filters=None)
    assert stats_ok["applied"] is True
    assert stats_ok["promoted_video_id"] == "v_win"
    p.close()


def test_rank1_864_sims_matches_direct_gemm_and_falls_back() -> None:
    rng = np.random.default_rng(123)
    raw = rng.standard_normal((24, 768)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    kf_map = {i: {"video_id": f"v{i // 4}", "pts_time": float(i % 4) * 2.0} for i in range(24)}
    objs = {
        f"v{v}": [{"class_name": "person", "pts_sec": 2.0, "confidence": 0.9, "bbox": (10.0, 10.0, 100.0, 200.0)}]
        for v in range(6)
    }
    durs = {f"v{v}": 10.0 for v in range(6)}
    aug, _, coco_idf = build_augmented_864d_embeddings(raw, kf_map, objs, durs)

    pipe = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipe.coco_idf_map = coco_idf
    pipe.publish_snapshot(all_embeddings=raw, augmented_embeddings=aug, keyframe_map=kf_map)
    assert pipe._ensure_rank1_864_cache() is True

    q7 = rng.standard_normal(768).astype(np.float32)
    q7 /= np.linalg.norm(q7)
    q8 = build_query_864d_vector(q7, "a person walking", coco_idf)
    s7, s8 = pipe._compute_768_and_864_sims(q7, q8)
    assert np.allclose(s7, raw @ q7, atol=1e-5)
    assert np.allclose(s8, aug @ q8, atol=1e-5)

    q7_batch = rng.standard_normal((3, 768)).astype(np.float32)
    q7_batch /= np.linalg.norm(q7_batch, axis=1, keepdims=True)
    q8_batch = np.vstack([build_query_864d_vector(q7_batch[k], "person running", coco_idf) for k in range(3)])
    s7_b, s8_b = pipe._compute_768_and_864_sims(q7_batch, q8_batch)
    assert np.allclose(s7_b, (raw @ q7_batch.T).T, atol=1e-5)
    assert np.allclose(s8_b, (aug @ q8_batch.T).T, atol=1e-5)

    aug_arbitrary = aug.copy()
    aug_arbitrary[:, :768] = rng.standard_normal((24, 768)).astype(np.float32)
    pipe.publish_snapshot(all_embeddings=raw, augmented_embeddings=aug_arbitrary)
    assert pipe._ensure_rank1_864_cache() is False
    _, s8_fb = pipe._compute_768_and_864_sims(q7, q8)
    assert np.allclose(s8_fb, aug_arbitrary @ q8, atol=1e-5)
    pipe.close()


def test_align_sequential_clauses_deque_and_vectorized_parity() -> None:
    rng = np.random.default_rng(99)
    k_v = 25
    ts = np.cumsum(rng.uniform(0.5, 4.0, size=k_v)).tolist()
    c2 = rng.random((2, k_v)).astype(np.float32)
    c2[1, 10] = c2[1, 11]  # Exact tie to verify earliest b tie-break
    dummy = np.zeros((k_v, 4), dtype=np.float32)
    dummy_c2 = [np.zeros(4, dtype=np.float32)] * 2

    def _ref_m2(ts_in: list[float], cc: np.ndarray, max_gap: float) -> tuple[float, tuple[float, float]]:
        best_s = -float("inf")
        best_p = (ts_in[0], ts_in[0])
        for a in range(len(ts_in)):
            for b in range(a, len(ts_in)):
                diff = ts_in[b] - ts_in[a]
                if 0.0 <= diff <= max_gap:
                    base = 0.5 * (float(cc[0, a]) + float(cc[1, b]))
                    sc = 0.92 * base if (len(ts_in) >= 2 and b == a) else base
                    if sc > best_s:
                        best_s = sc
                        best_p = (ts_in[a], ts_in[b])
        if best_s == -float("inf"):
            for a in range(len(ts_in)):
                for b in range(a, len(ts_in)):
                    base = 0.5 * (float(cc[0, a]) + float(cc[1, b]))
                    sc = 0.92 * base if (len(ts_in) >= 2 and b == a) else base
                    if sc > best_s:
                        best_s = sc
                        best_p = (ts_in[a], ts_in[b])
        return best_s, best_p

    for gap in (12.0, -1.0):
        sc_fast, pair_fast = align_sequential_keyframe_clauses(
            dummy, ts, dummy_c2, max_transition_sec=gap, precomputed_clause_curves=c2
        )
        sc_ref, pair_ref = _ref_m2(ts, c2, gap)
        assert sc_fast == pytest.approx(sc_ref, abs=1e-6)
        assert pair_fast == pair_ref

    c3 = rng.random((3, k_v)).astype(np.float32)
    dp = np.full((3, k_v), -np.inf, dtype=np.float32)
    parent = np.full((3, k_v), -1, dtype=np.int32)
    dp[0] = c3[0]
    for m in range(1, 3):
        for i in range(k_v):
            for j in range(i + 1):
                if 0.0 <= (ts[i] - ts[j]) <= 15.0 and dp[m - 1, j] != -np.inf:
                    step = 0.92 * c3[m, i] if j == i else c3[m, i]
                    if dp[m - 1, j] + step > dp[m, i]:
                        dp[m, i] = dp[m - 1, j] + step
                        parent[m, i] = j
    bi = int(np.argmax(dp[2]))
    ci = bi
    for m in (2, 1):
        ci = int(parent[m, ci])
    sc3_fast, pair3_fast = align_sequential_keyframe_clauses(
        dummy, ts, [np.zeros(4, dtype=np.float32)] * 3, max_transition_sec=15.0, precomputed_clause_curves=c3
    )
    assert sc3_fast == pytest.approx(float(dp[2, bi]) / 3.0, abs=1e-5)
    assert pair3_fast == (ts[ci], ts[bi])
