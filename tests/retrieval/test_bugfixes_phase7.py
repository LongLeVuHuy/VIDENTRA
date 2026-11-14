"""Phase 7.1 unit tests for Bug 1, Bug 2, Bug 6, and Bimodal Valley Multi-Window."""
from unittest.mock import MagicMock

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from videntra.api.search import SearchRequest, router
from videntra.retrieval.unified_pipeline import (
    DOMAIN_LEXICAL_SYNONYMS,
    UnifiedMultimodalSearchPipeline,
    _resolve_moment_frame_indices,
)


def _make_mock_pipeline(
    raw_768: np.ndarray,
    aug_864: np.ndarray,
    kf_map: dict[int, dict],
    vid_kfs: dict[str, list[dict]],
    durations: dict[str, float] | None = None,
    **pipeline_kwargs,
) -> UnifiedMultimodalSearchPipeline:
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend

    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False, **pipeline_kwargs)
    pipeline.query_decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(enabled=False),
        enable_disk_cache=False,
    )
    pipeline.all_embeddings = np.ascontiguousarray(raw_768, dtype=np.float32)
    pipeline.augmented_embeddings = np.ascontiguousarray(aug_864, dtype=np.float32)
    pipeline.keyframe_map = kf_map
    pipeline.video_keyframes = vid_kfs
    pipeline.video_durations = durations or {vid: 40.0 for vid in vid_kfs}
    pipeline.aug_index = faiss.IndexFlatIP(864)
    pipeline.aug_index.add(pipeline.augmented_embeddings)
    pipeline.index = faiss.IndexFlatIP(768)
    pipeline.index.add(pipeline.all_embeddings)
    return pipeline


def _constant_encoder(texts: list[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), 768), dtype=np.float32)
    vecs[:, 0] = 1.0
    return vecs


# ---------------------------------------------------------------------------
# Bug 1: _resolve_moment_frame_indices & Stage 6 source_frame_index
# ---------------------------------------------------------------------------

def test_resolve_moment_frame_indices_helper():
    kfs = [
        {"pts_time": 1.0, "source_frame_index": 0},
        {"pts_time": 4.0, "source_frame_index": 120},
        {"pts_time": 6.0, "source_frame_index": 180},
        {"pts_time": 8.0, "source_frame_index": 240},
        {"pts_time": 12.0, "source_frame_index": 360},
    ]
    s, e, r, ev = _resolve_moment_frame_indices(kfs, start_sec=3.5, end_sec=8.5, best_kf_pos=2)
    assert (s, e, r) == (120, 240, 180)
    assert ev == (120, 180, 240)

    # Empty window falls back to best_kf_pos (including source_frame_index=0)
    s0, e0, r0, ev0 = _resolve_moment_frame_indices(kfs, start_sec=50.0, end_sec=55.0, best_kf_pos=0)
    assert (s0, e0, r0, ev0) == (0, 0, 0, (0,))

    assert _resolve_moment_frame_indices([], 0.0, 1.0, 0) == (0, 0, 0, ())


def test_bug1_stage6_populates_source_frame_indices_for_primary_and_win2():
    # Bimodal video with non-sequential source_frame_index values
    pts_list = [2.0, 5.0, 8.0, 12.0, 18.0, 24.0, 27.0, 30.0]
    src_frames = [60, 150, 240, 360, 540, 720, 810, 900]
    sims = [0.20, 0.90, 0.25, 0.10, 0.10, 0.25, 0.85, 0.20]

    n = len(pts_list)
    raw_768 = np.zeros((n, 768), dtype=np.float32)
    aug_864 = np.zeros((n, 864), dtype=np.float32)
    raw_768[:, 0] = sims
    aug_864[:, 0] = sims

    kf_map = {}
    vid_kfs: list[dict] = []
    for fid, (pts, sf) in enumerate(zip(pts_list, src_frames)):
        kf_map[fid] = {"video_id": "vid_bimodal", "pts_time": pts, "source_frame_index": sf}
        vid_kfs.append({"faiss_id": fid, "pts_time": pts, "source_frame_index": sf})

    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, {"vid_bimodal": vid_kfs})
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    res = pipeline.search("person jumping on stage", top_k=5, ablation_config="A5")
    assert len(res) == 2
    primary, win2 = res[0], res[1]
    assert win2.moment_id.endswith("_win2")

    assert primary.representative_original_frame_index == 150
    assert primary.start_original_frame_index in src_frames
    assert primary.end_original_frame_index in src_frames
    assert primary.start_original_frame_index > 0
    assert len(primary.evidence_frame_indices) >= 1
    assert all(idx in src_frames for idx in primary.evidence_frame_indices)

    assert win2.representative_original_frame_index == 810
    assert win2.start_original_frame_index in src_frames
    assert win2.end_original_frame_index in src_frames
    assert len(win2.evidence_frame_indices) >= 1
    assert all(idx in src_frames for idx in win2.evidence_frame_indices)


# ---------------------------------------------------------------------------
# Bug 2: lookup_true_visual_sim & Stage 6 0.70*768d + 0.30*864d blend
# ---------------------------------------------------------------------------

def test_bug2_lookup_true_visual_sim_blends_768_and_864():
    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[:, 0] = [0.80, 0.40]
    aug_864[:, 0] = [0.20, 0.90]

    kf_map = {
        0: {"video_id": "vid_1", "pts_time": 5.0, "source_frame_index": 100},
        1: {"video_id": "vid_1", "pts_time": 20.0, "source_frame_index": 400},
    }
    vid_kfs = {
        "vid_1": [
            {"faiss_id": 0, "pts_time": 5.0, "source_frame_index": 100},
            {"faiss_id": 1, "pts_time": 20.0, "source_frame_index": 400},
        ]
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)

    q_768 = np.zeros((1, 768), dtype=np.float32)
    q_768[0, 0] = 1.0
    q_864 = np.zeros((864,), dtype=np.float32)
    q_864[0] = 1.0

    curve_cache: dict = {}
    sim_t5 = pipeline.lookup_true_visual_sim(
        "vid_1",
        ts=5.0,
        primary_q_864=q_864,
        primary_q_768=q_768,
        curve_cache=curve_cache,
    )
    expected_t5 = 0.70 * 0.80 + 0.30 * 0.20  # 0.62
    assert abs(sim_t5 - expected_t5) < 1e-5

    sim_t20 = pipeline.lookup_true_visual_sim(
        "vid_1",
        ts=20.0,
        primary_q_864=q_864,
        primary_q_768=q_768,
        curve_cache=curve_cache,
    )
    expected_t20 = 0.70 * 0.40 + 0.30 * 0.90  # 0.55
    assert abs(sim_t20 - expected_t20) < 1e-5
    assert sim_t5 > sim_t20


# ---------------------------------------------------------------------------
# Bug 6: search() filters, DOMAIN_LEXICAL_SYNONYMS, SearchRequest.rerank_with_relations
# ---------------------------------------------------------------------------

def test_bug6_search_filters_video_ids_prefix_and_duration():
    raw_768 = np.zeros((3, 768), dtype=np.float32)
    aug_864 = np.zeros((3, 864), dtype=np.float32)
    raw_768[:, 0] = [0.90, 0.80, 0.70]
    aug_864[:, 0] = [0.90, 0.80, 0.70]

    kf_map = {
        0: {"video_id": "msrvtt_001", "pts_time": 4.0, "source_frame_index": 10},
        1: {"video_id": "qvh_002", "pts_time": 4.0, "source_frame_index": 20},
        2: {"video_id": "avc_003", "pts_time": 4.0, "source_frame_index": 30},
    }
    vid_kfs = {
        "msrvtt_001": [{"faiss_id": 0, "pts_time": 4.0, "source_frame_index": 10}],
        "qvh_002": [{"faiss_id": 1, "pts_time": 4.0, "source_frame_index": 20}],
        "avc_003": [{"faiss_id": 2, "pts_time": 4.0, "source_frame_index": 30}],
    }
    durations = {"msrvtt_001": 12.0, "qvh_002": 150.0, "avc_003": 60.0}
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs, durations=durations)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    res_ids = pipeline.search("person walking", top_k=5, filters={"video_ids": ["avc_003"]})
    assert [m.video_id for m in res_ids] == ["avc_003"]

    res_prefix = pipeline.search("person walking", top_k=5, filters={"dataset_prefix": "qvh_"})
    assert [m.video_id for m in res_prefix] == ["qvh_002"]

    res_dur = pipeline.search(
        "person walking",
        top_k=5,
        mode="fast",
        filters={"min_duration": 30.0, "max_duration": 100.0},
    )
    assert [m.video_id for m in res_dur] == ["avc_003"]


def test_bug6_domain_lexical_synonyms_config_driven():
    assert "letterman" in DOMAIN_LEXICAL_SYNONYMS
    assert "news" in DOMAIN_LEXICAL_SYNONYMS
    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)

    fts_let, _, hi_let, _, _ = pipeline.prepare_fts5_query_and_tokens("David Letterman monologue")
    assert '"lateshow"' in fts_let
    assert "lateshow" in hi_let

    fts_news, _, hi_news, _, _ = pipeline.prepare_fts5_query_and_tokens("breaking news channel broadcast")
    assert '"cnn"' in fts_news and '"fox"' in fts_news
    assert {"cnn", "fox"}.issubset(hi_news)

    fts_io3, _, hi_io3, _, _ = pipeline.prepare_fts5_query_and_tokens("reaction with IO3 solution")
    assert '"hio"' in fts_io3
    assert "iodide" in hi_io3


def test_bug6_search_request_rerank_with_relations_forwarded():
    req_default = SearchRequest(query="a man next to a car")
    assert req_default.rerank_with_relations is False

    app = FastAPI()
    app.include_router(router)
    mock_pipeline = MagicMock()
    mock_pipeline.search_with_metadata.return_value = {
        "moments": [],
        "relation_assessments": [],
        "diagnostics": {"lexical_status": "complete"},
        "timings": {},
    }
    app.state.pipeline = mock_pipeline
    app.state.snapshot_id = "test_snap"

    client = TestClient(app)
    resp = client.post(
        "/v1/search",
        json={
            "query": "a man next to a car",
            "mode": "accurate",
            "verify_relations": True,
            "rerank_with_relations": True,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_pipeline.search_with_metadata.call_args.kwargs
    assert call_kwargs["rerank_with_relations"] is True


# ---------------------------------------------------------------------------
# Bimodal Valley Multi-Window Proposal
# ---------------------------------------------------------------------------

def test_bimodal_valley_suppresses_flat_curve_and_scales_bimodal_win2():
    # Case 1: Flat / plateau curve without a deep valley between primary window and second peak
    pts_flat = [2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    sims_flat = [0.75, 0.90, 0.85, 0.83, 0.82, 0.84, 0.78]
    n = len(pts_flat)
    raw_flat = np.zeros((n, 768), dtype=np.float32)
    aug_flat = np.zeros((n, 864), dtype=np.float32)
    raw_flat[:, 0] = sims_flat
    aug_flat[:, 0] = sims_flat
    kf_map_flat = {
        i: {"video_id": "vid_flat", "pts_time": p, "source_frame_index": i * 30}
        for i, p in enumerate(pts_flat)
    }
    vid_kfs_flat = {
        "vid_flat": [
            {"faiss_id": i, "pts_time": p, "source_frame_index": i * 30}
            for i, p in enumerate(pts_flat)
        ]
    }
    p_flat = _make_mock_pipeline(raw_flat, aug_flat, kf_map_flat, vid_kfs_flat)
    p_flat.encode_texts = _constant_encoder  # type: ignore[method-assign]
    res_flat = p_flat.search("person talking", top_k=5, ablation_config="A5")
    assert len(res_flat) == 1  # No _win2 emitted when valley condition fails

    # Case 2: Clear bimodal curve with deep valley -> _win2 emitted with 0.85 * (s_max2 / s_max1)^2
    pts_bi = [2.0, 5.0, 8.0, 12.0, 18.0, 24.0, 27.0, 30.0]
    sims_bi = [0.20, 0.90, 0.25, 0.10, 0.10, 0.25, 0.81, 0.20]
    m = len(pts_bi)
    raw_bi = np.zeros((m, 768), dtype=np.float32)
    aug_bi = np.zeros((m, 864), dtype=np.float32)
    raw_bi[:, 0] = sims_bi
    aug_bi[:, 0] = sims_bi
    kf_map_bi = {
        i: {"video_id": "vid_bi", "pts_time": p, "source_frame_index": i * 30}
        for i, p in enumerate(pts_bi)
    }
    vid_kfs_bi = {
        "vid_bi": [
            {"faiss_id": i, "pts_time": p, "source_frame_index": i * 30}
            for i, p in enumerate(pts_bi)
        ]
    }
    p_bi = _make_mock_pipeline(raw_bi, aug_bi, kf_map_bi, vid_kfs_bi)
    p_bi.encode_texts = _constant_encoder  # type: ignore[method-assign]
    res_bi = p_bi.search("person talking", top_k=5, ablation_config="A5")
    assert len(res_bi) == 2
    primary, win2 = res_bi[0], res_bi[1]
    expected_win2_score = round(primary.ranking_score * 0.85 * ((0.81 / 0.90) ** 2), 5)
    assert abs(win2.ranking_score - expected_win2_score) < 1e-4


# ---------------------------------------------------------------------------
# Phase 7.2 / 7.3 / 7.4 Integration Tests
# ---------------------------------------------------------------------------

def test_phase72_modality_routing_fts_and_verify_relations():
    import sqlite3
    from videntra.retrieval.unified_pipeline import _build_routed_fts5_queries

    raw_768 = np.zeros((4, 768), dtype=np.float32)
    aug_864 = np.zeros((4, 864), dtype=np.float32)
    raw_768[:, 0] = [0.85, 0.80, 0.75, 0.70]
    aug_864[:, 0] = [0.85, 0.80, 0.75, 0.70]

    kf_map = {
        0: {"video_id": "vid_r", "pts_time": 2.0, "source_frame_index": 30},
        1: {"video_id": "vid_r", "pts_time": 6.0, "source_frame_index": 90},
        2: {"video_id": "vid_r", "pts_time": 12.0, "source_frame_index": 180},
        3: {"video_id": "vid_r", "pts_time": 18.0, "source_frame_index": 270},
    }
    vid_kfs = {
        "vid_r": [
            {"faiss_id": 0, "pts_time": 2.0, "source_frame_index": 30},
            {"faiss_id": 1, "pts_time": 6.0, "source_frame_index": 90},
            {"faiss_id": 2, "pts_time": 12.0, "source_frame_index": 180},
            {"faiss_id": 3, "pts_time": 18.0, "source_frame_index": 270},
        ]
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE fts_ocr USING fts5(observation_id, video_id, timestamp_ms, normalized_text)")
    conn.execute("CREATE VIRTUAL TABLE fts_asr USING fts5(segment_id, video_id, start_ms, end_ms, normalized_text)")
    conn.execute("INSERT INTO fts_ocr VALUES ('o1', 'vid_r', 2000, 'welcome banner')")
    conn.execute("INSERT INTO fts_asr VALUES ('a1', 'vid_r', 5000, 7000, 'hello everyone')")
    pipeline.db_conn = conn

    out = pipeline.search_with_metadata(
        snapshot_id="test_snap",
        verify_relations=True,
        query='person wearing jacket says "hello everyone" and then walks away',
        top_k=3,
    )
    assert pipeline.last_routing_plan is not None
    assert pipeline.last_routing_plan.has_temporal_constraints is True
    assert len(out["moments"]) >= 1
    assert len(out["relation_assessments"]) == len(out["moments"])

    # Fallback mode preserves Phase 6 FTS5 parity (no cue stripping)
    assert pipeline.last_routing_plan.routing_mode == "fallback"
    fts_ocr_fb, fts_asr_fb = _build_routed_fts5_queries('"says" OR "wearing" OR "banner"', pipeline.last_routing_plan)
    assert fts_ocr_fb == '"says" OR "wearing" OR "banner"'
    assert fts_asr_fb == '"says" OR "wearing" OR "banner"'

    # llm_compact mode activates modality-specific FTS5 cue filtering
    from dataclasses import replace

    llm_plan = replace(pipeline.last_routing_plan, routing_mode="llm_compact")
    fts_ocr, fts_asr = _build_routed_fts5_queries('"says" OR "wearing" OR "banner"', llm_plan)
    assert '"says"' not in fts_ocr
    assert '"wearing"' not in fts_asr
    pipeline.close()


def test_phase73_ltr_fallback_and_active_scoring(tmp_path):
    import json
    from videntra.ranking.features import LTR_FEATURE_NAMES

    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[:, 0] = [0.80, 0.75]
    aug_864[:, 0] = [0.80, 0.75]

    kf_map = {
        0: {"video_id": "vid_1", "pts_time": 5.0, "source_frame_index": 10},
        1: {"video_id": "vid_2", "pts_time": 5.0, "source_frame_index": 20},
    }
    vid_kfs = {
        "vid_1": [{"faiss_id": 0, "pts_time": 5.0, "source_frame_index": 10}],
        "vid_2": [{"faiss_id": 1, "pts_time": 5.0, "source_frame_index": 20}],
    }
    pipeline = _make_mock_pipeline(
        raw_768,
        aug_864,
        kf_map,
        vid_kfs,
        ltr_model_path=tmp_path / "missing_ltr.json",
    )
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    # Fallback mode (no LTR model loaded)
    assert pipeline.ltr_ranker.is_loaded is False
    res_fallback = pipeline.search("person running", top_k=2)
    assert [m.video_id for m in res_fallback] == ["vid_1", "vid_2"]

    # Active LTR mode with custom weights
    model_file = tmp_path / "ltr.json"
    weights = [0.0] * len(LTR_FEATURE_NAMES)
    weights[0] = 1.0  # s_vis_blend_max
    model_file.write_text(
        json.dumps({"model_type": "pairwise_logistic", "feature_names": list(LTR_FEATURE_NAMES), "weights": weights}),
        encoding="utf-8",
    )
    pipeline.ltr_ranker.load(model_file)
    assert pipeline.ltr_ranker.is_loaded is True
    res_active = pipeline.search("person running", top_k=2)
    assert len(res_active) == 2
    assert res_active[0].ranking_score != res_fallback[0].ranking_score


def test_phase74_optional_qat_and_e5_hooks(tmp_path):
    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    kf_map = {
        0: {"video_id": "vid_q", "pts_time": 2.0, "source_frame_index": 10},
        1: {"video_id": "vid_q", "pts_time": 6.0, "source_frame_index": 20},
    }
    vid_kfs = {
        "vid_q": [
            {"faiss_id": 0, "pts_time": 2.0, "source_frame_index": 10},
            {"faiss_id": 1, "pts_time": 6.0, "source_frame_index": 20},
        ]
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)

    # Missing artifacts on disk: clean no-op
    status = pipeline.load_optional_phase74_artifacts(tmp_path)
    assert status == {"spatial_patches": False, "e5_text_index": False}
    assert pipeline._score_late_interaction_window("vid_q", 0.0, 10.0, np.ones(768, dtype=np.float32)) == 0.0
    assert pipeline._search_e5_text_index(np.ones(384, dtype=np.float32)) == []

    # Present spatial_patches_fp16.npy: loads and scores MaxSim
    patches = np.zeros((2, 9, 768), dtype=np.float16)
    patches[0, 3, 0] = 1.0
    np.save(tmp_path / "spatial_patches_fp16.npy", patches)
    status2 = pipeline.load_optional_phase74_artifacts(tmp_path)
    assert status2["spatial_patches"] is True
    q = np.zeros(768, dtype=np.float32)
    q[0] = 1.0
    maxsim = pipeline._score_late_interaction_window("vid_q", 0.0, 10.0, q)
    assert abs(maxsim - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Audit Fixes: Vấn đề 1, 2, 3, 4, 6
# ---------------------------------------------------------------------------

def test_issue6_fallback_parity_fts_and_seq_weight():
    from videntra.query.rules import RuleBasedQueryParser
    from videntra.query.schemas import ModalityRoutingPlan
    from videntra.retrieval.unified_pipeline import _build_routed_fts5_queries

    q = 'person wearing jacket says "hello" and then walks away'
    graph = RuleBasedQueryParser().parse(q)
    base = '"says" OR "wearing" OR "jacket"'

    for mode in ("fallback", "fast_rule", "tri_modal_broadcast"):
        plan = ModalityRoutingPlan(
            original_query=q,
            query_length_tier="medium",
            visual_clauses=(q,),
            ocr_phrases=(),
            asr_phrases=(),
            object_requirements=(),
            has_explicit_ocr_intent=False,
            has_explicit_asr_intent=False,
            has_spatial_constraints=False,
            has_temporal_constraints=True,
            query_graph=graph,
            routing_mode=mode,
        )
        assert _build_routed_fts5_queries(base, plan) == (base, base)

    llm_plan = ModalityRoutingPlan(
        original_query=q,
        query_length_tier="medium",
        visual_clauses=(q,),
        ocr_phrases=(),
        asr_phrases=(),
        object_requirements=(),
        has_explicit_ocr_intent=False,
        has_explicit_asr_intent=False,
        has_spatial_constraints=False,
        has_temporal_constraints=True,
        query_graph=graph,
        routing_mode="llm_compact",
    )
    fts_ocr, fts_asr = _build_routed_fts5_queries(base, llm_plan)
    assert '"says"' not in fts_ocr
    assert '"wearing"' not in fts_asr


def test_issue2_final_results_sorted_and_smax1_in_window():
    # vid_bi is bimodal (0.90 at t=5s, 0.80 at t=27s -> win2_score ~ 0.67 * vid_bi.score).
    # vid_mid has score 0.82 -> must rank 2nd ahead of vid_bi_win2 after sorting!
    pts_bi = [2.0, 5.0, 8.0, 12.0, 18.0, 24.0, 27.0, 30.0]
    sims_bi = [0.20, 0.90, 0.25, 0.10, 0.10, 0.25, 0.80, 0.20]
    n_bi = len(pts_bi)
    raw_768 = np.zeros((n_bi + 1, 768), dtype=np.float32)
    aug_864 = np.zeros((n_bi + 1, 864), dtype=np.float32)
    raw_768[:n_bi, 0] = sims_bi
    aug_864[:n_bi, 0] = sims_bi
    raw_768[n_bi, 0] = 0.82
    aug_864[n_bi, 0] = 0.82

    kf_map = {
        i: {"video_id": "vid_bi", "pts_time": p, "source_frame_index": i * 30}
        for i, p in enumerate(pts_bi)
    }
    kf_map[n_bi] = {"video_id": "vid_mid", "pts_time": 5.0, "source_frame_index": 30}
    vid_kfs = {
        "vid_bi": [
            {"faiss_id": i, "pts_time": p, "source_frame_index": i * 30}
            for i, p in enumerate(pts_bi)
        ],
        "vid_mid": [{"faiss_id": n_bi, "pts_time": 5.0, "source_frame_index": 30}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    res = pipeline.search("person jumping", top_k=5, ablation_config="A5", ranking_mode="diverse_video")
    scores = [m.ranking_score for m in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0].video_id == "vid_bi" and not res[0].moment_id.endswith("_win2")
    assert res[1].video_id == "vid_mid"
    assert res[2].moment_id.endswith("_win2")


def test_issue1_train_inference_feature_parity_in_diagnostics():
    import sqlite3
    from videntra.ranking.features import LTR_FEATURE_NAMES

    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[:, 0] = [0.60, 0.50]
    aug_864[:, 0] = [0.60, 0.50]
    kf_map = {
        0: {"video_id": "vid_a", "pts_time": 4.0, "source_frame_index": 10},
        1: {"video_id": "vid_b", "pts_time": 4.0, "source_frame_index": 20},
    }
    vid_kfs = {
        "vid_a": [{"faiss_id": 0, "pts_time": 4.0, "source_frame_index": 10}],
        "vid_b": [{"faiss_id": 1, "pts_time": 4.0, "source_frame_index": 20}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE fts_ocr USING fts5(observation_id, video_id, timestamp_ms, normalized_text)")
    conn.execute("CREATE VIRTUAL TABLE fts_asr USING fts5(segment_id, video_id, start_ms, end_ms, normalized_text)")
    # Only 1 of 2 high-spec tokens ("rocket" matched, "launch" missing)
    conn.execute("INSERT INTO fts_ocr VALUES ('o1', 'vid_a', 4000, 'rocket')")
    pipeline.db_conn = conn

    pipeline.search("rocket launch", top_k=5, ablation_config="A5")
    cand_feats = pipeline.last_diagnostics.get("candidate_ltr_features", {})
    assert "vid_a" in cand_feats and "vid_b" in cand_feats
    fa = cand_feats["vid_a"]
    for name in LTR_FEATURE_NAMES:
        assert name in fa
    # s_vis_top3_mean must reflect raw visual_sim (~0.59), NOT Stage 5 total score (which includes RRF+lex_boost)
    assert 0.55 < fa["s_vis_blend_max"] <= 0.61
    assert abs(fa["s_vis_top3_mean"] - fa["s_vis_blend_max"]) < 1e-6
    assert abs(fa["s_vis_top5_mean"] - fa["s_vis_blend_max"]) < 1e-6
    assert fa["high_spec_token_coverage"] < 1.0
    assert fa["anchor_lexical_match"] == 0.0
    pipeline.close()


def test_issue3_dense_e5_wired_into_stage3_with_strict_asr_gate():
    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[:, 0] = [0.80, 0.20]
    aug_864[:, 0] = [0.80, 0.20]
    kf_map = {
        0: {"video_id": "vid_vis", "pts_time": 4.0, "source_frame_index": 10},
        1: {"video_id": "vid_e5_asr", "pts_time": 4.0, "source_frame_index": 20},
    }
    vid_kfs = {
        "vid_vis": [{"faiss_id": 0, "pts_time": 4.0, "source_frame_index": 10}],
        "vid_e5_asr": [{"faiss_id": 1, "pts_time": 4.0, "source_frame_index": 20}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]

    e5_idx = faiss.IndexFlatIP(384)
    v_high = np.zeros((1, 384), dtype=np.float32)
    v_high[0, 0] = 0.90  # >= 0.835
    v_low = np.zeros((1, 384), dtype=np.float32)
    v_low[0, 0] = 0.70  # < 0.835
    e5_idx.add(np.vstack([v_high, v_low]))
    pipeline.e5_text_index = e5_idx
    pipeline.e5_text_manifest = [
        {"faiss_id": 0, "source_type": "ocr", "video_id": "vid_vis", "start_ms": 4000, "end_ms": 4000, "raw_text": "quantum physics"},
        {"faiss_id": 1, "source_type": "asr", "video_id": "vid_e5_asr", "start_ms": 4000, "end_ms": 6000, "raw_text": "unrelated speech"},
    ]
    q_e5 = np.zeros(384, dtype=np.float32)
    q_e5[0] = 1.0
    pipeline.encode_e5_query = lambda q: q_e5  # type: ignore[method-assign]

    res = pipeline.search("person explaining physics on board", top_k=5, ablation_config="A5")
    ocr_cands = pipeline.last_diagnostics["ocr_candidates"]
    asr_cands = pipeline.last_diagnostics["asr_candidates"]
    assert any(c.get("is_dense_e5") and c["video_id"] == "vid_vis" for c in ocr_cands)
    assert not any(c["video_id"] == "vid_e5_asr" for c in asr_cands)
    assert res[0].video_id == "vid_vis"


def test_issue4_temporal_context_and_late_interaction_wired(tmp_path):
    from videntra.indexing.augmented_vector import compute_multiscale_temporal_context

    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[:, 0] = [0.80, 0.60]
    aug_864[:, 0] = [0.80, 0.60]
    kf_map = {
        0: {"video_id": "vid_t", "pts_time": 2.0, "source_frame_index": 10},
        1: {"video_id": "vid_t", "pts_time": 4.0, "source_frame_index": 20},
    }
    vid_kfs = {
        "vid_t": [
            {"faiss_id": 0, "pts_time": 2.0, "source_frame_index": 10},
            {"faiss_id": 1, "pts_time": 4.0, "source_frame_index": 20},
        ]
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.encode_texts = _constant_encoder  # type: ignore[method-assign]
    pipeline.temporal_context_768 = compute_multiscale_temporal_context(raw_768, kf_map)

    pipeline.search("person running", top_k=2, ablation_config="A5")
    feats_ctx = pipeline.last_diagnostics["candidate_ltr_features"]["vid_t"]
    assert feats_ctx["s_late_interaction"] > 0.0

    # When spatial_patches is present, _score_late_interaction_window takes precedence
    patches = np.zeros((2, 9, 768), dtype=np.float16)
    patches[0, 2, 0] = 0.75
    patches[0, 2, 1] = float(np.sqrt(1.0 - 0.75**2))
    np.save(tmp_path / "spatial_patches_fp16.npy", patches)
    pipeline.load_optional_phase74_artifacts(tmp_path)
    pipeline.search("person running", top_k=2, ablation_config="A5")
    feats_patch = pipeline.last_diagnostics["candidate_ltr_features"]["vid_t"]
    assert abs(feats_patch["s_late_interaction"] - 0.75) < 1e-3
