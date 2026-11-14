"""Unit tests for Phần III research modules (CM-GD, SE-Viterbi, PQ-VCA) and LTR Monotonic Sign Projection."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_e5_text_index import build_structured_event_synopses
from scripts.kaggle_multimodal_offline_worker import extract_florence2_keyframe_captions
from scripts.train_ltr_ranker_cv import (
    NEGATIVE_MONOTONIC_FEATURES,
    POSITIVE_MONOTONIC_FEATURES,
    train_pairwise_logistic_l2,
)
from videntra.ranking.features import LTR_FEATURE_NAMES
from videntra.retrieval.alignment import (
    ViterbiLocalizationResult,
    compute_query_adaptive_csls,
    shot_aware_graph_diffusion,
    viterbi_subevent_localization,
)


def test_shot_aware_graph_diffusion_respects_shot_boundary():
    # 4 keyframes: frames 0,1,2 are in shot A (similar embeddings); frame 3 is shot B (orthogonal)
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.2, 0.0, 0.0],
            [0.95, 0.3, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ts = [0.0, 2.0, 4.0, 5.0]
    vis_scores = [0.2, 0.2, 0.2, 0.2]
    asr_scores = [0.0, 1.0, 0.0, 0.0]

    diffused = shot_aware_graph_diffusion(
        emb,
        ts,
        vis_scores,
        asr_scores=asr_scores,
        eta_asr=0.5,
        sigma_t=4.0,
        shot_sim_threshold=0.65,
        alpha_d=0.4,
    )
    assert diffused.shape == (4,)
    # Frame 0 and 2 (same shot as frame 1) receive diffused ASR boost
    assert diffused[0] > 0.2
    assert diffused[2] > 0.2
    # Frame 3 (hard shot cut) must NOT receive score leakage from frame 1
    assert diffused[3] == pytest.approx(0.2, abs=1e-6)

    # Edge cases: empty & single keyframe
    assert shot_aware_graph_diffusion(np.zeros((0, 4)), [], []).size == 0
    single = shot_aware_graph_diffusion(np.ones((1, 4)), [1.0], [0.5])
    assert single.shape == (1,) and single[0] == pytest.approx(0.5)


def test_compute_query_adaptive_csls_suppresses_local_hub():
    # Query points along dim 0
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Candidate 0: specific match to q; Candidate 1: generic hub aligned with all reference queries
    cands = np.array(
        [
            [0.80, 0.60, 0.0],
            [0.82, 0.0, 0.57],
        ],
        dtype=np.float32,
    )
    # Reference queries are all strongly aligned with dim 2 (making Candidate 1 a hub)
    ref_queries = np.array(
        [
            [0.2, 0.0, 0.98],
            [0.1, 0.0, 0.99],
            [0.3, 0.0, 0.95],
        ],
        dtype=np.float32,
    )

    csls_scores = compute_query_adaptive_csls(
        q, cands, reference_query_vecs=ref_queries, k_neighbors=2
    )
    assert csls_scores.shape == (2,)
    # Raw cosine had cand 1 > cand 0 (0.82 vs 0.80), but CSLS suppresses hub cand 1
    assert csls_scores[0] > csls_scores[1]

    # Local proxy mode (reference_query_vecs=None) & empty mode
    local_csls = compute_query_adaptive_csls(q, cands, k_neighbors=2)
    assert local_csls.shape == (2,)
    assert compute_query_adaptive_csls(q, np.zeros((0, 3))).size == 0


def test_viterbi_subevent_localization_multimodal_dp():
    ts = [0.0, 3.0, 6.0, 10.0, 14.0, 20.0]
    # Event 1: "screen shows ROUND 1" -> weak visual, strong OCR at t=6.0 (idx 2)
    # Event 2: "fighter kicks" -> strong visual at t=14.0 (idx 4)
    vis_curves = [
        [0.35, 0.30, 0.25, 0.20, 0.15, 0.10],
        [0.10, 0.15, 0.20, 0.25, 0.85, 0.20],
    ]
    ocr_curves = [
        [0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    asr_curves = [
        [0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.10, 0.0],
    ]

    res = viterbi_subevent_localization(
        ts,
        vis_curves,
        event_ocr_curves=ocr_curves,
        event_asr_curves=asr_curves,
        event_weights=[(0.5, 1.0, 0.3), (1.0, 0.2, 0.3)],
        max_gap_sec=15.0,
    )
    assert isinstance(res, ViterbiLocalizationResult)
    assert res.selected_indices == (2, 4)
    assert res.start_sec == 6.0
    assert res.end_sec == 14.0
    assert res.best_score > 0.0

    # Empty & 1-event edge cases
    assert viterbi_subevent_localization([], []) is None
    one = viterbi_subevent_localization([1.0, 5.0], [[0.2, 0.8]])
    assert one is not None and one.selected_indices == (1,)


def test_pq_vca_mode_a_build_structured_event_synopses(tmp_path: Path):
    keyframes = [
        {"keyframe_id": "kf_1", "video_id": "v1", "timestamp_ms": 5000},
        {"keyframe_id": "kf_2", "video_id": "v1", "timestamp_ms": 20000},
    ]
    objects = [
        {
            "video_id": "v1",
            "timestamp_ms": 4000,
            "class_name": "person",
            "bbox_x1": 10.0,
            "bbox_x2": 50.0,
            "image_width": 640,
            "confidence": 0.9,
        },
        {
            "video_id": "v1",
            "timestamp_ms": 6000,
            "class_name": "car",
            "bbox_x1": 500.0,
            "bbox_x2": 620.0,
            "image_width": 640,
            "confidence": 0.8,
        },
        {
            "video_id": "v1",
            "timestamp_ms": 5500,
            "class_name": "dog",
            "bbox_x1": 100.0,
            "bbox_x2": 200.0,
            "image_width": 640,
            "confidence": 0.10,  # Filtered out (< 0.25)
        },
    ]
    ocr = [{"video_id": "v1", "timestamp_ms": 5200, "raw_text": "ROUND 1"}]
    asr = [{"video_id": "v1", "start_ms": 3000, "end_ms": 7000, "raw_text": "Let the match begin"}]

    synopses = build_structured_event_synopses(
        keyframes=keyframes,
        frame_objects=objects,
        ocr_observations=ocr,
        asr_segments=asr,
        window_half_sec=4.0,
    )
    # kf_2 at 20000ms has no nearby evidence -> skipped
    assert len(synopses) == 1
    syn = synopses[0]
    assert syn["keyframe_id"] == "kf_1"
    assert "1 person (left)" in syn["synopsis_text"]
    assert "1 car (right)" in syn["synopsis_text"]
    assert "dog" not in syn["synopsis_text"]
    assert "OCR: ROUND 1." in syn["synopsis_text"]
    assert "Speech: Let the match begin." in syn["synopsis_text"]

    # Also verify SQLite db_path loading
    db_path = tmp_path / "mini.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE Keyframes (keyframe_id TEXT, video_id TEXT, timestamp_ms INTEGER)")
    conn.execute(
        "CREATE TABLE FrameObjects (video_id TEXT, keyframe_id TEXT, timestamp_ms INTEGER, "
        "class_name TEXT, bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL, confidence REAL, image_width INTEGER)"
    )
    conn.execute("CREATE TABLE OCRObservations (observation_id TEXT, video_id TEXT, timestamp_ms INTEGER, raw_text TEXT)")
    conn.execute("CREATE TABLE ASRSegments (segment_id TEXT, video_id TEXT, start_ms INTEGER, end_ms INTEGER, raw_text TEXT)")
    conn.execute("INSERT INTO Keyframes VALUES ('kf_1', 'v1', 5000)")
    conn.execute("INSERT INTO OCRObservations VALUES ('o1', 'v1', 5100, 'NEWS')")
    conn.commit()
    conn.close()

    db_syn = build_structured_event_synopses(db_path=db_path)
    assert len(db_syn) == 1
    assert "OCR: NEWS." in db_syn[0]["synopsis_text"]


def test_pq_vca_mode_b_extract_florence2_captions_mocked(tmp_path: Path):
    class _FakeInputs(dict):
        def to(self, _device: str):
            return self

    class _FakeProcessor:
        def __call__(self, text, images, return_tensors="pt"):
            return _FakeInputs({"pixel_values": np.zeros((len(images), 3, 8, 8), dtype=np.float32)})

        def batch_decode(self, generated_ids, skip_special_tokens=True):
            return [f"<DETAILED_CAPTION>A person cooking in a kitchen {i}" for i in range(len(generated_ids))]

    class _FakeModel:
        def generate(self, **kwargs):
            n = len(kwargs["pixel_values"])
            return [[1, 2, 3] for _ in range(n)]

    records = [
        {"faiss_id": 10, "video_id": "v1", "timestamp_ms": 1000, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
        {"faiss_id": 11, "video_id": "v1", "timestamp_ms": 3000, "image": np.zeros((16, 16, 3), dtype=np.uint8)},
    ]
    caps = extract_florence2_keyframe_captions(
        records,
        out_dir=tmp_path,
        processor=_FakeProcessor(),
        model=_FakeModel(),
        device="cpu",
        batch_size=1,
    )
    assert len(caps) == 2
    assert caps[0]["caption_text"] == "A person cooking in a kitchen 0"
    assert (tmp_path / "keyframe_captions.json").is_file()
    persisted = json.loads((tmp_path / "keyframe_captions.json").read_text(encoding="utf-8"))
    assert len(persisted) == 2


def test_ltr_monotonic_sign_projection_enforced():
    assert POSITIVE_MONOTONIC_FEATURES.isdisjoint(NEGATIVE_MONOTONIC_FEATURES)
    assert (POSITIVE_MONOTONIC_FEATURES | NEGATIVE_MONOTONIC_FEATURES) == set(LTR_FEATURE_NAMES)

    # Adversarial dataset where delta_z_vis (idx 2) and spike_drop_ratio (idx 7) are positively
    # correlated with label, and z_vis_max (idx 1) is negatively correlated with label.
    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 0.3, size=(40, len(LTR_FEATURE_NAMES)))
    y = np.array([1.0 if i % 4 == 0 else 0.0 for i in range(40)], dtype=np.float64)
    qids = [f"q_{i // 4}" for i in range(40)]

    X[y > 0, 0] += 1.0   # s_vis_blend_max (+)
    X[y > 0, 1] -= 1.5   # adversarial z_vis_max
    X[y > 0, 2] += 1.5   # adversarial delta_z_vis
    X[y > 0, 7] += 1.2   # adversarial spike_drop_ratio

    weights, _, _, _ = train_pairwise_logistic_l2(
        X, y, qids, l2_reg=0.1, learning_rate=0.1, max_iter=200, monotonic_constraints=True
    )
    for idx, name in enumerate(LTR_FEATURE_NAMES):
        if name in POSITIVE_MONOTONIC_FEATURES:
            assert weights[idx] >= 0.0, f"{name} violated >= 0 constraint: {weights[idx]}"
        elif name in NEGATIVE_MONOTONIC_FEATURES:
            assert weights[idx] <= 0.0, f"{name} violated <= 0 constraint: {weights[idx]}"


def test_compute_keyframe_hubness_radii_cross_video(tmp_path: Path):
    from videntra.retrieval.visual import build_or_load_keyframe_hubness_radii, compute_keyframe_hubness_radii

    rng = np.random.default_rng(7)
    vecs = rng.normal(0.0, 1.0, size=(30, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # Make index 0 a universal hub close to the mean direction of all other vectors
    mean_dir = np.mean(vecs[1:], axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    vecs[0] = mean_dir
    vids = [f"v{i // 3}" for i in range(30)]

    radii = compute_keyframe_hubness_radii(vecs, video_ids=vids, k_neighbors=5, sample_anchors=30)
    assert radii.shape == (30,)
    assert np.isfinite(radii).all()
    assert radii[0] > float(np.median(radii))

    cache_file = tmp_path / "hub_radii.npy"
    loaded_r, med_r = build_or_load_keyframe_hubness_radii(
        vecs, video_ids=vids, cache_path=cache_file, k_neighbors=5, sample_anchors=30
    )
    assert cache_file.is_file()
    assert np.allclose(loaded_r, radii)
    assert med_r == pytest.approx(float(np.median(radii)), abs=1e-6)

    # Empty edge case
    assert compute_keyframe_hubness_radii(np.zeros((0, 16), dtype=np.float32)).size == 0


def test_snap_boundaries_to_shot_cuts():
    from videntra.localization.refine import snap_boundaries_to_shot_cuts

    ts = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float32)
    # Shot 1: idx 0..1; Shot 2: idx 2..4; Shot 3: idx 5..6
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.99, 0.0],
        [0.0, 0.98, 0.1],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.99],
    ], dtype=np.float32)
    # Predicted window [2.8, 9.2] should snap start -> 4.0 (cut between 1 and 2) and end -> 8.0 (cut between 4 and 5)
    s_new, e_new = snap_boundaries_to_shot_cuts(ts, vecs, start_sec=2.8, end_sec=9.2, cut_threshold=0.28, max_snap_sec=3.0)
    assert s_new == pytest.approx(4.0, abs=1e-4)
    assert e_new == pytest.approx(8.0, abs=1e-4)


def test_se_viterbi_llm_compact_no_crash_and_e5_pure_visual_gate():
    import faiss
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend
    from videntra.query.schemas import ModalityRoutingPlan, QueryGraph
    from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline

    n = 24
    pts = [float(i * 2.0) for i in range(n)]
    raw_768 = np.zeros((n, 768), dtype=np.float32)
    aug_864 = np.zeros((n, 864), dtype=np.float32)
    # Create 3 distinct peaks at t=4, t=22, t=40 for NMS-MultiPeak
    for i in range(n):
        val = 0.2
        if i in (2, 11, 20):
            val = 0.95 - 0.02 * (i // 10)
        raw_768[i, 0] = val
        aug_864[i, 0] = val

    kf_map = {i: {"video_id": "v_long", "pts_time": pts[i], "source_frame_index": i * 60} for i in range(n)}
    vid_kfs = {"v_long": [{"faiss_id": i, "pts_time": pts[i], "source_frame_index": i * 60} for i in range(n)]}

    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipeline.all_embeddings = raw_768
    pipeline.augmented_embeddings = aug_864
    pipeline.keyframe_map = kf_map
    pipeline.video_keyframes = vid_kfs
    pipeline.video_durations = {"v_long": 50.0}
    pipeline.aug_index = faiss.IndexFlatIP(864)
    pipeline.aug_index.add(aug_864)
    pipeline.index = faiss.IndexFlatIP(768)
    pipeline.index.add(raw_768)
    pipeline.encode_texts = lambda texts: np.tile(np.array([[1.0] + [0.0] * 767], dtype=np.float32), (len(texts), 1))  # type: ignore[method-assign]

    # Mock E5 index + spy on encode_e5_query to verify pure-visual query does NOT call E5
    e5_calls: list[str] = []
    pipeline.e5_text_index = faiss.IndexFlatIP(384)
    pipeline.encode_e5_query = lambda q: e5_calls.append(q) or None  # type: ignore[method-assign]

    from videntra.query.rules import RuleBasedQueryParser
    parsed_graph = RuleBasedQueryParser().parse("a boy jumps then runs away")
    plan = ModalityRoutingPlan(
        original_query="a boy jumps then runs away",
        query_length_tier="medium",
        visual_clauses=["a boy jumps", "runs away"],
        ocr_phrases=[],
        asr_phrases=[],
        object_requirements=[],
        has_explicit_ocr_intent=False,
        has_explicit_asr_intent=False,
        has_spatial_constraints=False,
        has_temporal_constraints=True,
        query_graph=parsed_graph,
        routing_mode="llm_compact",
        cache_hit=True,
        decomposer_latency_ms=0.1,
    )
    pipeline.query_decomposer = AdaptiveQueryDecomposer(llm_backend=LocalSmallLLMBackend(enabled=False), enable_disk_cache=False)
    pipeline.query_decomposer.decompose = lambda q, mode="accurate": plan  # type: ignore[method-assign]

    res = pipeline.search("a boy jumps then runs away", top_k=5, ablation_config="A5")
    assert len(res) >= 2
    assert res[0].ranking_score > res[1].ranking_score
    assert e5_calls == []  # Pure-visual query must NOT invoke E5 encoder!

