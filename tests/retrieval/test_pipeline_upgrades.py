"""Regression and upgrade unit tests for Stage 2 768d+864d multi-clause blend, Stage 5 visual-floor gate, and Stage 6 local coherence."""
import faiss
import numpy as np

from videntra.retrieval.unified_pipeline import (
    UnifiedMultimodalSearchPipeline,
    _generate_morphological_stems,
    analyze_query_lexical_tokens,
)


def _make_mock_pipeline(raw_768: np.ndarray, aug_864: np.ndarray, kf_map: dict, vid_kfs: dict) -> UnifiedMultimodalSearchPipeline:
    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipeline.all_embeddings = np.ascontiguousarray(raw_768, dtype=np.float32)
    pipeline.augmented_embeddings = np.ascontiguousarray(aug_864, dtype=np.float32)
    pipeline.keyframe_map = kf_map
    pipeline.video_keyframes = vid_kfs
    pipeline.video_durations = {vid: 30.0 for vid in vid_kfs}
    pipeline.aug_index = faiss.IndexFlatIP(864)
    pipeline.aug_index.add(pipeline.augmented_embeddings)
    pipeline.index = faiss.IndexFlatIP(768)
    pipeline.index.add(pipeline.all_embeddings)
    return pipeline


def test_multi_clause_blends_768d_and_864d_embeddings():
    # Two videos: vid_A aligns in 768d space (w=0.70), vid_B only has higher tail in 864d (w=0.30).
    raw_768 = np.zeros((2, 768), dtype=np.float32)
    raw_768[0, 0] = 0.90  # vid_A strong 768d
    raw_768[1, 0] = 0.10  # vid_B weak 768d

    aug_864 = np.zeros((2, 864), dtype=np.float32)
    aug_864[0, 0] = 0.20
    aug_864[1, 0] = 0.50  # vid_B higher 864d dim 0

    kf_map = {
        0: {"video_id": "vid_A", "pts_time": 5.0, "source_frame_index": 0},
        1: {"video_id": "vid_B", "pts_time": 5.0, "source_frame_index": 0},
    }
    vid_kfs = {
        "vid_A": [{"faiss_id": 0, "pts_time": 5.0}],
        "vid_B": [{"faiss_id": 1, "pts_time": 5.0}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)

    def fake_encode(texts):
        vecs = np.zeros((len(texts), 768), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs

    pipeline.encode_texts = fake_encode  # type: ignore[method-assign]

    # Multi-clause query triggers Stage 2 sub_clauses path
    res = pipeline.search("a person walks in park and then sits down", top_k=2)
    assert len(res) == 2
    assert res[0].video_id == "vid_A"


def test_local_window_coherence_rewards_sustained_moment():
    # Both videos have a peak cluster at 0.80, but vid_coherent has a second peak within +/-3.5s
    # while vid_isolated has its second peak 12s away (beyond 3.5s and beyond 6.0s adjacency gap).
    raw_768 = np.zeros((4, 768), dtype=np.float32)
    aug_864 = np.zeros((4, 864), dtype=np.float32)
    raw_768[:, 0] = [0.80, 0.78, 0.80, 0.78]
    aug_864[:, 0] = [0.80, 0.78, 0.80, 0.78]

    kf_map = {
        0: {"video_id": "vid_coherent", "pts_time": 10.0, "source_frame_index": 0},
        1: {"video_id": "vid_coherent", "pts_time": 12.5, "source_frame_index": 1},  # within +/-3.5s
        2: {"video_id": "vid_isolated", "pts_time": 5.0, "source_frame_index": 0},
        3: {"video_id": "vid_isolated", "pts_time": 20.0, "source_frame_index": 1},  # far away (> 6s)
    }
    vid_kfs = {
        "vid_coherent": [{"faiss_id": 0, "pts_time": 10.0}, {"faiss_id": 1, "pts_time": 12.5}],
        "vid_isolated": [{"faiss_id": 2, "pts_time": 5.0}, {"faiss_id": 3, "pts_time": 20.0}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)

    def fake_encode(texts):
        vecs = np.zeros((len(texts), 768), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs

    pipeline.encode_texts = fake_encode  # type: ignore[method-assign]
    res = pipeline.search("person skiing downhill", top_k=2)
    assert len(res) == 2
    assert res[0].video_id == "vid_coherent"
    assert res[0].ranking_score > res[1].ranking_score


def test_visual_floor_gate_blocks_low_visual_lexical_distractor():
    import sqlite3

    raw_768 = np.zeros((2, 768), dtype=np.float32)
    aug_864 = np.zeros((2, 864), dtype=np.float32)
    raw_768[0, 0] = 0.60   # vid_vis strong visual
    aug_864[0, 0] = 0.60
    raw_768[1, 0] = 0.05   # vid_lex far below visual floor (0.60 - 0.065)
    aug_864[1, 0] = 0.05

    kf_map = {
        0: {"video_id": "vid_vis", "pts_time": 5.0, "source_frame_index": 0},
        1: {"video_id": "vid_lex", "pts_time": 5.0, "source_frame_index": 0},
    }
    vid_kfs = {
        "vid_vis": [{"faiss_id": 0, "pts_time": 5.0}],
        "vid_lex": [{"faiss_id": 1, "pts_time": 5.0}],
    }
    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE fts_ocr USING fts5(observation_id, video_id, timestamp_ms, normalized_text)")
    conn.execute("CREATE VIRTUAL TABLE fts_asr USING fts5(segment_id, video_id, start_ms, end_ms, normalized_text)")
    conn.execute("INSERT INTO fts_asr VALUES ('s1', 'vid_lex', 4000, 6000, 'tennis racket court game')")
    pipeline.db_conn = conn

    def fake_encode(texts):
        vecs = np.zeros((len(texts), 768), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs

    pipeline.encode_texts = fake_encode  # type: ignore[method-assign]
    res = pipeline.search("tennis player on court", top_k=2)
    assert res[0].video_id == "vid_vis"
    pipeline.close()


def test_morphological_stemming_is_general():
    assert "check" in _generate_morphological_stems("checking")
    assert "build" in _generate_morphological_stems("building")
    assert "phrenolog" in _generate_morphological_stems("phrenologists")
    assert "frenolog" in _generate_morphological_stems("phrenologists")
    assert "biolog" in _generate_morphological_stems("biologists")
    # Short or generic visual words must not be over-stemmed into noisy tokens
    assert _generate_morphological_stems("makes") == []
    assert _generate_morphological_stems("cooking") == []

    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    fts_q, _, _, _, _ = pipeline.prepare_fts5_query_and_tokens("a customer checking fresh groceries")
    assert "check*" in fts_q
    fts_late, _, _, _, _ = pipeline.prepare_fts5_query_and_tokens("audience laughing at late show")
    assert '"lateshow"' in fts_late


def test_sentence_initial_common_words_not_misclassified_as_proper_nouns():
    _, _, _, pn_cook = analyze_query_lexical_tokens("Cooking dinner in a modern kitchen")
    assert pn_cook == set()

    _, _, _, pn_cartoon = analyze_query_lexical_tokens("Cartoon animals talking loudly")
    assert pn_cartoon == set()

    _, _, _, pn_names = analyze_query_lexical_tokens("Gordon Ramsay hosting David Letterman on CNN")
    assert {"gordon", "ramsay", "david", "letterman", "cnn"}.issubset(pn_names)


def test_stage6_padded_keyframes_prevent_single_frame_outlier_bias():
    # Setup 2 videos with 5 keyframes each, and limit search_depth=2 so each video gets at most 1 cluster in top-2.
    # vid_outlier has 1 spike frame [0.52] and weak rest [0.12, 0.12, 0.12, 0.12].
    # vid_consistent has sustained strong frames [0.51, 0.49, 0.48, 0.47, 0.46].
    raw_768 = np.zeros((10, 768), dtype=np.float32)
    aug_864 = np.zeros((10, 864), dtype=np.float32)
    outlier_sims = [0.52, 0.12, 0.12, 0.12, 0.12]
    consistent_sims = [0.51, 0.49, 0.48, 0.47, 0.46]

    kf_map = {}
    vid_kfs = {"vid_outlier": [], "vid_consistent": []}
    for i, sim in enumerate(outlier_sims):
        raw_768[i, 0] = sim
        aug_864[i, 0] = sim
        kf_map[i] = {"video_id": "vid_outlier", "pts_time": i * 2.0, "source_frame_index": i}
        vid_kfs["vid_outlier"].append({"faiss_id": i, "pts_time": i * 2.0})
    for j, sim in enumerate(consistent_sims):
        idx = 5 + j
        raw_768[idx, 0] = sim
        aug_864[idx, 0] = sim
        kf_map[idx] = {"video_id": "vid_consistent", "pts_time": j * 2.0, "source_frame_index": j}
        vid_kfs["vid_consistent"].append({"faiss_id": idx, "pts_time": j * 2.0})

    pipeline = _make_mock_pipeline(raw_768, aug_864, kf_map, vid_kfs)
    pipeline.video_kf_fids = {
        "vid_outlier": list(range(0, 5)),
        "vid_consistent": list(range(5, 10)),
    }
    pipeline.video_kf_matrix = {
        "vid_outlier": aug_864[0:5],
        "vid_consistent": aug_864[5:10],
    }
    pipeline.search_depth = 2

    def fake_encode(texts):
        vecs = np.zeros((len(texts), 768), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs

    pipeline.encode_texts = fake_encode  # type: ignore[method-assign]
    res = pipeline.search("person skiing downhill", top_k=2)
    assert len(res) == 2
    assert res[0].video_id == "vid_consistent"
    assert res[0].ranking_score > res[1].ranking_score
