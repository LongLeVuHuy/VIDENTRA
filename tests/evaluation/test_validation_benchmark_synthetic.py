"""Protocol checks with deterministic predictions; these are not engine benchmarks."""
from scripts.run_validation_benchmark import evaluate_query_set
import pytest


class FakeRunner:
    indexed_videos = {"target"} | {f"d{i}" for i in range(11)}
    dataset_videos = {"synthetic": indexed_videos}
    timings = {"t_total_ms": 3, "t_non_neural_ms": 2, "t_faiss_ms": 1,
               "t_fts5_ms": 0, "t_fusion_ms": 1, "t_boundary_ms": 0}
    diagnostics = {}

    def search_multimodal(self, query, top_k, ablation_config):
        if query == "none":
            return []
        if query == "late":
            return [self._moment(f"d{i}") for i in range(10)] + [self._moment("target")]
        return [self._moment("target")]

    @staticmethod
    def _moment(video_id):
        return {"video_id": video_id, "start_sec": 1.0, "end_sec": 9.0,
                "ranking_score": 1.0, "modality_attribution": "visual"}


def _query(qid, text):
    return {"query_id": qid, "query_text": text, "dataset": "synthetic",
            "target_videos": ["target"], "target_windows": [[0.0, 10.0]]}


def test_empty_predictions_count_as_moment_miss_and_boundary_mae_is_per_endpoint():
    summary, records = evaluate_query_set(FakeRunner(), [_query("a", "hit"), _query("b", "none")], show_progress=False)
    metrics = summary["metrics_overall"]
    assert metrics["moment_hit_at_1_iou05"] == 0.5
    assert metrics["moment_map_at_05"] == 0.5
    assert metrics["mean_boundary_mae_sec"] == 1.0
    assert metrics["boundary_mae_n"] == 1
    assert metrics["boundary_mae_scope"] == "top1_correct_video_iou_at_least_0.5"
    assert records[1]["moment_hit_iou05"] == 0


def test_mrr_is_truncated_at_ten_unique_videos():
    summary, records = evaluate_query_set(FakeRunner(), [_query("a", "late")], show_progress=False)
    assert summary["metrics_overall"]["mean_reciprocal_rank_mrr"] == 0.0
    assert records[0]["matched_rank"] == 0


def test_unlabeled_or_ambiguous_moments_are_rejected():
    with pytest.raises(ValueError, match="known target videos"):
        evaluate_query_set(FakeRunner(), [{**_query("a", "none"), "target_videos": []}], show_progress=False)
    with pytest.raises(ValueError, match="unambiguous target"):
        evaluate_query_set(FakeRunner(), [{**_query("a", "none"), "target_videos": ["target", "d0"]}], show_progress=False)


def test_real_unified_pipeline_protocol_on_synthetic_vectors():
    import faiss
    import numpy as np
    from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline

    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, memory_cache_size=0)
    pipeline.augmented_embeddings = np.zeros((2, 864), dtype=np.float32)
    pipeline.augmented_embeddings[:, 0] = [0.9, 0.3]
    pipeline.aug_index = faiss.IndexFlatIP(864)
    pipeline.aug_index.add(pipeline.augmented_embeddings)
    pipeline.index = faiss.IndexFlatIP(768)
    pipeline.index.add(np.zeros((2, 768), dtype=np.float32))
    pipeline.keyframe_map = {i: {"video_id": f"v{i}", "pts_time": 2.0, "source_frame_index": i} for i in range(2)}
    pipeline.video_keyframes = {f"v{i}": [{"faiss_id": i, "pts_time": 2.0}] for i in range(2)}
    pipeline.video_durations = {"v0": 10.0, "v1": 10.0}
    def encode(texts):
        result = np.zeros((len(texts), 768), dtype=np.float32)
        result[:, 0] = 1
        return result
    pipeline.encode_texts = encode

    class SyntheticRunner:
        indexed_videos = {"v0", "v1"}
        dataset_videos = {"synthetic": indexed_videos}
        def search_multimodal(self, query, top_k, ablation_config):
            response = pipeline.search_with_metadata(snapshot_id="synthetic", query=query,
                top_k=top_k, mode="accurate", ranking_mode="diverse_video", ablation_config=ablation_config)
            self.timings, self.diagnostics = response["timings"], response["diagnostics"]
            return [{"video_id": m.video_id, "start_sec": m.moment_start_seconds,
                     "end_sec": m.moment_end_seconds, "ranking_score": m.ranking_score,
                     "modality_attribution": m.match_status} for m in response["moments"]]

    query = {"query_id": "q", "query_text": "ambient", "dataset": "synthetic",
             "target_videos": ["v0"], "target_windows": []}
    try:
        summary, records = evaluate_query_set(SyntheticRunner(), [query], show_progress=False)
    finally:
        pipeline.close()
    assert summary["metrics_overall"]["video_hit_at_1"] == 1.0
    assert records[0]["predicted_top1_video"] == "v0"


def test_eleven_moments_two_same_video_at_top_gives_video_hit10_1():
    import json

    class ElevenMomentRunner:
        indexed_videos = {"target"} | {f"d{i}" for i in range(10)}
        dataset_videos = {"synthetic": indexed_videos}
        timings = {
            "t_total_ms": 3.0, "t_non_neural_ms": 2.0, "t_faiss_ms": 1.0,
            "t_fts5_ms": 0.0, "t_fusion_ms": 1.0, "t_boundary_ms": 0.0,
        }
        diagnostics = {}
        ranked_videos = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "target"]

        def search_multimodal(self, query, top_k, ablation_config):
            moments = [
                {"video_id": "d0", "start_sec": 0.0, "end_sec": 4.0, "ranking_score": 0.99, "modality_attribution": "visual"},
                {"video_id": "d0", "start_sec": 10.0, "end_sec": 14.0, "ranking_score": 0.98, "modality_attribution": "visual"},
            ]
            for i in range(1, 9):
                moments.append({
                    "video_id": f"d{i}", "start_sec": 1.0, "end_sec": 5.0,
                    "ranking_score": 0.9 - i * 0.05, "modality_attribution": "visual",
                })
            moments.append({
                "video_id": "target", "start_sec": 0.0, "end_sec": 10.0,
                "ranking_score": 0.40, "modality_attribution": "visual",
            })
            return moments

    summary, records = evaluate_query_set(ElevenMomentRunner(), [_query("m11", "q")], show_progress=False)
    assert summary["metrics_overall"]["video_hit_at_10"] == 1.0
    assert summary["metrics_overall"]["mean_reciprocal_rank_mrr"] == 0.1
    top10_moments = json.loads(records[0]["ranked_predictions"])
    assert len(top10_moments) == 10
    assert sum(1 for m in top10_moments if m["video_id"] == "d0") == 2


def test_corpus_smaller_than_k_tie_break_and_unrounded_timestamps():
    import json

    class SmallTieRunner:
        indexed_videos = {"v_b", "v_a", "target"}
        dataset_videos = {"synthetic": indexed_videos}
        timings = {
            "t_total_ms": 3.0, "t_non_neural_ms": 2.0, "t_faiss_ms": 1.0,
            "t_fts5_ms": 0.0, "t_fusion_ms": 1.0, "t_boundary_ms": 0.0,
        }
        diagnostics = {}

        def search_multimodal(self, query, top_k, ablation_config):
            return [
                {"video_id": "v_b", "start_sec": 1.0, "end_sec": 3.0, "ranking_score": 0.5, "modality_attribution": "visual"},
                {"video_id": "target", "start_sec": 0.004, "end_sec": 9.996, "ranking_score": 0.5, "modality_attribution": "visual"},
                {"video_id": "v_a", "start_sec": 2.0, "end_sec": 4.0, "ranking_score": 0.5, "modality_attribution": "visual"},
            ]

    summary, records = evaluate_query_set(SmallTieRunner(), [_query("small", "q")], show_progress=False)
    assert summary["metrics_overall"]["video_hit_at_10"] == 1.0
    ranked_vids = json.loads(records[0]["ranked_videos"])
    assert ranked_vids == ["v_b", "target", "v_a"]
    assert records[0]["matched_rank"] == 2
    top_moments = json.loads(records[0]["ranked_predictions"])
    assert top_moments[1]["start_sec"] == 0.004
    assert top_moments[1]["end_sec"] == 9.996
