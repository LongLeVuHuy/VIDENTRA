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
