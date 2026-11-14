"""Run end-to-end multimodal search ablations and evaluation on VIDENTRA."""
import json
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import pyarrow.parquet as pq

from videntra.indexing.ids import IDMappingStore
from videntra.indexing.faiss_store import VisualIndexStore
from videntra.indexing.text_store import TextIndexStore
from videntra.retrieval.visual import VisualRetrievalEngine, RankedMoment, temporal_iou
from videntra.retrieval.branches import VisualBranch, OCRBranch, ASRBranch, BranchCandidate
from videntra.retrieval.union import IndependentBranchUnion
from videntra.retrieval.fusion import MultimodalMomentFusion
from videntra.query.rules import RuleBasedQueryParser
from videntra.ranking.ranker import RelationAwareRanker
from videntra.contracts.validation import validate_result

class PrecomputedQueryEncoder:
    """Encoder interface using precomputed Kaggle query embeddings strictly without local inference."""

    def __init__(self, query_embeddings_map: dict[str, np.ndarray], dimension: int = 768):
        self.embeddings = query_embeddings_map
        self.dimension = dimension

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            clean = t.strip()
            if clean in self.embeddings:
                vecs.append(self.embeddings[clean])
            else:
                matched = None
                for k, v in self.embeddings.items():
                    if k.lower() == clean.lower():
                        matched = v
                        break
                if matched is not None:
                    vecs.append(matched)
                else:
                    raise ValueError(
                        f"Query '{t}' has no precomputed Kaggle embedding. "
                        "Local CPU/GPU inference for new queries is strictly forbidden by policy."
                    )
        return np.vstack(vecs).astype(np.float32)

    def encode_images(self, images: list[Any]) -> np.ndarray:
        raise NotImplementedError("Image encoding on local CPU is disabled. Use Kaggle GPU embeddings.")

def evaluate_predictions(predictions: dict[str, list[dict]], ground_truth: list[dict]) -> dict[str, Any]:
    """Independent retrieval evaluator computing Video & Moment Hit@K, MRR, and tIoU."""
    total_queries = len(ground_truth)
    video_hit1 = 0
    video_hit5 = 0
    mrr_sum = 0.0
    moment_hit1_iou3 = 0
    moment_hit1_iou5 = 0
    moment_hit5_iou3 = 0
    moment_hit5_iou5 = 0

    per_query_results = []

    for gt_item in ground_truth:
        qid = gt_item["qid"]
        rel_videos = {r["video_id"]: r["relevant_windows"] for r in gt_item["relevant_videos"]}
        preds = predictions.get(qid, [])

        q_video_hit1 = False
        q_video_hit5 = False
        q_rr = 0.0
        q_m_hit1_iou3 = False
        q_m_hit1_iou5 = False
        q_m_hit5_iou3 = False
        q_m_hit5_iou5 = False

        for rank, p in enumerate(preds[:5], start=1):
            vid = p["video_id"]
            st = p["moment_start_seconds"]
            et = p["moment_end_seconds"]

            if vid in rel_videos:
                if rank == 1:
                    q_video_hit1 = True
                q_video_hit5 = True
                if q_rr == 0.0:
                    q_rr = 1.0 / rank

                # Check moment overlap
                gold_windows = rel_videos[vid]
                best_iou = max([temporal_iou(st, et, gw[0], gw[1]) for gw in gold_windows], default=0.0)

                if best_iou >= 0.3:
                    if rank == 1:
                        q_m_hit1_iou3 = True
                    q_m_hit5_iou3 = True
                if best_iou >= 0.5:
                    if rank == 1:
                        q_m_hit1_iou5 = True
                    q_m_hit5_iou5 = True

        if q_video_hit1:
            video_hit1 += 1
        if q_video_hit5:
            video_hit5 += 1
        mrr_sum += q_rr
        if q_m_hit1_iou3:
            moment_hit1_iou3 += 1
        if q_m_hit1_iou5:
            moment_hit1_iou5 += 1
        if q_m_hit5_iou3:
            moment_hit5_iou3 += 1
        if q_m_hit5_iou5:
            moment_hit5_iou5 += 1

        per_query_results.append({
            "qid": qid,
            "reciprocal_rank": round(q_rr, 4),
            "video_hit1": q_video_hit1,
            "video_hit5": q_video_hit5,
            "moment_hit1_iou3": q_m_hit1_iou3,
            "moment_hit1_iou5": q_m_hit1_iou5,
            "top_candidate": preds[0] if preds else None,
        })

    metrics = {
        "num_queries": total_queries,
        "video_hit@1": round(video_hit1 / total_queries, 4) if total_queries else 0.0,
        "video_hit@5": round(video_hit5 / total_queries, 4) if total_queries else 0.0,
        "mrr": round(mrr_sum / total_queries, 4) if total_queries else 0.0,
        "moment_hit@1_tiou0.3": round(moment_hit1_iou3 / total_queries, 4) if total_queries else 0.0,
        "moment_hit@1_tiou0.5": round(moment_hit1_iou5 / total_queries, 4) if total_queries else 0.0,
        "moment_hit@5_tiou0.3": round(moment_hit5_iou3 / total_queries, 4) if total_queries else 0.0,
        "moment_hit@5_tiou0.5": round(moment_hit5_iou5 / total_queries, 4) if total_queries else 0.0,
        "per_query": per_query_results,
    }
    return metrics

def run_evaluation(
    embeddings_dir: str = "work/benchmark/smoke/embeddings",
    inputs_dir: str = "work/kaggle/inputs",
    text_db: str = "work/benchmark/smoke/text_index.db",
    gt_file: str = "work/benchmark/smoke/ground_truth.jsonl",
    output_report: str = "work/experiments/run_multimodal_eval_01/report.json",
):
    print("=== Starting VIDENTRA Multimodal Search Evaluation ===")
    emb_p = Path(embeddings_dir)
    in_p = Path(inputs_dir)
    report_p = Path(output_report)
    report_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Precomputed Embeddings
    frame_embeddings = np.load(emb_p / "frame_embeddings.npy")
    query_embeddings = np.load(emb_p / "query_embeddings.npy")
    with open(emb_p / "frame_ids.json", "r", encoding="utf-8") as f:
        frame_ids = json.load(f)
    with open(emb_p / "query_ids.json", "r", encoding="utf-8") as f:
        query_ids = json.load(f)
    with open(in_p / "benchmark_queries.jsonl", "r", encoding="utf-8") as f:
        benchmark_queries = [json.loads(line) for line in f if line.strip()]
    with open(in_p / "keyframes_manifest.jsonl", "r", encoding="utf-8") as f:
        keyframes_manifest = {item["sampled_frame_id"]: item for item in [json.loads(l) for l in f if l.strip()]}
    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded: {len(frame_embeddings)} frame embeddings, {len(query_embeddings)} query embeddings")

    # Map query text -> embedding vector
    qid_to_vec = {qid: vec for qid, vec in zip(query_ids, query_embeddings)}
    query_text_to_vec = {q["query"]: qid_to_vec[q["qid"]] for q in benchmark_queries if q["qid"] in qid_to_vec}

    query_encoder = PrecomputedQueryEncoder(query_text_to_vec)

    # 2. Build CPU Visual FAISS Index
    id_mapping = IDMappingStore()
    visual_store = VisualIndexStore(dimension=768)

    int_ids = []
    valid_vectors = []
    for idx, s_id in enumerate(frame_ids):
        if s_id in keyframes_manifest:
            m = keyframes_manifest[s_id]
            int_id = id_mapping.register(
                video_id=m["video_id"],
                timeline_revision="rev_m1_1",
                ordinal=m["ordinal"],
                sampled_frame_id=s_id,
                timestamp_us=int(round(m["timestamp_seconds"] * 1_000_000)),
                timestamp_seconds=m["timestamp_seconds"],
            )
            int_ids.append(int_id)
            valid_vectors.append(frame_embeddings[idx])

    visual_store.add(np.array(valid_vectors), int_ids)
    print(f"Built CPU Visual FAISS index: {len(visual_store)} vectors registered.")

    # 3. Setup Branches and Rankers
    visual_engine = VisualRetrievalEngine(
        encoder=query_encoder,
        index_store=visual_store,
        id_mapping=id_mapping,
    )
    text_store = TextIndexStore(text_db)
    visual_branch = VisualBranch(visual_engine)
    ocr_branch = OCRBranch(text_store)
    asr_branch = ASRBranch(text_store)
    query_parser = RuleBasedQueryParser()
    relation_ranker = RelationAwareRanker()
    fusion_engine = MultimodalMomentFusion()

    # Load evidence for RelationAwareRanker
    evidence_by_video: dict[str, list[dict]] = {}
    evidence_dir = Path("work/benchmark/smoke/evidence")
    for asr_p in evidence_dir.glob("asr_segments/*.parquet"):
        t = pq.read_table(str(asr_p))
        vid = t.column("video_id")[0].as_py()
        evidence_by_video.setdefault(vid, []).extend([
            {
                "kind": "asr_segment",
                "text": t.column("text")[i].as_py(),
                "timestamp_seconds": t.column("start_seconds")[i].as_py(),
            }
            for i in range(len(t))
        ])
    for ocr_p in evidence_dir.glob("ocr_tracks/*.parquet"):
        t = pq.read_table(str(ocr_p))
        vid = t.column("video_id")[0].as_py()
        evidence_by_video.setdefault(vid, []).extend([
            {
                "kind": "ocr_track",
                "text": t.column("text")[i].as_py(),
                "timestamp_seconds": t.column("start_seconds")[i].as_py(),
            }
            for i in range(len(t))
        ])

    # 4. Run 4 Ablations
    ablations = [
        "visual_only",
        "visual_ocr",
        "visual_asr",
        "full_multimodal",
    ]

    ablation_predictions = {name: {} for name in ablations}
    ablation_timings = {name: [] for name in ablations}
    rescue_rates = []

    for q in benchmark_queries:
        qid = q["qid"]
        q_text = q["query"]

        # Run Visual Branch
        t0 = time.perf_counter()
        vis_cands, _, _ = visual_branch.retrieve(q_text, top_k=20)
        vis_time = (time.perf_counter() - t0) * 1000

        # Run OCR & ASR Branches
        t1 = time.perf_counter()
        ocr_cands, _, _ = ocr_branch.retrieve(q_text, top_k=20)
        ocr_time = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        asr_cands, _, _ = asr_branch.retrieve(q_text, top_k=20)
        asr_time = (time.perf_counter() - t2) * 1000

        # Ablation 1: Visual-only
        vis_moments = fusion_engine.fuse(vis_cands, [], [], top_k=10)
        ablation_predictions["visual_only"][qid] = [m.to_result_dict() for m in vis_moments]
        ablation_timings["visual_only"].append(vis_time)

        # Ablation 2: Visual + OCR
        vocr_moments = fusion_engine.fuse(vis_cands, ocr_cands, [], top_k=10)
        ablation_predictions["visual_ocr"][qid] = [m.to_result_dict() for m in vocr_moments]
        ablation_timings["visual_ocr"].append(vis_time + ocr_time)

        # Ablation 3: Visual + ASR
        vasr_moments = fusion_engine.fuse(vis_cands, [], asr_cands, top_k=10)
        ablation_predictions["visual_asr"][qid] = [m.to_result_dict() for m in vasr_moments]
        ablation_timings["visual_asr"].append(vis_time + asr_time)

        # Ablation 4: Full Multimodal (QueryGraph v1 + RelationAwareRanker)
        t3 = time.perf_counter()
        q_graph = query_parser.parse(q_text)
        initial_moments = fusion_engine.fuse(vis_cands, ocr_cands, asr_cands, top_k=10)
        full_moments = relation_ranker.rank(initial_moments, query_graph=q_graph, evidence_by_video=evidence_by_video)
        full_time = (time.perf_counter() - t3) * 1000 + vis_time + ocr_time + asr_time
        ablation_predictions["full_multimodal"][qid] = [m.to_result_dict() for m in full_moments]
        ablation_timings["full_multimodal"].append(full_time)

        # Compute rescue rate for this query
        fused_vids = [m.video_id for m in full_moments]
        rr = IndependentBranchUnion.compute_rescue_rate(fused_vids, vis_cands)
        rescue_rates.append(rr)

    # 5. Evaluate all Ablations
    evaluation_results = {}
    for name in ablations:
        metrics = evaluate_predictions(ablation_predictions[name], ground_truth)
        metrics["mean_latency_ms"] = round(float(np.mean(ablation_timings[name])), 2)
        metrics["p95_latency_ms"] = round(float(np.percentile(ablation_timings[name], 95)), 2)
        evaluation_results[name] = metrics

    # 6. Schema Validation on Sample Predictions
    for name in ablations:
        for qid, preds in ablation_predictions[name].items():
            for p in preds[:1]:
                validate_result(p, production=False)

    # 7. Summary and Final Report
    avg_rescue_rate = float(np.mean(rescue_rates)) if rescue_rates else 0.0
    final_report = {
        "benchmark_dataset": "QVHighlights Smoke Subset (4 clips, 400 keyframes)",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "index_metadata": {
            "num_indexed_keyframes": len(visual_store),
            "faiss_index_type": "IndexFlatIP (CPU)",
            "text_index_type": "SQLite FTS5 unicode61 (remove_diacritics=0)",
            "vector_dimension": 768,
        },
        "multimodal_rescue_rate": round(avg_rescue_rate, 4),
        "ablations": evaluation_results,
    }

    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n================== EVALUATION REPORT SUMMARY ==================")
    print(f"Multimodal Rescue Rate: {avg_rescue_rate * 100:.2f}%")
    for name, m in evaluation_results.items():
        print(f"\n--- Ablation: {name.upper()} ---")
        print(f"  Video Hit@1: {m['video_hit@1'] * 100:.1f}%")
        print(f"  Video Hit@5: {m['video_hit@5'] * 100:.1f}%")
        print(f"  Video MRR:   {m['mrr']:.4f}")
        print(f"  Moment Hit@1 (tIoU>=0.3): {m['moment_hit@1_tiou0.3'] * 100:.1f}%")
        print(f"  Moment Hit@1 (tIoU>=0.5): {m['moment_hit@1_tiou0.5'] * 100:.1f}%")
        print(f"  Moment Hit@5 (tIoU>=0.3): {m['moment_hit@5_tiou0.3'] * 100:.1f}%")
        print(f"  Moment Hit@5 (tIoU>=0.5): {m['moment_hit@5_tiou0.5'] * 100:.1f}%")
        print(f"  Mean Latency: {m['mean_latency_ms']} ms")
    print(f"\nFull JSON report written to: {report_p}")
    return final_report

if __name__ == "__main__":
    run_evaluation()
