"""Run end-to-end visual baseline retrieval experiment on real SigLIP 2 model weights."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import sys

from videntra.config.ingestion import IngestionConfig, SamplingConfig
from videntra.encoders.siglip2 import SigLIP2Encoder
from videntra.evaluation.evaluator import evaluate_predictions
from videntra.fixtures import build_fixtures
from videntra.ingestion.jobs import IngestionJobManager
from videntra.registry.access import MediaResolver
from videntra.registry.store import sha256_file
from videntra.retrieval.visual import VisualRetrievalEngine


def main():
    run_id = "run_m2_visual_01"
    run_dir = Path("work/experiments") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = run_dir / "fixtures"

    print(f"[{run_id}] Step 1: Building test fixtures...")
    if not fixtures_dir.exists() or not any(fixtures_dir.iterdir()):
        build_fixtures(fixtures_dir)
    else:
        print(f"[{run_id}] Fixtures already exist, reusing.")

    print(f"[{run_id}] Step 2: Ingesting assets...")
    cfg = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=0.25))
    ingest_out = run_dir / "ingest"
    mgr = IngestionJobManager(config=cfg, output_dir=ingest_out)

    cfr_path = fixtures_dir / "cfr.mkv"
    nz_path = fixtures_dir / "nonzero_pts.mkv"

    cfr_hash = sha256_file(cfr_path)
    nz_hash = sha256_file(nz_path)

    records = [
        {
            "schema_version": "1.0",
            "video_id": f"sha256:{cfr_hash}",
            "source_video_id": "fixture:cfr",
            "sha256": cfr_hash,
            "media_uri": "asset://fixtures/cfr.mkv",
            "video_stream_index": 0,
            "audio_stream_index": None,
            "source_offset_us": 0,
            "source_mapping_kind": "constant_offset",
            "timeline_origin": {"num": 0, "den": 1},
            "duration_us": 1000000,
            "rights_status": "allowed",
            "rights_reference": "license://fixture-test",
            "original_dataset_id": "FixtureSuite",
            "official_split": "train",
            "exposure": "train",
            "decode_status": "pending",
            "timeline_revision": None,
            "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
        },
        {
            "schema_version": "1.0",
            "video_id": f"sha256:{nz_hash}",
            "source_video_id": "fixture:nonzero_pts",
            "sha256": nz_hash,
            "media_uri": "asset://fixtures/nonzero_pts.mkv",
            "video_stream_index": 0,
            "audio_stream_index": None,
            "source_offset_us": 0,
            "source_mapping_kind": "constant_offset",
            "timeline_origin": {"num": 0, "den": 1},
            "duration_us": 1000000,
            "rights_status": "allowed",
            "rights_reference": "license://fixture-test",
            "original_dataset_id": "FixtureSuite",
            "official_split": "train",
            "exposure": "train",
            "decode_status": "pending",
            "timeline_revision": None,
            "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
        },
    ]

    resolver = MediaResolver({"fixtures": fixtures_dir}, records)
    ingest_report = mgr.execute_manifest(records, resolver=resolver, for_inference=True)
    print(f"[{run_id}] Ingestion complete. Committed: {ingest_report.committed_assets}/{ingest_report.total_assets}")

    print(f"[{run_id}] Step 3: Initializing real SigLIP 2 encoder on CPU...")
    siglip2_dir = os.environ.get("VIDENTRA_MODEL_DIR", "models/siglip2")
    encoder = SigLIP2Encoder(model_path=siglip2_dir, device="cpu")
    engine = VisualRetrievalEngine(encoder=encoder)

    print(f"[{run_id}] Step 4: Indexing sampled frames...")
    t0 = time.perf_counter()
    indexed_samples = 0
    for r in records:
        vid = r["video_id"]
        safe_vid = vid.replace(":", "_")
        sample_path = list(ingest_out.glob(f"samples/{safe_vid}/*/samples.parquet"))[0]
        timeline_path = list(ingest_out.glob(f"timeline/{safe_vid}/*/frames.parquet"))[0]
        media_path = resolver.resolve(vid, for_inference=True)
        count = engine.index_video(vid, media_path, sample_path, timeline_path)
        indexed_samples += count

    index_time = time.perf_counter() - t0
    print(f"[{run_id}] Indexed {indexed_samples} samples across 2 videos in {index_time:.2f}s")

    index_dir = run_dir / "index"
    engine.save(index_dir)
    print(f"[{run_id}] Index saved to {index_dir}")

    # Parity check
    parity_ok = engine.index_store.verify_parity(engine.encoder.encode_texts(["test parity"]))
    print(f"[{run_id}] FAISS vs NumPy parity verified: {parity_ok}")

    print(f"[{run_id}] Step 5: Executing visual queries...")
    queries = [
        {"qid": "q_vi_01", "text": "màn hình kiểm tra màu sắc chuẩn"},
        {"qid": "q_en_01", "text": "synthetic test pattern generator screen"},
    ]

    all_predictions = []
    t_search_total = 0.0
    for q in queries:
        t_s0 = time.perf_counter()
        moments = engine.search(q["text"], top_k_moments=5)
        dt_s = time.perf_counter() - t_s0
        t_search_total += dt_s
        for rank, m in enumerate(moments, 1):
            pred = m.to_result_dict()
            pred["query_id"] = q["qid"]
            pred["rank"] = rank
            all_predictions.append(pred)

    pred_file = run_dir / "predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[{run_id}] Generated {len(all_predictions)} predictions to {pred_file}")

    print(f"[{run_id}] Step 6: Evaluating predictions...")
    gt_records = [
        {
            "query_id": "q_vi_01",
            "video_id": f"sha256:{cfr_hash}",
            "start_seconds": 0.0,
            "end_seconds": 1.0,
        },
        {
            "query_id": "q_en_01",
            "video_id": f"sha256:{nz_hash}",
            "start_seconds": 2.0,
            "end_seconds": 3.0,
        },
    ]

    eval_report = evaluate_predictions(all_predictions, gt_records)
    eval_dict = asdict(eval_report)

    final_report = {
        "run_id": run_id,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "type": "temporary_machine",
            "cpu": "Intel Core i5-7300U (2 cores, 4 threads)",
            "gpu": "None (Intel HD Graphics 620, CPU mode only)",
            "ram_gb": 32,
            "note": "Target hardware RTX 3060 6GB will be benchmarked on target machine; no fake latencies claimed."
        },
        "model": {
            "name": "SigLIP 2",
            "checkpoint": siglip2_dir,
            "dimension": 768,
            "device": "cpu",
        },
        "indexing": {
            "videos": len(records),
            "samples_indexed": indexed_samples,
            "indexing_wall_seconds": round(index_time, 3),
            "faiss_parity_verified": parity_ok,
        },
        "search": {
            "total_queries": len(queries),
            "average_search_seconds": round(t_search_total / len(queries), 4),
        },
        "evaluation": eval_dict,
    }

    report_file = run_dir / "m2_experiment_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"[{run_id}] Experiment report saved: {report_file}")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
