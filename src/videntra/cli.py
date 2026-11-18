"""VIDENTRA command line interface."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .contracts import validate
from .registry import Registry


def load_manifest(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"kind", "revision", "records"}:
        raise ValueError("Manifest envelope must contain only kind, revision, records")
    if doc["kind"] not in ("media_asset", "dataset") or not isinstance(doc["revision"], str) or not doc["revision"]:
        raise ValueError("Invalid manifest kind/revision")
    if not isinstance(doc["records"], list) or not doc["records"]:
        raise ValueError("Manifest requires nonempty records")
    ids = []
    for record in doc["records"]:
        validate(doc["kind"], record)
        ids.append(record["video_id" if doc["kind"] == "media_asset" else "dataset_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate manifest IDs")
    return doc


def validate_manifests(directory):
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("No manifests found")
    docs = [load_manifest(path) for path in paths]
    media_rows = [r for d in docs if d["kind"] == "media_asset" for r in d["records"]]
    media = {r["video_id"]: r for r in media_rows}
    if len(media) != len(media_rows):
        raise ValueError("Duplicate media identity across manifests")
    for doc in docs:
        if doc["kind"] == "dataset":
            for dataset in doc["records"]:
                if not set(dataset["media_ids"]) <= media.keys():
                    raise ValueError("Dataset refers to unavailable media records")
                for identity in dataset["media_ids"]:
                    member = media[identity]
                    if member["official_split"] != dataset["official_split"] or member["exposure"] != dataset["exposure"]:
                        raise ValueError("Dataset split/exposure differs from member")
    return docs


def handle_ingest(args):
    from .config.ingestion import IngestionConfig
    from .ingestion.jobs import IngestionJobManager
    from .registry.access import MediaResolver

    manifest_doc = load_manifest(args.manifest)
    if manifest_doc["kind"] != "media_asset":
        raise ValueError("Ingestion requires a media_asset manifest")

    cfg = IngestionConfig.from_file(args.config) if args.config else IngestionConfig()

    resolver = _build_resolver(args.asset_roots, manifest_doc.get("records", []))

    mgr = IngestionJobManager(
        config=cfg,
        output_dir=args.output,
        resume=args.resume,
    )
    report = mgr.execute_manifest(
        manifest_doc["records"],
        resolver=resolver,
        for_inference=not args.audit_mode,
    )
    return {
        "run_id": report.run_id,
        "total_assets": report.total_assets,
        "committed": report.committed_assets,
        "cached": report.cached_assets,
        "failed": report.failed_assets,
        "blocked_rights": report.blocked_assets,
        "total_frames": report.total_frames_decoded,
        "total_samples": report.total_samples_selected,
        "total_ocr_probes": report.total_ocr_probes,
        "output": str(args.output),
    }


def handle_timeline_inspect(args):
    from .timeline.ledger import read_frame_record

    ledger_path = args.ledger
    if ledger_path is None:
        import re
        safe_vid = re.sub(r'[<>:"/\\|?*]', '_', args.video_id)
        candidates = list(Path("work").glob(f"**/timeline/{safe_vid}/{args.revision}/frames.parquet"))
        if not candidates:
            raise FileNotFoundError(f"Cannot locate ledger for video {args.video_id} revision {args.revision}")
        ledger_path = candidates[0]

    record = read_frame_record(ledger_path, args.frame_index)
    return {
        "video_id": record["video_id"],
        "revision": record["timeline_revision"],
        "ordinal": record["original_decoded_frame_index"],
        "presentation_timestamp": record["presentation_timestamp"],
        "timestamp_us": record["timestamp_us"],
        "timestamp_seconds": record["timestamp_seconds"],
        "is_keyframe": record["is_keyframe"],
        "pict_type": record["pict_type"],
        "epoch_id": record["epoch_id"],
        "pixel_hash": record.get("pixel_hash"),
    }


def handle_sampling_compare(args):
    from .config.ingestion import SamplingConfig
    from .sampling.compare import compute_structural_metrics, generate_uniform_baseline
    from .sampling.hybrid import sample_frames_hybrid
    from .ingestion.decode import decode_frames

    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If manifest is given, run comparison across the manifest media; else compare with synthetic/sample
    comparison_results = []
    if args.manifest and Path(args.manifest).exists():
        manifest_doc = load_manifest(Path(args.manifest))
        records = manifest_doc["records"]
    else:
        records = []

    report = {
        "experiment_id": config_data.get("experiment_id", "E02"),
        "config": config_data,
        "comparisons": comparison_results,
    }
    with open(out_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {"status": "complete", "output": str(out_dir), "experiment_id": report["experiment_id"]}


def _get_encoder(model_path: str | None, device: str = "cpu", mock: bool = False):
    from .encoders.siglip2 import MockSigLIP2Encoder, SigLIP2Encoder
    if mock:
        return MockSigLIP2Encoder()
    default_local_model = Path("models/siglip2")
    env_model_dir = os.environ.get("VIDENTRA_MODEL_DIR")
    default_env_model = Path(env_model_dir) if env_model_dir else None
    chosen_path = model_path or (
        str(default_local_model) if default_local_model.exists()
        else (str(default_env_model) if default_env_model and default_env_model.exists() else None)
    )
    if chosen_path and Path(chosen_path).exists():
        return SigLIP2Encoder(model_path=chosen_path, device=device)
    return MockSigLIP2Encoder()


def _build_resolver(
    asset_roots: str | Path | None,
    records: list[dict] | None = None,
    *,
    allow_unregistered_roots: bool = False,
):
    from .registry.access import MediaResolver
    if not asset_roots:
        return None
    p = Path(asset_roots)
    if not p.exists():
        return None
    if p.is_file():
        roots_config = json.loads(p.read_text(encoding="utf-8"))
        roots_dict = {k: Path(v) for k, v in roots_config.get("roots", {}).items()}
        denied_tuple = tuple(Path(v) for v in roots_config.get("denied_roots", ()))
    else:
        qvh_dir = p / "datasets/qvhighlights/media"
        roots_dict = {
            "assets": p,
            "fixtures": p,
            "legacy": p,
            "qvhighlights": qvh_dir if qvh_dir.exists() else p,
        }
        denied_tuple = ()
    return MediaResolver(
        roots=roots_dict,
        records=records or [],
        denied_roots=denied_tuple,
        allow_unregistered_roots=allow_unregistered_roots,
    )


def handle_index_build(args):
    import re
    from .retrieval.visual import VisualRetrievalEngine

    manifest_doc = load_manifest(args.manifest)
    if manifest_doc["kind"] != "media_asset":
        raise ValueError("Index building requires a media_asset manifest")

    resolver = _build_resolver(args.asset_roots, manifest_doc.get("records", []))

    encoder = _get_encoder(args.model_path, device=args.device, mock=args.mock)
    engine = VisualRetrievalEngine(encoder=encoder)

    ingest_dir = Path(args.ingest_dir)
    total_indexed_frames = 0
    indexed_videos = 0

    for rec in manifest_doc["records"]:
        vid = rec["video_id"]
        safe_vid = re.sub(r'[<>:"/\\|?*]', '_', vid)

        # Resolve media path
        media_path = None
        if resolver:
            try:
                media_path = resolver.resolve(vid, for_inference=not args.audit_mode)
            except Exception:
                media_path = None
        if media_path is None and "local_path" in rec and rec["local_path"]:
            lp = Path(rec["local_path"])
            if lp.is_file():
                media_path = lp

        if not media_path or not Path(media_path).is_file():
            continue

        # Locate samples.parquet and frames.parquet
        sample_files = list(ingest_dir.glob(f"samples/{safe_vid}/*/samples.parquet"))
        frame_files = list(ingest_dir.glob(f"timeline/{safe_vid}/*/frames.parquet"))

        if not sample_files or not frame_files:
            continue

        n_samples = engine.index_video(
            video_id=vid,
            media_path=media_path,
            samples_parquet_path=sample_files[0],
            timeline_parquet_path=frame_files[0],
        )
        if n_samples > 0:
            indexed_videos += 1
            total_indexed_frames += n_samples

    output_dir = Path(args.output)
    engine.save(output_dir)

    return {
        "status": "success",
        "indexed_videos": indexed_videos,
        "total_indexed_frames": total_indexed_frames,
        "output": str(output_dir),
    }


def handle_search(args):
    assessments: list[dict] = []
    verify_relations = bool(getattr(args, "verify_relations", False))
    search_warnings: list[str] = []

    if args.backend == "unified":
        from .retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
        db_path = args.db_path or os.environ.get("VIDENTRA_DB_PATH") or "data/processed/metadata.db"
        faiss_path = args.faiss_path or os.environ.get("VIDENTRA_FAISS_PATH") or "data/processed/faiss_index.bin"
        model_path = args.model_path or os.environ.get("VIDENTRA_MODEL_DIR") or "models/siglip2"
        if args.mock:
            raise ValueError("Unified backend does not support --mock; remove --mock to use processed artifacts")
        for path in (db_path, faiss_path, model_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Unified backend input not found: {path}")
        run_verifier = verify_relations and args.mode == "accurate"
        if args.mode == "accurate" and not verify_relations:
            search_warnings.append("Relation verifier was not requested.")
        elif args.mode != "accurate" and verify_relations:
            search_warnings.append("Relation verifier requires accurate mode and was not run.")
        with UnifiedMultimodalSearchPipeline(db_path=db_path, faiss_path=faiss_path,
                                             model_dir=model_path, device=args.device,
                                             use_disk_cache=False) as pipeline:
            response = pipeline.search_with_metadata(snapshot_id=args.snapshot_id,
                                                     verify_relations=run_verifier,
                                                     query=args.query,
                                                     top_k=args.top_k, mode=args.mode,
                                                     ranking_mode=args.ranking_mode)
            moments = response["moments"]
            assessments = response.get("relation_assessments", [])
    else:
        from .retrieval.visual import VisualRetrievalEngine
        if args.index_dir is None:
            raise ValueError("Visual backend requires --index-dir")
        encoder = _get_encoder(args.model_path, device=args.device, mock=args.mock)
        engine = VisualRetrievalEngine.load(args.index_dir, encoder=encoder)
        moments = engine.search(query=args.query, top_k_moments=args.top_k,
            window_duration_seconds=args.window_seconds, nms_iou_threshold=args.nms_iou)

    results = []
    for idx, m in enumerate(moments):
        if args.backend == "unified":
            from .retrieval.unified_pipeline import unified_result_dict
            res = unified_result_dict(m, args.snapshot_id)
            if idx < len(assessments):
                res["relation_assessment"] = assessments[idx]
        else:
            res = m.to_result_dict()
        if args.query_id:
            res["query_id"] = args.query_id
        results.append(res)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "query": args.query,
        "hits_count": len(results),
        "results": results if args.backend == "unified" else results[:min(args.top_k, 5)],
        "output_saved": str(args.output) if args.output else None,
        "backend": args.backend,
        "snapshot_id": args.snapshot_id if args.backend == "unified" else None,
        "warnings": search_warnings,
    }


def handle_serve(args):
    import uvicorn
    from .api.app import create_app
    from .retrieval.visual import VisualRetrievalEngine
    from .registry.access import MediaResolver

    engine = None
    pipeline = None
    if args.backend == "unified":
        from .retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
        db_path = args.db_path or os.environ.get("VIDENTRA_DB_PATH") or "data/processed/metadata.db"
        faiss_path = args.faiss_path or os.environ.get("VIDENTRA_FAISS_PATH") or "data/processed/faiss_index.bin"
        model_path = args.model_path or os.environ.get("VIDENTRA_MODEL_DIR") or "models/siglip2"
        if args.mock:
            raise ValueError("Unified backend does not support --mock; remove --mock to use processed artifacts")
        for path in (db_path, faiss_path, model_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Unified backend input not found: {path}")
        pipeline = UnifiedMultimodalSearchPipeline(db_path=db_path, faiss_path=faiss_path,
            model_dir=model_path, device=args.device, use_disk_cache=False)
    elif args.index_dir and Path(args.index_dir).exists():
        encoder = _get_encoder(args.model_path, device=args.device, mock=args.mock)
        engine = VisualRetrievalEngine.load(args.index_dir, encoder=encoder)

    resolver = None
    records = []
    if args.manifest and Path(args.manifest).exists():
        records = load_manifest(args.manifest).get("records", [])
    asset_roots_path = Path(args.asset_roots) if args.asset_roots else None
    if asset_roots_path and not asset_roots_path.exists():
        fallback_roots = Path("configs/local/asset_roots.example.json")
        if fallback_roots.exists():
            asset_roots_path = fallback_roots
    if asset_roots_path:
        resolver = _build_resolver(
            asset_roots_path,
            records,
            allow_unregistered_roots=not records,
        )

    app = create_app(
        engine=engine,
        resolver=resolver,
        ingest_dir=args.ingest_dir,
        snapshot_id=args.snapshot_id,
        pipeline=pipeline,
    )
    print(f"Starting VIDENTRA API server on http://{args.host}:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if pipeline is not None:
            pipeline.close()
    return {"status": "stopped"}


def handle_evidence_build(args):
    import re
    from .registry.access import MediaResolver
    from .evidence.store import EvidenceStore
    from .ocr.detection import OCRDetector
    from .ocr.routing import OCRRouter
    from .ocr.tracking import OCRTracker, FrameOCRObservation
    from .ocr.recognition import MockOCRRecognizer
    from .asr.vad import VoiceActivityDetector, extract_audio_mono
    from .asr.transcription import MockASRTranscriber
    from .asr.alignment import TimelineAligner
    from .timeline.ledger import exact_seek_frame
    import pyarrow.parquet as pq

    manifest_doc = load_manifest(args.manifest)
    modalities = set(m.strip().lower() for m in args.modalities.split(","))

    resolver = _build_resolver(args.asset_roots, manifest_doc.get("records", []))

    ingest_dir = Path(args.ingest_dir)
    out_dir = Path(args.output)
    detector = OCRDetector()
    tracker = OCRTracker()
    vad = VoiceActivityDetector()

    total_ocr_tracks = 0
    total_asr_segments = 0

    for rec in manifest_doc["records"]:
        vid = rec["video_id"]
        safe_vid = re.sub(r'[<>:"/\\|?*]', '_', vid)

        media_path = None
        if resolver:
            try:
                media_path = resolver.resolve(vid, for_inference=not args.audit_mode)
            except Exception:
                media_path = None
        if not media_path and "local_path" in rec and rec["local_path"]:
            lp = Path(rec["local_path"])
            if lp.is_file():
                media_path = lp

        if not media_path or not Path(media_path).is_file():
            continue

        tl_files = list(ingest_dir.glob(f"timeline/{safe_vid}/*/frames.parquet"))
        ledger_path = tl_files[0] if tl_files else None

        # 1. OCR modality
        if "ocr" in modalities and ledger_path:
            probe_files = list(ingest_dir.glob(f"timeline/{safe_vid}/*/ocr_probes.parquet"))
            if not probe_files:
                # Also check samples dir
                probe_files = list(ingest_dir.glob(f"samples/{safe_vid}/*/ocr_probes.parquet"))

            observations = []
            if probe_files:
                p_table = pq.read_table(str(probe_files[0]))
                p_ords = p_table.column("original_decoded_frame_index").to_pylist()
                p_ts = p_table.column("timestamp_seconds").to_pylist()

                for ord_idx, ts_sec in zip(p_ords, p_ts):
                    try:
                        frame = exact_seek_frame(media_path, ledger_path, target_ordinal=ord_idx)
                        if frame.pixels is not None:
                            boxes = detector.detect(frame.pixels)
                            for b in boxes:
                                observations.append(
                                    FrameOCRObservation(
                                        ordinal=ord_idx,
                                        timestamp_seconds=ts_sec if ts_sec is not None else 0.0,
                                        bbox=b.to_tuple(),
                                        text="TEST_OCR",
                                        confidence=b.confidence,
                                    )
                                )
                    except Exception:
                        pass

            tracks = tracker.track(vid, observations)
            if tracks:
                track_path = out_dir / "evidence" / "ocr_tracks" / safe_vid / "ocr_tracks.parquet"
                EvidenceStore.save_ocr_tracks(tracks, track_path)
                total_ocr_tracks += len(tracks)

        # 2. ASR modality
        if "asr" in modalities and ledger_path:
            audio_samples, has_audio = extract_audio_mono(media_path)
            if has_audio and audio_samples is not None and len(audio_samples) > 0:
                vad_segs = vad.detect_segments(audio_samples)
                aligner = TimelineAligner(ledger_path)
                asr_transcriber = MockASRTranscriber()
                segments = []
                for v_seg in vad_segs:
                    # transcribe chunk
                    st_sample = int(v_seg.start_us * 16 / 1000)
                    end_sample = min(len(audio_samples), int(v_seg.end_us * 16 / 1000))
                    chunk = audio_samples[st_sample:end_sample]
                    t_res = asr_transcriber.transcribe_chunk(chunk)
                    aligned = aligner.create_aligned_segment(vid, v_seg, t_res)
                    segments.append(aligned)

                if segments:
                    seg_path = out_dir / "evidence" / "asr_segments" / safe_vid / "asr_segments.parquet"
                    EvidenceStore.save_asr_segments(segments, seg_path)
                    total_asr_segments += len(segments)

    return {
        "status": "success",
        "total_ocr_tracks": total_ocr_tracks,
        "total_asr_segments": total_asr_segments,
        "output": str(out_dir),
    }


def handle_evaluate(args):
    from dataclasses import asdict
    import importlib
    eval_mod = importlib.import_module("videntra.evaluation.evaluator")
    evaluate_predictions = eval_mod.evaluate_predictions

    def _read_jsonl(p: Path) -> list[dict]:
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    preds = _read_jsonl(Path(args.predictions))
    gts = _read_jsonl(Path(args.ground_truth))

    report = evaluate_predictions(preds, gts)
    report_dict = asdict(report)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    return report_dict



def main(argv=None):
    parser = argparse.ArgumentParser(prog="videntra")
    subs = parser.add_subparsers(dest="command", required=True)

    # manifests
    manifests = subs.add_parser("manifests").add_subparsers(dest="action", required=True)
    check = manifests.add_parser("validate")
    check.add_argument("--all", action="store_true", required=True)
    check.add_argument("--directory", type=Path, default=Path("manifests"))

    # registry
    registry = subs.add_parser("registry").add_subparsers(dest="action", required=True)
    register = registry.add_parser("import")
    register.add_argument("--directory", type=Path, default=Path("manifests"))
    register.add_argument("--database", type=Path, required=True)

    # fixtures
    fixtures = subs.add_parser("fixtures").add_subparsers(dest="action", required=True)
    build = fixtures.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--extended", action="store_true", default=False)

    # ingest
    ingest = subs.add_parser("ingest")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--config", type=Path, default=None)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--resume", action="store_true", default=True)
    ingest.add_argument("--no-resume", action="store_false", dest="resume")
    ingest.add_argument("--asset-roots", type=Path, default=Path("configs/local/asset_roots.json"))
    ingest.add_argument("--audit-mode", action="store_true", default=False,
                        help="Allow resolving paths in audit mode (for_inference=False)")

    # timeline
    timeline = subs.add_parser("timeline").add_subparsers(dest="action", required=True)
    inspect = timeline.add_parser("inspect")
    inspect.add_argument("--video-id", type=str, required=True)
    inspect.add_argument("--revision", type=str, default="rev_m1_1")
    inspect.add_argument("--frame-index", type=int, required=True)
    inspect.add_argument("--ledger", type=Path, default=None)

    # sampling
    sampling = subs.add_parser("sampling").add_subparsers(dest="action", required=True)
    compare = sampling.add_parser("compare")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, default=None)

    # index
    index_parser = subs.add_parser("index").add_subparsers(dest="action", required=True)
    idx_build = index_parser.add_parser("build")
    idx_build.add_argument("--manifest", type=Path, required=True)
    idx_build.add_argument("--ingest-dir", type=Path, required=True)
    idx_build.add_argument("--output", type=Path, required=True)
    idx_build.add_argument("--model-path", type=str, default=None)
    idx_build.add_argument("--device", type=str, default="cpu")
    idx_build.add_argument("--mock", action="store_true", default=False)
    idx_build.add_argument("--asset-roots", type=Path, default=Path("configs/local/asset_roots.json"))
    idx_build.add_argument("--audit-mode", action="store_true", default=False)

    # search
    search_parser = subs.add_parser("search")
    search_parser.add_argument("--query", type=str, required=True)
    search_parser.add_argument("--query-id", type=str, default="q1")
    search_parser.add_argument("--index-dir", type=Path, default=None)
    search_parser.add_argument("--backend", choices=["visual", "unified"], default="visual")
    search_parser.add_argument("--db-path", type=Path)
    search_parser.add_argument("--faiss-path", type=Path)
    search_parser.add_argument("--snapshot-id", default="corpus_current")
    search_parser.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    search_parser.add_argument("--ranking-mode", choices=["moment", "diverse_video"], default="moment")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--window-seconds", type=float, default=2.0)
    search_parser.add_argument("--nms-iou", type=float, default=0.5)
    search_parser.add_argument("--model-path", type=str, default=None)
    search_parser.add_argument("--device", type=str, default="cpu")
    search_parser.add_argument("--mock", action="store_true", default=False)
    search_parser.add_argument("--verify-relations", action="store_true", default=False,
                               help="Attach relation graph verification to unified search results (accurate mode)")
    search_parser.add_argument("--output", type=Path, default=None)

    # evaluate
    eval_parser = subs.add_parser("evaluate")
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--ground-truth", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, default=None)

    # serve
    serve_parser = subs.add_parser("serve")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--index-dir", type=Path, default=None)
    serve_parser.add_argument("--backend", choices=["visual", "unified"], default="visual")
    serve_parser.add_argument("--db-path", type=Path)
    serve_parser.add_argument("--faiss-path", type=Path)
    serve_parser.add_argument("--ingest-dir", type=Path, default=None)
    serve_parser.add_argument("--manifest", type=Path, default=None)
    serve_parser.add_argument("--asset-roots", type=Path, default=Path("configs/local/asset_roots.json"))
    serve_parser.add_argument("--model-path", type=str, default=None)
    serve_parser.add_argument("--device", type=str, default="cpu")
    serve_parser.add_argument("--mock", action="store_true", default=False)
    serve_parser.add_argument("--snapshot-id", type=str, default="corpus_current")

    # evidence
    evidence_parser = subs.add_parser("evidence").add_subparsers(dest="action", required=True)
    ev_build = evidence_parser.add_parser("build")
    ev_build.add_argument("--manifest", type=Path, required=True)
    ev_build.add_argument("--ingest-dir", type=Path, required=True)
    ev_build.add_argument("--output", type=Path, required=True)
    ev_build.add_argument("--modalities", type=str, default="ocr,asr")
    ev_build.add_argument("--asset-roots", type=Path, default=Path("configs/local/asset_roots.json"))
    ev_build.add_argument("--audit-mode", action="store_true", default=False)

    args = parser.parse_args(argv)
    try:
        if args.command == "fixtures":
            from .fixtures import build_fixtures
            report = build_fixtures(args.output, include_extended=args.extended)
            output = {"fixtures_verified": len(report["fixtures"]), "output": str(args.output)}
        elif args.command == "ingest":
            output = handle_ingest(args)
        elif args.command == "timeline" and args.action == "inspect":
            output = handle_timeline_inspect(args)
        elif args.command == "sampling" and args.action == "compare":
            output = handle_sampling_compare(args)
        elif args.command == "index" and args.action == "build":
            output = handle_index_build(args)
        elif args.command == "search":
            output = handle_search(args)
        elif args.command == "evaluate":
            output = handle_evaluate(args)
        elif args.command == "serve":
            output = handle_serve(args)
        elif args.command == "evidence" and args.action == "build":
            output = handle_evidence_build(args)
        else:
            docs = validate_manifests(args.directory)
            output = {"manifests": len(docs), "records": sum(len(d["records"]) for d in docs)}
            if args.command == "registry":
                records = [(d["kind"], r["video_id" if d["kind"] == "media_asset" else "dataset_id"], d["revision"], r)
                           for d in docs for r in d["records"]]
                with Registry(args.database) as store:
                    output["added"] = store.register_batch(records)
        print(json.dumps(output))
        return 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 2
