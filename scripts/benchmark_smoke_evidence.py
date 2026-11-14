"""Extract OCR and ASR evidence and populate FTS5 text store for benchmark smoke subset."""
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

from videntra.ingestion.probe import probe_media
from videntra.timeline.ledger import exact_seek_frame
from videntra.ocr.detection import OCRDetector
from videntra.ocr.tracking import OCRTracker, FrameOCRObservation
from videntra.ocr.routing import OCRRouter
from videntra.ocr.recognition import GeneralOCRRecognizer, VietOCRRecognizer
from videntra.asr.vad import VoiceActivityDetector, extract_audio_mono
from videntra.asr.transcription import PhoWhisperTranscriber
from videntra.asr.alignment import TimelineAligner
from videntra.evidence.store import EvidenceStore
from videntra.indexing.text_store import TextIndexStore

def run_smoke_evidence():
    ingest_dir = Path("work/benchmark/smoke/ingest")
    evidence_dir = Path("work/benchmark/smoke/evidence")
    text_db_path = Path("work/benchmark/smoke/text_index.db")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path("manifests/benchmark/smoke/benchmark_smoke.media.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    media_root = Path(os.environ.get("VIDENTRA_QVH_MEDIA_DIR", "data/raw/qvhighlights/media"))
    asr_model_p = Path(os.environ.get("VIDENTRA_ASR_MODEL_DIR", "models/asr/phowhisper_small"))

    detector = OCRDetector()
    tracker = OCRTracker()
    router = OCRRouter(
        general_recognizer=GeneralOCRRecognizer(),
        vietnamese_recognizer=VietOCRRecognizer(),
    )
    vad = VoiceActivityDetector()
    transcriber = PhoWhisperTranscriber(model_path=asr_model_p, device="cpu")
    text_store = TextIndexStore(text_db_path)

    extracted_summary = []

    for rec in manifest_doc["records"]:
        vid = rec["video_id"]
        sha = rec["sha256"]
        media_filename = Path(rec["media_uri"]).name
        media_path = media_root / media_filename
        if not media_path.exists():
            continue

        tl_matches = list(ingest_dir.glob(f"timeline/*{sha}*/*/frames.parquet"))
        if not tl_matches:
            continue
        tl_file = tl_matches[0]
        aligner = TimelineAligner(tl_file)

        # 1. ASR Extraction
        audio_samples, has_audio = extract_audio_mono(media_path)
        asr_segments = []
        if has_audio and audio_samples is not None and len(audio_samples) > 0:
            vad_segs = vad.detect_segments(audio_samples)
            # Process up to 10 key speech segments per video for smoke
            for v_seg in vad_segs[:10]:
                st = int(v_seg.start_us * 16 / 1000)
                et = min(len(audio_samples), int(v_seg.end_us * 16 / 1000))
                chunk = audio_samples[st:et]
                t_res = transcriber.transcribe_chunk(chunk)
                if t_res.text.strip():
                    aligned = aligner.create_aligned_segment(vid, v_seg, t_res)
                    asr_segments.append(aligned)

        if asr_segments:
            asr_out = evidence_dir / "asr_segments" / f"{sha}.parquet"
            asr_out.parent.mkdir(parents=True, exist_ok=True)
            EvidenceStore.save_asr_segments(asr_segments, asr_out)

            # Add to FTS5 index
            asr_entries = [
                {
                    "id": s.segment_id,
                    "text": s.text,
                    "start_seconds": s.start_seconds,
                    "end_seconds": s.end_seconds,
                    "start_ordinal": s.start_ordinal,
                    "end_ordinal": s.end_ordinal,
                    "confidence": s.confidence,
                }
                for s in asr_segments
            ]
            text_store.add_entries(video_id=vid, source_type="asr", entries=asr_entries)

        # 2. OCR Extraction
        sample_matches = list(ingest_dir.glob(f"samples/*{sha}*/*/ocr_probes.parquet"))
        ocr_obs = []
        if sample_matches:
            p_table = pq.read_table(str(sample_matches[0]))
            p_ords = p_table.column("original_decoded_frame_index").to_pylist()
            p_ts = p_table.column("timestamp_seconds").to_pylist()

            # Examine sample probes
            for ord_idx, ts_sec in zip(p_ords[:15], p_ts[:15]):
                try:
                    frame = exact_seek_frame(media_path, tl_file, target_ordinal=ord_idx)
                    if frame.pixels is not None:
                        boxes = detector.detect(frame.pixels)
                        for b in boxes:
                            # Crop text patch if dimensions allow
                            h, w = frame.pixels.shape[:2]
                            y1, y2 = int(b.y1 * h), int(b.y2 * h)
                            x1, x2 = int(b.x1 * w), int(b.x2 * w)
                            crop = frame.pixels[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame.pixels
                            rec_res, _ = router.route_and_recognize(crop)
                            ocr_obs.append(
                                FrameOCRObservation(
                                    ordinal=ord_idx,
                                    timestamp_seconds=ts_sec or 0.0,
                                    bbox=b.to_tuple(),
                                    text=rec_res.text,
                                    confidence=rec_res.confidence,
                                )
                            )
                except Exception:
                    pass

        ocr_tracks = tracker.track(vid, ocr_obs)
        if ocr_tracks:
            ocr_out = evidence_dir / "ocr_tracks" / f"{sha}.parquet"
            ocr_out.parent.mkdir(parents=True, exist_ok=True)
            EvidenceStore.save_ocr_tracks(ocr_tracks, ocr_out)

            # Add to FTS5 index
            ocr_entries = [
                {
                    "id": t.track_id,
                    "text": t.text,
                    "start_seconds": t.start_seconds,
                    "end_seconds": t.end_seconds,
                    "start_ordinal": t.start_ordinal,
                    "end_ordinal": t.end_ordinal,
                    "confidence": t.confidence,
                }
                for t in ocr_tracks
            ]
            text_store.add_entries(video_id=vid, source_type="ocr", entries=ocr_entries)

        extracted_summary.append({
            "video_id": vid,
            "filename": media_filename,
            "asr_segments": len(asr_segments),
            "ocr_tracks": len(ocr_tracks),
        })

    receipt_path = evidence_dir / "extraction_summary.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(extracted_summary, f, indent=2, ensure_ascii=False)

    print("SUCCESS: Evidence extraction complete for smoke dataset.")
    for item in extracted_summary:
        print(f"Video {item['filename']}: ASR={item['asr_segments']} segments, OCR={item['ocr_tracks']} tracks")

if __name__ == "__main__":
    run_smoke_evidence()
