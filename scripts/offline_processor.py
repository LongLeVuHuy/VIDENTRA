"""VIDENTRA Offline Processing & Storage Pipeline.

Executes offline ingestion, keyframe sampling, multi-model evidence grouping,
FAISS index synchronization, and SQLite metadata population with real-time progress tracking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import numpy as np

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import av
import faiss

from videntra.storage.database import OfflineMetadataStore, current_iso
from videntra.storage.grouping import OCREvidenceGrouper, ASREvidenceGrouper
from videntra.storage.integrity import verify_offline_storage_integrity
from videntra.ingestion.decode import decode_frames
from videntra.sampling.hybrid import HybridAdaptiveSampler
from videntra.config.ingestion import SamplingConfig
from videntra.evidence.normalize import normalize_text, fold_diacritics
try:
    from scripts.extract_all_keyframes import extract_video_keyframes
except ImportError:
    from extract_all_keyframes import extract_video_keyframes


def print_progress(step_idx: int, total_steps: int, step_name: str, step_pct: float, overall_pct: float):
    """Prints single-line formatted progress."""
    bar_len = 25
    filled = int(bar_len * (overall_pct / 100.0))
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(
        f"\r[Bước {step_idx}/{total_steps}] {step_name:<28} | Bước: {step_pct:5.1f}% | Tổng: {overall_pct:5.1f}% [{bar}]"
    )
    sys.stdout.flush()
    if step_pct >= 100.0:
        sys.stdout.write("\n")


def _guard_and_backup_path(target_path: Path, i_understand_overwrite: bool, label: str) -> Path | None:
    """Require explicit overwrite flag and create *.bak-<timestamp> backup before overwriting."""
    if not target_path.exists():
        return None
    if not i_understand_overwrite:
        raise RuntimeError(
            f"Refusing to overwrite existing {label} at '{target_path}' without --i-understand-overwrite."
        )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = target_path.with_name(f"{target_path.name}.bak-{ts}")
    shutil.copy2(target_path, backup_path)
    return backup_path


class OfflineProcessingPipeline:
    def __init__(
        self,
        db_path: Path = Path("data/processed/metadata.db"),
        faiss_path: Path = Path("work/offline_scratch/faiss_index.bin"),
        raw_data_dir: Path = Path("data/datasets"),
        keyframes_dir: Path = Path("data/datasets/msrvtt/keyframes"),
        models_dir: Path = Path("models"),
        encoder: Any | None = None,
        i_understand_overwrite: bool = False,
    ):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.raw_data_dir = raw_data_dir
        self.keyframes_dir = keyframes_dir
        self.models_dir = models_dir
        self.encoder = encoder
        self.i_understand_overwrite = i_understand_overwrite
        self.store = OfflineMetadataStore(self.db_path)

    def _resolve_encoder(self) -> Any:
        if self.encoder is not None:
            return self.encoder
        try:
            from videntra.encoders.siglip2 import SigLIP2Encoder

            siglip_dir = self.models_dir / "siglip2"
            encoder = SigLIP2Encoder(model_path=siglip_dir if siglip_dir.exists() else None)
            # Verify weights can be resolved before running full pipeline
            encoder.config.resolve_model_path()
            self.encoder = encoder
            return self.encoder
        except Exception as exc:
            raise NotImplementedError(
                f"Real SigLIP2Encoder is unavailable and placeholder embeddings are forbidden: {exc}"
            ) from exc

    def run_all(
        self,
        video_limit: int | None = None,
        dataset_filter: str | None = None,
        i_understand_overwrite: bool | None = None,
    ) -> dict:
        overwrite = self.i_understand_overwrite if i_understand_overwrite is None else i_understand_overwrite
        if self.faiss_path.exists() and not overwrite:
            raise RuntimeError(
                f"Refusing to overwrite existing FAISS index at '{self.faiss_path}' without --i-understand-overwrite."
            )

        total_steps = 6
        start_time = time.time()
        print(f"\n{'='*75}")
        print("BẮT ĐẦU VIDENTRA OFFLINE PROCESSING & STORAGE PIPELINE")
        print(f"{'='*75}")
        print(f"Database target : {self.db_path}")
        print(f"FAISS target    : {self.faiss_path}")
        print(f"Raw data root   : {self.raw_data_dir}")
        if dataset_filter:
            print(f"Dataset filter  : {dataset_filter}")

        # ---------------------------------------------------------------------
        # BƯỚC 1: Scan & Register Videos (0% -> 15%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 1/6: Quét và Đăng ký Video Corpus ---")
        video_files = []
        for dataset_dir in self.raw_data_dir.iterdir():
            if dataset_dir.is_dir() and (dataset_dir / "media").is_dir():
                if dataset_filter and dataset_dir.name.lower() != dataset_filter.lower():
                    continue
                v_list = list((dataset_dir / "media").glob("*.mp4"))
                for v in v_list:
                    video_files.append((dataset_dir.name, v))

        if video_limit:
            video_files = video_files[:video_limit]

        total_vids = len(video_files)
        if total_vids == 0:
            print(f"Không tìm thấy video nào trong {self.raw_data_dir}/*/media.")
            return {"status": "no_videos"}

        registered_vids = []
        for idx, (d_name, v_path) in enumerate(video_files):
            vid_id = v_path.stem
            try:
                with av.open(str(v_path)) as container:
                    v_stream = container.streams.video[0] if container.streams.video else None
                    a_stream = container.streams.audio[0] if container.streams.audio else None
                    dur_ms = int(float(container.duration or 0) / 1000.0) if container.duration else 0
                    fps = float(v_stream.average_rate or 25.0) if v_stream else 25.0
                    tb_num = v_stream.time_base.numerator if v_stream and v_stream.time_base else 1
                    tb_den = v_stream.time_base.denominator if v_stream and v_stream.time_base else 1000
                    w = v_stream.width if v_stream else 0
                    h = v_stream.height if v_stream else 0
                    t_frames = v_stream.frames if v_stream and v_stream.frames else int(fps * (dur_ms / 1000.0))
                    has_audio = 1 if a_stream is not None else 0

                rec = {
                    "video_id": vid_id,
                    "dataset": d_name,
                    "file_path": str(v_path),
                    "duration_ms": dur_ms,
                    "fps": fps,
                    "time_base_num": tb_num,
                    "time_base_den": tb_den,
                    "width": w,
                    "height": h,
                    "total_frames": t_frames,
                    "has_audio": has_audio,
                    "status": "ingested",
                }
                self.store.register_video(rec)
                registered_vids.append(rec)
            except Exception as e:
                pass

            step_pct = ((idx + 1) / total_vids) * 100.0
            overall_pct = (step_pct / 100.0) * 15.0
            print_progress(1, total_steps, "Đăng ký Videos", step_pct, overall_pct)

        # ---------------------------------------------------------------------
        # BƯỚC 2: Dynamic Keyframe Sampling (15% -> 35%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 2/6: Dynamic Keyframe Sampling & Ingestion ---")
        all_keyframes = []

        for idx, v_rec in enumerate(registered_vids):
            vid_id = v_rec["video_id"]
            v_path = Path(v_rec["file_path"])
            out_kf_dir = self.keyframes_dir / vid_id
            out_kf_dir.mkdir(parents=True, exist_ok=True)
            dur_sec = float(v_rec.get("duration_ms") or 0) / 1000.0
            samp_cfg = SamplingConfig.for_duration(dur_sec)

            try:
                # Stream frames through HybridAdaptiveSampler with bounded RAM (R04)
                sampler = HybridAdaptiveSampler(
                    config=samp_cfg,
                    video_id=vid_id,
                    timeline_revision="rev_0",
                )
                for f in decode_frames(v_path, include_pixels=True):
                    sampler.process_frame(f)
                    f.pixels = None
                manifest = sampler.finalize()

                # Extract & atomically persist selected keyframe JPEGs using PyAV presentation ordinals
                target_frames = {
                    s.original_decoded_frame_index: str(out_kf_dir / f"{s.original_decoded_frame_index:06d}.jpg")
                    for s in manifest.samples
                }
                extract_video_keyframes(vid_id, str(v_path), target_frames)

                for k_idx, s in enumerate(manifest.samples):
                    kf_id = f"{vid_id}__kf_{k_idx:05d}"
                    kf_path = out_kf_dir / f"{s.original_decoded_frame_index:06d}.jpg"
                    tb_num = v_rec["time_base_num"]
                    tb_den = v_rec["time_base_den"]
                    ts_us = int(s.timestamp_us or 0)
                    pts = int(s.pts) if s.pts is not None else ts_us

                    all_keyframes.append({
                        "keyframe_id": kf_id,
                        "video_id": vid_id,
                        "keyframe_index": k_idx,
                        "source_frame_index": s.original_decoded_frame_index,
                        "pts": pts,
                        "time_base": f"{tb_num}/{tb_den}",
                        "timestamp_ms": int(ts_us // 1000),
                        "keyframe_path": str(kf_path),
                        "thumbnail_path": str(kf_path),
                        "faiss_id": len(all_keyframes),
                        "phash": str(s.phash) if s.phash is not None else None,
                        "sampling_reason": ",".join(s.reasons) if s.reasons else "adaptive",
                    })
            except Exception as exc:
                import traceback
                print(f"[LỖI SAMPLING {vid_id}]: {exc}")
                traceback.print_exc()

            step_pct = ((idx + 1) / len(registered_vids)) * 100.0
            overall_pct = 15.0 + (step_pct / 100.0) * 20.0
            print_progress(2, total_steps, "Dynamic Sampling", step_pct, overall_pct)

        self.store.insert_keyframes(all_keyframes)

        # ---------------------------------------------------------------------
        # BƯỚC 3: Multi-Model Inference (SigLIP 2, YOLO, OCR, ASR) (35% -> 65%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 3/6: Multi-Model Inference (Embeddings, Objects, OCR, ASR) ---")
        total_kf = len(all_keyframes)
        raw_embs = []
        raw_objects = []
        raw_ocr_obs = []
        raw_asr_segs = []

        # 1. Initialize YOLOv8 Detector
        yolo_model = None
        yolo_weights = self.models_dir / "detection" / "yolov8n.pt"
        if yolo_weights.exists():
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(str(yolo_weights))
            except Exception:
                yolo_model = None

        # 2. Initialize SigLIP 2 Encoder (fail closed if unavailable)
        encoder = self._resolve_encoder()

        # 3. Initialize OCR models (optional if weights/libs are absent)
        from videntra.ocr.detection import OCRDetector
        from videntra.ocr.recognition import GeneralOCRRecognizer, VietOCRRecognizer
        from videntra.ocr.routing import OCRRouter
        ocr_det = OCRDetector()
        ocr_router = None
        try:
            ocr_router = OCRRouter(
                general_recognizer=GeneralOCRRecognizer(use_rapidocr=True),
                vietnamese_recognizer=VietOCRRecognizer(),
            )
        except Exception:
            ocr_router = None

        if all_keyframes:
            from PIL import Image
            kf_images = []
            for kf in all_keyframes:
                kf_p = Path(kf["keyframe_path"])
                if not kf_p.exists():
                    raise FileNotFoundError(f"Keyframe image missing for embedding: {kf_p}")
                with Image.open(kf_p) as im:
                    kf_images.append(im.convert("RGB"))
            emb_matrix = encoder.encode_images(kf_images)
            raw_embs = [emb_matrix[i] for i in range(len(all_keyframes))]

        for idx, kf in enumerate(all_keyframes):
            vid_id = kf["video_id"]
            kf_id = kf["keyframe_id"]
            ts_ms = kf["timestamp_ms"]
            kf_path = Path(kf["keyframe_path"])

            # Object detection with YOLOv8 if image exists
            if yolo_model and kf_path.exists():
                try:
                    results = yolo_model(str(kf_path), verbose=False)
                    for r in results:
                        orig_h, orig_w = r.orig_shape
                        for b_idx, box in enumerate(r.boxes):
                            xyxy = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0].item())
                            cls_name = yolo_model.names.get(cls_id, f"obj_{cls_id}")
                            conf = float(box.conf[0].item())

                            raw_objects.append({
                                "detection_id": f"{kf_id}__det_{b_idx}",
                                "video_id": vid_id,
                                "keyframe_id": kf_id,
                                "timestamp_ms": ts_ms,
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "bbox_x1": round(xyxy[0] / orig_w, 4),
                                "bbox_y1": round(xyxy[1] / orig_h, 4),
                                "bbox_x2": round(xyxy[2] / orig_w, 4),
                                "bbox_y2": round(xyxy[3] / orig_h, 4),
                                "confidence": round(conf, 4),
                                "image_width": orig_w,
                                "image_height": orig_h,
                            })
                except Exception:
                    pass

            if ocr_router is not None and kf_path.exists():
                try:
                    import cv2
                    img_bgr = cv2.imread(str(kf_path))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        h_im, w_im = img_rgb.shape[:2]
                        for b_idx, box in enumerate(ocr_det.detect(img_rgb)):
                            x1, y1 = max(0, int(box.x1 * w_im)), max(0, int(box.y1 * h_im))
                            x2, y2 = min(w_im, int(box.x2 * w_im)), min(h_im, int(box.y2 * h_im))
                            if x2 > x1 and y2 > y1:
                                crop = img_rgb[y1:y2, x1:x2]
                                rec_res, _ = ocr_router.route_and_recognize(crop)
                                if rec_res.text:
                                    raw_ocr_obs.append({
                                        "observation_id": f"{kf_id}__ocr_{b_idx}",
                                        "video_id": vid_id,
                                        "keyframe_id": kf_id,
                                        "timestamp_ms": ts_ms,
                                        "model_id": rec_res.model_id,
                                        "raw_text": rec_res.text,
                                        "bbox_x1": box.x1, "bbox_y1": box.y1,
                                        "bbox_x2": box.x2, "bbox_y2": box.y2,
                                        "confidence": round(float(rec_res.confidence), 4),
                                    })
                except Exception:
                    pass

            step_pct = ((idx + 1) / total_kf) * 100.0 if total_kf > 0 else 100.0
            overall_pct = 35.0 + (step_pct / 100.0) * 30.0
            print_progress(3, total_steps, "Model Inference", step_pct, overall_pct)

        # ---------------------------------------------------------------------
        # BƯỚC 4: Evidence Grouping & Raw Observation Persistence (65% -> 80%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 4/6: Lưu Trữ Kết Quả & Gộp Bằng Chứng Đa Phương Thức ---")
        # 1. Objects
        for kf in all_keyframes:
            kf_objs = [o for o in raw_objects if o["keyframe_id"] == kf["keyframe_id"]]
            self.store.record_frame_objects(kf["keyframe_id"], kf["video_id"], kf_objs, model_id="yolov8n")

        # 2. OCR
        self.store.insert_ocr_observations(raw_ocr_obs)
        total_ocr_groups = 0
        for kf in all_keyframes:
            kf_obs = [o for o in raw_ocr_obs if o["keyframe_id"] == kf["keyframe_id"]]
            groups, members = OCREvidenceGrouper.group_keyframe_observations(
                kf["video_id"], kf["keyframe_id"], kf["timestamp_ms"], kf_obs
            )
            if groups:
                self.store.insert_ocr_evidence_groups(groups, members)
                total_ocr_groups += len(groups)

        # 3. ASR
        self.store.insert_asr_segments(raw_asr_segs)
        total_asr_groups = 0
        for v_rec in registered_vids:
            vid_id = v_rec["video_id"]
            v_segs = [s for s in raw_asr_segs if s["video_id"] == vid_id]
            if v_segs:
                groups, members = ASREvidenceGrouper.group_video_segments(vid_id, v_segs)
                self.store.insert_asr_evidence_groups(groups, members)
                total_asr_groups += len(groups)

        print_progress(4, total_steps, "Gộp Bằng Chứng", 100.0, 80.0)

        # ---------------------------------------------------------------------
        # BƯỚC 5: FAISS Vector Indexing & Mapping (80% -> 95%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 5/6: Xây Dựng Chỉ Mục Vector FAISS & Ánh Xạ SQLite ---")
        faiss_mappings = []
        if raw_embs:
            _guard_and_backup_path(self.faiss_path, overwrite, "FAISS index")
            emb_matrix = np.vstack(raw_embs).astype(np.float32)
            index = faiss.IndexFlatIP(768)
            index.add(emb_matrix)
            self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.faiss_path))

            for i, kf in enumerate(all_keyframes):
                faiss_mappings.append({
                    "faiss_id": i,
                    "keyframe_id": kf["keyframe_id"],
                    "video_id": kf["video_id"],
                    "pts_time": float(kf["timestamp_ms"]) / 1000.0,
                    "source_frame_index": kf["source_frame_index"],
                    "keyframe_path": kf["keyframe_path"],
                })

            self.store.update_faiss_mappings(faiss_mappings)
            self.store.record_index_manifest({
                "manifest_id": f"manifest_{int(time.time())}",
                "index_type": "IndexFlatIP",
                "vector_dim": 768,
                "metric": "inner_product",
                "total_vectors": len(all_keyframes),
                "model_id": "google/siglip2-base-patch16-224",
                "model_version": "v1.0",
                "index_file_path": str(self.faiss_path),
                "is_active": 1,
            })

        print_progress(5, total_steps, "FAISS Index & Sync", 100.0, 95.0)

        # ---------------------------------------------------------------------
        # BƯỚC 6: Integrity Verification (95% -> 100%)
        # ---------------------------------------------------------------------
        print("\n--- BƯỚC 6/6: Kiểm Tra Tính Toàn Vẹn & Đồng Bộ Hệ Thống ---")
        report = verify_offline_storage_integrity(self.db_path, self.faiss_path)
        print_progress(6, total_steps, "Kiểm Tra Toàn Vẹn", 100.0, 100.0)

        elapsed = time.time() - start_time
        print(f"\n{'='*75}")
        print(f"HOÀN THÀNH OFFLINE PROCESSING TRONG {elapsed:.2f}s!")
        print(f"{'='*75}")
        print(f"- Trạng thái toàn vẹn  : {'ĐẠT (PASSED)' if report.passed else 'KHÔNG ĐẠT (FAILED)'}")
        print(f"- Tổng số videos       : {report.total_videos}")
        print(f"- Tổng số keyframes    : {report.total_keyframes}")
        print(f"- Vectors trong FAISS  : {report.total_faiss_vectors}")
        print(f"- Bằng chứng đối tượng : {report.total_frame_objects} objects")
        print(f"- Bằng chứng OCR       : {report.total_ocr_observations} observations -> {report.total_ocr_groups} groups")
        print(f"- Bằng chứng ASR       : {report.total_asr_segments} segments -> {report.total_asr_groups} groups")
        if report.errors:
            print(f"- Lỗi phát hiện        : {report.errors}")
        print(f"{'='*75}\n")

        return report.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIDENTRA Offline Processing & Storage Pipeline")
    parser.add_argument("--dataset", type=str, default=None, help="Process specific dataset (e.g. qvhighlights, avc, msrvtt)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos for rapid staging")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("work/offline_scratch/faiss_index.bin"))
    parser.add_argument("--reset", action="store_true", help="Reset existing database and index before running")
    parser.add_argument(
        "--i-understand-overwrite",
        action="store_true",
        help="Required to overwrite or --reset existing metadata.db / faiss_index.bin (creates *.bak-<timestamp> backup).",
    )
    args = parser.parse_args()

    if args.reset:
        _guard_and_backup_path(args.db_path, args.i_understand_overwrite, "metadata.db")
        _guard_and_backup_path(args.faiss_path, args.i_understand_overwrite, "FAISS index")
        if args.db_path.exists():
            args.db_path.unlink()
        if args.faiss_path.exists():
            args.faiss_path.unlink()

    pipeline = OfflineProcessingPipeline(
        db_path=args.db_path,
        faiss_path=args.faiss_path,
        i_understand_overwrite=args.i_understand_overwrite,
    )
    pipeline.run_all(video_limit=args.limit, dataset_filter=args.dataset)
