"""Ingests Offline GPU processing outputs from Kaggle into local SQLite metadata.db and FAISS index."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import time
import numpy as np

import faiss

from videntra.storage.database import OfflineMetadataStore, current_iso
from videntra.storage.grouping import OCREvidenceGrouper, ASREvidenceGrouper
from videntra.storage.integrity import verify_offline_storage_integrity

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_progress(step_idx: int, total_steps: int, step_name: str, step_pct: float, overall_pct: float):
    bar_len = 25
    filled = int(bar_len * (overall_pct / 100.0))
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(
        f"\r[Bước {step_idx}/{total_steps}] {step_name:<28} | Bước: {step_pct:5.1f}% | Tổng: {overall_pct:5.1f}% [{bar}]"
    )
    sys.stdout.flush()
    if step_pct >= 100.0:
        sys.stdout.write("\n")


def validate_recorded_pts(outputs_dir: Path) -> None:
    """Reject inconsistent timestamps; a pass does not replace a decoded ledger audit."""
    with (outputs_dir / "videos.json").open(encoding="utf-8") as stream:
        videos = {row["video_id"]: row for row in json.load(stream)}
    with (outputs_dir / "keyframe_manifest.json").open(encoding="utf-8") as stream:
        frames = json.load(stream)
    mismatches = missing = 0
    for frame in frames:
        video = videos.get(frame.get("video_id"), {})
        pts, num, den = frame.get("pts"), video.get("time_base_num"), video.get("time_base_den")
        if pts is None or not num or not den or frame.get("timestamp_ms") is None:
            missing += 1
            continue
        origin = video.get("time_origin_pts", 0)
        if origin is None:
            missing += 1
            continue
        recorded_pts_ms = Fraction((int(pts) - int(origin)) * int(num) * 1000, int(den))
        if abs(recorded_pts_ms - Fraction(str(frame["timestamp_ms"]))) > 1:
            mismatches += 1
    if mismatches or missing:
        raise ValueError(f"PTS/time-base audit failed: {mismatches} mismatches and {missing} missing mappings; verify ledger and explicit origin before ingest")


def validate_worker_outputs(outputs_dir: Path) -> None:
    """Reject incomplete runs and values incompatible with legacy ASR confidence."""
    with (outputs_dir / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict) or summary.get("status") != "success":
        raise ValueError("Worker summary must explicitly report status=success before ingest")

    asr_file = outputs_dir / "asr_segments.json"
    if not asr_file.is_file():
        return
    with asr_file.open(encoding="utf-8") as stream:
        segments = json.load(stream)
    if not isinstance(segments, list):
        raise ValueError("ASR segments must be a list")
    for index, segment in enumerate(segments):
        confidence = segment.get("confidence") if isinstance(segment, dict) else None
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f"ASR segment {index} has invalid confidence; raw log probabilities cannot be ingested as legacy confidence")


def ingest_kaggle_offline_outputs(
    outputs_dir: Path = Path("work/kaggle/offline_outputs"),
    db_path: Path = Path("data/processed/metadata.db"),
    faiss_path: Path = Path("data/processed/faiss_index.bin"),
    reset: bool = False,
):
    # Never overwrite a previous snapshot or delete a database through an ingest flag.
    if reset:
        raise ValueError("--reset is disabled; choose new output paths for a new snapshot")
    existing = [p for p in (db_path, faiss_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")) if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing snapshot files: {existing}")
    for name in ("summary.json", "videos.json", "keyframe_manifest.json", "frame_embeddings.npy"):
        if not (outputs_dir / name).is_file():
            raise FileNotFoundError(f"Missing {outputs_dir / name}")
    validate_worker_outputs(outputs_dir)
    validate_recorded_pts(outputs_dir)
    total_steps = 6
    start_time = time.time()
    print(f"\n{'='*75}")
    print("BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU OFFLINE GPU TỪ KAGGLE VÀO HỆ THỐNG LOCAL")
    print(f"{'='*75}")
    print(f"Nguồn Kaggle outputs : {outputs_dir}")
    print(f"Cơ sở dữ liệu đích   : {db_path}")
    print(f"Chỉ mục FAISS đích   : {faiss_path}")

    store = OfflineMetadataStore(db_path)

    # 1. Ingest Videos (0% -> 15%)
    print("\n--- BƯỚC 1/6: Đồng bộ Danh mục Videos ---")
    videos_file = outputs_dir / "videos.json"
    if not videos_file.exists():
        raise FileNotFoundError(f"Missing {videos_file}")

    with open(videos_file, "r", encoding="utf-8") as f:
        videos = json.load(f)
    video_time_bases = {v["video_id"]: f"{int(v['time_base_num'])}/{int(v['time_base_den'])}" for v in videos}

    for idx, v in enumerate(videos):
        store.register_video({
            "video_id": v["video_id"],
            "dataset": "kaggle_batch",
            "file_path": v["file_path"],
            "duration_ms": v["duration_ms"],
            "fps": v["fps"],
            "time_base_num": v["time_base_num"],
            "time_base_den": v["time_base_den"],
            "width": v["width"],
            "height": v["height"],
            "total_frames": int(v["fps"] * (v["duration_ms"] / 1000.0)),
            "has_audio": v.get("has_audio", 0),
            "status": "completed",
        })
        step_pct = ((idx + 1) / len(videos)) * 100.0
        overall_pct = (step_pct / 100.0) * 15.0
        print_progress(1, total_steps, "Đồng bộ Videos", step_pct, overall_pct)

    # 2. Ingest Keyframes (15% -> 35%)
    print("\n--- BƯỚC 2/6: Đồng bộ Keyframes & PTS Timestamps ---")
    kf_file = outputs_dir / "keyframe_manifest.json"
    if not kf_file.exists():
        raise FileNotFoundError(f"Missing {kf_file}")

    with open(kf_file, "r", encoding="utf-8") as f:
        keyframes = json.load(f)

    for i, k in enumerate(keyframes):
        k["faiss_id"] = i
        k["time_base"] = video_time_bases[k["video_id"]]
        # This worker exports embeddings, not JPEGs; an empty path means unavailable.
        k["keyframe_path"] = ""
        k["thumbnail_path"] = ""
        k["sampling_reason"] = k.get("sampling_reason") or "dynamic_max_gap"

    store.insert_keyframes(keyframes)
    print_progress(2, total_steps, "Đồng bộ Keyframes", 100.0, 35.0)

    # 3. Ingest Frame Objects (YOLOv8) (35% -> 55%)
    print("\n--- BƯỚC 3/6: Đồng bộ Bằng chứng Đối tượng YOLOv8 ---")
    obj_file = outputs_dir / "frame_objects.json"
    frame_objects = []
    if obj_file.exists():
        with open(obj_file, "r", encoding="utf-8") as f:
            frame_objects = json.load(f)

    # Group by keyframe
    objs_by_kf = {}
    for obj in frame_objects:
        objs_by_kf.setdefault(obj["keyframe_id"], []).append(obj)

    for idx, kf in enumerate(keyframes):
        kf_id = kf["keyframe_id"]
        kf_objs = objs_by_kf.get(kf_id, [])
        store.record_frame_objects(kf_id, kf["video_id"], kf_objs, model_id="yolov8n", model_version="v8.3")
        if (idx + 1) % 50 == 0 or (idx + 1) == len(keyframes):
            step_pct = ((idx + 1) / len(keyframes)) * 100.0
            overall_pct = 35.0 + (step_pct / 100.0) * 20.0
            print_progress(3, total_steps, "Đồng bộ FrameObjects", step_pct, overall_pct)

    # 4. Ingest OCR & ASR Multi-Model (55% -> 75%)
    print("\n--- BƯỚC 4/6: Đồng bộ & Gộp Bằng chứng OCR/ASR Đa Mô hình ---")
    # OCR
    ocr_file = outputs_dir / "ocr_observations.json"
    if ocr_file.exists():
        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr_obs = json.load(f)
        store.insert_ocr_observations(ocr_obs)

        # Group observations per keyframe
        obs_by_kf = {}
        for obs in ocr_obs:
            obs_by_kf.setdefault(obs["keyframe_id"], []).append(obs)

        for kf in keyframes:
            kf_id = kf["keyframe_id"]
            if kf_id in obs_by_kf:
                groups, members = OCREvidenceGrouper.group_keyframe_observations(
                    kf["video_id"], kf_id, kf["timestamp_ms"], obs_by_kf[kf_id]
                )
                if groups:
                    store.insert_ocr_evidence_groups(groups, members)

    # ASR
    asr_file = outputs_dir / "asr_segments.json"
    if asr_file.exists():
        with open(asr_file, "r", encoding="utf-8") as f:
            asr_segs = json.load(f)
        store.insert_asr_segments(asr_segs)

        # Group segments per video
        segs_by_vid = {}
        for seg in asr_segs:
            segs_by_vid.setdefault(seg["video_id"], []).append(seg)

        for v in videos:
            vid_id = v["video_id"]
            if vid_id in segs_by_vid:
                groups, members = ASREvidenceGrouper.group_video_segments(vid_id, segs_by_vid[vid_id])
                if groups:
                    store.insert_asr_evidence_groups(groups, members)

    print_progress(4, total_steps, "Gộp Bằng Chứng", 100.0, 75.0)

    # 5. Build FAISS Index & Mappings (75% -> 90%)
    print("\n--- BƯỚC 5/6: Xây dựng Chỉ mục Vector FAISS & Ánh xạ SQLite ---")
    embs_file = outputs_dir / "frame_embeddings.npy"
    if not embs_file.exists():
        raise FileNotFoundError(f"Missing {embs_file}")

    embs = np.load(str(embs_file)).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(embs.shape[1])
    faiss_index.add(embs)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(faiss_path))

    faiss_mappings = [
        {
            "faiss_id": i,
            "keyframe_id": kf["keyframe_id"],
            "video_id": kf["video_id"],
            "pts_time": float(kf["timestamp_ms"]) / 1000.0,
            "source_frame_index": kf["source_frame_index"],
            "keyframe_path": kf["keyframe_path"],
        }
        for i, kf in enumerate(keyframes)
    ]
    store.update_faiss_mappings(faiss_mappings)

    store.record_index_manifest({
        "manifest_id": f"kaggle_gpu_manifest_{int(time.time())}",
        "index_type": "IndexFlatIP",
        "vector_dim": int(embs.shape[1]),
        "metric": "inner_product",
        "total_vectors": len(keyframes),
        "model_id": "google/siglip2-base-patch16-224",
        "model_version": "v1.0",
        "index_file_path": str(faiss_path),
        "is_active": 1,
    })
    print_progress(5, total_steps, "FAISS Index & Sync", 100.0, 90.0)

    # 6. Integrity Verification (90% -> 100%)
    print("\n--- BƯỚC 6/6: Kiểm tra Tính Toàn vẹn Hệ thống ---")
    report = verify_offline_storage_integrity(db_path, faiss_path)
    print_progress(6, total_steps, "Kiểm Tra Toàn Vẹn", 100.0, 100.0)

    elapsed = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"ĐỒNG BỘ THÀNH CÔNG TRONG {elapsed:.2f}s!")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("work/kaggle/offline_outputs"))
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    ingest_kaggle_offline_outputs(
        outputs_dir=args.outputs_dir,
        db_path=args.db_path,
        faiss_path=args.faiss_path,
        reset=args.reset,
    )
