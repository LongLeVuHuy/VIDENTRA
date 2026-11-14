"""Script to extract real OCR observations from keyframes and update SQLite metadata.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys
import time
import cv2
import numpy as np

from videntra.storage.database import OfflineMetadataStore
from videntra.storage.grouping import OCREvidenceGrouper

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Update real OCR observations into VIDENTRA metadata.db")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--keyframes-dir", type=Path, default=Path("data/raw/keyframes"))
    parser.add_argument("--sample-step", type=int, default=2, help="Sample keyframe 0 and every N-th keyframe per video")
    parser.add_argument("--max-keyframes", type=int, default=None)
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        return

    store = OfflineMetadataStore(args.db_path)

    # 1. Initialize RapidOCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()
        print("1. Khởi tạo RapidOCR (ONNX Runtime) thành công.")
    except Exception as e:
        print(f"Error initializing RapidOCR: {e}")
        return

    # 2. Quét danh sách keyframes
    conn = sqlite3.connect(str(args.db_path))
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT keyframe_id FROM OCRObservations;")
    existing_kf_ids = {r[0] for r in cur.fetchall()}

    cur.execute("""
        SELECT keyframe_id, video_id, timestamp_ms, keyframe_path, keyframe_index
        FROM Keyframes
        ORDER BY video_id, keyframe_index;
    """)
    all_kfs = cur.fetchall()
    conn.close()

    print(f"2. Tìm thấy {len(all_kfs)} keyframes trong DB ({len(existing_kf_ids)} đã có OCR).")

    # Filter to representative keyframes (frame 0 + every N-th frame)
    target_kfs = [
        kf for kf in all_kfs
        if (kf[4] == 0 or (kf[4] % args.sample_step == 0)) and kf[0] not in existing_kf_ids
    ]
    if args.max_keyframes:
        target_kfs = target_kfs[:args.max_keyframes]

    print(f"   Đã chọn {len(target_kfs)} keyframes đại diện để trích xuất OCR (sample_step={args.sample_step}).")

    total_inserted_obs = 0
    total_groups = 0
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    db_lock = threading.Lock()

    def process_single_kf(kf_item):
        k_id, vid, ts_ms, kpath, k_idx = kf_item
        kf_file = Path(kpath)
        if not kf_file.exists():
            cand = args.keyframes_dir / vid / kf_file.name
            if cand.exists():
                kf_file = cand
            else:
                return None, []

        try:
            img = cv2.imread(str(kf_file))
            if img is None:
                return None, []

            orig_h, orig_w = img.shape[:2]
            if max(orig_h, orig_w) > 640:
                scale = 640.0 / max(orig_h, orig_w)
                proc_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
            else:
                proc_img = img

            res, _ = ocr_engine(proc_img)
            if not res:
                return None, []

            scale_x = orig_w / float(proc_img.shape[1])
            scale_y = orig_h / float(proc_img.shape[0])

            kf_obs = []
            for h_idx, item in enumerate(res):
                pts, txt, conf = item
                clean_txt = str(txt).strip()
                if not clean_txt or float(conf) < 0.35:
                    continue

                xs = [p[0] * scale_x for p in pts]
                ys = [p[1] * scale_y for p in pts]
                bx1 = round(max(0.0, min(xs)) / orig_w, 4)
                by1 = round(max(0.0, min(ys)) / orig_h, 4)
                bx2 = round(min(float(orig_w), max(xs)) / orig_w, 4)
                by2 = round(min(float(orig_h), max(ys)) / orig_h, 4)

                obs_id = f"{k_id}__ocr_{h_idx}"
                kf_obs.append({
                    "observation_id": obs_id,
                    "video_id": vid,
                    "keyframe_id": k_id,
                    "timestamp_ms": ts_ms,
                    "model_id": "RapidOCR",
                    "raw_text": clean_txt,
                    "bbox_x1": bx1,
                    "bbox_y1": by1,
                    "bbox_x2": bx2,
                    "bbox_y2": by2,
                    "confidence": round(float(conf), 4),
                    "language": "en",
                })
            return (vid, k_id, ts_ms), kf_obs
        except Exception:
            return None, []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_single_kf, kf): kf for kf in target_kfs}
        for idx, fut in enumerate(as_completed(futures), 1):
            meta, kf_obs = fut.result()
            if meta and kf_obs:
                vid, k_id, ts_ms = meta
                with db_lock:
                    store.insert_ocr_observations(kf_obs)
                    groups, members = OCREvidenceGrouper.group_keyframe_observations(
                        vid, k_id, ts_ms, kf_obs
                    )
                    if groups:
                        store.insert_ocr_evidence_groups(groups, members)
                        total_groups += len(groups)
                    total_inserted_obs += len(kf_obs)

            if idx % 200 == 0 or idx == len(target_kfs):
                elapsed = time.time() - t0
                pct = (idx / len(target_kfs)) * 100.0
                print(f"   [{idx}/{len(target_kfs)}] ({pct:.1f}%) | OCR Obs: {total_inserted_obs} | Groups: {total_groups} | Elapsed: {elapsed:.1f}s")

    print(f"\nHOÀN TẤT CẬP NHẬT OCR THỰC TẾ:")
    print(f"- Tổng số observations nạp mới : {total_inserted_obs}")
    print(f"- Tổng số groups & FTS5        : {total_groups}")
    print(f"- Thời gian thực thi           : {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
