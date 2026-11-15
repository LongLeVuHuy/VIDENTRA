"""Script to extract real OCR observations from keyframes using OCRRouter and update SQLite metadata.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import cv2
import numpy as np

from videntra.evidence.normalize import has_vietnamese_diacritics, normalize_text
from videntra.ocr.recognition import GeneralOCRRecognizer, VietOCRRecognizer
from videntra.ocr.routing import OCRRouter
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
    parser.add_argument("--general-backend", choices=("paddleocr_en", "rapidocr"), default="rapidocr")
    parser.add_argument("--vietocr-weights", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--receipt-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        return

    store = OfflineMetadataStore(args.db_path)

    # 1. Initialize OCR detector/general recognizer + OCRRouter (with VietOCR specialist if available)
    use_rapidocr = args.general_backend == "rapidocr"
    general_rec = GeneralOCRRecognizer(
        model_id="RapidOCR" if use_rapidocr else "paddleocr_en",
        use_rapidocr=use_rapidocr,
        allow_rapidocr_fallback=True,
        device=args.device,
    )
    try:
        general_rec._ensure_loaded()
        print(f"1. Khởi tạo {general_rec.model_id} thành công.")
    except Exception as e:
        print(f"Error initializing general OCR ({args.general_backend}): {e}")
        return

    viet_rec = VietOCRRecognizer(model_path=args.vietocr_weights, device=args.device)
    vietocr_available = True
    try:
        viet_rec._ensure_loaded()
        print("   Khởi tạo VietOCR specialist thành công.")
    except Exception as e:
        vietocr_available = False
        print(f"   Cảnh báo: VietOCR chưa khả dụng ({e}); chỉ dùng {general_rec.model_id}.")

    ocr_router = OCRRouter(
        general_recognizer=general_rec,
        vietnamese_recognizer=viet_rec if vietocr_available else general_rec,
        prefer_vietnamese=vietocr_available,
    )
    ocr_engine = general_rec._engine
    backend_kind = general_rec._backend_kind

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

            if backend_kind == "rapidocr":
                res, _ = ocr_engine(proc_img)
                raw_items = res or []
            else:
                paddle_res = ocr_engine.ocr(proc_img, cls=True)
                raw_items = []
                for line_group in (paddle_res or []):
                    for entry in (line_group or []):
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            raw_items.append((entry[0], entry[1][0], entry[1][1]))

            if not raw_items:
                return None, []

            scale_x = orig_w / float(proc_img.shape[1])
            scale_y = orig_h / float(proc_img.shape[0])

            kf_obs = []
            for h_idx, item in enumerate(raw_items):
                pts, txt, conf = item
                clean_txt = normalize_text(str(txt).strip())
                conf_val = float(conf)
                model_used = general_rec.model_id

                xs = [p[0] * scale_x for p in pts]
                ys = [p[1] * scale_y for p in pts]
                x1_px = max(0, int(min(xs)))
                y1_px = max(0, int(min(ys)))
                x2_px = min(orig_w, int(max(xs)))
                y2_px = min(orig_h, int(max(ys)))

                # Route through OCRRouter -> VietOCR specialist on Vietnamese diacritics or low confidence
                if vietocr_available and x2_px > x1_px and y2_px > y1_px and (
                    has_vietnamese_diacritics(clean_txt) or conf_val < 0.6
                ):
                    crop_rgb = cv2.cvtColor(img[y1_px:y2_px, x1_px:x2_px], cv2.COLOR_BGR2RGB)
                    routed_res, _reason = ocr_router.route_and_recognize(crop_rgb, hint_text=clean_txt)
                    if routed_res.text and routed_res.confidence >= conf_val * 0.8:
                        clean_txt = routed_res.text
                        conf_val = float(routed_res.confidence)
                        model_used = routed_res.model_id

                if not clean_txt or conf_val < 0.35:
                    continue

                bx1 = round(max(0.0, min(xs)) / orig_w, 4)
                by1 = round(max(0.0, min(ys)) / orig_h, 4)
                bx2 = round(min(float(orig_w), max(xs)) / orig_w, 4)
                by2 = round(min(float(orig_h), max(ys)) / orig_h, 4)
                lang = "vi" if has_vietnamese_diacritics(clean_txt) else "en"

                obs_id = f"{k_id}__ocr_{h_idx}"
                kf_obs.append({
                    "observation_id": obs_id,
                    "video_id": vid,
                    "keyframe_id": k_id,
                    "timestamp_ms": ts_ms,
                    "model_id": model_used,
                    "raw_text": clean_txt,
                    "bbox_x1": bx1,
                    "bbox_y1": by1,
                    "bbox_x2": bx2,
                    "bbox_y2": by2,
                    "confidence": round(conf_val, 4),
                    "language": lang,
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

    receipt = {
        "model_id": general_rec.model_id,
        "revision": os.environ.get("VIDENTRA_OCR_REVISION", "rapidocr-onnxruntime-1.4.4" if backend_kind == "rapidocr" else "paddleocr-en"),
        "vietocr_model_id": viet_rec.model_id if vietocr_available else None,
        "vietocr_revision": "vgg_transformer" if vietocr_available else None,
        "device": args.device,
        "total_observations": total_inserted_obs,
        "total_groups": total_groups,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = args.receipt_path or (args.db_path.parent / "ocr_update_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHOÀN TẤT CẬP NHẬT OCR THỰC TẾ:")
    print(f"- Tổng số observations nạp mới : {total_inserted_obs}")
    print(f"- Tổng số groups & FTS5        : {total_groups}")
    print(f"- Receipt                      : {receipt_path}")
    print(f"- Thời gian thực thi           : {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
