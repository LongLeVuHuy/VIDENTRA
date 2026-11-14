"""Script to transcribe real audio with Faster-Whisper and update SQLite metadata.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from videntra.storage.database import OfflineMetadataStore
from videntra.storage.grouping import ASREvidenceGrouper

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Update real ASR transcripts into VIDENTRA metadata.db")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-size", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        return

    store = OfflineMetadataStore(args.db_path)

    # 1. Check Faster-Whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: 'faster_whisper' is not installed in the current environment.")
        return

    # Check for local small snapshot if requested
    model_identifier = args.model_size
    pointer_file = Path("models/asr/faster_whisper_small.pointer.json")
    if args.model_size == "small" and pointer_file.exists():
        try:
            with open(pointer_file, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
                snap_path = pdata.get("snapshot_path")
                if snap_path and Path(snap_path).exists():
                    model_identifier = snap_path
                    print(f"Using local cached snapshot: {snap_path}")
        except Exception:
            pass

    compute_type = "float16" if args.device == "cuda" else "int8"
    print(f"1. Khởi tạo Faster-Whisper ({args.model_size}, device={args.device}, compute={compute_type})...")
    model = WhisperModel(
        model_identifier,
        device=args.device,
        compute_type=compute_type,
        cpu_threads=args.cpu_threads,
    )
    print("   Faster-Whisper đã sẵn sàng.")

    # 2. Build local media map
    print("2. Đang quét các file video cục bộ trong data/raw...")
    media_map = {p.stem: p for p in args.media_dir.glob("**/*.mp4")}
    print(f"   Tìm thấy {len(media_map)} file video trong {args.media_dir}.")

    # 3. Retrieve videos with audio from DB
    conn = sqlite3.connect(str(args.db_path))
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT video_id FROM ASRSegments;")
    existing_vids = {r[0] for r in cur.fetchall()}

    cur.execute("SELECT video_id, has_audio FROM Videos WHERE has_audio = 1 ORDER BY video_id;")
    all_rows = cur.fetchall()
    conn.close()

    rows = [r for r in all_rows if r[0] not in existing_vids]
    print(f"3. Tìm thấy {len(all_rows)} videos có has_audio = 1 trong DB ({len(existing_vids)} đã có ASR, còn {len(rows)} cần xử lý).")
    if args.max_videos:
        rows = rows[:args.max_videos]
        print(f"   (Giới hạn xử lý {args.max_videos} videos theo tham số --max-videos)")

    total_inserted_segs = 0
    total_groups = 0
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    db_lock = threading.Lock()

    def process_single_video(row_item):
        vid = row_item[0]
        v_path = media_map.get(vid)
        if not v_path or not v_path.exists():
            return None, []

        try:
            segs, info = model.transcribe(
                str(v_path),
                beam_size=1,
                vad_filter=True,
                temperature=0.0,
            )
            vid_segs = []
            for s_idx, seg in enumerate(segs):
                txt = seg.text.strip()
                if not txt:
                    continue
                vid_segs.append({
                    "segment_id": f"{vid}__asr_{s_idx}",
                    "video_id": vid,
                    "model_id": f"faster_whisper_{args.model_size}",
                    "start_ms": int(seg.start * 1000),
                    "end_ms": int(seg.end * 1000),
                    "raw_text": txt,
                    "language": getattr(seg, "language", getattr(info, "language", "en")),
                    "confidence": round(float(getattr(seg, "avg_logprob", 0.9)), 4),
                })
            return vid, vid_segs
        except Exception as exc:
            return None, []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_single_video, r): r for r in rows}
        for idx, fut in enumerate(as_completed(futures), 1):
            vid, vid_segs = fut.result()
            if vid and vid_segs:
                with db_lock:
                    store.insert_asr_segments(vid_segs)
                    groups, members = ASREvidenceGrouper.group_video_segments(vid, vid_segs)
                    if groups:
                        store.insert_asr_evidence_groups(groups, members)
                        total_groups += len(groups)
                    total_inserted_segs += len(vid_segs)

            if idx % 50 == 0 or idx == len(rows):
                elapsed = time.time() - t0
                pct = (idx / len(rows)) * 100.0
                print(f"   [{idx}/{len(rows)}] ({pct:.1f}%) | Segments: {total_inserted_segs} | Groups: {total_groups} | Elapsed: {elapsed:.1f}s")

    print(f"\nHOÀN TẤT CẬP NHẬT ASR THỰC TẾ:")
    print(f"- Tổng số segments nạp mới : {total_inserted_segs}")
    print(f"- Tổng số groups & FTS5    : {total_groups}")
    print(f"- Thời gian thực thi       : {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
