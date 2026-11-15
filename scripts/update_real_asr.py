"""Script to transcribe real audio with Faster-Whisper or PhoWhisper and update SQLite metadata.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np

from videntra.asr.transcription import PhoWhisperTranscriber
from videntra.storage.database import OfflineMetadataStore
from videntra.storage.grouping import ASREvidenceGrouper

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _decode_audio_mono_16k(video_path: Path) -> np.ndarray:
    """Decode video audio track into mono float32 16kHz PCM array."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            return np.empty(0, dtype=np.float32)
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()
                chunks.append(arr.reshape(-1).astype(np.float32))
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description="Update real ASR transcripts into VIDENTRA metadata.db")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--backend", choices=("faster_whisper", "phowhisper"), default="faster_whisper")
    parser.add_argument("--model-size", type=str, default="tiny")
    parser.add_argument("--asr-model-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--receipt-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        return

    store = OfflineMetadataStore(args.db_path)

    model = None
    phowhisper = None
    model_id_used = ""
    revision_used = ""

    if args.backend == "phowhisper":
        phowhisper = PhoWhisperTranscriber(model_path=args.asr_model_dir, device=args.device)
        phowhisper._ensure_loaded()
        model_id_used = "phowhisper_small"
        revision_used = os.environ.get("VIDENTRA_PHOWHISPER_REVISION", "vinai/PhoWhisper-small")
        print(f"1. Khởi tạo PhoWhisper-small (device={args.device}) thành công.")
    else:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Error: 'faster_whisper' is not installed in the current environment.")
            return

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
        model_id_used = f"faster_whisper_{args.model_size}"
        revision_used = os.environ.get("VIDENTRA_ASR_REVISION", f"Systran/faster-whisper-{args.model_size}")
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
            if args.backend == "phowhisper":
                audio = _decode_audio_mono_16k(v_path)
                if len(audio) == 0:
                    return vid, []
                chunk_samples = 30 * 16000
                vid_segs = []
                for c_idx, start_sample in enumerate(range(0, len(audio), chunk_samples)):
                    chunk = audio[start_sample : start_sample + chunk_samples]
                    res = phowhisper.transcribe_chunk(chunk, sample_rate=16000)
                    if res.status == "ok" and res.text:
                        vid_segs.append({
                            "segment_id": f"{vid}__asr_{c_idx}",
                            "video_id": vid,
                            "model_id": model_id_used,
                            "start_ms": int((start_sample / 16000.0) * 1000),
                            "end_ms": int(((start_sample + len(chunk)) / 16000.0) * 1000),
                            "raw_text": res.text,
                            "language": res.language,
                            "confidence": round(float(res.confidence), 4),
                        })
                return vid, vid_segs

            segs, info = model.transcribe(
                str(v_path),
                beam_size=1,
                vad_filter=True,
                temperature=0.0,
            )
            detected_lang = getattr(info, "language", None) or "unknown"
            vid_segs = []
            for s_idx, seg in enumerate(segs):
                txt = seg.text.strip()
                if not txt:
                    continue
                avg_lp = float(getattr(seg, "avg_logprob", -0.1))
                conf = min(1.0, max(0.0, math.exp(avg_lp))) if avg_lp <= 0 else min(1.0, avg_lp)
                vid_segs.append({
                    "segment_id": f"{vid}__asr_{s_idx}",
                    "video_id": vid,
                    "model_id": model_id_used,
                    "start_ms": int(seg.start * 1000),
                    "end_ms": int(seg.end * 1000),
                    "raw_text": txt,
                    "language": detected_lang,
                    "confidence": round(conf, 4),
                })
            return vid, vid_segs
        except Exception:
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

    receipt = {
        "backend": args.backend,
        "model_id": model_id_used,
        "revision": revision_used,
        "device": args.device,
        "total_segments": total_inserted_segs,
        "total_groups": total_groups,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = args.receipt_path or (args.db_path.parent / "asr_update_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHOÀN TẤT CẬP NHẬT ASR THỰC TẾ:")
    print(f"- Tổng số segments nạp mới : {total_inserted_segs}")
    print(f"- Tổng số groups & FTS5    : {total_groups}")
    print(f"- Receipt                  : {receipt_path}")
    print(f"- Thời gian thực thi       : {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
