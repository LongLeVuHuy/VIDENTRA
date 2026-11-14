"""Extract all 11,581 keyframe images from local raw MP4 videos into data/raw/keyframes/<video_id>/<frame_idx:06d>.jpg."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sqlite3
import sys
import time

import cv2

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_video_keyframes(
    video_id: str,
    video_path: str,
    target_frames: dict[int, str],
) -> int:
    """Extract specified target frame indices from a video file and save as JPGs."""
    needed = {f_idx: path for f_idx, path in target_frames.items() if not Path(path).exists()}
    if not needed:
        return 0

    max_target_idx = max(needed.keys())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    saved_count = 0
    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx in needed:
            out_path = Path(needed[f_idx])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

        if f_idx >= max_target_idx:
            break

        f_idx += 1

    cap.release()
    return saved_count


def extract_all_keyframes(
    db_path: Path = Path("data/processed/metadata.db"),
    raw_dir: Path = Path("data/raw"),
    max_workers: int = 6,
) -> None:
    start_time = time.time()
    print("=" * 75)
    print("BẮT ĐẦU TRÍCH XUẤT TOÀN BỘ ẢNH KEYFRAMES VÀO THƯ MỤC DATA/RAW/KEYFRAMES")
    print("=" * 75)

    if not db_path.exists():
        raise FileNotFoundError(f"Missing SQLite database at {db_path}")

    # 1. Load target keyframes from DB
    print("1. Đang truy vấn danh sách keyframes từ SQLite metadata.db...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, source_frame_index, keyframe_path FROM Keyframes ORDER BY video_id, source_frame_index;")
    all_rows = cursor.fetchall()
    conn.close()

    total_keyframes = len(all_rows)
    print(f"   Tổng số keyframes trong cơ sở dữ liệu: {total_keyframes}")

    # Group by video_id
    targets_by_video: dict[str, dict[int, str]] = {}
    for vid, f_idx, kf_path in all_rows:
        targets_by_video.setdefault(vid, {})[int(f_idx)] = kf_path

    total_videos = len(targets_by_video)
    print(f"   Tổng số videos cần trích xuất: {total_videos}")

    # 2. Map video files on disk
    print("2. Đang quét các file video gốc trong data/raw/...")
    video_paths: dict[str, str] = {}
    for p in raw_dir.rglob("*.mp4"):
        video_paths[p.stem] = str(p)

    matched_videos = {vid: video_paths[vid] for vid in targets_by_video if vid in video_paths}
    missing_videos = set(targets_by_video.keys()) - set(matched_videos.keys())
    print(f"   Đã tìm thấy video file cho {len(matched_videos)}/{total_videos} videos.")
    if missing_videos:
        print(f"   Cảnh báo: Có {len(missing_videos)} videos chưa tìm thấy file mp4: {list(missing_videos)[:5]}")

    # 3. Check already existing frames
    existing_count = sum(
        1 for vid, frames in targets_by_video.items() for f_path in frames.values() if Path(f_path).exists()
    )
    print(f"   Số ảnh đã tồn tại sẵn trên đĩa: {existing_count}")
    print(f"   Số ảnh cần trích xuất mới     : {total_keyframes - existing_count}")

    if total_keyframes - existing_count == 0:
        print("\n>>> Toàn bộ 11.581 keyframes đã tồn tại đầy đủ! Hoàn thành.")
        return

    # 4. Multi-threaded extraction
    print(f"\n3. Bắt đầu trích xuất đa luồng (workers={max_workers})...")
    total_saved = 0
    completed_videos = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_video_keyframes,
                vid,
                matched_videos[vid],
                targets_by_video[vid],
            ): vid
            for vid in matched_videos
        }

        for future in as_completed(futures):
            saved = future.result()
            total_saved += saved
            completed_videos += 1

            if completed_videos % 50 == 0 or completed_videos == len(matched_videos):
                pct = (completed_videos / len(matched_videos)) * 100.0
                curr_total = existing_count + total_saved
                sys.stdout.write(
                    f"\r   Tiến độ: {completed_videos}/{len(matched_videos)} videos ({pct:5.1f}%) | Ảnh đã lưu: {curr_total}/{total_keyframes}"
                )
                sys.stdout.flush()

    sys.stdout.write("\n")

    # Final verification
    final_existing = sum(
        1 for vid, frames in targets_by_video.items() for f_path in frames.values() if Path(f_path).exists()
    )
    elapsed = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"HOÀN THÀNH TRÍCH XUẤT TRONG {elapsed:.2f}s!")
    print(f"{'='*75}")
    print(f"- Tổng số keyframes yêu cầu : {total_keyframes}")
    print(f"- Tổng số ảnh thực tế trên đĩa: {final_existing}")
    print(f"- Tỷ lệ hoàn thành          : {(final_existing / total_keyframes) * 100.0:.2f}%")
    print(f"- Thư mục lưu trữ           : {raw_dir / 'keyframes'}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    extract_all_keyframes(db_path=args.db_path, raw_dir=args.raw_dir, max_workers=args.workers)
