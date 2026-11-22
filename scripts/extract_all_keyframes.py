"""Extract keyframe images from raw videos using PyAV presentation-order ordinals and atomic JPEG writes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sqlite3
import sys
import time

import av
from PIL import Image

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")


def _is_valid_jpeg(path: Path) -> bool:
    """Return True if path exists, is non-empty, and has valid JPEG SOI/EOI markers."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size <= 32:
            return False
        with path.open("rb") as fh:
            head = fh.read(2)
            fh.seek(-2, os.SEEK_END)
            tail = fh.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except OSError:
        return False


def _write_jpeg_atomic(rgb_pixels, out_path: Path, quality: int = 90) -> None:
    """Save RGB ndarray to JPEG via same-directory temporary file and atomic replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.jpg")
    try:
        Image.fromarray(rgb_pixels).save(tmp_path, format="JPEG", quality=int(quality))
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def extract_video_keyframes(
    video_id: str,
    video_path: str,
    target_frames: dict[int, str],
    *,
    jpeg_quality: int = 90,
) -> int:
    """Extract specified presentation-order frame indices with PyAV and save valid JPEGs atomically."""
    needed = {
        int(f_idx): Path(path)
        for f_idx, path in target_frames.items()
        if not _is_valid_jpeg(Path(path))
    }
    if not needed:
        return 0

    max_target_idx = max(needed.keys())
    saved_count = 0
    container = None
    try:
        container = av.open(str(video_path))
        if not container.streams.video:
            return 0
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for f_idx, frame in enumerate(container.decode(stream)):
            if f_idx in needed:
                rgb = frame.to_ndarray(format="rgb24")
                _write_jpeg_atomic(rgb, needed[f_idx], quality=jpeg_quality)
                saved_count += 1
                if saved_count >= len(needed):
                    break
            if f_idx >= max_target_idx:
                break
    except Exception:
        return saved_count
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
    return saved_count


def _discover_video_paths(conn: sqlite3.Connection, raw_dir: Path) -> dict[str, str]:
    """Resolve video file paths prioritizing DB Videos.file_path, then scanning raw_dir."""
    video_paths: dict[str, str] = {}
    try:
        for vid, fp in conn.execute("SELECT video_id, file_path FROM Videos;").fetchall():
            if fp and Path(str(fp)).is_file():
                video_paths[str(vid)] = str(fp)
    except sqlite3.Error:
        pass

    if raw_dir.exists():
        for ext in _VIDEO_EXTENSIONS:
            for p in raw_dir.rglob(f"*{ext}"):
                video_paths.setdefault(p.stem, str(p))
    return video_paths


def extract_all_keyframes(
    db_path: Path = Path("data/processed/catalog/corpus_metadata.db"),
    raw_dir: Path = Path("data/datasets"),
    max_workers: int = 6,
) -> None:
    start_time = time.time()
    print("=" * 75)
    print("BẮT ĐẦU TRÍCH XUẤT TOÀN BỘ ẢNH KEYFRAMES VÀO THƯ MỤC DATA/DATASETS/*/KEYFRAMES")
    print("=" * 75)

    if not db_path.exists():
        raise FileNotFoundError(f"Missing SQLite database at {db_path}")

    # 1. Load target keyframes and known media paths from DB
    print("1. Đang truy vấn danh sách keyframes từ SQLite metadata.db...")
    with sqlite3.connect(str(db_path)) as conn:
        all_rows = conn.execute(
            "SELECT video_id, source_frame_index, keyframe_path FROM Keyframes ORDER BY video_id, source_frame_index;"
        ).fetchall()
        video_paths = _discover_video_paths(conn, raw_dir)

    total_keyframes = len(all_rows)
    print(f"   Tổng số keyframes trong cơ sở dữ liệu: {total_keyframes}")

    targets_by_video: dict[str, dict[int, str]] = {}
    for vid, f_idx, kf_path in all_rows:
        targets_by_video.setdefault(vid, {})[int(f_idx)] = kf_path

    total_videos = len(targets_by_video)
    print(f"   Tổng số videos cần trích xuất: {total_videos}")

    matched_videos = {vid: video_paths[vid] for vid in targets_by_video if vid in video_paths}
    missing_videos = set(targets_by_video.keys()) - set(matched_videos.keys())
    print(f"   Đã tìm thấy video file cho {len(matched_videos)}/{total_videos} videos.")
    if missing_videos:
        print(f"   Cảnh báo: Có {len(missing_videos)} videos chưa tìm thấy file media: {list(missing_videos)[:5]}")

    existing_count = sum(
        1 for frames in targets_by_video.values() for f_path in frames.values() if _is_valid_jpeg(Path(f_path))
    )
    print(f"   Số ảnh hợp lệ đã tồn tại trên đĩa: {existing_count}")
    print(f"   Số ảnh cần trích xuất / sửa hỏng   : {total_keyframes - existing_count}")

    if total_keyframes - existing_count == 0:
        print("\n>>> Toàn bộ keyframes đã tồn tại và hợp lệ! Hoàn thành.")
        return

    print(f"\n3. Bắt đầu trích xuất PyAV đa luồng (workers={max_workers})...")
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
                    f"\r   Tiến độ: {completed_videos}/{len(matched_videos)} videos ({pct:5.1f}%) | Ảnh hợp lệ: {curr_total}/{total_keyframes}"
                )
                sys.stdout.flush()

    sys.stdout.write("\n")

    final_existing = sum(
        1 for frames in targets_by_video.values() for f_path in frames.values() if _is_valid_jpeg(Path(f_path))
    )
    elapsed = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"HOÀN THÀNH TRÍCH XUẤT TRONG {elapsed:.2f}s!")
    print(f"{'='*75}")
    print(f"- Tổng số keyframes yêu cầu    : {total_keyframes}")
    print(f"- Tổng số ảnh hợp lệ trên đĩa : {final_existing}")
    print(f"- Tỷ lệ hoàn thành             : {(final_existing / max(1, total_keyframes)) * 100.0:.2f}%")
    print(f"- Thư mục dữ liệu gốc           : {raw_dir}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/catalog/corpus_metadata.db"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/datasets"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    extract_all_keyframes(db_path=args.db_path, raw_dir=args.raw_dir, max_workers=args.workers)
