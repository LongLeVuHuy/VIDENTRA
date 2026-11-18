"""Rebuild full 1,324-video VIDENTRA corpus under tuned hybrid_v2_adaptive keyframes.

Subcommands:
    sample-local   Sample all videos locally via PyAV + HybridAdaptiveSampler and write atomic JPEGs.
    prepare-kaggle Package manifest (<1.5 MB) into Kaggle GPU launcher.py for SigLIP2 + YOLOv8n extraction.
    ingest-kaggle  Verify Kaggle receipt/SHA-256, backup prior state, atomically rebuild DB/FAISS/864d/hubness.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Any, Sequence
import zipfile

import av
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.extract_all_keyframes import _is_valid_jpeg, _write_jpeg_atomic  # noqa: E402
from videntra.config.ingestion import DecodeConfig, SamplingConfig  # noqa: E402
from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.ingestion.decode import decode_frames  # noqa: E402
from videntra.ingestion.probe import ProbeResult, StreamInfo  # noqa: E402
from videntra.sampling.hybrid import HybridAdaptiveSampler  # noqa: E402


SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"

DEFAULT_DB_PATH = Path("data/processed/metadata.db")
DEFAULT_FAISS_PATH = Path("data/processed/faiss_index.bin")
DEFAULT_AUGMENTED_PATH = Path("data/processed/augmented_embeddings.npy")
DEFAULT_HUBNESS_PATH = Path("data/cache/keyframe_hubness_radii.npy")
DEFAULT_KEYFRAMES_DIR = Path("data/raw/keyframes")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_STAGING_DIR = Path("work/kaggle/hybrid_v2_staging")
DEFAULT_KERNEL_DIR = Path("work/kaggle/hybrid_v2_kernel")
DEFAULT_KAGGLE_OUT_DIR = Path("work/kaggle/hybrid_v2_out")
DEFAULT_BACKUP_ROOT = Path("work/backups")
DEFAULT_KAGGLE_SLUG = "lvhuylong/videntra-hybrid-v2-full-1324"
DEFAULT_DATASETS = (
    "lvhuylong/msrvtt-1ka-original-videos-v1",
    "lvhuylong/phase8-avc-pilot300-bundle-20261102",
    "lvhuylong/qvhighlights-practice-videos-v1",
)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _atomic_write_via_temp(target: Path, writer_fn) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        writer_fn(tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _probe_via_pyav(video_path: Path) -> tuple[ProbeResult, float, int]:
    """Probe media stream metadata directly using PyAV without spawning ffprobe."""
    resolved = video_path.resolve()
    with av.open(str(resolved)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {resolved}")
        v_stream = container.streams.video[0]
        tb = v_stream.time_base
        tb_num = int(tb.numerator) if tb is not None else 1
        tb_den = int(tb.denominator) if tb is not None else 1000
        avg_rate = float(v_stream.average_rate) if v_stream.average_rate else 30.0

        dur_s = 0.0
        if v_stream.duration is not None and tb is not None:
            dur_s = float(v_stream.duration * tb)
        elif container.duration is not None:
            dur_s = float(container.duration) / float(av.time_base)

        has_audio = len(container.streams.audio) > 0
        a_stream = container.streams.audio[0] if has_audio else None
        a_idx = int(a_stream.index) if a_stream is not None else None
        total_frames = int(v_stream.frames or 0)

        v_info = StreamInfo(
            index=int(v_stream.index),
            codec_name=str(v_stream.codec_context.name if v_stream.codec_context else "unknown"),
            codec_type="video",
            time_base_num=tb_num,
            time_base_den=tb_den,
            width=int(v_stream.width or 0),
            height=int(v_stream.height or 0),
            avg_frame_rate=f"{avg_rate:.6f}",
            nb_frames=total_frames,
        )
        a_info = None
        if a_stream is not None:
            a_tb = a_stream.time_base
            a_info = StreamInfo(
                index=int(a_stream.index),
                codec_name=str(a_stream.codec_context.name if a_stream.codec_context else "unknown"),
                codec_type="audio",
                time_base_num=int(a_tb.numerator) if a_tb else 1,
                time_base_den=int(a_tb.denominator) if a_tb else 1000,
            )

        probe = ProbeResult(
            video_path=resolved,
            sha256="",
            container_format="mp4",
            duration_seconds=dur_s,
            duration_us=int(round(dur_s * 1_000_000)),
            video_stream=v_info,
            audio_stream=a_info,
            has_video=True,
            has_audio=has_audio,
            selected_video_stream_index=int(v_stream.index),
            selected_audio_stream_index=a_idx,
            raw_probe={},
        )
        return probe, avg_rate, total_frames


def _detect_dataset_group(mp4_path: Path) -> str:
    parts_lower = [p.lower() for p in mp4_path.parts]
    for name in ("msrvtt", "avc", "qvhighlights"):
        if name in parts_lower:
            return name
    return "other"


def _sample_single_video(args: tuple[str, str, int, str, str]) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    """Worker function: sample one video via HybridAdaptiveSampler and ensure valid JPEGs."""
    video_id, mp4_str, db_duration_ms, db_dataset, keyframes_root_str = args
    mp4_path = Path(mp4_str)
    keyframes_root = Path(keyframes_root_str)

    probe, avg_rate, stream_total_frames = _probe_via_pyav(mp4_path)
    duration_s = float(probe.duration_seconds or (db_duration_ms / 1000.0) or 0.0)
    cfg = SamplingConfig.for_duration(duration_s)
    if cfg.strategy != "hybrid_v2_adaptive":
        raise RuntimeError(f"Expected hybrid_v2_adaptive config, got {cfg.strategy}")

    decode_cfg = DecodeConfig(thread_count=2)
    sampler = HybridAdaptiveSampler(config=cfg, video_id=video_id, timeline_revision="rev_0")

    decoded_count = 0
    last_ts_sec = 0.0
    cache_pixels = stream_total_frames <= 1200
    pixels_cache: dict[int, np.ndarray] = {}
    for df in decode_frames(mp4_path, probe_result=probe, recipe=decode_cfg, include_pixels=True):
        decoded_count += 1
        if df.timestamp_seconds is not None:
            last_ts_sec = float(df.timestamp_seconds)
        if cache_pixels and df.pixels is not None:
            if decoded_count <= 1200:
                pixels_cache[df.ordinal] = df.pixels
            else:
                pixels_cache.clear()
                cache_pixels = False
        sampler.process_frame(df)

    manifest = sampler.finalize()
    if not manifest.samples:
        raise RuntimeError(f"No samples produced for {video_id}")

    actual_dur_ms = max(int(round(max(duration_s, last_ts_sec) * 1000.0)), int(db_duration_ms or 0))
    v_stream = probe.video_stream
    assert v_stream is not None
    video_meta = {
        "video_id": video_id,
        "dataset": db_dataset,
        "dataset_group": _detect_dataset_group(mp4_path),
        "duration_ms": actual_dur_ms,
        "fps": float(avg_rate),
        "time_base_num": int(v_stream.time_base_num),
        "time_base_den": int(v_stream.time_base_den),
        "width": int(v_stream.width or 0),
        "height": int(v_stream.height or 0),
        "total_frames": int(decoded_count or stream_total_frames),
        "has_audio": 1 if probe.has_audio else 0,
    }

    kf_dir = keyframes_root / video_id
    kf_dir.mkdir(parents=True, exist_ok=True)
    needed_ordinals: dict[int, Path] = {}
    reused_count = 0
    written_count = 0
    tb_str = f"{v_stream.time_base_num}/{v_stream.time_base_den}"

    kf_rows: list[dict[str, Any]] = []
    for k_idx, s in enumerate(manifest.samples):
        ord_idx = int(s.original_decoded_frame_index)
        rel_posix = f"data/raw/keyframes/{video_id}/{ord_idx:06d}.jpg"
        abs_jpg = _ROOT / rel_posix
        if _is_valid_jpeg(abs_jpg):
            reused_count += 1
        elif ord_idx in pixels_cache:
            _write_jpeg_atomic(pixels_cache[ord_idx], abs_jpg, quality=90)
            written_count += 1
        else:
            needed_ordinals[ord_idx] = abs_jpg

        ts_ms = int(round(s.timestamp_us / 1000.0)) if s.timestamp_us is not None else 0
        pts_val = (
            int(s.pts)
            if s.pts is not None
            else int(round(s.timestamp_seconds * v_stream.time_base_den / max(1, v_stream.time_base_num)))
        )
        kf_rows.append({
            "keyframe_id": f"{video_id}__kf_{k_idx:05d}",
            "video_id": video_id,
            "keyframe_index": k_idx,
            "source_frame_index": ord_idx,
            "pts": pts_val,
            "time_base": tb_str,
            "timestamp_ms": ts_ms,
            "pts_time": round(float(s.timestamp_seconds), 6),
            "keyframe_path": rel_posix,
            "phash": s.phash,
            "sampling_reason": ",".join(s.reasons) if s.reasons else "hybrid_v2_adaptive",
        })

    pixels_cache.clear()
    if needed_ordinals:
        max_wanted = max(needed_ordinals)
        for df in decode_frames(mp4_path, probe_result=probe, recipe=decode_cfg, include_pixels=True):
            if df.ordinal in needed_ordinals:
                assert df.pixels is not None, f"Null pixels on frame {df.ordinal} of {video_id}"
                _write_jpeg_atomic(df.pixels, needed_ordinals[df.ordinal], quality=90)
                written_count += 1
            if df.ordinal >= max_wanted:
                break

    for row in kf_rows:
        abs_jpg = _ROOT / row["keyframe_path"]
        if not _is_valid_jpeg(abs_jpg):
            raise RuntimeError(f"Missing or corrupt JPEG after sampling: {abs_jpg}")

    return video_meta, kf_rows, reused_count, written_count


def run_sample_local(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    keyframes_dir: Path = DEFAULT_KEYFRAMES_DIR,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Sample all corpus videos locally under tuned hybrid_v2_adaptive and stage manifests."""
    t0 = time.time()
    conn = sqlite3.connect(str(db_path))
    db_rows = conn.execute(
        "SELECT video_id, duration_ms, dataset FROM Videos ORDER BY video_id ASC;"
    ).fetchall()
    conn.close()

    if limit is not None:
        db_rows = db_rows[:limit]

    local_mp4s = {p.stem: p for p in raw_dir.rglob("*.mp4")}
    jobs: list[tuple[str, str, int, str, str]] = []
    missing_vids: list[str] = []
    for vid, dur_ms, ds in db_rows:
        mp4 = local_mp4s.get(vid)
        if mp4 is None:
            missing_vids.append(vid)
        else:
            jobs.append((str(vid), str(mp4), int(dur_ms or 0), str(ds or "kaggle_batch"), str(keyframes_dir)))

    if missing_vids:
        raise FileNotFoundError(f"Missing {len(missing_vids)} local mp4 files, e.g. {missing_vids[:5]}")

    videos_meta: list[dict[str, Any]] = []
    all_keyframes: list[dict[str, Any]] = []
    total_reused = 0
    total_written = 0
    counts_by_group: dict[str, list[int]] = {}

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done_idx, (v_meta, kf_rows, reused_n, written_n) in enumerate(
                pool.map(_sample_single_video, jobs, chunksize=2), start=1
            ):
                videos_meta.append(v_meta)
                all_keyframes.extend(kf_rows)
                total_reused += reused_n
                total_written += written_n
                grp = v_meta["dataset_group"]
                counts_by_group.setdefault(grp, []).append(len(kf_rows))
                if done_idx % 100 == 0 or done_idx == len(jobs):
                    print(
                        f"[sample-local] {done_idx}/{len(jobs)} videos | "
                        f"{len(all_keyframes)} keyframes | reused={total_reused} new_jpg={total_written}",
                        flush=True,
                    )
    else:
        for done_idx, job in enumerate(jobs, start=1):
            v_meta, kf_rows, reused_n, written_n = _sample_single_video(job)
            videos_meta.append(v_meta)
            all_keyframes.extend(kf_rows)
            total_reused += reused_n
            total_written += written_n
            grp = v_meta["dataset_group"]
            counts_by_group.setdefault(grp, []).append(len(kf_rows))

    # Assign contiguous global faiss_id sorted by (video_id, keyframe_index)
    all_keyframes.sort(key=lambda r: (r["video_id"], r["keyframe_index"]))
    for fid, row in enumerate(all_keyframes):
        row["faiss_id"] = fid

    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(staging_dir / "videos_meta.jsonl", videos_meta)
    _write_jsonl_gz(staging_dir / "keyframes_manifest.jsonl.gz", all_keyframes)

    group_stats = {
        grp: {
            "videos": len(cnts),
            "keyframes": int(sum(cnts)),
            "avg_per_video": round(sum(cnts) / max(1, len(cnts)), 2),
            "min": int(min(cnts)),
            "max": int(max(cnts)),
        }
        for grp, cnts in sorted(counts_by_group.items())
    }
    summary = {
        "strategy": "hybrid_v2_adaptive",
        "video_count": len(videos_meta),
        "keyframe_count": len(all_keyframes),
        "reused_jpegs": total_reused,
        "written_jpegs": total_written,
        "elapsed_seconds": round(time.time() - t0, 2),
        "group_stats": group_stats,
        "manifest_sha256": sha256_file(staging_dir / "keyframes_manifest.jsonl.gz"),
    }
    (staging_dir / "sample_local_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


_KAGGLE_RUNNER_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from importlib import metadata as importlib_metadata
import zipfile

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_hybrid_v2")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["HF_HUB_DISABLE_XET"] = "1"
constraints = []
for pkg in ("numpy", "torch", "torchvision", "transformers", "opencv-python", "opencv-contrib-python", "scipy", "pillow"):
    try:
        constraints.append(f"{pkg}=={importlib_metadata.version(pkg)}")
    except importlib_metadata.PackageNotFoundError:
        pass
constraints_file = TEMP / "constraints.txt"
constraints_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints_file), "av==16.0.1", "ultralytics==8.4.202"],
    check=True,
)

import av
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp_box(coords):
    x1, y1, x2, y2 = [max(0.0, min(1.0, round(float(c), 4))) for c in coords]
    if x2 <= x1:
        x2 = min(1.0, round(x1 + 0.0001, 4))
    if x1 >= x2:
        x1 = max(0.0, round(x2 - 0.0001, 4))
    if y2 <= y1:
        y2 = min(1.0, round(y1 + 0.0001, 4))
    if y1 >= y2:
        y1 = max(0.0, round(y2 - 0.0001, 4))
    return x1, y1, x2, y2


t_start = time.time()
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

# 1. Verify pinned model weights
weights_local = Path(
    hf_hub_download(repo_id=SIGLIP_MODEL_ID, filename="model.safetensors", revision=SIGLIP_REVISION)
)
actual_siglip_sha = sha256_file(weights_local)
assert actual_siglip_sha == SIGLIP_WEIGHTS_SHA256, f"SigLIP2 SHA mismatch: {actual_siglip_sha}"

yolo_path = TEMP / "yolov8n.pt"
if not yolo_path.is_file():
    urllib.request.urlretrieve(YOLO_URL, yolo_path)
actual_yolo_sha = sha256_file(yolo_path)
assert actual_yolo_sha == YOLO_SHA256, f"YOLOv8n SHA mismatch: {actual_yolo_sha}"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, revision=SIGLIP_REVISION)
siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, revision=SIGLIP_REVISION).eval().to(device)
yolo = YOLO(str(yolo_path))

# 2. Index mounted Kaggle video files
mp4_index = {}
for root in (Path("/kaggle/input"),):
    if root.exists():
        for mp4 in root.rglob("*.mp4"):
            mp4_index[mp4.stem] = mp4
print(f"Discovered {len(mp4_index)} mp4 files on Kaggle input", flush=True)

# 3. Load compact keyframes manifest
with gzip.open(TEMP / "compact_manifest.json.gz", "rt", encoding="utf-8") as f:
    compact = json.load(f)

kfs_by_vid = {}
total_kfs = 0
for vid, tuples in compact.items():
    rows = []
    for k_idx, (ord_idx, ts_ms, fid) in enumerate(tuples):
        rows.append({
            "video_id": vid,
            "keyframe_id": f"{vid}__kf_{k_idx:05d}",
            "source_frame_index": int(ord_idx),
            "timestamp_ms": int(ts_ms),
            "faiss_id": int(fid),
        })
        total_kfs += 1
    kfs_by_vid[vid] = rows

vid_list = sorted(kfs_by_vid.keys())
missing = [v for v in vid_list if v not in mp4_index]
assert not missing, f"Missing {len(missing)} videos on Kaggle input: {missing[:5]}"
print(f"Loaded manifest: {len(vid_list)} videos, {total_kfs} keyframes", flush=True)

embeddings_out = np.zeros((total_kfs, 768), dtype=np.float32)
objects_out = []
runs_out = []


def decode_video_keyframes(vid: str):
    rows = kfs_by_vid[vid]
    wanted = {int(r["source_frame_index"]): r for r in rows}
    max_ord = max(wanted.keys())
    found = {}
    with av.open(str(mp4_index[vid])) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        ordinal = 0
        for frame in container.decode(stream):
            if ordinal in wanted:
                found[ordinal] = frame.to_ndarray(format="rgb24")
            if ordinal >= max_ord:
                break
            ordinal += 1
    assert len(found) == len(wanted), f"Video {vid}: decoded {len(found)}/{len(wanted)} keyframes"
    return [(r, Image.fromarray(found[int(r["source_frame_index"])])) for r in rows]


SIGLIP_BS = 128
YOLO_BS = 64
done_vids = 0
CHUNK_VIDS = 64

for c_st in range(0, len(vid_list), CHUNK_VIDS):
    chunk_vids = vid_list[c_st : c_st + CHUNK_VIDS]
    chunk_items = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in pool.map(decode_video_keyframes, chunk_vids):
            chunk_items.extend(res)

    done_vids += len(chunk_vids)
    # SigLIP2 FP16 CUDA -> FP32 L2-normalized embeddings (identical to SigLIP2Encoder.encode_images)
    for b_st in range(0, len(chunk_items), SIGLIP_BS):
        sub = chunk_items[b_st : b_st + SIGLIP_BS]
        imgs = [it[1] for it in sub]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = siglip_model.get_image_features(**inputs)
            else:
                out = siglip_model.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
            emb = emb.float()
            norm_emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            arr = norm_emb.cpu().numpy().astype(np.float32)
        for j, (row, _) in enumerate(sub):
            embeddings_out[int(row["faiss_id"])] = arr[j]

    # YOLOv8n batch detection on PIL RGB images
    for b_st in range(0, len(chunk_items), YOLO_BS):
        sub = chunk_items[b_st : b_st + YOLO_BS]
        imgs = [it[1] for it in sub]
        yolo_res = yolo(imgs, verbose=False, device=0 if device == "cuda" else "cpu", conf=0.25)
        for (row, _), r in zip(sub, yolo_res):
            orig_h, orig_w = r.orig_shape
            obj_cnt = 0
            for d_idx, box in enumerate(r.boxes):
                conf = float(box.conf[0].item())
                if conf < 0.25:
                    continue
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = clamp_box([xyxy[0] / orig_w, xyxy[1] / orig_h, xyxy[2] / orig_w, xyxy[3] / orig_h])
                if x1 >= x2 or y1 >= y2:
                    continue
                cls_id = int(box.cls[0].item())
                cls_name = yolo.names.get(cls_id, f"obj_{cls_id}")
                objects_out.append({
                    "detection_id": f"{row['keyframe_id']}__det_{d_idx}",
                    "video_id": row["video_id"],
                    "keyframe_id": row["keyframe_id"],
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "class_id": cls_id,
                    "class_name": str(cls_name),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "confidence": round(conf, 4),
                    "image_width": int(orig_w),
                    "image_height": int(orig_h),
                })
                obj_cnt += 1
            runs_out.append({
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "object_count": obj_cnt,
            })

    print(
        f"Chunk {done_vids}/{len(vid_list)} videos | objs={len(objects_out)} | elapsed={time.time() - t_start:.1f}s",
        flush=True,
    )

assert np.isfinite(embeddings_out).all(), "Non-finite SigLIP2 embeddings detected"
norms = np.linalg.norm(embeddings_out, axis=1)
assert np.allclose(norms, 1.0, atol=1e-3), f"Bad norm range: min={norms.min()} max={norms.max()}"

out_dir = WORK / "hybrid_v2_out"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "frame_embeddings.npy", embeddings_out)

with gzip.open(out_dir / "yolo_objects.jsonl.gz", "wt", encoding="utf-8") as f:
    for rec in objects_out:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with gzip.open(out_dir / "object_runs.jsonl.gz", "wt", encoding="utf-8") as f:
    for rec in runs_out:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

artifact_names = ("frame_embeddings.npy", "yolo_objects.jsonl.gz", "object_runs.jsonl.gz")
files_meta = {
    name: {"sha256": sha256_file(out_dir / name), "bytes": (out_dir / name).stat().st_size}
    for name in artifact_names
}
receipt = {
    "status": "SUCCESS",
    "strategy": "hybrid_v2_adaptive",
    "video_count": len(vid_list),
    "keyframe_count": total_kfs,
    "object_count": len(objects_out),
    "run_count": len(runs_out),
    "siglip_model_id": SIGLIP_MODEL_ID,
    "siglip_revision": SIGLIP_REVISION,
    "siglip_weights_sha256": actual_siglip_sha,
    "yolo_model": "yolov8n",
    "yolo_version": "v8.3",
    "yolo_sha256": actual_yolo_sha,
    "elapsed_seconds": round(time.time() - t_start, 2),
    "files": files_meta,
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("DONE:", json.dumps(receipt), flush=True)
'''


def run_prepare_kaggle(
    *,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    kernel_dir: Path = DEFAULT_KERNEL_DIR,
    kernel_id: str = DEFAULT_KAGGLE_SLUG,
) -> dict[str, Any]:
    """Build self-contained Kaggle GPU launcher.py + kernel-metadata.json (<1 MB payload)."""
    manifest_gz = staging_dir / "keyframes_manifest.jsonl.gz"
    videos_meta = staging_dir / "videos_meta.jsonl"
    if not manifest_gz.is_file() or not videos_meta.is_file():
        raise FileNotFoundError(f"Missing staged files in {staging_dir}. Run sample-local first.")

    rows = _read_jsonl_gz(manifest_gz)
    rows.sort(key=lambda r: (str(r["video_id"]), int(r["keyframe_index"])))
    compact: dict[str, list[list[int]]] = {}
    for row in rows:
        compact.setdefault(str(row["video_id"]), []).append([
            int(row["source_frame_index"]),
            int(row["timestamp_ms"]),
            int(row["faiss_id"]),
        ])

    compact_bytes = gzip.compress(
        json.dumps(compact, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
        compresslevel=9,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("compact_manifest.json.gz", compact_bytes)

    b64_source = base64.b64encode(buf.getvalue()).decode("ascii")
    launcher_code = f"SOURCE = {b64_source!r}\n" + _KAGGLE_RUNNER_TEMPLATE.lstrip("\n")
    launcher_size = len(launcher_code.encode("utf-8"))
    if launcher_size >= 950_000:
        raise ValueError(f"launcher.py size {launcher_size} exceeds Kaggle 1 MB kernel source limit")

    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "launcher.py").write_text(launcher_code, encoding="utf-8")

    title = kernel_id.split("/")[-1].replace("-", " ").title()
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": "launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(DEFAULT_DATASETS),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "kernel_dir": str(kernel_dir),
        "kernel_id": kernel_id,
        "launcher_bytes": launcher_size,
        "compact_gz_bytes": len(compact_bytes),
        "datasets": list(DEFAULT_DATASETS),
    }


def _resolve_kaggle_out_dir(base_dir: Path) -> Path:
    if (base_dir / "receipt.json").is_file():
        return base_dir
    if (base_dir / "hybrid_v2_out" / "receipt.json").is_file():
        return base_dir / "hybrid_v2_out"
    raise FileNotFoundError(f"receipt.json not found under {base_dir}")


def _clamp_norm_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float] | None:
    vals = [x1, y1, x2, y2]
    if any(not math.isfinite(float(v)) for v in vals):
        return None
    cx1 = max(0.0, min(1.0, round(float(x1), 4)))
    cy1 = max(0.0, min(1.0, round(float(y1), 4)))
    cx2 = max(0.0, min(1.0, round(float(x2), 4)))
    cy2 = max(0.0, min(1.0, round(float(y2), 4)))
    if cx2 <= cx1:
        cx2 = min(1.0, round(cx1 + 0.0001, 4))
    if cx1 >= cx2:
        cx1 = max(0.0, round(cx2 - 0.0001, 4))
    if cy2 <= cy1:
        cy2 = min(1.0, round(cy1 + 0.0001, 4))
    if cy1 >= cy2:
        cy1 = max(0.0, round(cy2 - 0.0001, 4))
    if not (0.0 <= cx1 < cx2 <= 1.0 and 0.0 <= cy1 < cy2 <= 1.0):
        return None
    return cx1, cy1, cx2, cy2


def _remap_ocr_observations(
    old_ocr_rows: list[tuple[Any, ...]],
    keyframes_by_vid: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """Remap legacy OCRObservations to nearest new keyframe per video and deduplicate."""
    ts_index_by_vid: dict[str, list[int]] = {
        vid: [int(r["timestamp_ms"]) for r in rows]
        for vid, rows in keyframes_by_vid.items()
    }
    remapped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, float, float, float, float]] = set()
    per_kf_counter: dict[str, int] = {}
    dropped = 0

    for row in old_ocr_rows:
        vid, ts_ms, model_id, raw_text, bx1, by1, bx2, by2, conf, lang, created_at = row
        kfs = keyframes_by_vid.get(str(vid))
        if not kfs:
            dropped += 1
            continue
        ts_arr = ts_index_by_vid[str(vid)]
        pos = bisect_left(ts_arr, int(ts_ms))
        if pos <= 0:
            best_kf = kfs[0]
        elif pos >= len(kfs):
            best_kf = kfs[-1]
        else:
            prev_kf, next_kf = kfs[pos - 1], kfs[pos]
            best_kf = prev_kf if abs(int(ts_ms) - int(prev_kf["timestamp_ms"])) <= abs(int(next_kf["timestamp_ms"]) - int(ts_ms)) else next_kf

        box = _clamp_norm_bbox(float(bx1), float(by1), float(bx2), float(by2))
        clean_raw = str(raw_text or "").strip()
        if box is None or not clean_raw:
            dropped += 1
            continue

        dedup_key = (str(best_kf["keyframe_id"]), str(model_id), clean_raw, *box)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        kf_id = str(best_kf["keyframe_id"])
        idx = per_kf_counter.get(kf_id, 0)
        per_kf_counter[kf_id] = idx + 1
        safe_conf = float(conf) if conf is not None and math.isfinite(float(conf)) else 1.0
        safe_conf = max(0.0, min(1.0, safe_conf))

        remapped.append({
            "observation_id": f"{kf_id}__ocr_{idx}",
            "video_id": str(vid),
            "keyframe_id": kf_id,
            "timestamp_ms": int(best_kf["timestamp_ms"]),
            "model_id": str(model_id),
            "raw_text": clean_raw,
            "bbox_x1": box[0],
            "bbox_y1": box[1],
            "bbox_x2": box[2],
            "bbox_y2": box[3],
            "confidence": safe_conf,
            "language": str(lang or "en"),
            "created_at": str(created_at or datetime.now(timezone.utc).isoformat()),
        })
    return remapped, dropped


def _create_backup(
    backup_root: Path,
    *,
    db_path: Path,
    faiss_path: Path,
    augmented_path: Path,
    hubness_path: Path,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"pre_hybrid_v2_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if db_path.is_file():
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()

    candidates = [
        db_path,
        faiss_path,
        augmented_path,
        db_path.parent / "faiss_text_e5_384.bin",
        db_path.parent / "e5_text_manifest.json",
        hubness_path,
        Path("models/ranking/videntra_ltr_v1.json"),
    ]
    for src in candidates:
        if src.is_file():
            shutil.copy2(src, backup_dir / src.name)
    return backup_dir


def run_ingest_kaggle(
    *,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    kaggle_out_dir: Path = DEFAULT_KAGGLE_OUT_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    faiss_path: Path = DEFAULT_FAISS_PATH,
    augmented_path: Path = DEFAULT_AUGMENTED_PATH,
    hubness_path: Path = DEFAULT_HUBNESS_PATH,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
) -> dict[str, Any]:
    """Verify Kaggle output, backup live indexes, rebuild DB/FAISS/864d/hubness atomically."""
    import faiss
    from videntra.indexing.augmented_vector import build_augmented_864d_embeddings
    from videntra.retrieval.visual import build_or_load_keyframe_hubness_radii
    from videntra.storage.database import OfflineMetadataStore, current_iso
    from videntra.storage.grouping import OCREvidenceGrouper
    from videntra.storage.integrity import verify_offline_storage_integrity

    src_dir = _resolve_kaggle_out_dir(kaggle_out_dir)
    receipt = json.loads((src_dir / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "SUCCESS":
        raise RuntimeError(f"Kaggle receipt not SUCCESS: {receipt}")
    if receipt.get("siglip_revision") != SIGLIP_REVISION:
        raise ValueError(f"SigLIP revision mismatch: {receipt.get('siglip_revision')}")
    if receipt.get("siglip_weights_sha256") != SIGLIP_WEIGHTS_SHA256:
        raise ValueError(f"SigLIP SHA mismatch: {receipt.get('siglip_weights_sha256')}")

    files_meta = receipt.get("files") or {}
    for fname, meta in files_meta.items():
        fpath = src_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing Kaggle artifact: {fpath}")
        if meta.get("bytes") is not None and fpath.stat().st_size != int(meta["bytes"]):
            raise ValueError(f"Byte size mismatch for {fname}")
        if meta.get("sha256") and sha256_file(fpath) != meta["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {fname}")

    videos_meta = _read_jsonl(staging_dir / "videos_meta.jsonl")
    keyframes = _read_jsonl_gz(staging_dir / "keyframes_manifest.jsonl.gz")
    keyframes.sort(key=lambda r: int(r["faiss_id"]))
    n_kf = len(keyframes)
    if n_kf != int(receipt["keyframe_count"]):
        raise ValueError(f"Manifest keyframe count {n_kf} != receipt {receipt['keyframe_count']}")

    raw_embs = np.load(src_dir / "frame_embeddings.npy").astype(np.float32)
    if raw_embs.shape != (n_kf, 768) or not np.isfinite(raw_embs).all():
        raise ValueError(f"Invalid frame_embeddings shape/values: {raw_embs.shape}")
    norms = np.linalg.norm(raw_embs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"Non-unit embedding norms: min={norms.min()}, max={norms.max()}")

    yolo_objects = _read_jsonl_gz(src_dir / "yolo_objects.jsonl.gz")
    object_runs = _read_jsonl_gz(src_dir / "object_runs.jsonl.gz")

    # Verify every keyframe JPEG exists cleanly on local disk before mutating DB
    for row in keyframes:
        jpg = _ROOT / row["keyframe_path"]
        if not _is_valid_jpeg(jpg):
            raise FileNotFoundError(f"Missing or corrupt local keyframe JPEG: {jpg}")

    backup_dir = _create_backup(
        backup_root,
        db_path=db_path,
        faiss_path=faiss_path,
        augmented_path=augmented_path,
        hubness_path=hubness_path,
    )

    staged_db = staging_dir / "metadata.db.staging"
    staged_faiss = staging_dir / "faiss_index.bin.staging"
    staged_aug = staging_dir / "augmented_embeddings.npy.staging"
    staged_hub = staging_dir / "keyframe_hubness_radii.npy.staging"
    shutil.copy2(db_path, staged_db)

    # Build staged FAISS 768d index first so its SHA-256 can be recorded in IndexManifest
    faiss_index = faiss.IndexFlatIP(768)
    faiss_index.add(raw_embs)
    faiss.write_index(faiss_index, str(staged_faiss))
    faiss_sha = sha256_file(staged_faiss)

    # Stage DB updates atomically while keeping Videos/ASR* rows intact
    store = OfflineMetadataStore(staged_db)
    keyframes_by_vid: dict[str, list[dict[str, Any]]] = {}
    for kf in keyframes:
        keyframes_by_vid.setdefault(kf["video_id"], []).append(kf)

    now = current_iso()
    with store.transaction() as conn:
        old_ocr_rows = conn.execute(
            """
            SELECT video_id, timestamp_ms, model_id, raw_text,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language, created_at
            FROM OCRObservations
            ORDER BY video_id, timestamp_ms, observation_id;
            """
        ).fetchall()

        remapped_ocr, dropped_ocr = _remap_ocr_observations(old_ocr_rows, keyframes_by_vid)

        for v in videos_meta:
            conn.execute(
                """
                UPDATE Videos
                SET duration_ms = ?, fps = ?, time_base_num = ?, time_base_den = ?,
                    width = ?, height = ?, total_frames = ?, has_audio = ?,
                    status = 'completed', updated_at = ?
                WHERE video_id = ?;
                """,
                (
                    int(v["duration_ms"]),
                    float(v["fps"]),
                    int(v["time_base_num"]),
                    int(v["time_base_den"]),
                    int(v["width"]),
                    int(v["height"]),
                    int(v["total_frames"]),
                    int(v["has_audio"]),
                    now,
                    str(v["video_id"]),
                ),
            )

        for tbl in (
            "OCREvidenceMembers",
            "OCREvidenceGroups",
            "OCRObservations",
            "FrameObjectRuns",
            "FrameObjects",
            "KeyframeMap",
            "Keyframes",
            "IndexManifest",
        ):
            conn.execute(f"DELETE FROM {tbl};")
        conn.execute("DELETE FROM fts_ocr;")

        conn.executemany(
            """
            INSERT INTO Keyframes (
                keyframe_id, video_id, keyframe_index, source_frame_index,
                pts, time_base, timestamp_ms, keyframe_path, thumbnail_path,
                faiss_id, phash, sampling_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    kf["keyframe_id"],
                    kf["video_id"],
                    int(kf["keyframe_index"]),
                    int(kf["source_frame_index"]),
                    int(kf["pts"]),
                    kf["time_base"],
                    int(kf["timestamp_ms"]),
                    kf["keyframe_path"],
                    kf["keyframe_path"],
                    int(kf["faiss_id"]),
                    kf.get("phash"),
                    kf.get("sampling_reason"),
                    now,
                )
                for kf in keyframes
            ],
        )

        conn.executemany(
            """
            INSERT INTO KeyframeMap (
                faiss_id, keyframe_id, video_id, pts_time, source_frame_index, keyframe_path
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    int(kf["faiss_id"]),
                    kf["keyframe_id"],
                    kf["video_id"],
                    float(kf["pts_time"]),
                    int(kf["source_frame_index"]),
                    kf["keyframe_path"],
                )
                for kf in keyframes
            ],
        )

        if yolo_objects:
            conn.executemany(
                """
                INSERT INTO FrameObjects (
                    detection_id, video_id, keyframe_id, timestamp_ms, class_id, class_name,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, image_width, image_height,
                    model_id, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        obj["detection_id"],
                        obj["video_id"],
                        obj["keyframe_id"],
                        int(obj["timestamp_ms"]),
                        int(obj["class_id"]),
                        obj["class_name"],
                        float(obj["bbox_x1"]),
                        float(obj["bbox_y1"]),
                        float(obj["bbox_x2"]),
                        float(obj["bbox_y2"]),
                        float(obj["confidence"]),
                        int(obj["image_width"]),
                        int(obj["image_height"]),
                        "yolov8n",
                        "v8.3",
                        now,
                    )
                    for obj in yolo_objects
                ],
            )

        if object_runs:
            conn.executemany(
                """
                INSERT INTO FrameObjectRuns (
                    keyframe_id, video_id, object_count, model_id, processed_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                [
                    (r["keyframe_id"], r["video_id"], int(r["object_count"]), "yolov8n", now)
                    for r in object_runs
                ],
            )

        if remapped_ocr:
            store.insert_ocr_observations(remapped_ocr, conn=conn)
            obs_by_kf: dict[str, list[dict[str, Any]]] = {}
            for obs in remapped_ocr:
                obs_by_kf.setdefault(obs["keyframe_id"], []).append(obs)
            for kf in keyframes:
                kf_obs = obs_by_kf.get(kf["keyframe_id"])
                if kf_obs:
                    groups, members = OCREvidenceGrouper.group_keyframe_observations(
                        kf["video_id"], kf["keyframe_id"], int(kf["timestamp_ms"]), kf_obs
                    )
                    if groups:
                        store.insert_ocr_evidence_groups(groups, members, conn=conn, update_fts=True)

        conn.execute(
            """
            INSERT INTO IndexManifest (
                manifest_id, index_type, vector_dim, metric, total_vectors,
                model_id, model_version, weights_hash, index_file_path, checksum, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                f"hybrid_v2_manifest_{int(time.time())}",
                "IndexFlatIP",
                768,
                "inner_product",
                n_kf,
                SIGLIP_MODEL_ID,
                SIGLIP_REVISION,
                SIGLIP_WEIGHTS_SHA256,
                str(faiss_path),
                faiss_sha,
                1,
                now,
            ),
        )

        # Read OCR/ASR inputs inside staged transaction for 864d vector build
        asr_rows = conn.execute("SELECT video_id, start_ms, end_ms FROM ASRSegments;").fetchall()
        ocr_grp_rows = conn.execute("SELECT video_id, timestamp_ms FROM OCREvidenceGroups;").fetchall()

    # Checkpoint staged DB into clean single file before integrity verification & move
    chk_conn = sqlite3.connect(str(staged_db))
    chk_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    chk_conn.execute("PRAGMA journal_mode=DELETE;")
    chk_conn.close()
    for suf in ("-wal", "-shm"):
        p_wal = Path(f"{staged_db}{suf}")
        if p_wal.exists():
            p_wal.unlink()

    staged_report = verify_offline_storage_integrity(staged_db, staged_faiss)
    if not staged_report.passed:
        raise RuntimeError(f"Staged DB integrity failed: {staged_report}")

    # Build 864d augmented embeddings and hubness radii
    keyframe_map_dict = {int(kf["faiss_id"]): kf for kf in keyframes}
    videos_duration = {str(v["video_id"]): float(v["duration_ms"]) / 1000.0 for v in videos_meta}
    ocr_by_vid: dict[str, list[float]] = {}
    for vid, ts_ms in ocr_grp_rows:
        ocr_by_vid.setdefault(str(vid), []).append(float(ts_ms) / 1000.0)
    asr_by_vid: dict[str, list[tuple[float, float]]] = {}
    for vid, st_ms, end_ms in asr_rows:
        asr_by_vid.setdefault(str(vid), []).append((float(st_ms) / 1000.0, float(end_ms) / 1000.0))

    aug_864d, _, _ = build_augmented_864d_embeddings(
        raw_embeddings=raw_embs,
        keyframe_map=keyframe_map_dict,
        frame_objects=yolo_objects,
        videos_duration=videos_duration,
        ocr_data=ocr_by_vid,
        asr_data=asr_by_vid,
    )
    if aug_864d.shape != (n_kf, 864) or not np.isfinite(aug_864d).all():
        raise ValueError(f"Invalid 864d augmented embeddings: {aug_864d.shape}")
    with staged_aug.open("wb") as f:
        np.save(f, aug_864d.astype(np.float32))

    kf_vids = [str(kf["video_id"]) for kf in keyframes]
    radii, mean_rad = build_or_load_keyframe_hubness_radii(
        raw_embs,
        video_ids=kf_vids,
        cache_path=None,
    )
    if radii.shape != (n_kf,) or not np.isfinite(radii).all():
        raise ValueError(f"Invalid hubness radii shape: {radii.shape}")
    with staged_hub.open("wb") as f:
        np.save(f, radii.astype(np.float32))

    # Atomically swap staged artifacts into live paths
    for suf in ("-wal", "-shm"):
        live_wal = Path(f"{db_path}{suf}")
        if live_wal.exists():
            live_wal.unlink()
    shutil.move(str(staged_db), str(db_path))
    shutil.move(str(staged_faiss), str(faiss_path))
    os.replace(staged_aug, augmented_path)
    os.replace(staged_hub, hubness_path)

    live_report = verify_offline_storage_integrity(db_path, faiss_path)
    if not live_report.passed:
        raise RuntimeError(f"Live DB integrity check failed after swap: {live_report}")

    return {
        "backup_dir": str(backup_dir),
        "video_count": len(videos_meta),
        "keyframe_count": n_kf,
        "yolo_objects": len(yolo_objects),
        "ocr_remapped": len(remapped_ocr),
        "ocr_dropped": dropped_ocr,
        "augmented_shape": list(aug_864d.shape),
        "hubness_mean": round(float(mean_rad), 6),
        "integrity_passed": live_report.passed,
        "integrity_summary": {
            "total_videos": live_report.total_videos,
            "total_keyframes": live_report.total_keyframes,
            "total_faiss_vectors": live_report.total_faiss_vectors,
            "total_frame_objects": live_report.total_frame_objects,
            "total_ocr_observations": live_report.total_ocr_observations,
            "total_ocr_groups": live_report.total_ocr_groups,
            "total_asr_segments": live_report.total_asr_segments,
            "warnings": live_report.warnings,
            "errors": live_report.errors,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild corpus under tuned hybrid_v2_adaptive keyframes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_loc = sub.add_parser("sample-local", help="Sample local mp4 corpus via hybrid_v2_adaptive.")
    p_loc.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p_loc.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_loc.add_argument("--keyframes-dir", type=Path, default=DEFAULT_KEYFRAMES_DIR)
    p_loc.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    p_loc.add_argument("--workers", type=int, default=6)
    p_loc.add_argument("--limit", type=int, default=None)

    p_prep = sub.add_parser("prepare-kaggle", help="Build Kaggle GPU launcher.py package.")
    p_prep.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    p_prep.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    p_prep.add_argument("--kernel-id", type=str, default=DEFAULT_KAGGLE_SLUG)

    p_ing = sub.add_parser("ingest-kaggle", help="Verify and atomically ingest Kaggle outputs.")
    p_ing.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    p_ing.add_argument("--kaggle-out-dir", type=Path, default=DEFAULT_KAGGLE_OUT_DIR)
    p_ing.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p_ing.add_argument("--faiss-path", type=Path, default=DEFAULT_FAISS_PATH)
    p_ing.add_argument("--augmented-path", type=Path, default=DEFAULT_AUGMENTED_PATH)
    p_ing.add_argument("--hubness-path", type=Path, default=DEFAULT_HUBNESS_PATH)
    p_ing.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)

    args = parser.parse_args(argv)
    if args.cmd == "sample-local":
        res = run_sample_local(
            db_path=args.db_path,
            raw_dir=args.raw_dir,
            keyframes_dir=args.keyframes_dir,
            staging_dir=args.staging_dir,
            workers=args.workers,
            limit=args.limit,
        )
    elif args.cmd == "prepare-kaggle":
        res = run_prepare_kaggle(
            staging_dir=args.staging_dir,
            kernel_dir=args.kernel_dir,
            kernel_id=args.kernel_id,
        )
    else:
        res = run_ingest_kaggle(
            staging_dir=args.staging_dir,
            kaggle_out_dir=args.kaggle_out_dir,
            db_path=args.db_path,
            faiss_path=args.faiss_path,
            augmented_path=args.augmented_path,
            hubness_path=args.hubness_path,
            backup_root=args.backup_root,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
