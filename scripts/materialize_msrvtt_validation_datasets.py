"""Materialize self-contained MSR-VTT validation tuning & confirm datasets locally.

Splits built (strictly disjoint from sealed test_1ka):
- msrvtt_val_tune    : tune_g1 (1,000) + tune_g2 (1,000) = 2,000 videos -> validation_tuning
- msrvtt_val_confirm : confirm_g1 (1,000) + confirm_g2 (1,000) = 2,000 videos -> validation_confirm

Each split contains original .mp4 videos, presentation-order .jpg keyframes, SQLite WAL
catalog (Videos, Keyframes, KeyframeMap, FrameObjects, OCRObservations, ASRSegments, FTS5),
all pre-extracted visual/text FAISS & NumPy embeddings, raw evidence JSONLs, and annotations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import av
import faiss
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from videntra.evidence.normalize import fold_diacritics, normalize_text  # noqa: E402
from videntra.registry import DatasetRegistryManager  # noqa: E402
from videntra.retrieval.msrvtt_matrix import _find_shard_subdir  # noqa: E402
from videntra.sampling.hybrid import compute_phash_64  # noqa: E402
from videntra.storage.database import OfflineMetadataStore, current_iso  # noqa: E402
from videntra.storage.grouping import ASREvidenceGrouper, OCREvidenceGrouper  # noqa: E402


PROTOCOL_ROOT = Path("work/msrvtt_protocol_20261122")
DEFAULT_DOWNLOAD_DIR = Path("work/_tmp_msrvtt_download")


@dataclass(frozen=True)
class ValidationSplitSpec:
    dataset_id: str
    split_role: str
    description: str
    gallery_names: tuple[str, str]
    mm_shard_name: str
    vp_shard_name: str
    vl2_shard_name: str
    cap_shard_name: str
    asr_shard_name: str
    validation_dir: Path


SPLIT_SPECS: tuple[ValidationSplitSpec, ...] = (
    ValidationSplitSpec(
        dataset_id="msrvtt_val_tune",
        split_role="validation_tuning",
        description="MSR-VTT validation tuning split (tune_g1 + tune_g2: 2,000 videos, hyperparameter tuning only)",
        gallery_names=("tune_g1", "tune_g2"),
        mm_shard_name="mm_val_tune",
        vp_shard_name="vp_val_tune",
        vl2_shard_name="vl2_val_tune",
        cap_shard_name="cap_val_tune",
        asr_shard_name="asr_val_4000",
        validation_dir=Path("validation/msrvtt/valid_tuning"),
    ),
    ValidationSplitSpec(
        dataset_id="msrvtt_val_confirm",
        split_role="validation_confirm",
        description="MSR-VTT validation confirmation split (confirm_g1 + confirm_g2: 2,000 videos, confirm & lock only)",
        gallery_names=("confirm_g1", "confirm_g2"),
        mm_shard_name="mm_val_confirm",
        vp_shard_name="vp_val_confirm",
        vl2_shard_name="vl2_val_confirm",
        cap_shard_name="cap_val_confirm",
        asr_shard_name="asr_val_4000",
        validation_dir=Path("validation/msrvtt/valid_confirm"),
    ),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _build_l2_faiss_index(vecs: np.ndarray, out_path: Path) -> int:
    arr = np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D vector array for {out_path}, got shape={arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(1e-12, norms)
    index = faiss.IndexFlatIP(int(arr.shape[1]))
    index.add(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(out_path)
    return int(index.ntotal)


def _load_split_video_ids(spec: ValidationSplitSpec, protocol_root: Path) -> list[str]:
    vids: set[str] = set()
    for gname in spec.gallery_names:
        mpath = protocol_root / "manifests" / gname / "gallery_manifest.jsonl"
        for row in _read_jsonl(mpath):
            vids.add(str(row["video_id"]))
    return sorted(vids)


def _extract_media_from_zip(
    zip_path: Path,
    split_targets: dict[str, tuple[list[str], Path]],
) -> dict[str, int]:
    """Stream-extract needed .mp4 videos from Kaggle archive to data/datasets/<split>/media/."""
    needed_vids: set[str] = set()
    for vids, media_dir in split_targets.values():
        media_dir.mkdir(parents=True, exist_ok=True)
        for vid in vids:
            needed_vids.add(vid)

    if not zip_path.is_file():
        missing_already = [
            vid
            for vids, media_dir in split_targets.values()
            for vid in vids
            if not (media_dir / f"{vid}.mp4").is_file() or (media_dir / f"{vid}.mp4").stat().st_size == 0
        ]
        if not missing_already:
            return {split_id: len(vids) for split_id, (vids, _) in split_targets.items()}
        raise FileNotFoundError(f"Missing Kaggle zip {zip_path} and {len(missing_already)} media files on disk")

    def _entry_priority(name: str) -> tuple[int, int, str]:
        low = name.lower()
        if "trainvalvideo/" in low or "testvideo/" in low:
            return (0, len(name), name)
        return (1, len(name), name)

    extracted_counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        best_info: dict[str, zipfile.ZipInfo] = {}
        for zinfo in zf.infolist():
            if zinfo.is_dir() or not zinfo.filename.lower().endswith(".mp4"):
                continue
            stem = Path(zinfo.filename).stem
            if stem not in needed_vids:
                continue
            prev = best_info.get(stem)
            if prev is None or _entry_priority(zinfo.filename) < _entry_priority(prev.filename):
                best_info[stem] = zinfo

        missing_in_zip = sorted(needed_vids - set(best_info.keys()))
        if missing_in_zip:
            raise FileNotFoundError(
                f"{len(missing_in_zip)} required videos missing in {zip_path}: {missing_in_zip[:5]}"
            )

        for split_id, (vids, media_dir) in split_targets.items():
            done = 0
            for idx, vid in enumerate(vids, start=1):
                dst = media_dir / f"{vid}.mp4"
                zinfo = best_info[vid]
                if not dst.is_file() or dst.stat().st_size != zinfo.file_size or dst.stat().st_size == 0:
                    tmp = dst.with_name(f"{dst.name}.tmp")
                    with zf.open(zinfo, "r") as src_f, tmp.open("wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
                    if tmp.stat().st_size != zinfo.file_size or tmp.stat().st_size == 0:
                        raise IOError(f"Corrupt extraction for {vid}: {tmp.stat().st_size} != {zinfo.file_size}")
                    tmp.replace(dst)
                done += 1
                if idx % 500 == 0 or idx == len(vids):
                    print(f"[{split_id}] extracted media: {idx}/{len(vids)}", flush=True)
            extracted_counts[split_id] = done
    return extracted_counts


def _process_single_video_keyframes(
    *,
    video_id: str,
    dataset_id: str,
    media_path: Path,
    keyframes_root: Path,
    kf_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Decode one video in presentation order, write missing JPEGs atomically, and collect DB metadata."""
    targets_by_ord: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in kf_rows:
        targets_by_ord[int(row["source_frame_index"])].append(row)

    vid_kf_dir = keyframes_root / video_id
    vid_kf_dir.mkdir(parents=True, exist_ok=True)
    need_ords = set(targets_by_ord.keys())

    container = av.open(str(media_path))
    try:
        v_streams = container.streams.video
        if not v_streams:
            raise RuntimeError(f"No video stream in {media_path}")
        v_stream = v_streams[0]
        tb = v_stream.time_base
        tb_num = int(tb.numerator) if tb is not None else 1
        tb_den = int(tb.denominator) if tb is not None else 1000
        tb_str = f"{tb_num}/{tb_den}"
        avg_rate = float(v_stream.average_rate or v_stream.guessed_rate or 30.0)
        has_audio = 1 if container.streams.audio else 0
        width = int(v_stream.width or 0)
        height = int(v_stream.height or 0)

        decoded_count = 0
        last_pts = 0
        last_dur = 0
        seen_ords: set[int] = set()
        kf_records: list[dict[str, Any]] = []

        for frame_idx, frame in enumerate(container.decode(v_stream)):
            decoded_count = frame_idx + 1
            if frame.pts is not None:
                last_pts = int(frame.pts)
                last_dur = int(getattr(frame, "duration", 0) or 0)
            if frame_idx not in need_ords:
                continue
            seen_ords.add(frame_idx)
            img_rgb = frame.to_ndarray(format="rgb24")
            if width == 0 or height == 0:
                height, width = img_rgb.shape[:2]
            phash_hex = f"{compute_phash_64(img_rgb):016x}"

            dst_jpg = vid_kf_dir / f"{frame_idx:06d}.jpg"
            if not dst_jpg.is_file() or dst_jpg.stat().st_size == 0:
                tmp_jpg = dst_jpg.with_name(f"{dst_jpg.name}.tmp")
                Image.fromarray(img_rgb).save(tmp_jpg, format="JPEG", quality=90)
                tmp_jpg.replace(dst_jpg)

            pts_val = int(frame.pts) if frame.pts is not None else int(round(float(kf_rows_ts) * tb_den / tb_num)) if (kf_rows_ts := targets_by_ord[frame_idx][0].get("pts_sec")) is not None else 0
            rel_kf_path = dst_jpg.resolve().relative_to(_ROOT).as_posix()
            for r in targets_by_ord[frame_idx]:
                ts_ms = int(r.get("timestamp_ms", round(float(r.get("pts_sec", 0.0)) * 1000.0)))
                kf_records.append({
                    "keyframe_id": str(r["keyframe_id"]),
                    "video_id": video_id,
                    "keyframe_index": int(r["keyframe_index"]),
                    "source_frame_index": frame_idx,
                    "pts": pts_val,
                    "pts_time": float(r.get("pts_sec", ts_ms / 1000.0)),
                    "time_base": tb_str,
                    "timestamp_ms": ts_ms,
                    "keyframe_path": rel_kf_path,
                    "faiss_id": int(r["faiss_id"]),
                    "phash": phash_hex,
                    "sampling_reason": str(r.get("sampling_reason") or "hybrid_v2_adaptive"),
                })

        if seen_ords != need_ords:
            missing_ords = sorted(need_ords - seen_ords)
            raise RuntimeError(f"Video {video_id} decoded {decoded_count} frames, missing target ordinals {missing_ords}")

        if last_pts > 0 and tb is not None:
            dur_ms = int(round((last_pts + last_dur) * float(tb) * 1000.0))
        elif container.duration is not None:
            dur_ms = int(round(float(container.duration) / 1000.0))
        else:
            dur_ms = max([r["timestamp_ms"] for r in kf_records], default=0)

        video_rec = {
            "video_id": video_id,
            "dataset": dataset_id,
            "file_path": media_path.resolve().relative_to(_ROOT).as_posix(),
            "duration_ms": max(0, dur_ms),
            "fps": avg_rate,
            "time_base_num": tb_num,
            "time_base_den": tb_den,
            "width": width,
            "height": height,
            "total_frames": decoded_count,
            "has_audio": has_audio,
            "status": "completed",
        }
        return video_rec, kf_records
    finally:
        container.close()


def _extract_keyframes_for_split(
    spec: ValidationSplitSpec,
    split_vids: list[str],
    ds_root: Path,
    mm_sd: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kf_rows = _read_jsonl(mm_sd / "keyframes_manifest.jsonl.gz")
    by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kf_rows:
        by_vid[str(row["video_id"])].append(row)

    missing_vid_kfs = [v for v in split_vids if v not in by_vid]
    if missing_vid_kfs:
        raise RuntimeError(f"[{spec.dataset_id}] {len(missing_vid_kfs)} videos missing from keyframes_manifest")

    media_dir = ds_root / "media"
    keyframes_dir = ds_root / "keyframes"
    video_records: list[dict[str, Any]] = []
    all_kf_records: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(
                _process_single_video_keyframes,
                video_id=vid,
                dataset_id=spec.dataset_id,
                media_path=media_dir / f"{vid}.mp4",
                keyframes_root=keyframes_dir,
                kf_rows=by_vid[vid],
            ): vid
            for vid in split_vids
        }
        done = 0
        for fut in as_completed(futs):
            vrec, klist = fut.result()
            video_records.append(vrec)
            all_kf_records.extend(klist)
            done += 1
            if done % 250 == 0 or done == len(split_vids):
                print(f"[{spec.dataset_id}] keyframes decoded: {done}/{len(split_vids)} ({len(all_kf_records)} frames)", flush=True)

    video_records.sort(key=lambda r: r["video_id"])
    all_kf_records.sort(key=lambda r: int(r["faiss_id"]))
    return video_records, all_kf_records


def _populate_embeddings_and_evidence(
    spec: ValidationSplitSpec,
    split_vids: set[str],
    ds_root: Path,
    protocol_root: Path,
) -> dict[str, Any]:
    k_out = protocol_root / "kaggle_out"
    mm_sd = _find_shard_subdir(k_out / spec.mm_shard_name)
    vp_sd = _find_shard_subdir(k_out / spec.vp_shard_name)
    vl2_sd = _find_shard_subdir(k_out / spec.vl2_shard_name)
    cap_sd = _find_shard_subdir(k_out / spec.cap_shard_name)
    asr_sd = _find_shard_subdir(k_out / spec.asr_shard_name)

    emb_kf_dir = ds_root / "embeddings" / "visual_keyframes"
    emb_clip_dir = ds_root / "embeddings" / "visual_clips"
    emb_txt_dir = ds_root / "embeddings" / "text_evidence"
    ev_dir = ds_root / "evidence"
    ann_dir = ds_root / "annotations"

    # 1. SigLIP2 Keyframe FAISS + manifests + query vectors
    kf_emb = np.load(str(mm_sd / "keyframe_embeddings_fp16.npy"), mmap_mode="r")
    siglip_kf_ntotal = _build_l2_faiss_index(kf_emb, emb_kf_dir / "siglip2_keyframes_768d.faiss")
    _atomic_copy_file(mm_sd / "keyframes_manifest.jsonl.gz", emb_kf_dir / "siglip2_keyframes_manifest.jsonl.gz")
    _atomic_copy_file(mm_sd / "siglip_query_vectors.npy", emb_kf_dir / "siglip2_query_768d.npy")
    _atomic_copy_file(mm_sd / "query_rows.jsonl", emb_kf_dir / "siglip2_query_rows.jsonl")

    # 2. VideoPrism whole/window/query + VL2 (PE-Core-L-14-336 & SigLIP2-SO400M-384)
    vp_vecs = np.load(str(vp_sd / "video_vectors.npy"), mmap_mode="r")
    vp_ntotal = _build_l2_faiss_index(vp_vecs, emb_clip_dir / "videoprism_whole_clips_768d.faiss")
    for src_name, dst_name in (
        ("video_rows.jsonl", "videoprism_whole_clips_manifest.jsonl"),
        ("window_vectors.npy", "videoprism_windows_768d.npy"),
        ("window_rows.jsonl", "videoprism_windows_manifest.jsonl"),
        ("query_vectors.npy", "videoprism_query_768d.npy"),
        ("query_rows.jsonl", "videoprism_query_rows.jsonl"),
    ):
        _atomic_copy_file(vp_sd / src_name, emb_clip_dir / dst_name)

    for src_name, dst_name in (
        ("pe_video_vectors.npy", "pecore_l14_336_video_1024d.npy"),
        ("pe_frame_vectors_fp16.npy", "pecore_l14_336_frames_1024d_fp16.npy"),
        ("pe_query_vectors.npy", "pecore_l14_336_query_1024d.npy"),
        ("sig400_video_vectors.npy", "siglip2_so400m_384_video_1152d.npy"),
        ("sig400_frame_vectors_fp16.npy", "siglip2_so400m_384_frames_1152d_fp16.npy"),
        ("sig400_query_vectors.npy", "siglip2_so400m_384_query_1152d.npy"),
        ("video_rows.jsonl", "vl2_video_rows.jsonl"),
        ("query_rows.jsonl", "vl2_query_rows.jsonl"),
    ):
        _atomic_copy_file(vl2_sd / src_name, emb_clip_dir / dst_name)

    # 3. Florence-2 captions & E5 384d FAISS strictly filtered to split_vids
    cap_rows_all = _read_jsonl(cap_sd / "florence2_captions.jsonl.gz")
    cap_vecs_all = np.load(str(cap_sd / "caption_e5_vectors.npy"), mmap_mode="r")
    kept_indices: list[int] = []
    cap_rows_kept: list[dict[str, Any]] = []
    e5_manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cap_rows_all):
        vid = str(row["video_id"])
        if vid in split_vids:
            kept_indices.append(idx)
            cap_rows_kept.append(row)
            e5_manifest_rows.append({
                "faiss_id": len(e5_manifest_rows),
                "video_id": vid,
                "source": "florence2_caption",
                "merged_caption": str(row.get("merged_caption") or ""),
            })
    cap_vecs_split = np.asarray(cap_vecs_all[kept_indices], dtype=np.float32)
    e5_ntotal = _build_l2_faiss_index(cap_vecs_split, emb_txt_dir / "e5_captions_384d.faiss")
    _write_jsonl_gz(emb_txt_dir / "e5_captions_manifest.jsonl.gz", e5_manifest_rows)
    _atomic_copy_file(cap_sd / "query_e5_vectors.npy", emb_txt_dir / "e5_query_384d.npy")
    _atomic_copy_file(cap_sd / "query_rows.jsonl", emb_txt_dir / "e5_query_rows.jsonl")
    _write_jsonl_gz(ev_dir / "florence2_captions.jsonl.gz", cap_rows_kept)

    # 4. Evidence JSONLs (OCR, YOLO, Kalman from mm_sd; ASR strictly filtered from asr_val_4000)
    for fname in ("ocr_observations.jsonl.gz", "ocr_video_status.json", "yolo_objects.jsonl.gz", "kalman_tracks.jsonl.gz"):
        _atomic_copy_file(mm_sd / fname, ev_dir / fname)

    asr_rows_split = [r for r in _read_jsonl(asr_sd / "asr_segments.jsonl.gz") if str(r["video_id"]) in split_vids]
    _write_jsonl_gz(ev_dir / "asr_segments.jsonl.gz", asr_rows_split)
    asr_status_all = json.loads((asr_sd / "asr_video_status.json").read_text(encoding="utf-8"))
    asr_status_split = {vid: asr_status_all[vid] for vid in sorted(split_vids) if vid in asr_status_all}
    _atomic_write_json(ev_dir / "asr_video_status.json", asr_status_split)

    # 5. Annotations & Validation Role Splits
    spec.validation_dir.mkdir(parents=True, exist_ok=True)
    gk = spec.validation_dir / ".gitkeep"
    if not gk.exists():
        gk.write_text("", encoding="utf-8")

    e_dir = protocol_root / "eval_only"
    m_dir = protocol_root / "manifests"
    for idx_g, gname in enumerate(spec.gallery_names, start=1):
        prefix = f"g{idx_g}"
        _atomic_copy_file(m_dir / gname / "gallery_manifest.jsonl", ann_dir / f"{prefix}_{gname}_gallery_manifest.jsonl")
        _atomic_copy_file(m_dir / gname / "queries.jsonl", ann_dir / f"{prefix}_{gname}_queries.jsonl")

        _atomic_copy_file(m_dir / gname / "gallery_manifest.jsonl", spec.validation_dir / f"{gname}_gallery_manifest.jsonl")
        _atomic_copy_file(m_dir / gname / "queries.jsonl", spec.validation_dir / f"{gname}_queries.jsonl")
        _atomic_copy_file(e_dir / f"{gname}_labels.jsonl", spec.validation_dir / f"{gname}_labels.jsonl")
        _atomic_copy_file(e_dir / f"{gname}_split_lock.json", spec.validation_dir / f"{gname}_split_lock.json")

    return {
        "siglip2_keyframes_ntotal": siglip_kf_ntotal,
        "videoprism_whole_clips_ntotal": vp_ntotal,
        "e5_captions_ntotal": e5_ntotal,
        "florence2_caption_videos": len(cap_rows_kept),
        "asr_segments_count": len(asr_rows_split),
        "mm_subdir": mm_sd,
        "ev_dir": ev_dir,
        "emb_kf_faiss": emb_kf_dir / "siglip2_keyframes_768d.faiss",
    }


def _populate_sqlite_catalog(
    *,
    spec: ValidationSplitSpec,
    ds_root: Path,
    video_records: list[dict[str, Any]],
    kf_records: list[dict[str, Any]],
    ev_dir: Path,
    emb_kf_faiss: Path,
) -> dict[str, int]:
    catalog_dir = ds_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    db_path = catalog_dir / "corpus_metadata.db"
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.is_file():
            p.unlink()

    store = OfflineMetadataStore(db_path)
    now = current_iso()
    vid_dims = {v["video_id"]: (int(v["width"] or 320), int(v["height"] or 240)) for v in video_records}
    kf_meta = {k["keyframe_id"]: (k["video_id"], int(k["timestamp_ms"])) for k in kf_records}

    yolo_rows = _read_jsonl(ev_dir / "yolo_objects.jsonl.gz")
    ocr_rows = _read_jsonl(ev_dir / "ocr_observations.jsonl.gz")
    asr_rows = _read_jsonl(ev_dir / "asr_segments.jsonl.gz")

    yolo_by_kf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for det in yolo_rows:
        yolo_by_kf[str(det["keyframe_id"])].append(det)

    ocr_by_kf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in ocr_rows:
        ocr_by_kf[str(obs["keyframe_id"])].append(obs)

    asr_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in asr_rows:
        asr_by_vid[str(seg["video_id"])].append(seg)

    with store.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO Videos (
                video_id, dataset, file_path, duration_ms, fps, time_base_num, time_base_den,
                width, height, total_frames, has_audio, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    v["video_id"],
                    v["dataset"],
                    v["file_path"],
                    v["duration_ms"],
                    v["fps"],
                    v["time_base_num"],
                    v["time_base_den"],
                    v["width"],
                    v["height"],
                    v["total_frames"],
                    v["has_audio"],
                    v["status"],
                    now,
                    now,
                )
                for v in video_records
            ],
        )

        conn.executemany(
            """
            INSERT INTO Keyframes (
                keyframe_id, video_id, keyframe_index, source_frame_index, pts, time_base,
                timestamp_ms, keyframe_path, thumbnail_path, faiss_id, phash, sampling_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    k["keyframe_id"],
                    k["video_id"],
                    k["keyframe_index"],
                    k["source_frame_index"],
                    k["pts"],
                    k["time_base"],
                    k["timestamp_ms"],
                    k["keyframe_path"],
                    "",
                    k["faiss_id"],
                    k["phash"],
                    k["sampling_reason"],
                    now,
                )
                for k in kf_records
            ],
        )

        conn.executemany(
            """
            INSERT INTO KeyframeMap (
                faiss_id, keyframe_id, video_id, pts_time, source_frame_index, keyframe_path
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            [
                (k["faiss_id"], k["keyframe_id"], k["video_id"], k["pts_time"], k["source_frame_index"], k["keyframe_path"])
                for k in kf_records
            ],
        )

        conn.execute(
            """
            INSERT INTO IndexManifest (
                manifest_id, index_type, vector_dim, metric, total_vectors,
                model_id, model_version, weights_hash, index_file_path, checksum, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                f"{spec.dataset_id}_siglip2_keyframes",
                "IndexFlatIP",
                768,
                "inner_product",
                len(kf_records),
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d",
                "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
                emb_kf_faiss.resolve().relative_to(_ROOT).as_posix(),
                "",
                1,
                now,
            ),
        )

        det_tuples = []
        for det in yolo_rows:
            vid = str(det["video_id"])
            w, h = vid_dims.get(vid, (320, 240))
            det_tuples.append((
                str(det["detection_id"]),
                vid,
                str(det["keyframe_id"]),
                int(det["timestamp_ms"]),
                int(det["class_id"]),
                str(det["class_name"]),
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
                float(det["confidence"]),
                w,
                h,
                "yolov8n",
                "v8.3",
                now,
            ))
        if det_tuples:
            conn.executemany(
                """
                INSERT INTO FrameObjects (
                    detection_id, video_id, keyframe_id, timestamp_ms, class_id, class_name,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, image_width, image_height,
                    model_id, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                det_tuples,
            )

        conn.executemany(
            """
            INSERT INTO FrameObjectRuns (keyframe_id, video_id, object_count, model_id, processed_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                (k["keyframe_id"], k["video_id"], len(yolo_by_kf.get(k["keyframe_id"], ())), "yolov8n", now)
                for k in kf_records
            ],
        )

        ocr_obs_tuples = []
        ocr_grp_tuples = []
        ocr_mem_tuples = []
        fts_ocr_tuples = []
        for kf_id, obs_list in ocr_by_kf.items():
            vid, ts_ms = kf_meta.get(kf_id, (str(obs_list[0]["video_id"]), int(obs_list[0]["timestamp_ms"])))
            prepped = []
            for o in obs_list:
                raw_txt = str(o.get("raw_text") or "")
                norm_txt = normalize_text(raw_txt)
                obs_id = str(o["observation_id"])
                item = {
                    "observation_id": obs_id,
                    "video_id": vid,
                    "keyframe_id": kf_id,
                    "timestamp_ms": int(o.get("timestamp_ms", ts_ms)),
                    "model_id": "RapidOCR",
                    "raw_text": raw_txt,
                    "bbox_x1": float(o["bbox_x1"]),
                    "bbox_y1": float(o["bbox_y1"]),
                    "bbox_x2": float(o["bbox_x2"]),
                    "bbox_y2": float(o["bbox_y2"]),
                    "confidence": float(o.get("confidence", 0.0)),
                }
                prepped.append(item)
                ocr_obs_tuples.append((
                    obs_id, vid, kf_id, item["timestamp_ms"], "RapidOCR",
                    raw_txt, norm_txt, item["bbox_x1"], item["bbox_y1"], item["bbox_x2"], item["bbox_y2"],
                    item["confidence"], "en", now,
                ))
            groups, members = OCREvidenceGrouper.group_keyframe_observations(vid, kf_id, ts_ms, prepped)
            for g in groups:
                cnorm = normalize_text(g["consensus_text"])
                ocr_grp_tuples.append((
                    g["group_id"], g["video_id"], g["keyframe_id"], int(g["timestamp_ms"]),
                    g["consensus_text"], cnorm, str(g["consensus_bbox"]), g["status"],
                    g.get("track_id"), g.get("estimated_start_ms"), g.get("estimated_end_ms"),
                    g["models_involved"], now,
                ))
                if g["consensus_text"].strip():
                    fts_ocr_tuples.append((
                        cnorm, fold_diacritics(cnorm), g["video_id"], g["group_id"], g["keyframe_id"], int(g["timestamp_ms"]),
                    ))
            ocr_mem_tuples.extend(members)

        if ocr_obs_tuples:
            conn.executemany(
                """
                INSERT INTO OCRObservations (
                    observation_id, video_id, keyframe_id, timestamp_ms, model_id, raw_text,
                    normalized_text, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                ocr_obs_tuples,
            )
            conn.executemany(
                """
                INSERT INTO OCREvidenceGroups (
                    group_id, video_id, keyframe_id, timestamp_ms, consensus_text,
                    normalized_text, consensus_bbox, status, track_id,
                    estimated_start_ms, estimated_end_ms, models_involved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                ocr_grp_tuples,
            )
            conn.executemany(
                "INSERT INTO OCREvidenceMembers (group_id, observation_id, model_id) VALUES (?, ?, ?);",
                ocr_mem_tuples,
            )
            conn.executemany(
                "INSERT INTO fts_ocr (normalized_text, unaccented_text, video_id, group_id, keyframe_id, timestamp_ms) VALUES (?, ?, ?, ?, ?, ?);",
                fts_ocr_tuples,
            )

        asr_seg_tuples = []
        asr_grp_tuples = []
        asr_mem_tuples = []
        fts_asr_tuples = []
        for vid, seg_list in asr_by_vid.items():
            prepped_s = []
            for s in seg_list:
                raw_txt = str(s.get("raw_text") or "")
                norm_txt = normalize_text(raw_txt)
                seg_id = str(s["segment_id"])
                item = {
                    "segment_id": seg_id,
                    "video_id": vid,
                    "model_id": "faster_whisper_small",
                    "start_ms": int(s["start_ms"]),
                    "end_ms": int(s["end_ms"]),
                    "raw_text": raw_txt,
                    "language": str(s.get("language") or "en"),
                    "confidence": float(s.get("confidence", 0.0)),
                }
                prepped_s.append(item)
                asr_seg_tuples.append((
                    seg_id, vid, "faster_whisper_small", item["start_ms"], item["end_ms"],
                    raw_txt, norm_txt, item["language"], item["confidence"], None, now,
                ))
            groups, members = ASREvidenceGrouper.group_video_segments(vid, prepped_s)
            for g in groups:
                cnorm = normalize_text(g["consensus_text"])
                asr_grp_tuples.append((
                    g["group_id"], g["video_id"], int(g["start_ms"]), int(g["end_ms"]),
                    g["consensus_text"], cnorm, g["status"], g["models_involved"],
                    float(g.get("temporal_overlap_ratio", 1.0)), float(g.get("text_similarity", 1.0)), now,
                ))
                if g["consensus_text"].strip():
                    fts_asr_tuples.append((
                        cnorm, fold_diacritics(cnorm), g["video_id"], g["group_id"], "",
                        int(g["start_ms"]), int(g["end_ms"]), g["models_involved"],
                    ))
            asr_mem_tuples.extend(members)

        if asr_seg_tuples:
            conn.executemany(
                """
                INSERT INTO ASRSegments (
                    segment_id, video_id, model_id, start_ms, end_ms, raw_text,
                    normalized_text, language, confidence, speaker_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                asr_seg_tuples,
            )
            conn.executemany(
                """
                INSERT INTO ASREvidenceGroups (
                    group_id, video_id, start_ms, end_ms, consensus_text, normalized_text,
                    status, models_involved, temporal_overlap_ratio, text_similarity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                asr_grp_tuples,
            )
            conn.executemany(
                "INSERT INTO ASREvidenceMembers (group_id, segment_id, model_id) VALUES (?, ?, ?);",
                asr_mem_tuples,
            )
            conn.executemany(
                "INSERT INTO fts_asr (normalized_text, unaccented_text, video_id, group_id, segment_id, start_ms, end_ms, model_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                fts_asr_tuples,
            )

    return {
        "videos": len(video_records),
        "keyframes": len(kf_records),
        "frame_objects": len(yolo_rows),
        "ocr_observations": len(ocr_rows),
        "asr_segments": len(asr_rows),
    }


def _stage_test_1ka_vault_folder(protocol_root: Path) -> None:
    test_dir = Path("test_queries/msrvtt/test_1ka_benchmark")
    test_dir.mkdir(parents=True, exist_ok=True)
    gk = test_dir / ".gitkeep"
    if not gk.exists():
        gk.write_text("", encoding="utf-8")
    _atomic_copy_file(protocol_root / "manifests" / "test_1ka" / "gallery_manifest.jsonl", test_dir / "test_1ka_gallery_manifest.jsonl")
    _atomic_copy_file(protocol_root / "eval_only" / "test_1ka_queries_vault.jsonl", test_dir / "test_1ka_queries_vault.jsonl")
    _atomic_copy_file(protocol_root / "eval_only" / "test_1ka_labels.jsonl", test_dir / "test_1ka_labels.jsonl")
    _atomic_copy_file(protocol_root / "eval_only" / "test_1ka_split_lock.json", test_dir / "test_1ka_split_lock.json")


def materialize_all(
    *,
    protocol_root: Path = PROTOCOL_ROOT,
    zip_path: Path = DEFAULT_DOWNLOAD_DIR / "khoa" / "msrvtt.zip",
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    workers: int = 6,
    cleanup_download: bool = True,
) -> dict[str, Any]:
    mgr = DatasetRegistryManager()
    split_vids_list: dict[str, list[str]] = {}
    split_media_targets: dict[str, tuple[list[str], Path]] = {}

    for spec in SPLIT_SPECS:
        vids = _load_split_video_ids(spec, protocol_root)
        split_vids_list[spec.dataset_id] = vids
        paths = mgr.dataset_paths(spec.dataset_id)
        mgr.add_dataset(
            spec.dataset_id,
            description=spec.description,
            license_name="research",
            source="https://www.kaggle.com/datasets/khoahunhtngng/msrvtt",
        )
        split_media_targets[spec.dataset_id] = (vids, paths.media_dir)

    print("=== Step 1/4: Extracting original .mp4 videos ===", flush=True)
    _extract_media_from_zip(zip_path, split_media_targets)

    # Verify both media dirs are complete before deleting temporary zip download
    for spec in SPLIT_SPECS:
        media_dir = mgr.dataset_paths(spec.dataset_id).media_dir
        mp4s = [p for p in media_dir.glob("*.mp4") if p.stat().st_size > 0]
        if len(mp4s) != len(split_vids_list[spec.dataset_id]):
            raise RuntimeError(f"[{spec.dataset_id}] expected {len(split_vids_list[spec.dataset_id])} mp4s, found {len(mp4s)}")

    if cleanup_download and download_dir.exists():
        print(f"=== Reclaiming disk space: deleting temporary download dir {download_dir} ===", flush=True)
        shutil.rmtree(download_dir)

    _stage_test_1ka_vault_folder(protocol_root)

    summary: dict[str, Any] = {"materialized_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "splits": {}}
    for spec in SPLIT_SPECS:
        print(f"\n=== Materializing {spec.dataset_id} ({spec.split_role}) ===", flush=True)
        paths = mgr.dataset_paths(spec.dataset_id)
        vids = split_vids_list[spec.dataset_id]
        vid_set = set(vids)

        artifact_stats = _populate_embeddings_and_evidence(spec, vid_set, paths.root, protocol_root)
        video_recs, kf_recs = _extract_keyframes_for_split(spec, vids, paths.root, artifact_stats["mm_subdir"], workers=workers)
        db_stats = _populate_sqlite_catalog(
            spec=spec,
            ds_root=paths.root,
            video_records=video_recs,
            kf_records=kf_recs,
            ev_dir=artifact_stats["ev_dir"],
            emb_kf_faiss=artifact_stats["emb_kf_faiss"],
        )

        # Enrich dataset.json with split_role + catalog/embedding paths + counts
        ds_meta = json.loads(paths.metadata_file.read_text(encoding="utf-8"))
        ds_meta["split_role"] = spec.split_role
        ds_meta["galleries"] = list(spec.gallery_names)
        ds_meta["paths"].update({
            "catalog_db": (paths.root / "catalog" / "corpus_metadata.db").as_posix(),
            "embeddings": (paths.root / "embeddings").as_posix(),
            "evidence": (paths.root / "evidence").as_posix(),
            "validation_queries": spec.validation_dir.as_posix(),
        })
        ds_meta["counts"] = {
            **db_stats,
            "siglip2_keyframes_ntotal": artifact_stats["siglip2_keyframes_ntotal"],
            "videoprism_whole_clips_ntotal": artifact_stats["videoprism_whole_clips_ntotal"],
            "e5_captions_ntotal": artifact_stats["e5_captions_ntotal"],
            "florence2_caption_videos": artifact_stats["florence2_caption_videos"],
        }
        _atomic_write_json(paths.metadata_file, ds_meta)

        verify_res = mgr.verify(spec.dataset_id, db_path=paths.root / "catalog" / "corpus_metadata.db")
        if not verify_res["ok"]:
            raise RuntimeError(f"[{spec.dataset_id}] verify failed: {verify_res}")

        receipt = {
            "schema_version": 1,
            "created_at_utc": summary["materialized_at_utc"],
            "dataset_id": spec.dataset_id,
            "split_role": spec.split_role,
            "counts": ds_meta["counts"],
            "verify": verify_res,
        }
        _atomic_write_json(paths.provenance_dir / "materialization_receipt.json", receipt)
        summary["splits"][spec.dataset_id] = receipt
        print(f"[{spec.dataset_id}] VERIFIED OK: {ds_meta['counts']}", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize MSR-VTT validation tuning & confirm datasets locally.")
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_DOWNLOAD_DIR / "khoa" / "msrvtt.zip")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--workers", type=int, default=max(4, min(8, os.cpu_count() or 4)))
    parser.add_argument("--keep-download", action="store_true")
    args = parser.parse_args()

    res = materialize_all(
        protocol_root=args.protocol_root,
        zip_path=args.zip_path,
        download_dir=args.download_dir,
        workers=args.workers,
        cleanup_download=not args.keep_download,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
