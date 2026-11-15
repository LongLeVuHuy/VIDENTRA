"""Ingests Offline GPU processing outputs from Kaggle into local SQLite metadata.db and FAISS index."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from fractions import Fraction
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Any, Callable
import numpy as np

import faiss

# Ensure src/ is importable when invoked as `python scripts/ingest_kaggle_offline_results.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.evidence.normalize import fold_diacritics, normalize_text  # noqa: E402
from videntra.storage.database import OfflineMetadataStore, PayloadConflictError, current_iso  # noqa: E402
from videntra.storage.grouping import OCREvidenceGrouper, ASREvidenceGrouper  # noqa: E402
from videntra.storage.integrity import verify_offline_storage_integrity  # noqa: E402

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


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy src to dst atomically via temp file + os.replace."""
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _ensure_caption_schema(conn: sqlite3.Connection) -> None:
    """Create KeyframeCaptions and fts_vlm_caption tables if not already present."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS KeyframeCaptions (
        caption_id TEXT PRIMARY KEY,
        keyframe_id TEXT,
        video_id TEXT NOT NULL,
        faiss_id INTEGER NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        caption_text TEXT NOT NULL,
        normalized_text TEXT NOT NULL,
        model_id TEXT NOT NULL DEFAULT 'microsoft/Florence-2-base',
        created_at TEXT NOT NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kf_captions_video ON KeyframeCaptions(video_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kf_captions_faiss ON KeyframeCaptions(faiss_id);")
    try:
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_vlm_caption USING fts5(
            normalized_text,
            unaccented_text,
            video_id UNINDEXED,
            keyframe_id UNINDEXED,
            faiss_id UNINDEXED,
            timestamp_ms UNINDEXED,
            tokenize='unicode61'
        );
        """)
    except sqlite3.OperationalError:
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_vlm_caption USING fts5(
            normalized_text,
            unaccented_text,
            video_id UNINDEXED,
            keyframe_id UNINDEXED,
            faiss_id UNINDEXED,
            timestamp_ms UNINDEXED
        );
        """)


def ingest_p85_artifacts(
    outputs_dir: Path,
    db_path: Path,
    processed_dir: Path | None = None,
    *,
    allow_partial_keyframes: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Ingest P8-5 artifacts (spatial_patches_fp16.npy, keyframe_captions.json, faiss_e5_caption_384.bin).

    Safely updates an existing metadata.db (adding KeyframeCaptions + fts_vlm_caption)
    and copies spatial_patches_fp16.npy / faiss_e5_caption_384.bin into processed_dir.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")

    target_dir = processed_dir or db_path.parent
    result: dict[str, object] = {
        "outputs_dir": str(outputs_dir),
        "db_path": str(db_path),
        "processed_dir": str(target_dir),
        "spatial_patches_shape": None,
        "captions_ingested": 0,
        "e5_caption_vectors": 0,
        "dry_run": dry_run,
    }

    with sqlite3.connect(str(db_path), timeout=30.0) as conn:
        kf_count = int(conn.execute("SELECT COUNT(*) FROM Keyframes").fetchone()[0])

    # 1. Validate & copy spatial_patches_fp16.npy
    patches_src = outputs_dir / "spatial_patches_fp16.npy"
    if patches_src.is_file():
        patches_arr = np.load(str(patches_src), mmap_mode="r")
        if patches_arr.ndim != 3 or patches_arr.shape[1] != 9:
            raise ValueError(
                f"spatial_patches_fp16.npy must have shape (N, 9, D), got {tuple(patches_arr.shape)}"
            )
        if patches_arr.dtype != np.float16:
            raise ValueError(f"spatial_patches_fp16.npy must be float16, got {patches_arr.dtype}")
        if not allow_partial_keyframes and kf_count > 0 and patches_arr.shape[0] != kf_count:
            raise ValueError(
                f"spatial_patches_fp16.npy row count ({patches_arr.shape[0]}) != Keyframes count ({kf_count})"
            )
        result["spatial_patches_shape"] = list(patches_arr.shape)
        del patches_arr
        if not dry_run:
            _atomic_copy_file(patches_src, target_dir / "spatial_patches_fp16.npy")

    # 2. Validate & ingest keyframe_captions.json -> KeyframeCaptions + fts_vlm_caption
    captions_src = outputs_dir / "keyframe_captions.json"
    if captions_src.is_file():
        raw_captions = json.loads(captions_src.read_text(encoding="utf-8"))
        if not isinstance(raw_captions, list):
            raise ValueError("keyframe_captions.json must be a JSON list")

        for idx, rec in enumerate(raw_captions):
            if not isinstance(rec, dict) or "video_id" not in rec or "timestamp_ms" not in rec or "caption_text" not in rec:
                raise ValueError(f"Invalid caption record at index {idx}")

        if not dry_run:
            now = current_iso()
            with sqlite3.connect(str(db_path), timeout=30.0) as conn:
                _ensure_caption_schema(conn)
                conn.execute("DELETE FROM fts_vlm_caption;")
                cap_rows = []
                fts_rows = []
                for idx, rec in enumerate(raw_captions):
                    vid = str(rec["video_id"])
                    fid = int(rec.get("faiss_id", idx))
                    kf_id = str(rec.get("keyframe_id") or f"{vid}__kf_{fid}")
                    ts_ms = int(rec["timestamp_ms"])
                    cap_txt = str(rec["caption_text"]).strip()
                    norm_txt = normalize_text(cap_txt)
                    unacc_txt = fold_diacritics(norm_txt)
                    cap_id = f"{kf_id}__florence2"
                    cap_rows.append((
                        cap_id,
                        kf_id,
                        vid,
                        fid,
                        ts_ms,
                        cap_txt,
                        norm_txt,
                        "microsoft/Florence-2-base",
                        now,
                    ))
                    if norm_txt:
                        fts_rows.append((norm_txt, unacc_txt, vid, kf_id, str(fid), str(ts_ms)))

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO KeyframeCaptions (
                        caption_id, keyframe_id, video_id, faiss_id, timestamp_ms,
                        caption_text, normalized_text, model_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    cap_rows,
                )
                if fts_rows:
                    conn.executemany(
                        """
                        INSERT INTO fts_vlm_caption (
                            normalized_text, unaccented_text, video_id, keyframe_id, faiss_id, timestamp_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        fts_rows,
                    )
                conn.commit()
        result["captions_ingested"] = len(raw_captions)

    # 3. Validate & copy faiss_e5_caption_384.bin + e5_caption_manifest.json
    cap_faiss_src = outputs_dir / "faiss_e5_caption_384.bin"
    cap_manifest_src = outputs_dir / "e5_caption_manifest.json"
    if cap_faiss_src.is_file():
        cap_idx = faiss.read_index(str(cap_faiss_src))
        if int(cap_idx.d) != 384:
            raise ValueError(f"faiss_e5_caption_384.bin must be 384d, got {cap_idx.d}d")
        if cap_manifest_src.is_file():
            cap_manifest = json.loads(cap_manifest_src.read_text(encoding="utf-8"))
            if not isinstance(cap_manifest, list) or len(cap_manifest) != int(cap_idx.ntotal):
                raise ValueError(
                    f"e5_caption_manifest.json length ({len(cap_manifest) if isinstance(cap_manifest, list) else 'invalid'}) != FAISS ntotal ({cap_idx.ntotal})"
                )
            if not dry_run:
                _atomic_copy_file(cap_manifest_src, target_dir / "e5_caption_manifest.json")
        result["e5_caption_vectors"] = int(cap_idx.ntotal)
        if not dry_run:
            _atomic_copy_file(cap_faiss_src, target_dir / "faiss_e5_caption_384.bin")

    return result


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name}:{line_no} must be a JSON object")
            records.append(obj)
    return records


def _rebuild_ocr_asr_fts(conn: sqlite3.Connection) -> None:
    """Rebuild fts_ocr and fts_asr cleanly from OCREvidenceGroups and ASREvidenceGroups."""
    conn.execute("DELETE FROM fts_ocr;")
    ocr_groups = conn.execute(
        "SELECT group_id, video_id, keyframe_id, timestamp_ms, consensus_text, normalized_text FROM OCREvidenceGroups;"
    ).fetchall()
    fts_ocr_rows = [
        (
            norm_txt or normalize_text(cons_txt),
            fold_diacritics(norm_txt or normalize_text(cons_txt)),
            vid_id,
            grp_id,
            kf_id,
            int(ts_ms),
        )
        for grp_id, vid_id, kf_id, ts_ms, cons_txt, norm_txt in ocr_groups
        if (cons_txt or "").strip()
    ]
    if fts_ocr_rows:
        conn.executemany(
            "INSERT INTO fts_ocr (normalized_text, unaccented_text, video_id, group_id, keyframe_id, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            fts_ocr_rows,
        )

    conn.execute("DELETE FROM fts_asr;")
    asr_groups = conn.execute(
        "SELECT group_id, video_id, start_ms, end_ms, consensus_text, normalized_text, models_involved FROM ASREvidenceGroups;"
    ).fetchall()
    fts_asr_rows = [
        (
            norm_txt or normalize_text(cons_txt),
            fold_diacritics(norm_txt or normalize_text(cons_txt)),
            vid_id,
            grp_id,
            f"grp_{grp_id}",
            int(start_ms),
            int(end_ms),
            models_inv or "",
        )
        for grp_id, vid_id, start_ms, end_ms, cons_txt, norm_txt, models_inv in asr_groups
        if (cons_txt or "").strip()
    ]
    if fts_asr_rows:
        conn.executemany(
            "INSERT INTO fts_asr (normalized_text, unaccented_text, video_id, group_id, segment_id, start_ms, end_ms, model_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            fts_asr_rows,
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _preflight_ocr_asr_artifacts(
    outputs_dir: Path,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Validate receipt, file hashes, line counts, media/keyframe ownership, OCR boxes, and ASR timestamps."""
    ocr_file = outputs_dir / "ocr_observations.jsonl"
    asr_file = outputs_dir / "asr_segments.jsonl"
    if not ocr_file.is_file() and not asr_file.is_file():
        raise FileNotFoundError(f"Neither ocr_observations.jsonl nor asr_segments.jsonl found in {outputs_dir}")

    ocr_obs = _load_jsonl(ocr_file) if ocr_file.is_file() else []
    asr_segs = _load_jsonl(asr_file) if asr_file.is_file() else []

    receipt_path = outputs_dir / "ocr_asr_receipt.json"
    legacy_unverified = False
    provenance_status = "verified"

    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or receipt.get("status") != "success":
            raise ValueError("ocr_asr_receipt.json must report status='success' before ingest")
        receipt_version = receipt.get("receipt_version")
        if receipt_version is not None and str(receipt_version) not in {"1.0", "2.0"}:
            raise ValueError(f"Unsupported ocr_asr_receipt.json version: {receipt_version}")

        artifacts = receipt.get("artifacts")
        if str(receipt_version) == "2.0" or artifacts is not None:
            if not isinstance(artifacts, dict):
                raise ValueError("ocr_asr_receipt.json 'artifacts' must be an object")
            for key, path, loaded_rows in (
                ("ocr_observations", ocr_file, ocr_obs),
                ("asr_segments", asr_file, asr_segs),
            ):
                art_meta = artifacts.get(key)
                if art_meta is None:
                    continue
                if not isinstance(art_meta, dict):
                    raise ValueError(f"ocr_asr_receipt.json artifacts.{key} must be an object")
                expected_count = art_meta.get("line_count")
                if expected_count is not None and int(expected_count) != len(loaded_rows):
                    raise ValueError(
                        f"Line count mismatch for {path.name}: receipt={expected_count}, actual={len(loaded_rows)}"
                    )
                expected_sha = art_meta.get("sha256")
                if path.is_file() and expected_sha is not None:
                    actual_sha = _sha256_file(path)
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"SHA-256 mismatch for {path.name}: receipt={expected_sha}, actual={actual_sha}"
                        )
                elif not path.is_file() and expected_count:
                    raise FileNotFoundError(f"Receipt lists {expected_count} rows for missing {path.name}")

        if ocr_obs and not isinstance(receipt.get("ocr_models"), dict):
            raise ValueError("ocr_asr_receipt.json missing 'ocr_models' provenance for OCR observations")
        if asr_segs and not isinstance(receipt.get("asr_models"), dict):
            raise ValueError("ocr_asr_receipt.json missing 'asr_models' provenance for ASR segments")
        if receipt_version is None and artifacts is None:
            provenance_status = "legacy_receipt"
    else:
        legacy_unverified = True
        provenance_status = "unverified"

    video_durations: dict[str, int] = {
        str(row[0]): int(row[1] or 0)
        for row in conn.execute("SELECT video_id, duration_ms FROM Videos;").fetchall()
    }
    keyframe_meta: dict[str, tuple[str, int]] = {
        str(row[0]): (str(row[1]), int(row[2]))
        for row in conn.execute("SELECT keyframe_id, video_id, timestamp_ms FROM Keyframes;").fetchall()
    }

    for idx, obs in enumerate(ocr_obs):
        for req in ("observation_id", "video_id", "keyframe_id", "timestamp_ms", "model_id", "raw_text",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            if req not in obs:
                raise ValueError(f"OCR observation {idx} missing required field '{req}'")
        if not isinstance(obs["model_id"], str) or not obs["model_id"].strip():
            raise ValueError(f"OCR observation {idx} has invalid model_id")
        conf = obs.get("confidence", 1.0)
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"OCR observation {idx} has invalid confidence: {conf}")

        vid_id = str(obs["video_id"])
        kf_id = str(obs["keyframe_id"])
        if vid_id not in video_durations:
            raise ValueError(f"OCR observation {idx} references unknown video_id '{vid_id}'")
        if kf_id not in keyframe_meta:
            raise ValueError(f"OCR observation {idx} references unknown keyframe_id '{kf_id}'")
        if keyframe_meta[kf_id][0] != vid_id:
            raise ValueError(
                f"OCR observation {idx} keyframe_id '{kf_id}' belongs to video '{keyframe_meta[kf_id][0]}', not '{vid_id}'"
            )
        ts_ms = obs["timestamp_ms"]
        if isinstance(ts_ms, bool) or not isinstance(ts_ms, (int, float)) or not math.isfinite(ts_ms) or ts_ms < 0:
            raise ValueError(f"OCR observation {idx} has invalid timestamp_ms: {ts_ms}")

        coords = [obs["bbox_x1"], obs["bbox_y1"], obs["bbox_x2"], obs["bbox_y2"]]
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in coords):
            raise ValueError(f"OCR observation {idx} has non-finite bbox coordinates: {coords}")
        x1, y1, x2, y2 = (float(c) for c in coords)
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"OCR observation {idx} has invalid normalized bbox: {(x1, y1, x2, y2)}")

    asr_words: list[dict[str, Any]] = []
    for idx, seg in enumerate(asr_segs):
        for req in ("segment_id", "video_id", "model_id", "start_ms", "end_ms", "raw_text"):
            if req not in seg:
                raise ValueError(f"ASR segment {idx} missing required field '{req}'")
        if not isinstance(seg["model_id"], str) or not seg["model_id"].strip():
            raise ValueError(f"ASR segment {idx} has invalid model_id")
        conf = seg.get("confidence", 1.0)
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"ASR segment {idx} has invalid confidence: {conf}")

        vid_id = str(seg["video_id"])
        if vid_id not in video_durations:
            raise ValueError(f"ASR segment {idx} references unknown video_id '{vid_id}'")
        st_ms, end_ms = seg["start_ms"], seg["end_ms"]
        if (
            isinstance(st_ms, bool) or isinstance(end_ms, bool)
            or not isinstance(st_ms, (int, float)) or not isinstance(end_ms, (int, float))
            or not math.isfinite(st_ms) or not math.isfinite(end_ms)
            or st_ms < 0 or st_ms >= end_ms
        ):
            raise ValueError(f"ASR segment {idx} has invalid interval start_ms={st_ms}, end_ms={end_ms}")
        time_basis = seg.get("time_basis")
        if time_basis is not None and time_basis != "ms":
            raise ValueError(f"ASR segment {idx} has incompatible time_basis={time_basis!r}")
        dur_ms = video_durations[vid_id]
        if dur_ms > 0 and end_ms > dur_ms + 2000:
            raise ValueError(
                f"ASR segment {idx} end_ms={end_ms} exceeds video '{vid_id}' duration_ms={dur_ms}"
            )

        seg_words = seg.get("words") or []
        if not isinstance(seg_words, list):
            raise ValueError(f"ASR segment {idx} 'words' must be a list")
        for w_idx, w in enumerate(seg_words):
            if not isinstance(w, dict):
                raise ValueError(f"ASR segment {idx} word {w_idx} must be an object")
            for w_req in ("word_id", "word", "start_ms", "end_ms"):
                if w_req not in w:
                    raise ValueError(f"ASR segment {idx} word {w_idx} missing '{w_req}'")
            w_st, w_end = int(w["start_ms"]), int(w["end_ms"])
            if w_st < int(st_ms) or w_end > int(end_ms) or w_st > w_end:
                raise ValueError(
                    f"ASR segment {idx} word {w_idx} interval ({w_st}, {w_end}) outside segment ({st_ms}, {end_ms})"
                )
            asr_words.append({**w, "segment_id": w.get("segment_id", seg["segment_id"])})

    return {
        "ocr_obs": ocr_obs,
        "asr_segs": asr_segs,
        "asr_words": asr_words,
        "provenance_status": provenance_status,
        "legacy_unverified": legacy_unverified,
    }


def ingest_ocr_asr_artifacts(
    outputs_dir: Path,
    db_path: Path,
    *,
    dry_run: bool = False,
    _before_fts_hook: Callable[[sqlite3.Connection], None] | None = None,
) -> dict[str, object]:
    """Safely merge Kaggle OCR/ASR JSONL outputs into an existing metadata.db and rebuild FTS5."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")

    store = OfflineMetadataStore(db_path)
    with store._get_connection() as preflight_conn:
        preflight = _preflight_ocr_asr_artifacts(outputs_dir, preflight_conn)

    ocr_obs: list[dict[str, Any]] = preflight["ocr_obs"]
    asr_segs: list[dict[str, Any]] = preflight["asr_segs"]
    asr_words: list[dict[str, Any]] = preflight["asr_words"]
    provenance_status: str = preflight["provenance_status"]
    legacy_unverified: bool = preflight["legacy_unverified"]

    ocr_groups_count = 0
    asr_groups_count = 0
    ocr_inserted = 0
    asr_inserted = 0
    fts_rebuilt = False

    if not dry_run:
        with store.transaction() as conn:
            if ocr_obs:
                ocr_inserted = store.insert_ocr_observations(ocr_obs, conn=conn)
                updated_kfs = sorted({(str(obs["video_id"]), str(obs["keyframe_id"])) for obs in ocr_obs})
                for vid, kf_id in updated_kfs:
                    existing_group_cnt = conn.execute(
                        "SELECT COUNT(*) FROM OCREvidenceGroups WHERE keyframe_id = ?;",
                        (kf_id,),
                    ).fetchone()[0]
                    if ocr_inserted == 0 and existing_group_cnt > 0:
                        continue
                    all_kf_rows = conn.execute(
                        """
                        SELECT observation_id, video_id, keyframe_id, timestamp_ms, model_id, raw_text,
                               bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language
                        FROM OCRObservations
                        WHERE keyframe_id = ?
                        ORDER BY model_id, observation_id;
                        """,
                        (kf_id,),
                    ).fetchall()
                    all_kf_obs = [dict(r) for r in all_kf_rows]
                    ts_ms = int(all_kf_obs[0]["timestamp_ms"])
                    conn.execute("DELETE FROM OCREvidenceGroups WHERE keyframe_id = ?;", (kf_id,))
                    groups, members = OCREvidenceGrouper.group_keyframe_observations(vid, kf_id, ts_ms, all_kf_obs)
                    if groups:
                        store.insert_ocr_evidence_groups(groups, members, conn=conn, update_fts=False)
                        ocr_groups_count += len(groups)

            if asr_segs:
                asr_inserted = store.insert_asr_segments(asr_segs, words=asr_words or None, conn=conn)
                updated_vids = sorted({str(seg["video_id"]) for seg in asr_segs})
                for vid in updated_vids:
                    existing_group_cnt = conn.execute(
                        "SELECT COUNT(*) FROM ASREvidenceGroups WHERE video_id = ?;",
                        (vid,),
                    ).fetchone()[0]
                    if asr_inserted == 0 and existing_group_cnt > 0:
                        continue
                    all_seg_rows = conn.execute(
                        """
                        SELECT segment_id, video_id, model_id, start_ms, end_ms, raw_text,
                               language, confidence, speaker_id
                        FROM ASRSegments
                        WHERE video_id = ?
                        ORDER BY model_id, start_ms, segment_id;
                        """,
                        (vid,),
                    ).fetchall()
                    all_vid_segs = [dict(r) for r in all_seg_rows]
                    conn.execute("DELETE FROM ASREvidenceGroups WHERE video_id = ?;", (vid,))
                    groups, members = ASREvidenceGrouper.group_video_segments(vid, all_vid_segs)
                    if groups:
                        store.insert_asr_evidence_groups(groups, members, conn=conn, update_fts=False)
                        asr_groups_count += len(groups)

            if _before_fts_hook is not None:
                _before_fts_hook(conn)

            if ocr_inserted > 0 or asr_inserted > 0 or ocr_groups_count > 0 or asr_groups_count > 0:
                _rebuild_ocr_asr_fts(conn)
                fts_rebuilt = True

    return {
        "mode": "ocr_asr",
        "outputs_dir": str(outputs_dir),
        "db_path": str(db_path),
        "provenance_status": provenance_status,
        "legacy_unverified": legacy_unverified,
        "ocr_observations_ingested": len(ocr_obs),
        "ocr_observations_inserted": ocr_inserted,
        "ocr_groups_upserted": ocr_groups_count,
        "asr_segments_ingested": len(asr_segs),
        "asr_records_inserted": asr_inserted,
        "asr_groups_upserted": asr_groups_count,
        "fts_rebuilt": fts_rebuilt,
        "dry_run": dry_run,
    }


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

    # 5b. Copy optional E5 text index and ingest P8-5 artifacts if present
    for opt_name in ("faiss_text_e5_384.bin", "e5_text_manifest.json"):
        opt_src = outputs_dir / opt_name
        if opt_src.is_file():
            _atomic_copy_file(opt_src, faiss_path.parent / opt_name)

    if any((outputs_dir / n).is_file() for n in ("spatial_patches_fp16.npy", "keyframe_captions.json", "faiss_e5_caption_384.bin")):
        ingest_p85_artifacts(
            outputs_dir=outputs_dir,
            db_path=db_path,
            processed_dir=faiss_path.parent,
            allow_partial_keyframes=False,
            dry_run=False,
        )

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
    parser = argparse.ArgumentParser(description="Ingest Kaggle offline GPU outputs into local storage.")
    parser.add_argument("--mode", choices=("full", "p85", "ocr_asr"), default="full")
    parser.add_argument("--outputs-dir", type=Path, default=Path("work/kaggle/offline_outputs"))
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--faiss-path", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--processed-dir", type=Path, default=None, help="Target dir for P8-5 artifacts (defaults to db_path.parent).")
    parser.add_argument("--allow-partial-keyframes", action="store_true", help="Allow spatial_patches row count != Keyframes count in p85 mode.")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifacts without modifying DB or files.")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.mode == "ocr_asr":
        res = ingest_ocr_asr_artifacts(
            outputs_dir=args.outputs_dir,
            db_path=args.db_path,
            dry_run=args.dry_run,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.mode == "p85":
        res = ingest_p85_artifacts(
            outputs_dir=args.outputs_dir,
            db_path=args.db_path,
            processed_dir=args.processed_dir,
            allow_partial_keyframes=args.allow_partial_keyframes,
            dry_run=args.dry_run,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        ingest_kaggle_offline_outputs(
            outputs_dir=args.outputs_dir,
            db_path=args.db_path,
            faiss_path=args.faiss_path,
            reset=args.reset,
        )
