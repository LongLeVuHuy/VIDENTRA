"""Build local E5 dense text FAISS index (faiss_text_e5_384.bin + e5_text_manifest.json) from metadata.db.

Reuses `build_e5_text_faiss_index` (and `encode_e5_passages`) from
`scripts/kaggle_multimodal_offline_worker.py` so the local artifact format is
100% identical to the Kaggle offline worker output.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kaggle_multimodal_offline_worker import (  # noqa: E402
    E5_MODEL_ID,
    E5_REVISION,
    build_e5_text_faiss_index,
)


def load_text_rows(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ASRSegments and OCRObservations from metadata.db in deterministic order."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        asr_rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT segment_id, video_id, start_ms, end_ms, raw_text
                FROM ASRSegments
                WHERE TRIM(COALESCE(raw_text, '')) != ''
                ORDER BY video_id, start_ms, segment_id
                """
            ).fetchall()
        ]
        ocr_rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT observation_id, video_id, timestamp_ms, raw_text
                FROM OCRObservations
                WHERE TRIM(COALESCE(raw_text, '')) != ''
                ORDER BY video_id, timestamp_ms, observation_id
                """
            ).fetchall()
        ]
    finally:
        conn.close()
    return asr_rows, ocr_rows


def _spatial_zone_from_bbox(obj: dict[str, Any]) -> str:
    """Map object horizontal bbox center to left / center / right."""
    img_w = float(obj.get("image_width") or 0.0)
    x1 = obj.get("bbox_x1")
    x2 = obj.get("bbox_x2")
    if img_w > 0.0 and x1 is not None and x2 is not None:
        cx = 0.5 * (float(x1) + float(x2)) / img_w
    elif x1 is not None and x2 is not None and 0.0 <= float(x2) <= 1.0:
        cx = 0.5 * (float(x1) + float(x2))
    else:
        return "center"
    if cx < 0.33:
        return "left"
    if cx > 0.67:
        return "right"
    return "center"


def _load_synopsis_inputs_from_db(
    db_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load Keyframes, FrameObjects, OCRObservations, ASRSegments from metadata.db."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        keyframes = [
            dict(r)
            for r in conn.execute(
                "SELECT keyframe_id, video_id, timestamp_ms FROM Keyframes ORDER BY video_id, timestamp_ms, keyframe_id"
            ).fetchall()
        ]
        frame_objects = [
            dict(r)
            for r in conn.execute(
                """
                SELECT video_id, keyframe_id, timestamp_ms, class_name,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, image_width
                FROM FrameObjects
                ORDER BY video_id, timestamp_ms
                """
            ).fetchall()
        ]
        asr_rows, ocr_rows = load_text_rows(db_path)
    finally:
        conn.close()
    return keyframes, frame_objects, ocr_rows, asr_rows


def build_structured_event_synopses(
    db_path: Path | None = None,
    *,
    keyframes: Sequence[dict[str, Any]] | None = None,
    frame_objects: Sequence[dict[str, Any]] | None = None,
    ocr_observations: Sequence[dict[str, Any]] | None = None,
    asr_segments: Sequence[dict[str, Any]] | None = None,
    window_half_sec: float = 4.0,
    min_obj_confidence: float = 0.25,
) -> list[dict[str, Any]]:
    """Build PQ-VCA Mode A Structured Event Synopses over [t - 4s, t + 4s] windows.

    Combines YOLOv8 FrameObjects (class + count + 2D spatial zone),
    OCRObservations, and ASRSegments into compact text synopses suitable
    for multilingual-e5-small 384d encoding.
    """
    if db_path is not None:
        db_kf, db_obj, db_ocr, db_asr = _load_synopsis_inputs_from_db(db_path)
        keyframes = db_kf if keyframes is None else keyframes
        frame_objects = db_obj if frame_objects is None else frame_objects
        ocr_observations = db_ocr if ocr_observations is None else ocr_observations
        asr_segments = db_asr if asr_segments is None else asr_segments

    kfs = list(keyframes or [])
    if not kfs:
        return []

    objs_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in frame_objects or ():
        if float(o.get("confidence", 1.0)) >= float(min_obj_confidence):
            objs_by_vid[str(o["video_id"])].append(o)

    ocr_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ocr_observations or ():
        txt = str(r.get("raw_text") or "").strip()
        if txt:
            ocr_by_vid[str(r["video_id"])].append(r)

    asr_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in asr_segments or ():
        txt = str(s.get("raw_text") or "").strip()
        if txt:
            asr_by_vid[str(s["video_id"])].append(s)

    half_ms = int(round(max(0.0, float(window_half_sec)) * 1000.0))
    synopses: list[dict[str, Any]] = []

    for kf in kfs:
        vid = str(kf["video_id"])
        t_ms = int(kf["timestamp_ms"])
        win_start = max(0, t_ms - half_ms)
        win_end = t_ms + half_ms

        obj_counts: Counter[tuple[str, str]] = Counter()
        for o in objs_by_vid.get(vid, ()):
            o_ms = int(o["timestamp_ms"])
            if win_start <= o_ms <= win_end:
                cname = str(o.get("class_name", "")).strip()
                if cname:
                    obj_counts[(cname, _spatial_zone_from_bbox(o))] += 1

        ocr_texts: list[str] = []
        seen_ocr: set[str] = set()
        for r in ocr_by_vid.get(vid, ()):
            r_ms = int(r["timestamp_ms"])
            if win_start <= r_ms <= win_end:
                txt = str(r["raw_text"]).strip()
                if txt and txt not in seen_ocr:
                    seen_ocr.add(txt)
                    ocr_texts.append(txt)

        asr_texts: list[str] = []
        seen_asr: set[str] = set()
        for s in asr_by_vid.get(vid, ()):
            s_start = int(s.get("start_ms", 0))
            s_end = int(s.get("end_ms", s_start))
            if s_end >= win_start and s_start <= win_end:
                txt = str(s["raw_text"]).strip()
                if txt and txt not in seen_asr:
                    seen_asr.add(txt)
                    asr_texts.append(txt)

        parts: list[str] = []
        if obj_counts:
            obj_desc = ", ".join(
                f"{cnt} {cname} ({zone})"
                for (cname, zone), cnt in sorted(obj_counts.items())
            )
            parts.append(f"Objects: {obj_desc}.")
        if ocr_texts:
            parts.append(f"OCR: {' | '.join(ocr_texts)}.")
        if asr_texts:
            parts.append(f"Speech: {' '.join(asr_texts)}.")

        if not parts:
            continue

        synopses.append({
            "video_id": vid,
            "keyframe_id": str(kf.get("keyframe_id", f"{vid}_{t_ms}")),
            "timestamp_ms": t_ms,
            "start_ms": win_start,
            "end_ms": win_end,
            "synopsis_text": " ".join(parts),
            "modality": "ses",
        })

    return synopses


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E5 dense text FAISS index from metadata.db")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-id", type=str, default=E5_MODEL_ID)
    parser.add_argument("--revision", type=str, default=E5_REVISION)
    args = parser.parse_args()

    t0 = time.perf_counter()
    asr_rows, ocr_rows = load_text_rows(args.db_path)
    print(f"Loaded {len(asr_rows)} ASRSegments and {len(ocr_rows)} OCRObservations from {args.db_path}")

    info = build_e5_text_faiss_index(
        asr_segments=asr_rows,
        ocr_observations=ocr_rows,
        out_dir=args.out_dir,
        device=args.device,
        model_id=args.model_id,
        revision=args.revision,
    )
    elapsed = time.perf_counter() - t0
    info["elapsed_seconds"] = round(elapsed, 2)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
