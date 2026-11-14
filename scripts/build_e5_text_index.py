"""Build local E5 dense text FAISS index (faiss_text_e5_384.bin + e5_text_manifest.json) from metadata.db.

Reuses `build_e5_text_faiss_index` (and `encode_e5_passages`) from
`scripts/kaggle_multimodal_offline_worker.py` so the local artifact format is
100% identical to the Kaggle offline worker output.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

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
