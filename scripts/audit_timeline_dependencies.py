"""Read-only impact audit for a REVIEW_REQUIRED timeline proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from videntra.evaluation.provenance import sha256_file


def audit(proposal_path: Path, db_path: Path) -> dict:
    proposal_hash, db_hash = sha256_file(proposal_path), sha256_file(db_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if proposal.get("status") != "REVIEW_REQUIRED_NOT_APPLIED":
        raise ValueError("Expected a review-only timeline proposal")
    rows = proposal["proposed_rows"]
    old = {row["keyframe_id"]: row["old_timestamp_ms"] for row in rows}
    if len(old) != len(rows):
        raise ValueError("Duplicate proposed keyframe IDs")
    wal = Path(str(db_path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Nonempty WAL: freeze database before audit")
    tables = {"KeyframeMap": "pts_time", "Keyframes": "timestamp_ms",
              "OCRObservations": "timestamp_ms", "FrameObjects": "timestamp_ms",
              "OCREvidenceGroups": "timestamp_ms", "fts_ocr": "timestamp_ms"}
    result = {}
    keys = list(old)
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        for table, timestamp_field in tables.items():
            counts = {"rows": 0, "unique_keyframes": 0, "equal_old_timestamp": 0, "different_timestamp": 0}
            seen = set()
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                for key, timestamp in conn.execute(f"SELECT keyframe_id, {timestamp_field} FROM {table} WHERE keyframe_id IN ({placeholders})", chunk):
                    counts["rows"] += 1
                    seen.add(key)
                    value_ms = timestamp * 1000 if table == "KeyframeMap" and timestamp is not None else timestamp
                    if value_ms is not None and abs(float(value_ms) - float(old[key])) <= 0.001:
                        counts["equal_old_timestamp"] += 1
                    else:
                        counts["different_timestamp"] += 1
            counts["unique_keyframes"] = len(seen)
            result[table] = counts
    if (wal.exists() and wal.stat().st_size) or sha256_file(proposal_path) != proposal_hash or sha256_file(db_path) != db_hash:
        raise RuntimeError("Input changed during audit")
    return {"status": "REVIEW_REQUIRED_NOT_APPLIED", "proposal_rows": len(rows),
            "proposal_sha256": proposal_hash, "database_sha256": db_hash, "dependencies": result,
            "limitations": ["Counts do not prove clip/source/audio mappings or media identity.",
                            "OCR groups, tracks and FTS must be rebuilt or verified in a new snapshot.",
                            "No proposal, database, evidence or index was changed."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.proposal, args.db), indent=2))
