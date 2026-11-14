"""Evaluator-only export of query IDs and original text; never export gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import verify_split_lock


def export(lock_path: Path, annotations: Path, subset: str, output: Path) -> dict:
    if subset not in {"dev", "test"} or not verify_split_lock(lock_path):
        raise ValueError("Invalid split manifest or subset")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(annotations) != lock.get("metadata", {}).get("source_files_sha256", {}).get(subset):
        raise ValueError("Annotation file hash mismatch")
    with annotations.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    by_id = {f"qvh_{row['qid']}": row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source query IDs")
    query_ids = [item["query_id"] for item in lock["records"] if item["split"] == subset]
    if not query_ids or any(qid not in by_id for qid in query_ids):
        raise ValueError("Empty or unmatched split")
    result = [{"query_id": qid, "text": by_id[qid]["query"]} for qid in query_ids]
    if any(not isinstance(row["text"], str) or not row["text"].strip() for row in result):
        raise ValueError("Empty query text")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {"queries": len(result), "sha256": sha256_file(output), "fields": ["query_id", "text"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.split_lock, args.annotations, args.subset, args.output), indent=2))
