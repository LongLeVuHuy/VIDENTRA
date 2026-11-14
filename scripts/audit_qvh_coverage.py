"""Evaluator-only QVH split coverage of a frozen SQLite/FAISS snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import faiss

from videntra.evaluation.corpus_gate import evaluate_coverage
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import verify_split_lock, qvh_source_id


def audit(split_lock: Path, subset: str, annotations: Path, db_path: Path, index_path: Path) -> dict:
    if subset not in {"dev", "test"} or not verify_split_lock(split_lock):
        raise ValueError("Invalid subset or split lock")
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    if sha256_file(annotations) != lock.get("metadata", {}).get("source_files_sha256", {}).get(subset):
        raise ValueError("Annotation source differs from the split manifest")
    wal = Path(str(db_path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Snapshot has a SQLite WAL; checkpoint/freeze it separately first")
    snapshot_hashes = {"db_sha256": sha256_file(db_path), "index_sha256": sha256_file(index_path)}
    index = faiss.read_index(str(index_path))
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        n, low, high, distinct = conn.execute("SELECT COUNT(*),MIN(faiss_id),MAX(faiss_id),COUNT(DISTINCT faiss_id) FROM KeyframeMap").fetchone()
        if n != index.ntotal or distinct != n or (n and (low != 0 or high != n - 1)):
            raise ValueError("Index and keyframe mapping are inconsistent")
        indexed = {r[0] for r in conn.execute("SELECT DISTINCT video_id FROM KeyframeMap")}

    # Gold is joined only after the snapshot's candidate universe has been fixed.
    with annotations.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    by_id = {f"qvh_{row['qid']}": row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate query IDs in annotation source")
    qrels = []
    for item in lock["records"]:
        if item["split"] != subset:
            continue
        record = by_id.get(item["query_id"])
        if record is None or qvh_source_id(record["vid"]) != item["source_group"]:
            raise ValueError("Split/source group does not match the annotation source")
        qrels.append({"query_id": item["query_id"], "dataset": "qvh_" + subset,
                      "target_videos": [record["vid"]]})
    if (wal.exists() and wal.stat().st_size) or snapshot_hashes != {"db_sha256": sha256_file(db_path), "index_sha256": sha256_file(index_path)}:
        raise RuntimeError("Snapshot changed during the coverage audit")
    return {"subset": subset, "coverage": evaluate_coverage(qrels, indexed),
            "snapshot_hashes": snapshot_hashes, "split_sha256": sha256_file(split_lock),
            "limitations": ["Coverage is evaluator-only; it is not retrieval quality.",
                            "Grouping, rights, model provenance and timeline accuracy remain unverified."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=["dev", "test"])
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.split_lock, args.subset, args.annotations, args.db, args.index), indent=2))
