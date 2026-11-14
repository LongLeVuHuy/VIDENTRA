"""Read-only model/index audit; a matching count is not provenance proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import faiss

from videntra.evaluation.provenance import hash_paths


def audit(db_path: Path, index_path: Path, model_dir: Path) -> dict:
    if not db_path.is_file() or not index_path.is_file() or not model_dir.is_dir():
        raise FileNotFoundError("Database, index, or model directory not found")
    index = faiss.read_index(str(index_path))
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        count, low, high, unique = conn.execute(
            "SELECT count(*), min(faiss_id), max(faiss_id), count(distinct faiss_id) FROM KeyframeMap").fetchone()
        manifests = conn.execute("SELECT model_id, model_version, weights_hash, checksum, total_vectors, vector_dim, metric FROM IndexManifest WHERE is_active=1").fetchall()
        model_versions = conn.execute("SELECT count(*) FROM ModelVersions").fetchone()[0]
    finally:
        conn.close()
    hashes = hash_paths([index_path, model_dir])
    model_hash = hashes.get(str(model_dir / "model.safetensors"))
    index_hash = hashes[str(index_path)]
    active = manifests[0] if len(manifests) == 1 else None
    return {"index": {"count": index.ntotal, "dimension": index.d, "metric": int(index.metric_type), "sha256": index_hash},
            "mapping": {"count": count, "min_id": low, "max_id": high, "unique": unique},
            "mapping_consistent": count == index.ntotal and unique == count and (count == 0 or (low == 0 and high == count - 1)),
            "active_manifests": manifests, "model_versions_count": model_versions,
            "model_files_sha256": {k: v for k, v in hashes.items() if k != str(index_path)},
            "declared_index_hash_matches": (active[3] == index_hash) if active and active[3] else None,
            "declared_weights_hash_matches": (active[2] == model_hash) if active and active[2] and model_hash else None,
            "model_index_compatibility": "UNVERIFIED",
            "note": "Local hashes identify bytes; they do not prove the encoder that produced the index, revision or license."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.db, args.index, args.model_dir), indent=2))
