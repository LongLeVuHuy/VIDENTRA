"""Read-only audit of saved Kaggle vectors, row mapping and recorded time bases."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sqlite3

import faiss
import numpy as np

from videntra.evaluation.provenance import sha256_file


def audit(outputs: Path, db_path: Path, index_path: Path) -> dict:
    with (outputs / "keyframe_manifest.json").open(encoding="utf-8") as stream:
        frames = json.load(stream)
    with (outputs / "videos.json").open(encoding="utf-8") as stream:
        videos = {row["video_id"]: row for row in json.load(stream)}
    vectors_path = outputs / "frame_embeddings.npy"
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    index = faiss.read_index(str(index_path))
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT faiss_id,keyframe_id,video_id,pts_time,source_frame_index FROM KeyframeMap ORDER BY faiss_id").fetchall()

    shape_matches = vectors.shape == (index.ntotal, index.d)
    unequal = 0
    nonfinite = 0
    if shape_matches:
        for start in range(0, index.ntotal, 512):
            chunk = np.asarray(vectors[start:start + 512], dtype=np.float32)
            nonfinite += int((~np.isfinite(chunk)).sum())
            unequal += int(np.count_nonzero(chunk != index.reconstruct_n(start, len(chunk))))
    mapping_mismatches = abs(len(rows) - len(frames))
    pts_mismatch = fps_match = missing_timebase = 0
    impacted_videos: set[str] = set()
    max_error_ms = 0.0
    for idx, frame in enumerate(frames):
        if idx < len(rows):
            fid, key, vid, ts, ordinal = rows[idx]
            if (fid != idx or key != frame["keyframe_id"] or vid != frame["video_id"] or
                ordinal != frame["source_frame_index"] or abs(ts * 1000 - frame["timestamp_ms"]) > 0.001):
                mapping_mismatches += 1
        video = videos.get(frame["video_id"], {})
        pts, num, den = frame.get("pts"), video.get("time_base_num"), video.get("time_base_den")
        if pts is None or not num or not den:
            missing_timebase += 1
            continue
        error = abs(float(Fraction(int(pts) * int(num) * 1000, int(den))) - float(frame["timestamp_ms"]))
        if error > 1:
            pts_mismatch += 1
            impacted_videos.add(frame["video_id"])
            max_error_ms = max(max_error_ms, error)
            fps = video.get("fps")
            if fps and abs(frame["source_frame_index"] / fps * 1000 - frame["timestamp_ms"]) <= 1:
                fps_match += 1
    return {"vector_shape_matches": shape_matches, "vector_unequal_elements": unequal if shape_matches else None,
            "nonfinite_elements": nonfinite if shape_matches else None,
            "mapping_mismatches": mapping_mismatches, "rows": len(frames),
            "timestamp_pts_mismatches_gt_1ms": pts_mismatch, "impacted_videos": len(impacted_videos),
            "mismatches_matching_frame_over_fps": fps_match, "missing_pts_or_timebase": missing_timebase,
            "max_timestamp_error_ms": round(max_error_ms, 3),
            "sha256": {p.name: sha256_file(p) for p in (vectors_path, outputs / "keyframe_manifest.json", outputs / "videos.json", index_path)},
            "limitations": "Matches identify saved artifacts; they do not prove the encoder revision or decoded ledger correctness."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.outputs, args.db, args.index), indent=2))
