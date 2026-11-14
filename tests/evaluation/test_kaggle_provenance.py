import json
import sqlite3

import faiss
import numpy as np

from scripts.audit_kaggle_provenance import audit


def test_audit_detects_pts_and_vector_mismatch(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "videos.json").write_text(json.dumps([{"video_id": "v", "fps": 24,
        "time_base_num": 1, "time_base_den": 1000}]), encoding="utf-8")
    frames = [{"keyframe_id": "k", "video_id": "v", "pts": 1000,
               "timestamp_ms": 2000, "source_frame_index": 48}]
    (outputs / "keyframe_manifest.json").write_text(json.dumps(frames), encoding="utf-8")
    np.save(outputs / "frame_embeddings.npy", np.array([[1, 2]], dtype=np.float32))
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 3]], dtype=np.float32))
    index_path = tmp_path / "idx.bin"
    faiss.write_index(index, str(index_path))
    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE KeyframeMap (faiss_id INTEGER, keyframe_id TEXT, video_id TEXT, pts_time REAL, source_frame_index INTEGER)")
        conn.execute("INSERT INTO KeyframeMap VALUES (0, 'k', 'v', 2.0, 48)")
    report = audit(outputs, db, index_path)
    assert report["vector_unequal_elements"] == 1
    assert report["mapping_mismatches"] == 0
    assert report["timestamp_pts_mismatches_gt_1ms"] == 1
    assert report["mismatches_matching_frame_over_fps"] == 1
