import json
import sqlite3

import faiss
import numpy as np
import pytest

from scripts.audit_qvh_coverage import audit
from scripts.freeze_qvh_source_split import build


def test_qvh_coverage_is_computed_against_frozen_snapshot(tmp_path):
    dev, test = tmp_path / "dev.jsonl", tmp_path / "test.jsonl"
    dev.write_text(json.dumps({"qid": 1, "vid": "abcdefghijk_0.0_10.0"}) + "\n", encoding="utf-8")
    test.write_text(json.dumps({"qid": 2, "vid": "lmnopqrstuv_0.0_10.0"}) + "\n", encoding="utf-8")
    lock = tmp_path / "split.json"
    build(dev, test, lock)
    db = tmp_path / "metadata.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE KeyframeMap (faiss_id INTEGER, video_id TEXT)")
        conn.execute("INSERT INTO KeyframeMap VALUES (0, 'abcdefghijk_0.0_10.0')")
    index = faiss.IndexFlatIP(2)
    index.add(np.ones((1, 2), dtype=np.float32))
    index_path = tmp_path / "index.bin"
    faiss.write_index(index, str(index_path))
    result = audit(lock, "test", test, db, index_path)
    assert result["coverage"]["qvh_test"]["coverage"] == 0
    assert result["coverage"]["qvh_test"]["not_covered"] == 1
    wal = tmp_path / "metadata.db-wal"
    wal.write_bytes(b"not-frozen")
    with pytest.raises(ValueError, match="SQLite WAL"):
        audit(lock, "test", test, db, index_path)
    wal.unlink()
    test.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        audit(lock, "test", test, db, index_path)
