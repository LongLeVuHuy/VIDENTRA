import json
import sqlite3

from scripts.audit_timeline_dependencies import audit


def test_dependency_audit_reports_mismatched_timestamp(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"status": "REVIEW_REQUIRED_NOT_APPLIED", "proposed_rows": [
        {"keyframe_id": "k1", "old_timestamp_ms": 1000}]}), encoding="utf-8")
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE KeyframeMap(keyframe_id TEXT, pts_time REAL)")
        conn.execute("INSERT INTO KeyframeMap VALUES('k1', 1.0)")
        for table in ["Keyframes", "OCRObservations", "FrameObjects", "OCREvidenceGroups", "fts_ocr"]:
            conn.execute(f"CREATE TABLE {table}(keyframe_id TEXT, timestamp_ms INTEGER)")
        conn.execute("INSERT INTO OCRObservations VALUES('k1', 900)")
    report = audit(proposal, db)
    assert report["proposal_rows"] == 1
    assert report["dependencies"]["KeyframeMap"]["equal_old_timestamp"] == 1
    assert report["dependencies"]["OCRObservations"]["different_timestamp"] == 1
    assert report["dependencies"]["FrameObjects"]["rows"] == 0
