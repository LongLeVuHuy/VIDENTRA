import json
import sqlite3
import numpy as np
import pytest

from scripts.ingest_kaggle_offline_results import (
    ingest_kaggle_offline_outputs,
    validate_recorded_pts,
    validate_worker_outputs,
)


def test_ingest_refuses_existing_snapshot_and_reset(tmp_path):
    db, index = tmp_path / "existing.db", tmp_path / "existing.index"
    db.write_bytes(b"original-db")
    index.write_bytes(b"original-index")
    with pytest.raises(FileExistsError):
        ingest_kaggle_offline_outputs(tmp_path, db, index)
    with pytest.raises(ValueError, match="reset is disabled"):
        ingest_kaggle_offline_outputs(tmp_path, db, index, reset=True)
    assert db.read_bytes() == b"original-db"
    assert index.read_bytes() == b"original-index"


def test_pts_preflight_blocks_inconsistency_and_missing(tmp_path):
    (tmp_path / "videos.json").write_text(json.dumps([{"video_id": "v", "time_base_num": 1, "time_base_den": 1000}]))
    frames = [{"video_id": "v", "pts": 1000, "timestamp_ms": 1000}]
    path = tmp_path / "keyframe_manifest.json"
    path.write_text(json.dumps(frames))
    validate_recorded_pts(tmp_path)
    frames[0]["timestamp_ms"] = 1020
    path.write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="1 mismatches"):
        validate_recorded_pts(tmp_path)
    frames[0]["pts"] = None
    path.write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="1 missing"):
        validate_recorded_pts(tmp_path)


def test_pts_preflight_uses_explicit_nonzero_origin(tmp_path):
    (tmp_path / "videos.json").write_text(json.dumps([{"video_id": "v", "time_base_num": 1,
                                                        "time_base_den": 1000, "time_origin_pts": 10000}]))
    (tmp_path / "keyframe_manifest.json").write_text(json.dumps([{"video_id": "v", "pts": 11000, "timestamp_ms": 1000}]))
    validate_recorded_pts(tmp_path)


def _write_preflight_outputs(outputs, summary, segments):
    outputs.mkdir()
    (outputs / "summary.json").write_text(json.dumps(summary))
    (outputs / "videos.json").write_text("[]")
    (outputs / "keyframe_manifest.json").write_text("[]")
    np.save(outputs / "frame_embeddings.npy", np.empty((0, 768), dtype=np.float32))
    (outputs / "asr_segments.json").write_text(json.dumps(segments))


@pytest.mark.parametrize("summary", [{"status": "partial"}, {}, {"status": "failed"}, []])
def test_ingest_rejects_non_success_summary_before_creating_snapshot(tmp_path, summary):
    outputs = tmp_path / "outputs"
    _write_preflight_outputs(outputs, summary, [])
    db, index = tmp_path / "snapshot.db", tmp_path / "snapshot.faiss"
    with pytest.raises(ValueError, match="status=success"):
        ingest_kaggle_offline_outputs(outputs, db, index)
    assert not db.exists() and not index.exists()
    assert not (tmp_path / "snapshot.db-wal").exists()
    assert not (tmp_path / "snapshot.db-shm").exists()


def test_ingest_rejects_missing_summary_before_creating_snapshot(tmp_path):
    outputs = tmp_path / "outputs"
    _write_preflight_outputs(outputs, {"status": "success"}, [])
    (outputs / "summary.json").unlink()
    db, index = tmp_path / "snapshot.db", tmp_path / "snapshot.faiss"
    with pytest.raises(FileNotFoundError, match="summary.json"):
        ingest_kaggle_offline_outputs(outputs, db, index)
    assert not db.exists() and not index.exists()


@pytest.mark.parametrize("confidence", [-0.42, 1.1, float("nan"), float("inf"), "0.5", True, None])
def test_ingest_rejects_invalid_asr_confidence_before_creating_snapshot(tmp_path, confidence):
    outputs = tmp_path / "outputs"
    _write_preflight_outputs(outputs, {"status": "success"}, [{"confidence": confidence}])
    db, index = tmp_path / "snapshot.db", tmp_path / "snapshot.faiss"
    with pytest.raises(ValueError, match="invalid confidence"):
        ingest_kaggle_offline_outputs(outputs, db, index)
    assert not db.exists() and not index.exists()
    assert not (tmp_path / "snapshot.db-wal").exists()
    assert not (tmp_path / "snapshot.db-shm").exists()


def test_worker_preflight_accepts_finite_confidence_range(tmp_path):
    outputs = tmp_path / "outputs"
    _write_preflight_outputs(outputs, {"status": "success"},
                             [{"confidence": 0}, {"confidence": 0.5}, {"confidence": 1}])
    validate_worker_outputs(outputs)


def test_ingest_preserves_pts_timebase_from_video_metadata(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    videos = [{"video_id": "v", "file_path": "fixture.mp4", "duration_ms": 10000,
               "fps": 24, "time_base_num": 1, "time_base_den": 12288,
               "width": 64, "height": 64, "has_audio": 0}]
    frames = [{"keyframe_id": "k", "video_id": "v", "keyframe_index": 0,
               "source_frame_index": 48, "pts": 24576, "timestamp_ms": 2000}]
    (outputs / "videos.json").write_text(json.dumps(videos))
    (outputs / "keyframe_manifest.json").write_text(json.dumps(frames))
    (outputs / "summary.json").write_text(json.dumps({"status": "success"}))
    np.save(outputs / "frame_embeddings.npy", np.ones((1, 768), dtype=np.float32))
    db, index = tmp_path / "snapshot.db", tmp_path / "snapshot.faiss"
    ingest_kaggle_offline_outputs(outputs, db, index)
    with sqlite3.connect(db) as conn:
        pts, time_base, timestamp_ms, keyframe_path = conn.execute("SELECT pts,time_base,timestamp_ms,keyframe_path FROM Keyframes").fetchone()
    assert (pts, time_base, timestamp_ms) == (24576, "1/12288", 2000)
    assert keyframe_path == ""
