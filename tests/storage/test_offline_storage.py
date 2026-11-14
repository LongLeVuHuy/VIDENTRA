"""Tests for Offline Metadata Storage, Multi-Model Grouping, and Integrity Checks."""
import sqlite3
from pathlib import Path
import numpy as np
import pytest

from videntra.storage.database import OfflineMetadataStore
from videntra.storage.grouping import OCREvidenceGrouper, ASREvidenceGrouper, text_similarity
from videntra.storage.integrity import verify_offline_storage_integrity


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "metadata.db"
    store = OfflineMetadataStore(db_file)
    return store, db_file


def test_video_and_keyframe_storage(temp_db):
    store, db_file = temp_db

    # 1. Register video
    vid_record = {
        "video_id": "test_video_01",
        "dataset": "msrvtt",
        "file_path": "/data/raw/msrvtt/media/video0.mp4",
        "duration_ms": 10500,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 1280,
        "height": 720,
        "total_frames": 260,
        "has_audio": 1,
        "status": "ingested",
    }
    store.register_video(vid_record)

    # 2. Insert keyframes
    kfs = [
        {
            "keyframe_id": "test_video_01__kf_00000",
            "video_id": "test_video_01",
            "keyframe_index": 0,
            "source_frame_index": 0,
            "pts": 0,
            "time_base": "1/1000",
            "timestamp_ms": 0,
            "keyframe_path": "/data/raw/keyframes/test_video_01/0000.jpg",
            "faiss_id": 0,
            "sampling_reason": "start",
        },
        {
            "keyframe_id": "test_video_01__kf_00001",
            "video_id": "test_video_01",
            "keyframe_index": 1,
            "source_frame_index": 48,
            "pts": 1920,
            "time_base": "1/1000",
            "timestamp_ms": 1920,
            "keyframe_path": "/data/raw/keyframes/test_video_01/0048.jpg",
            "faiss_id": 1,
            "sampling_reason": "adaptive_luma",
        },
    ]
    store.insert_keyframes(kfs)

    # 3. Update FAISS mapping
    mappings = [
        {
            "faiss_id": 0,
            "keyframe_id": "test_video_01__kf_00000",
            "video_id": "test_video_01",
            "pts_time": 0.0,
            "source_frame_index": 0,
            "keyframe_path": "/data/raw/keyframes/test_video_01/0000.jpg",
        },
        {
            "faiss_id": 1,
            "keyframe_id": "test_video_01__kf_00001",
            "video_id": "test_video_01",
            "pts_time": 1.92,
            "source_frame_index": 48,
            "keyframe_path": "/data/raw/keyframes/test_video_01/0048.jpg",
        },
    ]
    store.update_faiss_mappings(mappings)

    # 4. Check integrity
    report = verify_offline_storage_integrity(db_file)
    assert report.passed is True
    assert report.total_videos == 1
    assert report.total_keyframes == 2
    assert len(report.errors) == 0


def test_object_detection_storage(temp_db):
    store, db_file = temp_db
    store.register_video({
        "video_id": "v_obj",
        "dataset": "avc",
        "file_path": "/data/raw/avc/media/v1.mp4",
        "duration_ms": 5000,
        "fps": 30.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 1920,
        "height": 1080,
        "total_frames": 150,
    })
    store.insert_keyframes([{
        "keyframe_id": "v_obj__kf_00000",
        "video_id": "v_obj",
        "keyframe_index": 0,
        "source_frame_index": 0,
        "pts": 0,
        "time_base": "1/1000",
        "timestamp_ms": 0,
        "keyframe_path": "/kf.jpg",
        "faiss_id": 0,
    }])
    store.update_faiss_mappings([{
        "faiss_id": 0,
        "keyframe_id": "v_obj__kf_00000",
        "video_id": "v_obj",
        "pts_time": 0.0,
        "source_frame_index": 0,
        "keyframe_path": "/kf.jpg",
    }])

    # Add object detections
    detections = [
        {
            "detection_id": "det_001",
            "timestamp_ms": 0,
            "class_id": 2,
            "class_name": "car",
            "bbox_x1": 0.1,
            "bbox_y1": 0.2,
            "bbox_x2": 0.4,
            "bbox_y2": 0.6,
            "confidence": 0.92,
        },
        {
            "detection_id": "det_002",
            "timestamp_ms": 0,
            "class_id": 0,
            "class_name": "person",
            "bbox_x1": 0.5,
            "bbox_y1": 0.3,
            "bbox_x2": 0.6,
            "bbox_y2": 0.8,
            "confidence": 0.88,
        },
    ]
    store.record_frame_objects("v_obj__kf_00000", "v_obj", detections, model_id="yolov8x")

    report = verify_offline_storage_integrity(db_file)
    assert report.passed is True
    assert report.total_frame_objects == 2


def test_ocr_multi_model_grouping(temp_db):
    store, db_file = temp_db
    store.register_video({
        "video_id": "v_ocr",
        "dataset": "textvr",
        "file_path": "/data/v.mp4",
        "duration_ms": 5000,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 1280,
        "height": 720,
        "total_frames": 100,
    })
    store.insert_keyframes([{
        "keyframe_id": "v_ocr__kf_00000",
        "video_id": "v_ocr",
        "keyframe_index": 0,
        "source_frame_index": 0,
        "pts": 0,
        "time_base": "1/1000",
        "timestamp_ms": 0,
        "keyframe_path": "/kf.jpg",
        "faiss_id": 0,
    }])
    store.update_faiss_mappings([{
        "faiss_id": 0,
        "keyframe_id": "v_ocr__kf_00000",
        "video_id": "v_ocr",
        "pts_time": 0.0,
        "source_frame_index": 0,
        "keyframe_path": "/kf.jpg",
    }])

    # Observations from Model A (PP-OCR) and Model B (VietOCR)
    obs = [
        # Box 1: Agreement
        {
            "observation_id": "ocr_a_1",
            "video_id": "v_ocr",
            "keyframe_id": "v_ocr__kf_00000",
            "timestamp_ms": 0,
            "model_id": "OCR_A",
            "raw_text": "Cửa Hàng",
            "bbox_x1": 0.10, "bbox_y1": 0.10, "bbox_x2": 0.30, "bbox_y2": 0.20,
            "confidence": 0.85,
        },
        {
            "observation_id": "ocr_b_1",
            "video_id": "v_ocr",
            "keyframe_id": "v_ocr__kf_00000",
            "timestamp_ms": 0,
            "model_id": "OCR_B",
            "raw_text": "Cửa hàng",
            "bbox_x1": 0.11, "bbox_y1": 0.10, "bbox_x2": 0.31, "bbox_y2": 0.21,
            "confidence": 0.95,
        },
        # Box 2: Disputed
        {
            "observation_id": "ocr_a_2",
            "video_id": "v_ocr",
            "keyframe_id": "v_ocr__kf_00000",
            "timestamp_ms": 0,
            "model_id": "OCR_A",
            "raw_text": "STOP 100",
            "bbox_x1": 0.70, "bbox_y1": 0.70, "bbox_x2": 0.90, "bbox_y2": 0.85,
            "confidence": 0.70,
        },
        {
            "observation_id": "ocr_b_2",
            "video_id": "v_ocr",
            "keyframe_id": "v_ocr__kf_00000",
            "timestamp_ms": 0,
            "model_id": "OCR_B",
            "raw_text": "KHẨN CẤP",
            "bbox_x1": 0.71, "bbox_y1": 0.69, "bbox_x2": 0.89, "bbox_y2": 0.84,
            "confidence": 0.90,
        },
        # Box 3: Single model (only Model A saw it)
        {
            "observation_id": "ocr_a_3",
            "video_id": "v_ocr",
            "keyframe_id": "v_ocr__kf_00000",
            "timestamp_ms": 0,
            "model_id": "OCR_A",
            "raw_text": "SALE 50%",
            "bbox_x1": 0.40, "bbox_y1": 0.40, "bbox_x2": 0.55, "bbox_y2": 0.50,
            "confidence": 0.80,
        },
    ]
    store.insert_ocr_observations(obs)

    # Run Grouper
    groups, members = OCREvidenceGrouper.group_keyframe_observations("v_ocr", "v_ocr__kf_00000", 0, obs)
    assert len(groups) == 3

    statuses = [g["status"] for g in groups]
    assert "agreed" in statuses
    assert "disputed" in statuses
    assert "single_model" in statuses

    store.insert_ocr_evidence_groups(groups, members)

    report = verify_offline_storage_integrity(db_file)
    assert report.passed is True
    assert report.total_ocr_observations == 5
    assert report.total_ocr_groups == 3


def test_asr_multi_model_grouping(temp_db):
    store, db_file = temp_db
    store.register_video({
        "video_id": "v_asr",
        "dataset": "tvr",
        "file_path": "/data/v.mp4",
        "duration_ms": 10000,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 1280,
        "height": 720,
        "total_frames": 250,
        "has_audio": 1,
    })

    segs = [
        # Match 1: Agreed
        {
            "segment_id": "asr_a_1",
            "video_id": "v_asr",
            "model_id": "ASR_A",
            "start_ms": 1000,
            "end_ms": 3500,
            "raw_text": "Xin chào quý vị và các bạn",
            "confidence": 0.90,
        },
        {
            "segment_id": "asr_b_1",
            "video_id": "v_asr",
            "model_id": "ASR_B",
            "start_ms": 1050,
            "end_ms": 3450,
            "raw_text": "Xin chào quý vị và các bạn",
            "confidence": 0.96,
        },
        # Match 2: Single model
        {
            "segment_id": "asr_b_2",
            "video_id": "v_asr",
            "model_id": "ASR_B",
            "start_ms": 6000,
            "end_ms": 8500,
            "raw_text": "Chúng tôi đang theo dõi trực tiếp",
            "confidence": 0.88,
        },
    ]
    store.insert_asr_segments(segs)

    groups, members = ASREvidenceGrouper.group_video_segments("v_asr", segs)
    assert len(groups) == 2
    assert groups[0]["status"] == "agreed"
    assert groups[1]["status"] == "single_model"

    store.insert_asr_evidence_groups(groups, members)

    report = verify_offline_storage_integrity(db_file)
    assert report.passed is True
    assert report.total_asr_segments == 3
    assert report.total_asr_groups == 2


def test_fts5_accent_and_unaccent_search(temp_db):
    store, db_file = temp_db

    # Register video and keyframe first to satisfy foreign key constraints
    store.register_video({
        "video_id": "vid_fts",
        "dataset": "textvr",
        "file_path": "/data/v.mp4",
        "duration_ms": 5000,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 1280,
        "height": 720,
        "total_frames": 100,
    })
    store.insert_keyframes([{
        "keyframe_id": "kf_fts_1",
        "video_id": "vid_fts",
        "keyframe_index": 0,
        "source_frame_index": 0,
        "pts": 1000,
        "time_base": "1/1000",
        "timestamp_ms": 1000,
        "keyframe_path": "/kf.jpg",
        "faiss_id": 0,
    }])

    # Query OCR via FTS5 with Vietnamese diacritics
    obs = [{
        "observation_id": "obs_1",
        "video_id": "vid_fts",
        "keyframe_id": "kf_fts_1",
        "timestamp_ms": 1000,
        "model_id": "OCR_A",
        "raw_text": "Bệnh viện Đa khoa",
        "bbox_x1": 0.1, "bbox_y1": 0.1, "bbox_x2": 0.5, "bbox_y2": 0.3,
    }]
    store.insert_ocr_observations(obs)

    groups, members = OCREvidenceGrouper.group_keyframe_observations("vid_fts", "kf_fts_1", 1000, obs)
    store.insert_ocr_evidence_groups(groups, members)

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Search with accent: "Bệnh viện"
    res1 = cur.execute("SELECT OCREvidenceGroups.video_id, consensus_text FROM fts_ocr JOIN OCREvidenceGroups ON fts_ocr.group_id = OCREvidenceGroups.group_id WHERE fts_ocr MATCH 'Bệnh*'").fetchall()
    assert len(res1) >= 1
    assert "Bệnh viện" in res1[0][1]

    # Search unaccented: "Benh vien" -> should match via unaccented_text column
    res2 = cur.execute("SELECT OCREvidenceGroups.video_id, consensus_text FROM fts_ocr JOIN OCREvidenceGroups ON fts_ocr.group_id = OCREvidenceGroups.group_id WHERE fts_ocr MATCH 'Benh*'").fetchall()
    assert len(res2) >= 1
    assert "Bệnh viện" in res2[0][1]
    conn.close()
