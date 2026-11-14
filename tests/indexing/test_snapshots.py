"""Tests for SQLite FTS5 text indexing and atomic snapshot manifests."""
import json
from pathlib import Path
import pytest

from videntra.indexing.text_store import TextIndexStore
from videntra.indexing.snapshots import build_snapshot_manifest, SnapshotManifest, verify_snapshot_artifacts


def test_text_index_store_fts5_bm25(tmp_path):
    db_file = tmp_path / "test_fts.db"
    store = TextIndexStore(db_file)

    entries = [
        {
            "id": "ocr_1",
            "text": "Siêu thị giảm giá SALE 50%",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "start_ordinal": 10,
            "end_ordinal": 30,
            "confidence": 0.95,
        },
        {
            "id": "asr_1",
            "text": "Chào mừng bạn đến với cửa hàng của chúng tôi",
            "start_seconds": 2.0,
            "end_seconds": 5.0,
            "start_ordinal": 20,
            "end_ordinal": 50,
            "confidence": 0.90,
        },
    ]

    added = store.add_entries(video_id="video_01", source_type="ocr", entries=[entries[0]])
    added += store.add_entries(video_id="video_01", source_type="asr", entries=[entries[1]])
    assert added == 2

    # Search OCR keyword
    hits = store.search("giảm giá")
    assert len(hits) == 1
    assert hits[0].source_type == "ocr"
    assert "SALE" in hits[0].text
    assert hits[0].score > 0.0

    # Search ASR keyword with diacritics
    asr_hits = store.search("chào mừng", source_type="asr")
    assert len(asr_hits) == 1
    assert asr_hits[0].source_type == "asr"

    # Search non-existent
    empty = store.search("hoàn toàn không có")
    assert len(empty) == 0

    store.close()


def test_snapshot_manifest_build_and_load(tmp_path):
    vis_dir = tmp_path / "visual_idx"
    vis_dir.mkdir()
    id_map = vis_dir / "id_mapping.parquet"
    id_map.write_bytes(b"dummy_parquet_data")
    (vis_dir / "index").mkdir()
    index_file = vis_dir / "index" / "index_numpy.npz"
    index_file.write_bytes(b"fixture_vectors")

    text_idx = tmp_path / "text.db"
    text_idx.write_bytes(b"dummy_sqlite_data")

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    manifest = build_snapshot_manifest(
        snapshot_id="snapshot_test_01",
        visual_index_dir=vis_dir,
        text_index_path=text_idx,
        evidence_dir=evidence_dir,
        video_ids=["video_A", "video_B"],
    )

    manifest_file = tmp_path / "snapshot.json"
    manifest.save(manifest_file)

    with pytest.raises(FileExistsError):
        manifest.save(manifest_file)

    loaded = SnapshotManifest.load(manifest_file)
    assert loaded.snapshot_id == "snapshot_test_01"
    assert "video_A" in loaded.video_ids
    assert "text_index" in loaded.content_hashes
    assert "visual_id_mapping" in loaded.content_hashes
    assert verify_snapshot_artifacts(loaded)
    unrecorded_faiss = vis_dir / "index" / "index.faiss"
    unrecorded_faiss.write_bytes(b"unrecorded")
    assert not verify_snapshot_artifacts(loaded)
    unrecorded_faiss.unlink()
    index_file.write_bytes(b"modified")
    assert not verify_snapshot_artifacts(loaded)
