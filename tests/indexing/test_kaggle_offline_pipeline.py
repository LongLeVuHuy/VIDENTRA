"""CPU-safe unit tests for Kaggle offline packaging, worker P8-5 schema, ingestion, and LLM cache prewarm."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest
import torch

from scripts.ingest_kaggle_offline_results import ingest_p85_artifacts
from scripts.kaggle_multimodal_offline_worker import (
    build_e5_caption_faiss_index,
    ensure_output_files_absent,
    extract_florence2_keyframe_captions,
    extract_spatial_patches_3x3,
)
from scripts.package_kaggle_inputs import package_db_keyframes, package_inputs
from scripts.prewarm_llm_query_cache import (
    atomic_write_cache,
    load_cache_records,
    load_queries_jsonl,
    load_split_queries,
    main as prewarm_main,
    merge_cache_files,
)
from videntra.storage.database import OfflineMetadataStore


def _seed_minimal_db(db_path: Path, num_keyframes: int = 2) -> None:
    store = OfflineMetadataStore(db_path)
    store.register_video({
        "video_id": "vid_001",
        "dataset": "test",
        "file_path": "vid_001.mp4",
        "duration_ms": 5000,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 320,
        "height": 240,
        "total_frames": 125,
        "has_audio": 0,
        "status": "completed",
    })
    kfs = [
        {
            "keyframe_id": f"vid_001__kf_{i:05d}",
            "video_id": "vid_001",
            "keyframe_index": i,
            "source_frame_index": i * 25,
            "pts": i * 1000,
            "time_base": "1/1000",
            "timestamp_ms": i * 1000,
            "keyframe_path": "",
            "thumbnail_path": "",
            "faiss_id": i,
            "sampling_reason": "uniform",
        }
        for i in range(num_keyframes)
    ]
    store.insert_keyframes(kfs)


def test_package_inputs_dry_run_and_db_dry_run(tmp_path: Path):
    manifest = tmp_path / "media.json"
    manifest.write_text(json.dumps({"records": [{"video_id": "v1", "sha256": "abc", "media_uri": "v1.mp4"}]}), encoding="utf-8")
    res = package_inputs(
        manifest_path=str(manifest),
        ingest_dir=str(tmp_path / "ingest"),
        media_root=str(tmp_path / "media"),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
    )
    assert res["dry_run"] is True
    assert res["num_videos"] == 1

    db_path = tmp_path / "metadata.db"
    _seed_minimal_db(db_path, num_keyframes=3)
    db_res = package_db_keyframes(db_path=db_path, output_dir=tmp_path / "db_out", dry_run=True)
    assert db_res["dry_run"] is True
    assert db_res["num_keyframes"] == 3


def test_worker_p85_spatial_patches_florence2_dtype_and_e5_caption_index(tmp_path: Path):
    # 1. SigLIP2 3x3 spatial patches
    hidden = torch.randn(2, 196, 768, dtype=torch.float32)
    patches = extract_spatial_patches_3x3(hidden)
    assert patches.shape == (2, 9, 768)
    assert patches.dtype == np.float16
    norms = np.linalg.norm(patches.astype(np.float32), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-2)

    # 2. Florence-2 captioner casts float32 pixel_values to model.dtype (float16)
    seen_dtypes: list[torch.dtype] = []

    class _FakeProc:
        def __call__(self, text, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros((len(images), 3, 8, 8), dtype=torch.float32)}

        def batch_decode(self, generated_ids, skip_special_tokens=True):
            return ["<DETAILED_CAPTION>A chef slicing vegetables" for _ in generated_ids]

    class _FakeModel:
        dtype = torch.float16

        def generate(self, **kwargs):
            seen_dtypes.append(kwargs["pixel_values"].dtype)
            return [[1, 2] for _ in range(len(kwargs["pixel_values"]))]

    records = [
        {
            "faiss_id": 0,
            "keyframe_id": "vid_001__kf_00000",
            "video_id": "vid_001",
            "timestamp_ms": 0,
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
        }
    ]
    captions = extract_florence2_keyframe_captions(
        records,
        out_dir=tmp_path,
        processor=_FakeProc(),
        model=_FakeModel(),
        device="cpu",
        batch_size=1,
    )
    assert seen_dtypes == [torch.float16]
    assert captions[0]["keyframe_id"] == "vid_001__kf_00000"
    assert captions[0]["caption_text"] == "A chef slicing vegetables"

    # 3. E5 caption FAISS index
    class _FakeTok:
        def __call__(self, batch, **_kwargs):
            n = len(batch)
            return {
                "input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
            }

    class _FakeE5Model:
        def __call__(self, **kwargs):
            n = kwargs["input_ids"].shape[0]
            return type("Out", (), {"last_hidden_state": torch.ones((n, 4, 384), dtype=torch.float32)})()

    idx_info = build_e5_caption_faiss_index(
        captions,
        out_dir=tmp_path,
        tokenizer=_FakeTok(),
        model=_FakeE5Model(),
        device="cpu",
    )
    assert idx_info["ntotal"] == 1
    assert (tmp_path / "faiss_e5_caption_384.bin").is_file()
    manifest = json.loads((tmp_path / "e5_caption_manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["keyframe_faiss_id"] == 0
    assert manifest[0]["caption_text"] == "A chef slicing vegetables"

    with pytest.raises(FileExistsError):
        ensure_output_files_absent(tmp_path)


def test_ingest_p85_artifacts_updates_db_and_validates_schemas(tmp_path: Path):
    db_path = tmp_path / "processed" / "metadata.db"
    _seed_minimal_db(db_path, num_keyframes=2)

    outputs = tmp_path / "kaggle_out"
    outputs.mkdir()

    # Valid spatial patches (2, 9, 768) float16
    patches = np.ones((2, 9, 768), dtype=np.float16)
    np.save(outputs / "spatial_patches_fp16.npy", patches)

    # Valid captions
    captions = [
        {"faiss_id": 0, "keyframe_id": "vid_001__kf_00000", "video_id": "vid_001", "timestamp_ms": 0, "caption_text": "Red car parked outside"},
        {"faiss_id": 1, "keyframe_id": "vid_001__kf_00001", "video_id": "vid_001", "timestamp_ms": 1000, "caption_text": "Người đàn ông đang nấu ăn"},
    ]
    (outputs / "keyframe_captions.json").write_text(json.dumps(captions), encoding="utf-8")

    # Valid 384d FAISS caption index + manifest
    idx = faiss.IndexFlatIP(384)
    vecs = np.zeros((2, 384), dtype=np.float32)
    vecs[:, 0] = 1.0
    idx.add(vecs)
    faiss.write_index(idx, str(outputs / "faiss_e5_caption_384.bin"))
    (outputs / "e5_caption_manifest.json").write_text(json.dumps(captions), encoding="utf-8")

    res = ingest_p85_artifacts(outputs_dir=outputs, db_path=db_path)
    assert res["spatial_patches_shape"] == [2, 9, 768]
    assert res["captions_ingested"] == 2
    assert res["e5_caption_vectors"] == 2
    assert (db_path.parent / "spatial_patches_fp16.npy").is_file()
    assert (db_path.parent / "faiss_e5_caption_384.bin").is_file()

    with sqlite3.connect(str(db_path)) as conn:
        cap_count = conn.execute("SELECT COUNT(*) FROM KeyframeCaptions").fetchone()[0]
        fts_hits = conn.execute(
            "SELECT video_id FROM fts_vlm_caption WHERE fts_vlm_caption MATCH 'nguoi'"
        ).fetchall()
    assert cap_count == 2
    assert len(fts_hits) == 1

    # Mismatched row count should fail closed
    np.save(outputs / "spatial_patches_fp16.npy", np.ones((1, 9, 768), dtype=np.float16))
    with pytest.raises(ValueError, match="row count"):
        ingest_p85_artifacts(outputs_dir=outputs, db_path=db_path)


def test_prewarm_split_no_gt_leakage_and_atomic_merge(tmp_path: Path):
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "test_queries"
    (val_dir / "qvhighlights").mkdir(parents=True)
    (test_dir / "qvhighlights").mkdir(parents=True)

    # Include GT fields (relevant_windows) to verify they are NEVER leaked
    (val_dir / "qvhighlights" / "highlight_val_release.jsonl").write_text(
        json.dumps({"qid": 1, "vid": "v1", "query": "A dog running", "relevant_windows": [[1.0, 5.0]]}) + "\n",
        encoding="utf-8",
    )
    (test_dir / "qvhighlights" / "highlight_val_release.jsonl").write_text(
        json.dumps({"qid": 2, "vid": "v2", "query": "A cat sleeping", "relevant_windows": [[2.0, 6.0]]}) + "\n",
        encoding="utf-8",
    )

    all_q = load_split_queries("all_1060", val_dir, test_dir, indexed_videos={"v1", "v2"})
    assert len(all_q) == 2
    assert all(set(q.keys()) == {"dataset", "query_id", "query_text"} for q in all_q)

    # Export GT-free queries JSONL + dry-run
    exported = tmp_path / "queries_1060.jsonl"
    assert prewarm_main([
        "--split", "all_1060",
        "--db-path", str(tmp_path / "missing.db"),
        "--allow-missing-db",
        "--val-dir", str(val_dir),
        "--test-dir", str(test_dir),
        "--export-queries-only", str(exported),
    ]) == 0
    loaded_q = load_queries_jsonl(exported)
    assert len(loaded_q) == 2

    # Atomic cache load (skips corrupted line) + atomic merge
    local_cache = tmp_path / "local_cache.jsonl"
    local_cache.write_text(
        json.dumps({"key": "k1", "plan": {"original_query": "A dog running"}}) + "\n{corrupted\n",
        encoding="utf-8",
    )
    assert list(load_cache_records(local_cache).keys()) == ["k1"]

    kaggle_cache = tmp_path / "kaggle_cache.jsonl"
    atomic_write_cache(kaggle_cache, {
        "k1": {"key": "k1", "plan": {"original_query": "A dog running"}},
        "k2": {"key": "k2", "plan": {"original_query": "A cat sleeping"}},
    })
    stats = merge_cache_files(local_cache, [kaggle_cache])
    assert stats["total"] == 2
    assert stats["added"] == 1
    assert prewarm_main([
        "--queries-file", str(exported),
        "--cache-path", str(local_cache),
        "--dry-run",
    ]) == 0
