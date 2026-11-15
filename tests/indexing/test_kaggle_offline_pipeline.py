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
    run_p85_keyframe_backfill,
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


def _write_valid_p85_sidecar_and_receipt(outputs: Path, patches: np.ndarray) -> None:
    import hashlib

    map_rows = [
        {
            "row_index": i,
            "faiss_id": i,
            "keyframe_id": f"vid_001__kf_{i:05d}",
            "video_id": "vid_001",
            "timeline_revision": "tl_v1",
            "image_sha256": f"sha_{i}",
        }
        for i in range(patches.shape[0])
    ]
    map_path = outputs / "spatial_patches_map.jsonl"
    map_path.write_text("".join(json.dumps(r) + "\n" for r in map_rows), encoding="utf-8")
    patches_path = outputs / "spatial_patches_fp16.npy"
    receipt = {
        "receipt_version": "1.0",
        "status": "success",
        "mode": "p85_keyframe_backfill",
        "artifacts": {
            "spatial_patches_fp16": {
                "filename": "spatial_patches_fp16.npy",
                "sha256": hashlib.sha256(patches_path.read_bytes()).hexdigest(),
                "row_count": int(patches.shape[0]),
            },
            "spatial_patches_map": {
                "filename": "spatial_patches_map.jsonl",
                "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "line_count": len(map_rows),
            },
        },
    }
    (outputs / "p85_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def test_ingest_p85_artifacts_updates_db_and_validates_schemas(tmp_path: Path):
    db_path = tmp_path / "processed" / "metadata.db"
    _seed_minimal_db(db_path, num_keyframes=2)

    outputs = tmp_path / "kaggle_out"
    outputs.mkdir()

    # Valid spatial patches (2, 9, 768) float16 + sidecar + receipt
    patches = np.ones((2, 9, 768), dtype=np.float16)
    np.save(outputs / "spatial_patches_fp16.npy", patches)
    _write_valid_p85_sidecar_and_receipt(outputs, patches)

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

    dry_res = ingest_p85_artifacts(outputs_dir=outputs, db_path=db_path, dry_run=True)
    res = ingest_p85_artifacts(outputs_dir=outputs, db_path=db_path, dry_run=False)
    for key in ("spatial_patches_shape", "captions_ingested", "e5_caption_vectors", "receipt_status"):
        assert dry_res[key] == res[key]
    assert res["spatial_patches_shape"] == [2, 9, 768]
    assert res["captions_ingested"] == 2
    assert res["e5_caption_vectors"] == 2
    assert res["receipt_status"] == "verified"
    assert (db_path.parent / "spatial_patches_fp16.npy").is_file()
    assert (db_path.parent / "spatial_patches_map.jsonl").is_file()
    assert (db_path.parent / "p85_receipt.json").is_file()
    assert (db_path.parent / "faiss_e5_caption_384.bin").is_file()

    with sqlite3.connect(str(db_path)) as conn:
        cap_count = conn.execute("SELECT COUNT(*) FROM KeyframeCaptions").fetchone()[0]
        fts_hits = conn.execute(
            "SELECT video_id FROM fts_vlm_caption WHERE fts_vlm_caption MATCH 'nguoi'"
        ).fetchall()
    assert cap_count == 2
    assert len(fts_hits) == 1

    # Mismatched row count should fail closed in both dry-run and real ingest
    np.save(outputs / "spatial_patches_fp16.npy", np.ones((1, 9, 768), dtype=np.float16))
    for dry in (True, False):
        with pytest.raises(ValueError, match="row count|SHA-256"):
            ingest_p85_artifacts(outputs_dir=outputs, db_path=db_path, dry_run=dry)


def test_p85_backfill_out_of_order_manifest_aligns_faiss_id_and_ingest_parity(tmp_path: Path):
    """Section 2.3: out-of-order manifest writes rows at exact faiss_id, verified end-to-end."""
    import hashlib
    from PIL import Image

    db_path = tmp_path / "processed" / "metadata.db"
    _seed_minimal_db(db_path, num_keyframes=2)

    vis_idx = faiss.IndexFlatIP(768)
    vis_idx.add(np.eye(2, 768, dtype=np.float32))
    vis_idx_path = db_path.parent / "faiss_index.bin"
    faiss.write_index(vis_idx, str(vis_idx_path))

    img_dir = tmp_path / "pkg" / "images"
    img_dir.mkdir(parents=True)
    img0 = img_dir / "vid_001__kf_00000.jpg"
    img1 = img_dir / "vid_001__kf_00001.jpg"
    Image.fromarray(np.full((16, 16, 3), 20, dtype=np.uint8)).save(img0)
    Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)).save(img1)

    # Deliberately out-of-order manifest: faiss_id=1 before faiss_id=0
    manifest_path = tmp_path / "pkg" / "keyframes_manifest.jsonl"
    manifest_rows = [
        {
            "faiss_id": 1,
            "keyframe_id": "vid_001__kf_00001",
            "video_id": "vid_001",
            "timestamp_ms": 1000,
            "timeline_revision": "tl_v1",
            "image_file": img1.name,
        },
        {
            "faiss_id": 0,
            "keyframe_id": "vid_001__kf_00000",
            "video_id": "vid_001",
            "timestamp_ms": 0,
            "timeline_revision": "tl_v1",
            "image_file": img0.name,
        },
    ]
    manifest_path.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows), encoding="utf-8")

    class _FakeSiglipProc:
        def __call__(self, images, return_tensors="pt"):
            arrs = [np.asarray(im, dtype=np.float32).mean() for im in images]
            pix = torch.tensor(arrs, dtype=torch.float32).view(len(images), 1, 1, 1).expand(len(images), 3, 8, 8)
            return {"pixel_values": pix}

    class _FakeVisionModel:
        def __call__(self, pixel_values):
            bsz = pixel_values.shape[0]
            hidden = torch.zeros((bsz, 196, 768), dtype=torch.float32)
            for b in range(bsz):
                mean_val = float(pixel_values[b, 0, 0, 0].item())
                # Distinct signature per frame: dim 0 for low-intensity img0, dim 1 for high-intensity img1
                if mean_val < 100.0:
                    hidden[b, :, 0] = 1.0
                else:
                    hidden[b, :, 1] = 1.0
            return type("Out", (), {"last_hidden_state": hidden})()

    class _FakeSiglipModel:
        def __init__(self):
            self.vision_model = _FakeVisionModel()

    out_dir = tmp_path / "p85_out"
    summary = run_p85_keyframe_backfill(
        manifest_path,
        out_dir,
        images_dir=img_dir,
        siglip_batch_size=2,
        enable_florence2=False,
        visual_index_path=vis_idx_path,
        siglip_processor=_FakeSiglipProc(),
        siglip_model=_FakeSiglipModel(),
    )
    assert summary["total_keyframes"] == 2

    # Check sidecar and receipt
    map_lines = [json.loads(l) for l in (out_dir / "spatial_patches_map.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["faiss_id"] for r in map_lines] == [0, 1]
    assert [r["row_index"] for r in map_lines] == [0, 1]
    assert map_lines[0]["image_sha256"] == hashlib.sha256(img0.read_bytes()).hexdigest()
    assert map_lines[1]["image_sha256"] == hashlib.sha256(img1.read_bytes()).hexdigest()

    receipt = json.loads((out_dir / "p85_receipt.json").read_text(encoding="utf-8"))
    assert receipt["artifacts"]["visual_index"]["sha256"] == hashlib.sha256(vis_idx_path.read_bytes()).hexdigest()

    # Dry-run & real ingest parity + faiss_id lookup check on final output
    dry_res = ingest_p85_artifacts(outputs_dir=out_dir, db_path=db_path, dry_run=True)
    real_res = ingest_p85_artifacts(outputs_dir=out_dir, db_path=db_path, dry_run=False)
    assert dry_res["spatial_patches_shape"] == real_res["spatial_patches_shape"] == [2, 9, 768]
    assert dry_res["receipt_status"] == real_res["receipt_status"] == "verified"

    final_patches = np.load(str(db_path.parent / "spatial_patches_fp16.npy"))
    # faiss_id 0 (img0) must have energy on dim 0; faiss_id 1 (img1) must have energy on dim 1
    assert float(final_patches[0, 0, 0]) > 0.9 and float(final_patches[0, 0, 1]) == 0.0
    assert float(final_patches[1, 0, 1]) > 0.9 and float(final_patches[1, 0, 0]) == 0.0


def test_p85_backfill_rejects_duplicate_keyframe_id_and_missing_batch_features(tmp_path: Path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img0 = img_dir / "kf0.jpg"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(img0)

    # 1. Duplicate keyframe_id
    dup_manifest = tmp_path / "dup_manifest.jsonl"
    dup_manifest.write_text(
        json.dumps({"faiss_id": 0, "keyframe_id": "dup_kf", "video_id": "vid_001", "timestamp_ms": 0, "image_file": "kf0.jpg"}) + "\n"
        + json.dumps({"faiss_id": 1, "keyframe_id": "dup_kf", "video_id": "vid_001", "timestamp_ms": 1000, "image_file": "kf0.jpg"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate keyframe_id"):
        run_p85_keyframe_backfill(
            dup_manifest,
            tmp_path / "out_dup",
            images_dir=img_dir,
            enable_florence2=False,
            siglip_processor=object(),
            siglip_model=object(),
        )

    # 2. Batch without features (last_hidden_state is None)
    ok_manifest = tmp_path / "ok_manifest.jsonl"
    ok_manifest.write_text(
        json.dumps({"faiss_id": 0, "keyframe_id": "vid_001__kf_00000", "video_id": "vid_001", "timestamp_ms": 0, "image_file": "kf0.jpg"}) + "\n",
        encoding="utf-8",
    )

    class _Proc:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros((len(images), 3, 8, 8), dtype=torch.float32)}

    class _NoHiddenModel:
        def __call__(self, **_kwargs):
            return type("Out", (), {"last_hidden_state": None})()

    with pytest.raises(ValueError, match="missing last_hidden_state"):
        run_p85_keyframe_backfill(
            ok_manifest,
            tmp_path / "out_nofeat",
            images_dir=img_dir,
            enable_florence2=False,
            siglip_processor=_Proc(),
            siglip_model=_NoHiddenModel(),
        )


def test_p85_ingest_rejects_missing_sidecar_missing_row_and_invalid_caption_manifest(tmp_path: Path):
    db_path = tmp_path / "processed" / "metadata.db"
    _seed_minimal_db(db_path, num_keyframes=2)

    # 1. Missing spatial_patches_map.jsonl for a 2-row float16 .npy
    out_nomap = tmp_path / "out_nomap"
    out_nomap.mkdir()
    np.save(out_nomap / "spatial_patches_fp16.npy", np.ones((2, 9, 768), dtype=np.float16))
    for dry in (True, False):
        with pytest.raises(FileNotFoundError, match="spatial_patches_map.jsonl"):
            ingest_p85_artifacts(outputs_dir=out_nomap, db_path=db_path, dry_run=dry)

    # 2. Missing row in spatial_patches_map.jsonl (1 map row vs 2 npy rows)
    out_missing_row = tmp_path / "out_missing_row"
    out_missing_row.mkdir()
    np.save(out_missing_row / "spatial_patches_fp16.npy", np.ones((2, 9, 768), dtype=np.float16))
    (out_missing_row / "spatial_patches_map.jsonl").write_text(
        json.dumps({"row_index": 0, "faiss_id": 0, "keyframe_id": "vid_001__kf_00000", "video_id": "vid_001"}) + "\n",
        encoding="utf-8",
    )
    for dry in (True, False):
        with pytest.raises(ValueError, match="row count"):
            ingest_p85_artifacts(outputs_dir=out_missing_row, db_path=db_path, dry_run=dry)

    # 3. Invalid caption manifest mapping (wrong video_id for keyframe_id)
    out_bad_cap = tmp_path / "out_bad_cap"
    out_bad_cap.mkdir()
    idx = faiss.IndexFlatIP(384)
    idx.add(np.ones((1, 384), dtype=np.float32))
    faiss.write_index(idx, str(out_bad_cap / "faiss_e5_caption_384.bin"))
    (out_bad_cap / "e5_caption_manifest.json").write_text(
        json.dumps([{"keyframe_faiss_id": 0, "keyframe_id": "vid_001__kf_00000", "video_id": "wrong_vid", "caption_text": "test"}]),
        encoding="utf-8",
    )
    for dry in (True, False):
        with pytest.raises(ValueError, match="belongs to"):
            ingest_p85_artifacts(outputs_dir=out_bad_cap, db_path=db_path, dry_run=dry)


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
