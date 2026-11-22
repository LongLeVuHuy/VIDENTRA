import json
from pathlib import Path
import sqlite3

import faiss
import numpy as np

from videntra.config.paths import (
    CACHE_SUBDIRS,
    DEFAULT_AUGMENTED_EMBEDDINGS_PATH,
    DEFAULT_E5_FAISS_PATH,
    DEFAULT_E5_MANIFEST_PATH,
    DEFAULT_HUBNESS_RADII_PATH,
    DEFAULT_KEYFRAME_FAISS_PATH,
    DEFAULT_METADATA_DB_PATH,
    DEFAULT_QUERY_PLANS_CACHE_PATH,
    DEFAULT_SIGLIP2_QUERY_CACHE_PATH,
    DEFAULT_VIDEOPRISM_CLIP_FAISS_PATH,
    DEFAULT_VIDEOPRISM_CLIP_MANIFEST_PATH,
    DEFAULT_VIDEOPRISM_QUERY_CACHE_PATH,
    DEFAULT_VIDEOPRISM_SUBCLIPS_MANIFEST_PATH,
    DEFAULT_VIDEOPRISM_SUBCLIPS_NPY_PATH,
    PROCESSED_SUBDIRS,
    resolve_artifact_path,
    resolve_env_db_path,
    resolve_env_faiss_path,
    resolve_keyframe_faiss_path,
    resolve_metadata_db_path,
    resolve_processed_root,
)
from videntra.registry import DatasetRegistryManager


def _norm(shape: tuple[int, ...], seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    return arr / np.maximum(1e-12, np.linalg.norm(arr, axis=-1, keepdims=True))


def test_canonical_self_documenting_constants() -> None:
    assert PROCESSED_SUBDIRS == ("catalog", "visual_keyframes", "visual_clips", "text_evidence")
    assert CACHE_SUBDIRS == ("query_embeddings", "query_plans", "calibration")
    assert DEFAULT_METADATA_DB_PATH == Path("data/processed/catalog/corpus_metadata.db")
    assert DEFAULT_KEYFRAME_FAISS_PATH == Path("data/processed/visual_keyframes/siglip2_keyframes_768d.faiss")
    assert DEFAULT_AUGMENTED_EMBEDDINGS_PATH == Path("data/processed/visual_keyframes/siglip2_yolo_augmented_864d.npy")
    assert DEFAULT_VIDEOPRISM_CLIP_FAISS_PATH == Path("data/processed/visual_clips/videoprism_whole_clips_768d.faiss")
    assert DEFAULT_VIDEOPRISM_CLIP_MANIFEST_PATH == Path("data/processed/visual_clips/videoprism_whole_clips_manifest.json")
    assert DEFAULT_VIDEOPRISM_SUBCLIPS_NPY_PATH == Path("data/processed/visual_clips/videoprism_subclips_768d.npy")
    assert DEFAULT_VIDEOPRISM_SUBCLIPS_MANIFEST_PATH == Path("data/processed/visual_clips/videoprism_subclips_manifest.jsonl")
    assert DEFAULT_E5_FAISS_PATH == Path("data/processed/text_evidence/e5_ocr_asr_384d.faiss")
    assert DEFAULT_E5_MANIFEST_PATH == Path("data/processed/text_evidence/e5_ocr_asr_manifest.json")
    assert DEFAULT_SIGLIP2_QUERY_CACHE_PATH == Path("data/cache/query_embeddings/siglip2_query_768d_cache.npz")
    assert DEFAULT_VIDEOPRISM_QUERY_CACHE_PATH == Path("data/cache/query_embeddings/videoprism_query_768d_cache.npz")
    assert DEFAULT_QUERY_PLANS_CACHE_PATH == Path("data/cache/query_plans/query_decomposition_plans.jsonl")
    assert DEFAULT_HUBNESS_RADII_PATH == Path("data/cache/calibration/siglip2_keyframe_hubness_radii.npy")


def test_resolve_processed_root_and_legacy_fallback(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "processed"
    (proc / "catalog").mkdir(parents=True)
    nested_db = proc / "catalog" / "corpus_metadata.db"
    nested_db.write_bytes(b"sqlite")

    assert resolve_processed_root(proc) == proc
    assert resolve_processed_root(proc / "catalog") == proc
    assert resolve_processed_root(nested_db) == proc
    assert resolve_metadata_db_path(proc) == nested_db

    flat_dir = tmp_path / "flat_proc"
    flat_dir.mkdir()
    legacy_db = flat_dir / "metadata.db"
    legacy_idx = flat_dir / "faiss_index.bin"
    legacy_db.write_bytes(b"sqlite")
    legacy_idx.write_bytes(b"faiss")
    assert resolve_processed_root(legacy_db) == flat_dir
    assert resolve_metadata_db_path(flat_dir) == legacy_db
    assert resolve_keyframe_faiss_path(flat_dir) == legacy_idx
    assert resolve_artifact_path(flat_dir, "text_evidence/e5_ocr_asr_384d.faiss", "faiss_text_e5_384.bin") == (
        flat_dir / "text_evidence/e5_ocr_asr_384d.faiss"
    )

    monkeypatch.setenv("VIDENTRA_DB_PATH", str(nested_db))
    monkeypatch.setenv("VIDENTRA_FAISS_PATH", str(legacy_idx))
    assert resolve_env_db_path() == nested_db
    assert resolve_env_faiss_path() == legacy_idx


def test_dataset_manager_prunes_structured_processed_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    proc = data_root / "processed"
    for sub in PROCESSED_SUBDIRS:
        (proc / sub).mkdir(parents=True, exist_ok=True)
    roots_cfg = tmp_path / "configs/local/asset_roots.json"

    mgr = DatasetRegistryManager(data_root=data_root, asset_roots_path=roots_cfg)
    mgr.add_dataset("ds_keep")
    mgr.add_dataset("ds_drop")

    (mgr.dataset_paths("ds_keep").media_dir / "vk1.mp4").write_bytes(b"k1")
    (mgr.dataset_paths("ds_keep").keyframes_dir / "vk1").mkdir()
    (mgr.dataset_paths("ds_keep").keyframes_dir / "vk1/000000.jpg").write_bytes(b"jk1")

    db_path = proc / "catalog/corpus_metadata.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE Videos (video_id TEXT PRIMARY KEY, dataset TEXT, file_path TEXT)")
        conn.execute("CREATE TABLE Keyframes (keyframe_id TEXT PRIMARY KEY, video_id TEXT, faiss_id INTEGER, keyframe_path TEXT)")
        conn.execute("CREATE TABLE KeyframeMap (faiss_id INTEGER PRIMARY KEY, keyframe_id TEXT, video_id TEXT)")
        conn.execute("CREATE TABLE IndexManifest (snapshot_id TEXT PRIMARY KEY, total_vectors INTEGER)")
        conn.execute("INSERT INTO Videos VALUES (?, ?, ?)", ("vd1", "ds_drop", str(mgr.dataset_paths("ds_drop").media_dir / "vd1.mp4")))
        conn.execute("INSERT INTO Videos VALUES (?, ?, ?)", ("vk1", "ds_keep", str(mgr.dataset_paths("ds_keep").media_dir / "vk1.mp4")))
        conn.execute("INSERT INTO Keyframes VALUES (?, ?, ?, ?)", ("vd1__0", "vd1", 0, str(mgr.dataset_paths("ds_drop").keyframes_dir / "vd1/000000.jpg")))
        conn.execute("INSERT INTO Keyframes VALUES (?, ?, ?, ?)", ("vk1__0", "vk1", 1, str(mgr.dataset_paths("ds_keep").keyframes_dir / "vk1/000000.jpg")))
        conn.execute("INSERT INTO KeyframeMap VALUES (0, 'vd1__0', 'vd1')")
        conn.execute("INSERT INTO KeyframeMap VALUES (1, 'vk1__0', 'vk1')")
        conn.execute("INSERT INTO IndexManifest VALUES ('snap', 2)")
        conn.commit()

    aug = _norm((2, 16), seed=1)
    np.save(proc / "visual_keyframes/siglip2_yolo_augmented_864d.npy", aug)
    idx_kf = faiss.IndexFlatIP(16)
    idx_kf.add(aug)
    faiss.write_index(idx_kf, str(proc / "visual_keyframes/siglip2_keyframes_768d.faiss"))

    idx_e5 = faiss.IndexFlatIP(8)
    idx_e5.add(_norm((2, 8), seed=2))
    faiss.write_index(idx_e5, str(proc / "text_evidence/e5_ocr_asr_384d.faiss"))
    (proc / "text_evidence/e5_ocr_asr_manifest.json").write_text(
        json.dumps([{"faiss_id": 0, "video_id": "vd1"}, {"faiss_id": 1, "video_id": "vk1"}]),
        encoding="utf-8",
    )

    idx_vp = faiss.IndexFlatIP(8)
    idx_vp.add(_norm((2, 8), seed=3))
    faiss.write_index(idx_vp, str(proc / "visual_clips/videoprism_whole_clips_768d.faiss"))
    (proc / "visual_clips/videoprism_whole_clips_manifest.json").write_text(
        json.dumps({
            "video_ids": ["vd1", "vk1"],
            "video_count": 2,
            "rows": [{"row_id": 0, "video_id": "vd1"}, {"row_id": 1, "video_id": "vk1"}],
        }),
        encoding="utf-8",
    )
    np.save(proc / "visual_clips/videoprism_subclips_768d.npy", _norm((2, 8), seed=4))
    (proc / "visual_clips/videoprism_subclips_manifest.jsonl").write_text(
        json.dumps({"row_id": 0, "video_id": "vd1"}) + "\n" + json.dumps({"row_id": 1, "video_id": "vk1"}) + "\n",
        encoding="utf-8",
    )

    res = mgr.remove_dataset("ds_drop", confirm=True, prune_processed=True)
    assert res["removed"] is True
    assert res["processed_prune"]["remaining_videos"] == 1
    assert faiss.read_index(str(proc / "visual_keyframes/siglip2_keyframes_768d.faiss")).ntotal == 1
    assert np.allclose(np.load(proc / "visual_keyframes/siglip2_yolo_augmented_864d.npy")[0], aug[1])
    assert faiss.read_index(str(proc / "text_evidence/e5_ocr_asr_384d.faiss")).ntotal == 1
    assert faiss.read_index(str(proc / "visual_clips/videoprism_whole_clips_768d.faiss")).ntotal == 1
    assert np.load(proc / "visual_clips/videoprism_subclips_768d.npy").shape == (1, 8)
    assert mgr.verify("ds_keep")["ok"] is True
