import json
from pathlib import Path
import sqlite3

import faiss
import numpy as np
import pytest

from videntra.api.frames import _lookup_unified_keyframe
from videntra.cli import main as cli_main
from videntra.registry import (
    DatasetRegistryManager,
    resolve_keyframe_roots,
    validate_dataset_id,
)


def _norm(shape: tuple[int, ...], seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    return arr / np.maximum(1e-12, np.linalg.norm(arr, axis=-1, keepdims=True))


def test_validate_dataset_id_rejects_traversal_and_special_chars() -> None:
    assert validate_dataset_id("MSRVTT_1ka-v2") == "msrvtt_1ka-v2"
    for bad in ("", "   ", "../evil", "a/b", "a\\b", "-leading", "_leading", "has space"):
        with pytest.raises(ValueError):
            validate_dataset_id(bad)


def test_dataset_registry_add_list_verify_and_keyframe_lookup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    roots_cfg = tmp_path / "configs/local/asset_roots.json"
    roots_cfg.parent.mkdir(parents=True, exist_ok=True)
    roots_cfg.write_text(
        json.dumps({"roots": {"m1_fixtures": "work/m1_fixtures"}, "denied_roots": ["validation"]}),
        encoding="utf-8",
    )

    mgr = DatasetRegistryManager(data_root=data_root, asset_roots_path=roots_cfg)
    meta = mgr.add_dataset(
        "demo_ds",
        description="Synthetic demo dataset",
        license_name="apache-2.0",
        source="local-unit-test",
    )
    assert meta["dataset_id"] == "demo_ds"

    ds_paths = mgr.dataset_paths("demo_ds")
    assert ds_paths.media_dir.is_dir()
    assert ds_paths.keyframes_dir.is_dir()
    assert ds_paths.annotations_dir.is_dir()
    assert ds_paths.provenance_dir.is_dir()
    assert ds_paths.snapshots_dir.is_dir()

    media_file = ds_paths.media_dir / "v001.mp4"
    media_file.write_bytes(b"mp4-bytes")
    kf_file = ds_paths.keyframes_dir / "v001" / "000000.jpg"
    kf_file.parent.mkdir(parents=True, exist_ok=True)
    kf_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\xff\xd9")

    # Verify DB + disk integrity
    db_path = data_root / "processed/metadata.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE Videos (video_id TEXT PRIMARY KEY, dataset TEXT, file_path TEXT)")
        conn.execute("CREATE TABLE Keyframes (keyframe_id TEXT PRIMARY KEY, video_id TEXT, source_frame_index INTEGER, keyframe_path TEXT, pts INTEGER, timestamp_ms INTEGER)")
        conn.execute("INSERT INTO Videos VALUES (?, ?, ?)", ("v001", "demo_ds", "data/datasets/demo_ds/media/v001.mp4"))
        conn.execute(
            "INSERT INTO Keyframes VALUES (?, ?, ?, ?, ?, ?)",
            ("v001__kf_0", "v001", 0, "data/datasets/demo_ds/keyframes/v001/000000.jpg", 0, 0),
        )
        conn.commit()

    listed = mgr.list_datasets()
    assert len(listed) == 1
    assert listed[0]["dataset_id"] == "demo_ds"
    assert listed[0]["asset_roots_synced"] is True
    assert listed[0]["subdirs"]["media"]["file_count"] == 1
    assert listed[0]["subdirs"]["keyframes"]["file_count"] == 1

    ver = mgr.verify("demo_ds", db_path=db_path)
    assert ver["ok"] is True
    assert ver["datasets"][0]["db_videos_checked"] == 1
    assert ver["datasets"][0]["db_missing_media"] == 0
    assert ver["datasets"][0]["db_missing_keyframes"] == 0

    # Verify asset_roots preserved m1_fixtures
    ar_doc = json.loads(roots_cfg.read_text(encoding="utf-8"))
    assert ar_doc["roots"]["m1_fixtures"] == "work/m1_fixtures"
    assert ar_doc["roots"]["demo_ds"].endswith("data/datasets/demo_ds/media")

    # Verify keyframe discovery without DB fallback works via resolve_keyframe_roots()
    assert any(p.resolve() == ds_paths.keyframes_dir.resolve() for p in resolve_keyframe_roots(data_root))
    found = _lookup_unified_keyframe(None, "v001", 0)
    assert found is not None and found[0].resolve() == kf_file.resolve()


def test_remove_dataset_dry_run_and_processed_prune(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    roots_cfg = tmp_path / "configs/local/asset_roots.json"
    proc_dir = data_root / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    mgr = DatasetRegistryManager(data_root=data_root, asset_roots_path=roots_cfg)
    mgr.add_dataset("ds_keep")
    mgr.add_dataset("ds_drop")

    (mgr.dataset_paths("ds_keep").media_dir / "vk1.mp4").write_bytes(b"k1")
    (mgr.dataset_paths("ds_drop").media_dir / "vd1.mp4").write_bytes(b"d1")
    (mgr.dataset_paths("ds_keep").keyframes_dir / "vk1").mkdir()
    (mgr.dataset_paths("ds_keep").keyframes_dir / "vk1/000000.jpg").write_bytes(b"jk1")
    (mgr.dataset_paths("ds_drop").keyframes_dir / "vd1").mkdir()
    (mgr.dataset_paths("ds_drop").keyframes_dir / "vd1/000000.jpg").write_bytes(b"jd1")

    # Build 2-dataset SQLite + FAISS + E5 + VideoPrism artifacts (drop first, keep second)
    db_path = proc_dir / "metadata.db"
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
    np.save(proc_dir / "augmented_embeddings.npy", aug)
    idx_kf = faiss.IndexFlatIP(16)
    idx_kf.add(aug)
    faiss.write_index(idx_kf, str(proc_dir / "faiss_index.bin"))

    e5_vecs = _norm((2, 8), seed=2)
    idx_e5 = faiss.IndexFlatIP(8)
    idx_e5.add(e5_vecs)
    faiss.write_index(idx_e5, str(proc_dir / "faiss_text_e5_384.bin"))
    (proc_dir / "e5_text_manifest.json").write_text(
        json.dumps([{"faiss_id": 0, "video_id": "vd1"}, {"faiss_id": 1, "video_id": "vk1"}]),
        encoding="utf-8",
    )

    vp_vecs = _norm((2, 8), seed=3)
    idx_vp = faiss.IndexFlatIP(8)
    idx_vp.add(vp_vecs)
    faiss.write_index(idx_vp, str(proc_dir / "faiss_videoprism_clip_768.bin"))
    (proc_dir / "videoprism_clip_manifest.json").write_text(
        json.dumps({
            "video_ids": ["vd1", "vk1"],
            "video_count": 2,
            "rows": [{"row_id": 0, "video_id": "vd1"}, {"row_id": 1, "video_id": "vk1"}],
        }),
        encoding="utf-8",
    )

    np.save(proc_dir / "videoprism_subclips_768.npy", _norm((2, 8), seed=4))
    (proc_dir / "videoprism_subclips_map.jsonl").write_text(
        json.dumps({"row_id": 0, "video_id": "vd1"}) + "\n" + json.dumps({"row_id": 1, "video_id": "vk1"}) + "\n",
        encoding="utf-8",
    )

    # Dry-run preview must NOT delete anything
    dry = mgr.remove_dataset("ds_drop", confirm=False, prune_processed=True)
    assert dry["removed"] is False
    assert dry["db_videos_matching"] == 1
    assert mgr.dataset_paths("ds_drop").root.exists()

    # Confirmed deletion removes ds_drop completely & preserves ds_keep vector/row alignment
    res = mgr.remove_dataset("ds_drop", confirm=True, prune_processed=True)
    assert res["removed"] is True
    assert not mgr.dataset_paths("ds_drop").root.exists()
    assert res["processed_prune"]["remaining_videos"] == 1
    assert faiss.read_index(str(proc_dir / "faiss_index.bin")).ntotal == 1
    assert np.allclose(np.load(proc_dir / "augmented_embeddings.npy")[0], aug[1])
    assert faiss.read_index(str(proc_dir / "faiss_text_e5_384.bin")).ntotal == 1
    assert faiss.read_index(str(proc_dir / "faiss_videoprism_clip_768.bin")).ntotal == 1
    assert np.load(proc_dir / "videoprism_subclips_768.npy").shape == (1, 8)

    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT video_id FROM Videos").fetchall() == [("vk1",)]
        assert conn.execute("SELECT faiss_id, keyframe_id FROM KeyframeMap").fetchall() == [(0, "vk1__0")]

    assert mgr.verify("ds_keep", db_path=db_path)["ok"] is True


def test_cli_dataset_subcommands(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    roots_cfg = tmp_path / "configs/local/asset_roots.json"
    assert cli_main(["dataset", "add", "--data-root", str(data_root), "--asset-roots", str(roots_cfg), "--name", "cli_ds"]) == 0
    assert cli_main(["dataset", "list", "--data-root", str(data_root), "--asset-roots", str(roots_cfg)]) == 0
    assert cli_main(["dataset", "verify", "--data-root", str(data_root), "--asset-roots", str(roots_cfg), "--name", "cli_ds"]) == 0
    assert cli_main(["dataset", "remove", "--data-root", str(data_root), "--asset-roots", str(roots_cfg), "--name", "cli_ds", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "cli_ds" in out


def test_list_and_verify_self_contained_catalog_db(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    roots_cfg = tmp_path / "configs/local/asset_roots.json"
    mgr = DatasetRegistryManager(data_root=data_root, asset_roots_path=roots_cfg)
    mgr.add_dataset("sc_ds")
    ds_paths = mgr.dataset_paths("sc_ds")

    (ds_paths.media_dir / "v100.mp4").write_bytes(b"mp4")
    kf = ds_paths.keyframes_dir / "v100" / "000000.jpg"
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_bytes(b"jpg")

    for extra_rel in ("catalog/corpus_metadata.db", "embeddings/clip.npy", "evidence/ocr.jsonl"):
        p = ds_paths.root / extra_rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not extra_rel.endswith(".db"):
            p.write_bytes(b"data")

    with sqlite3.connect(str(ds_paths.root / "catalog/corpus_metadata.db")) as conn:
        conn.execute("CREATE TABLE Videos (video_id TEXT PRIMARY KEY, dataset TEXT, file_path TEXT)")
        conn.execute("CREATE TABLE Keyframes (keyframe_id TEXT PRIMARY KEY, video_id TEXT, keyframe_path TEXT)")
        conn.execute("INSERT INTO Videos VALUES (?, ?, ?)", ("v100", "sc_ds", "data/datasets/sc_ds/media/v100.mp4"))
        conn.execute("INSERT INTO Keyframes VALUES (?, ?, ?)", ("v100__0", "v100", "data/datasets/sc_ds/keyframes/v100/000000.jpg"))
        conn.commit()

    listed = mgr.list_datasets()
    assert len(listed) == 1
    assert listed[0]["subdirs"]["catalog"]["file_count"] == 1
    assert listed[0]["subdirs"]["embeddings"]["file_count"] == 1
    assert listed[0]["subdirs"]["evidence"]["file_count"] == 1

    ver = mgr.verify("sc_ds")
    assert ver["ok"] is True
    assert ver["datasets"][0]["db_videos_checked"] == 1
    assert ver["datasets"][0]["db_missing_media"] == 0
    assert ver["datasets"][0]["db_missing_keyframes"] == 0
