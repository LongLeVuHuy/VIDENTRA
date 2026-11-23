"""Modular per-dataset storage registry and lifecycle management for VIDENTRA."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Mapping, Sequence

from videntra.config.paths import (
    resolve_artifact_path,
    resolve_keyframe_faiss_path,
    resolve_metadata_db_path,
    resolve_processed_root,
)


DATASET_SUBDIRS: tuple[str, ...] = (
    "media",
    "keyframes",
    "annotations",
    "provenance",
    "snapshots",
)
DEFAULT_DENIED_ROOTS: tuple[str, ...] = (
    "work/evaluation-only",
    "validation",
    "test_queries",
)
_DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_dataset_id(dataset_id: str) -> str:
    """Validate and normalize a dataset identifier."""
    norm = str(dataset_id or "").strip().lower()
    if not norm or not _DATASET_ID_RE.fullmatch(norm):
        raise ValueError(
            f"Invalid dataset_id {dataset_id!r}: must match ^[a-z0-9][a-z0-9_-]*$"
        )
    return norm


@dataclass(frozen=True)
class DatasetPaths:
    """Standardized on-disk folder structure for a single dataset."""

    dataset_id: str
    root: Path

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def keyframes_dir(self) -> Path:
        return self.root / "keyframes"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def provenance_dir(self) -> Path:
        return self.root / "provenance"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def metadata_file(self) -> Path:
        return self.root / "dataset.json"

    @property
    def gitkeep_file(self) -> Path:
        return self.root / ".gitkeep"

    def subdir_map(self) -> dict[str, Path]:
        return {
            "media": self.media_dir,
            "keyframes": self.keyframes_dir,
            "annotations": self.annotations_dir,
            "provenance": self.provenance_dir,
            "snapshots": self.snapshots_dir,
        }


def resolve_keyframe_roots(data_root: Path | str = Path("data")) -> list[Path]:
    """Return all active dataset keyframe roots plus optional legacy fallback."""
    base = Path(data_root)
    roots: list[Path] = []
    seen: set[Path] = set()

    ds_base = base / "datasets"
    if ds_base.is_dir():
        for ds_dir in sorted(p for p in ds_base.iterdir() if p.is_dir()):
            kf_dir = ds_dir / "keyframes"
            if kf_dir.is_dir():
                resolved = kf_dir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    roots.append(kf_dir)

    legacy_kf = base / "raw" / "keyframes"
    if legacy_kf.is_dir():
        resolved = legacy_kf.resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(legacy_kf)

    return roots


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _dir_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for f in path.rglob("*"):
        if f.is_file():
            file_count += 1
            total_bytes += f.stat().st_size
    return file_count, total_bytes


def _rebuild_flat_ip_faiss(src_index_path: Path, keep_indices: Sequence[int]) -> int:
    import faiss  # type: ignore
    import numpy as np

    idx = faiss.read_index(str(src_index_path))
    dim = int(idx.d)
    if len(keep_indices) > 0 and idx.ntotal > 0:
        vecs = idx.reconstruct_n(0, idx.ntotal)
        kept = np.ascontiguousarray(vecs[np.asarray(keep_indices, dtype=np.int64)], dtype=np.float32)
    else:
        kept = np.zeros((0, dim), dtype=np.float32)
    new_idx = faiss.IndexFlatIP(dim)
    if len(kept) > 0:
        new_idx.add(kept)
    tmp_path = src_index_path.with_name(f"{src_index_path.name}.tmp")
    faiss.write_index(new_idx, str(tmp_path))
    os.replace(tmp_path, src_index_path)
    return int(new_idx.ntotal)


def _atomic_slice_npy(npy_path: Path, keep_indices: Sequence[int]) -> int:
    import numpy as np

    arr = np.load(str(npy_path))
    idx_arr = np.asarray(keep_indices, dtype=np.int64)
    kept = np.ascontiguousarray(arr[idx_arr], dtype=arr.dtype) if len(idx_arr) > 0 else np.zeros((0, *arr.shape[1:]), dtype=arr.dtype)
    tmp_path = npy_path.with_name(f"{npy_path.name}.tmp.npy")
    np.save(str(tmp_path), kept)
    os.replace(tmp_path, npy_path)
    return int(kept.shape[0])


def _prune_json_manifest_and_index(
    manifest_path: Path,
    index_path: Path | None,
    npy_path: Path | None,
    remove_vids: set[str],
    *,
    id_key: str = "faiss_id",
) -> int | None:
    if not manifest_path.is_file():
        return None
    if manifest_path.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep_pos: list[int] = []
        new_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if str(row.get("video_id", "")) not in remove_vids:
                keep_pos.append(int(row.get(id_key, idx)))
                rc = dict(row)
                rc[id_key] = len(new_rows)
                new_rows.append(rc)
        _atomic_write_text(
            manifest_path,
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + ("\n" if new_rows else ""),
        )
    else:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(doc, list):
            keep_pos = []
            new_rows = []
            for idx, row in enumerate(doc):
                if str(row.get("video_id", "")) not in remove_vids:
                    keep_pos.append(int(row.get(id_key, idx)))
                    rc = dict(row)
                    rc[id_key] = len(new_rows)
                    new_rows.append(rc)
            _atomic_write_text(manifest_path, json.dumps(new_rows, indent=2, ensure_ascii=False))
        else:
            old_vids = [str(v) for v in doc.get("video_ids", [])]
            keep_pos = [i for i, v in enumerate(old_vids) if v not in remove_vids]
            new_vids = [old_vids[i] for i in keep_pos]
            new_rows = []
            for row in doc.get("rows", []):
                if str(row.get("video_id", "")) not in remove_vids:
                    rc = dict(row)
                    rc[id_key] = len(new_rows)
                    new_rows.append(rc)
            doc["video_ids"] = new_vids
            doc["video_count"] = len(new_vids)
            doc["rows"] = new_rows
            _atomic_write_json(manifest_path, doc)

    if index_path is not None and index_path.is_file():
        _rebuild_flat_ip_faiss(index_path, keep_pos)
    if npy_path is not None and npy_path.is_file():
        _atomic_slice_npy(npy_path, keep_pos)
    return len(keep_pos)


def _prune_processed_for_dataset(processed_dir: Path, dataset_id: str) -> dict[str, Any]:
    """Remove all database rows and vector entries belonging to dataset_id."""
    proc_root = resolve_processed_root(processed_dir)
    db_path = resolve_metadata_db_path(proc_root)
    if not db_path.is_file():
        return {"skipped": f"missing {db_path}"}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    existing_tables = {
        r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "Videos" not in existing_tables:
        conn.close()
        return {"skipped": "Videos table not found"}

    remove_vids = {
        str(r[0]) for r in cur.execute("SELECT video_id FROM Videos WHERE dataset = ?", (dataset_id,)).fetchall()
    }
    if not remove_vids:
        remaining = int(cur.execute("SELECT COUNT(*) FROM Videos").fetchone()[0])
        conn.close()
        return {"removed_videos": 0, "remaining_videos": remaining}

    summary: dict[str, Any] = {"removed_videos": len(remove_vids)}

    keep_kf_pos: list[int] = []
    keep_kf_ids: list[str] = []
    if "KeyframeMap" in existing_tables:
        for fid, kfid, vid in cur.execute(
            "SELECT faiss_id, keyframe_id, video_id FROM KeyframeMap ORDER BY faiss_id ASC"
        ).fetchall():
            if str(vid) not in remove_vids:
                keep_kf_pos.append(int(fid))
                keep_kf_ids.append(str(kfid))
        aug_path = resolve_artifact_path(
            proc_root, "visual_keyframes/siglip2_yolo_augmented_864d.npy", "augmented_embeddings.npy"
        )
        if aug_path.is_file():
            summary["augmented_embeddings_rows"] = _atomic_slice_npy(aug_path, keep_kf_pos)
        faiss_kf_path = resolve_keyframe_faiss_path(proc_root)
        if faiss_kf_path.is_file():
            summary["faiss_index_ntotal"] = _rebuild_flat_ip_faiss(faiss_kf_path, keep_kf_pos)

    e5_rem = _prune_json_manifest_and_index(
        resolve_artifact_path(proc_root, "text_evidence/e5_ocr_asr_manifest.json", "e5_text_manifest.json"),
        resolve_artifact_path(proc_root, "text_evidence/e5_ocr_asr_384d.faiss", "faiss_text_e5_384.bin"),
        None,
        remove_vids,
        id_key="faiss_id",
    )
    if e5_rem is not None:
        summary["e5_text_manifest_rows"] = e5_rem

    vp_rem = _prune_json_manifest_and_index(
        resolve_artifact_path(proc_root, "visual_clips/videoprism_whole_clips_manifest.json", "videoprism_clip_manifest.json"),
        resolve_artifact_path(proc_root, "visual_clips/videoprism_whole_clips_768d.faiss", "faiss_videoprism_clip_768.bin"),
        None,
        remove_vids,
        id_key="row_id",
    )
    if vp_rem is not None:
        summary["videoprism_clip_manifest_videos"] = vp_rem

    sub_rem = _prune_json_manifest_and_index(
        resolve_artifact_path(proc_root, "visual_clips/videoprism_subclips_manifest.jsonl", "videoprism_subclips_map.jsonl"),
        None,
        resolve_artifact_path(proc_root, "visual_clips/videoprism_subclips_768d.npy", "videoprism_subclips_768.npy"),
        remove_vids,
        id_key="row_id",
    )
    if sub_rem is not None:
        summary["videoprism_subclips_rows"] = sub_rem

    cur.execute("CREATE TEMP TABLE _prune_vids (video_id TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO _prune_vids (video_id) VALUES (?)", [(v,) for v in sorted(remove_vids)])

    for tbl in (
        "Videos",
        "Keyframes",
        "KeyframeMap",
        "FrameObjects",
        "FrameObjectRuns",
        "OCRObservations",
        "OCREvidenceGroups",
        "ASRSegments",
        "ASREvidenceGroups",
        "fts_ocr",
        "fts_asr",
    ):
        if tbl in existing_tables:
            cur.execute(f"DELETE FROM {tbl} WHERE video_id IN (SELECT video_id FROM _prune_vids)")

    if "OCREvidenceMembers" in existing_tables and "OCREvidenceGroups" in existing_tables:
        cur.execute("DELETE FROM OCREvidenceMembers WHERE group_id NOT IN (SELECT group_id FROM OCREvidenceGroups)")
    if "ASRWords" in existing_tables and "ASRSegments" in existing_tables:
        cur.execute("DELETE FROM ASRWords WHERE segment_id NOT IN (SELECT segment_id FROM ASRSegments)")
    if "ASREvidenceMembers" in existing_tables and "ASREvidenceGroups" in existing_tables:
        cur.execute("DELETE FROM ASREvidenceMembers WHERE group_id NOT IN (SELECT group_id FROM ASREvidenceGroups)")

    if keep_kf_ids:
        remap = [(new_fid, kfid) for new_fid, kfid in enumerate(keep_kf_ids)]
        if "KeyframeMap" in existing_tables:
            cur.executemany("UPDATE KeyframeMap SET faiss_id = ? WHERE keyframe_id = ?", remap)
        if "Keyframes" in existing_tables:
            kf_cols = {r[1] for r in cur.execute("PRAGMA table_info(Keyframes)").fetchall()}
            if "faiss_id" in kf_cols:
                cur.executemany("UPDATE Keyframes SET faiss_id = ? WHERE keyframe_id = ?", remap)
    if "IndexManifest" in existing_tables:
        cur.execute("UPDATE IndexManifest SET total_vectors = ?", (len(keep_kf_ids),))
    if "fts_ocr" in existing_tables:
        cur.execute("INSERT INTO fts_ocr(fts_ocr) VALUES('optimize')")
    if "fts_asr" in existing_tables:
        cur.execute("INSERT INTO fts_asr(fts_asr) VALUES('optimize')")
    conn.commit()
    cur.execute("VACUUM")
    summary["remaining_videos"] = int(cur.execute("SELECT COUNT(*) FROM Videos").fetchone()[0])
    conn.close()
    return summary


class DatasetRegistryManager:
    """Manage self-contained per-dataset directories under data/datasets/<id>/."""

    def __init__(
        self,
        data_root: Path | str = Path("data"),
        asset_roots_path: Path | str = Path("configs/local/asset_roots.json"),
    ) -> None:
        self.data_root = Path(data_root)
        self.datasets_root = self.data_root / "datasets"
        self.registry_path = self.data_root / "datasets.json"
        self.asset_roots_path = Path(asset_roots_path)

    def dataset_paths(self, dataset_id: str) -> DatasetPaths:
        ds_id = validate_dataset_id(dataset_id)
        return DatasetPaths(dataset_id=ds_id, root=self.datasets_root / ds_id)

    def _load_registry_doc(self) -> dict[str, Any]:
        datasets: dict[str, dict[str, Any]] = {}
        if self.registry_path.is_file():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for ds_id, meta in (raw.get("datasets") or {}).items():
                try:
                    norm = validate_dataset_id(ds_id)
                except ValueError:
                    continue
                if isinstance(meta, dict):
                    datasets[norm] = dict(meta)

        if self.datasets_root.is_dir():
            for ds_dir in sorted(p for p in self.datasets_root.iterdir() if p.is_dir()):
                try:
                    ds_id = validate_dataset_id(ds_dir.name)
                except ValueError:
                    continue
                meta_file = ds_dir / "dataset.json"
                if ds_id not in datasets and meta_file.is_file():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if isinstance(meta, dict):
                        datasets[ds_id] = meta

        return {"schema_version": 1, "datasets": datasets}

    def _save_registry_doc(self, datasets: Mapping[str, Mapping[str, Any]]) -> None:
        ordered = {k: dict(datasets[k]) for k in sorted(datasets)}
        _atomic_write_json(
            self.registry_path,
            {"schema_version": 1, "updated_at_utc": _utc_now_iso(), "datasets": ordered},
        )

    def _load_asset_roots_doc(self) -> dict[str, Any]:
        if self.asset_roots_path.is_file():
            doc = json.loads(self.asset_roots_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return {
                    "roots": dict(doc.get("roots") or {}),
                    "denied_roots": list(doc.get("denied_roots") or DEFAULT_DENIED_ROOTS),
                }
        return {"roots": {}, "denied_roots": list(DEFAULT_DENIED_ROOTS)}

    def _sync_asset_roots(self, registered_datasets: Mapping[str, Any], removed_id: str | None = None) -> None:
        doc = self._load_asset_roots_doc()
        roots: dict[str, str] = doc["roots"]
        if removed_id is not None:
            roots.pop(removed_id, None)
        for ds_id in registered_datasets:
            roots[ds_id] = self.dataset_paths(ds_id).media_dir.as_posix()
        denied = list(dict.fromkeys([*DEFAULT_DENIED_ROOTS, *doc["denied_roots"]]))
        _atomic_write_json(self.asset_roots_path, {"roots": roots, "denied_roots": denied})

    def list_datasets(self) -> list[dict[str, Any]]:
        doc = self._load_registry_doc()
        asset_roots = self._load_asset_roots_doc()["roots"]
        rows: list[dict[str, Any]] = []
        for ds_id in sorted(doc["datasets"]):
            meta = dict(doc["datasets"][ds_id])
            paths = self.dataset_paths(ds_id)
            subdir_stats: dict[str, dict[str, Any]] = {}
            total_files = 0
            total_bytes = 0
            for name, sdir in paths.subdir_map().items():
                fc, tb = _dir_stats(sdir)
                total_files += fc
                total_bytes += tb
                subdir_stats[name] = {
                    "exists": sdir.is_dir(),
                    "file_count": fc,
                    "total_mb": round(tb / (1024.0 * 1024.0), 2),
                }
            for extra_name in ("catalog", "embeddings", "evidence"):
                edir = paths.root / extra_name
                if edir.is_dir():
                    fc, tb = _dir_stats(edir)
                    total_files += fc
                    total_bytes += tb
                    subdir_stats[extra_name] = {
                        "exists": True,
                        "file_count": fc,
                        "total_mb": round(tb / (1024.0 * 1024.0), 2),
                    }
            rows.append({
                "dataset_id": ds_id,
                "root": paths.root.as_posix(),
                "description": str(meta.get("description") or ""),
                "license": str(meta.get("license") or "unknown"),
                "source": str(meta.get("source") or ""),
                "created_at_utc": meta.get("created_at_utc"),
                "total_files": total_files,
                "total_mb": round(total_bytes / (1024.0 * 1024.0), 2),
                "asset_roots_synced": asset_roots.get(ds_id) == paths.media_dir.as_posix(),
                "subdirs": subdir_stats,
            })
        return rows

    def add_dataset(
        self,
        dataset_id: str,
        *,
        description: str = "",
        license_name: str = "unknown",
        source: str = "",
        video_pattern: str = "*.mp4",
        create_gitkeep: bool = True,
    ) -> dict[str, Any]:
        paths = self.dataset_paths(dataset_id)
        for sdir in paths.subdir_map().values():
            sdir.mkdir(parents=True, exist_ok=True)
        if create_gitkeep and not paths.gitkeep_file.exists():
            paths.gitkeep_file.write_text("", encoding="utf-8")

        existing_meta: dict[str, Any] = {}
        if paths.metadata_file.is_file():
            existing_meta = json.loads(paths.metadata_file.read_text(encoding="utf-8"))

        now = _utc_now_iso()
        meta = {
            "schema_version": 1,
            "dataset_id": paths.dataset_id,
            "description": description or existing_meta.get("description") or f"VIDENTRA dataset {paths.dataset_id}",
            "license": license_name if license_name != "unknown" else existing_meta.get("license", "unknown"),
            "source": source or existing_meta.get("source") or "",
            "video_pattern": video_pattern,
            "created_at_utc": existing_meta.get("created_at_utc") or now,
            "updated_at_utc": now,
            "paths": {k: v.as_posix() for k, v in paths.subdir_map().items()},
        }
        _atomic_write_json(paths.metadata_file, meta)

        doc = self._load_registry_doc()
        doc["datasets"][paths.dataset_id] = {
            "description": meta["description"],
            "license": meta["license"],
            "source": meta["source"],
            "created_at_utc": meta["created_at_utc"],
            "root": paths.root.as_posix(),
            "media_dir": paths.media_dir.as_posix(),
            "keyframes_dir": paths.keyframes_dir.as_posix(),
        }
        self._save_registry_doc(doc["datasets"])
        self._sync_asset_roots(doc["datasets"])
        return meta

    def remove_dataset(
        self,
        dataset_id: str,
        *,
        confirm: bool = False,
        prune_processed: bool = False,
        processed_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        paths = self.dataset_paths(dataset_id)
        fc, tb = _dir_stats(paths.root)
        proc_dir = resolve_processed_root(processed_dir) if processed_dir is not None else (self.data_root / "processed")
        db_path = resolve_metadata_db_path(proc_dir)

        db_videos = 0
        db_keyframes = 0
        if db_path.is_file():
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cur = conn.cursor()
            tbls = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "Videos" in tbls:
                db_videos = int(
                    cur.execute("SELECT COUNT(*) FROM Videos WHERE dataset = ?", (paths.dataset_id,)).fetchone()[0]
                )
            if "Keyframes" in tbls and "Videos" in tbls:
                db_keyframes = int(
                    cur.execute(
                        "SELECT COUNT(*) FROM Keyframes WHERE video_id IN (SELECT video_id FROM Videos WHERE dataset = ?)",
                        (paths.dataset_id,),
                    ).fetchone()[0]
                )
            conn.close()

        report: dict[str, Any] = {
            "dataset_id": paths.dataset_id,
            "dataset_root": paths.root.as_posix(),
            "confirmed": bool(confirm),
            "files_to_remove": fc,
            "bytes_to_remove": tb,
            "db_videos_matching": db_videos,
            "db_keyframes_matching": db_keyframes,
            "removed": False,
        }
        if not confirm:
            report["message"] = "Dry-run preview only. Pass --yes (confirm=True) to delete files and unregister."
            return report

        if paths.root.exists():
            shutil.rmtree(paths.root)
        doc = self._load_registry_doc()
        doc["datasets"].pop(paths.dataset_id, None)
        self._save_registry_doc(doc["datasets"])
        self._sync_asset_roots(doc["datasets"], removed_id=paths.dataset_id)

        if prune_processed:
            report["processed_prune"] = _prune_processed_for_dataset(proc_dir, paths.dataset_id)
        report["removed"] = True
        return report

    def verify(
        self,
        dataset_id: str | None = None,
        *,
        db_path: Path | str | None = None,
    ) -> dict[str, Any]:
        doc = self._load_registry_doc()
        target_ids = [validate_dataset_id(dataset_id)] if dataset_id is not None else sorted(doc["datasets"])
        if not target_ids:
            return {"ok": False, "error": "No datasets registered", "datasets": []}

        asset_roots = self._load_asset_roots_doc()["roots"]
        resolved_db = Path(db_path) if db_path is not None else resolve_metadata_db_path(self.data_root / "processed")
        base_anchor = self.data_root.parent

        results: list[dict[str, Any]] = []
        all_ok = True
        for ds_id in target_ids:
            paths = self.dataset_paths(ds_id)
            errors: list[str] = []
            if ds_id not in doc["datasets"]:
                errors.append(f"Not registered in {self.registry_path}")
            for sub_name, sdir in paths.subdir_map().items():
                if not sdir.is_dir():
                    errors.append(f"Missing subdirectory: {sdir} ({sub_name})")
            if not paths.metadata_file.is_file():
                errors.append(f"Missing dataset metadata: {paths.metadata_file}")
            expected_root = paths.media_dir.as_posix()
            if asset_roots.get(ds_id) != expected_root:
                errors.append(f"asset_roots.json mismatch for {ds_id}: expected {expected_root!r}, got {asset_roots.get(ds_id)!r}")

            ds_db = resolved_db
            if db_path is None:
                local_catalog_db = paths.root / "catalog" / "corpus_metadata.db"
                if local_catalog_db.is_file():
                    ds_db = local_catalog_db

            db_checked = 0
            db_missing_media = 0
            db_missing_kfs = 0
            if ds_db.is_file():
                conn = sqlite3.connect(f"file:{ds_db}?mode=ro", uri=True)
                cur = conn.cursor()
                tbls = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "Videos" in tbls:
                    vrows = cur.execute(
                        "SELECT video_id, file_path FROM Videos WHERE dataset = ?", (ds_id,)
                    ).fetchall()
                    db_checked = len(vrows)
                    for _, fpath_str in vrows:
                        p = Path(str(fpath_str))
                        candidate = p if p.is_absolute() else (base_anchor / p)
                        if not candidate.is_file():
                            db_missing_media += 1
                    if db_missing_media > 0:
                        errors.append(f"DB Videos missing on disk: {db_missing_media}/{db_checked}")
                if "Keyframes" in tbls and "Videos" in tbls:
                    krows = cur.execute(
                        "SELECT k.keyframe_path FROM Keyframes k JOIN Videos v ON k.video_id = v.video_id WHERE v.dataset = ?",
                        (ds_id,),
                    ).fetchall()
                    for (kpath_str,) in krows:
                        p = Path(str(kpath_str))
                        candidate = p if p.is_absolute() else (base_anchor / p)
                        if not candidate.is_file():
                            db_missing_kfs += 1
                    if db_missing_kfs > 0:
                        errors.append(f"DB Keyframes missing on disk: {db_missing_kfs}/{len(krows)}")
                conn.close()

            ds_ok = len(errors) == 0
            all_ok = all_ok and ds_ok
            results.append({
                "dataset_id": ds_id,
                "ok": ds_ok,
                "errors": errors,
                "db_videos_checked": db_checked,
                "db_missing_media": db_missing_media,
                "db_missing_keyframes": db_missing_kfs,
            })

        return {"ok": all_ok, "datasets": results}
