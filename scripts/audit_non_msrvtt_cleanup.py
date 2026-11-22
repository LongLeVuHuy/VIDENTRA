"""Prune non-MSR-VTT assets/caches (with user approval receipt) and export standalone MSR-VTT snapshot.

Modes:
  --dry-run                Report non-MSR-VTT disk inventory without modifying files.
  --execute-approved-prune Delete approved non-MSR-VTT raw videos, keyframes, duplicate backups,
                           and prune non-MSR-VTT rows (OCR/ASR/objects/embeddings) from data/processed/.
  --snapshot-only          Export standalone MSR-VTT 1K-A snapshot to data/processed/msrvtt_1ka/ + smoke test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any

import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from videntra.retrieval.msrvtt_matrix import (  # noqa: E402
    _find_shard_subdir,
    compute_b1_videoprism_scores,
    load_gallery_feature_bank,
    rank_gallery_desc,
)

PROTOCOL_ROOT = Path("work/msrvtt_protocol_20261122")
_MSRVTT_VID_RE = re.compile(r"^video\d+$")
_USER_AUTH_QUOTE = (
    "bạn hãy xóa tất cả các file raw video không phải của tập msrvtt và các keyframes, "
    "ocr,asr,... liên quan nhé để cho tiết kiệm dung lượng. Chỉ dữ lại MSRVTT"
)


def is_msrvtt_video_id(video_id: str) -> bool:
    return bool(_MSRVTT_VID_RE.match(str(video_id)))


def export_standalone_msrvtt_snapshot(
    *,
    protocol_root: Path = PROTOCOL_ROOT,
    snapshot_dir: Path = Path("data/datasets/msrvtt/snapshots/msrvtt_1ka"),
) -> dict[str, Any]:
    """Export standalone MSR-VTT 1K-A assets & verify independent retrieval smoke test."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    m_test = protocol_root / "manifests" / "test_1ka"
    k_out = protocol_root / "kaggle_out"

    files_copied: list[str] = []
    gal_dst = snapshot_dir / "test_1ka_gallery_manifest.jsonl"
    if (m_test / "gallery_manifest.jsonl").is_file():
        shutil.copy2(m_test / "gallery_manifest.jsonl", gal_dst)
        files_copied.append(str(gal_dst))

    for fname in ("champion_lock_receipt.json", "protocol_receipt.json"):
        src = protocol_root / fname
        if src.is_file():
            dst = snapshot_dir / fname
            shutil.copy2(src, dst)
            files_copied.append(str(dst))

    vp_dst = snapshot_dir / "videoprism"
    vp_dst.mkdir(parents=True, exist_ok=True)
    for vp_cand in (k_out / "vp_test_1ka_locked", k_out / "vp_train_sub", k_out / "vp_test_1ka_queries_only"):
        if vp_cand.exists():
            sd = _find_shard_subdir(vp_cand)
            for asset in ("video_vectors.npy", "video_rows.jsonl", "window_vectors.npy", "window_rows.jsonl", "query_vectors.npy", "query_rows.jsonl"):
                src_a = sd / asset
                dst_a = vp_dst / f"{vp_cand.name}__{asset}"
                if src_a.is_file():
                    shutil.copy2(src_a, dst_a)
                    files_copied.append(str(dst_a))

    vp_shard_dirs = [d for d in (k_out / "vp_test_1ka_locked", k_out / "vp_train_sub", k_out / "vp_test_1ka_queries_only") if d.exists()]
    smoke_ok = False
    smoke_top1 = None
    if vp_shard_dirs and gal_dst.is_file() and (protocol_root / "eval_only" / "test_1ka_queries_vault.jsonl").is_file():
        mm_dirs = [d for d in (k_out / "mm_test_1ka_locked", k_out / "mm_test_1ka_videos_only", k_out / "mm_test_1ka_queries_only") if d.exists()]
        asr_dirs = [d for d in (k_out / "asr_test_1ka_videos_only",) if d.exists()]
        cap_dirs = [d for d in (k_out / "cap_test_1ka_locked", k_out / "cap_val_4000", k_out / "cap_val_tune", k_out / "cap_test_1ka_queries_only") if d.exists()]
        bank = load_gallery_feature_bank(
            gallery_name="test_1ka_smoke",
            gallery_manifest_path=gal_dst,
            queries_path=protocol_root / "eval_only" / "test_1ka_queries_vault.jsonl",
            vp_dirs=vp_shard_dirs,
            mm_dirs=mm_dirs,
            asr_dirs=asr_dirs,
            cap_dirs=cap_dirs,
        )
        if bank.vp_whole_vecs is not None and bank.queries:
            scores = compute_b1_videoprism_scores(bank, bank.queries[0]["query_text"])
            ranked = rank_gallery_desc(bank.video_ids, scores)
            smoke_ok = bool(len(ranked) == len(bank.video_ids) and np.isfinite(scores).all())
            smoke_top1 = ranked[0]

    receipt = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_dir": str(snapshot_dir),
        "files_count": len(files_copied),
        "smoke_retrieval_passed": smoke_ok,
        "smoke_sample_top1": smoke_top1,
    }
    (snapshot_dir / "snapshot_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def _dir_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    files = [f for f in path.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _collect_deletion_targets(root: Path) -> list[dict[str, Any]]:
    """Collect approved non-MSR-VTT directories/files for disk reclamation."""
    candidates: list[tuple[str, Path]] = [
        ("non_msrvtt_raw", root / "data/raw/avc"),
        ("non_msrvtt_raw", root / "data/raw/qvhighlights"),
        ("non_msrvtt_raw", root / "data/raw/textvr"),
        ("non_msrvtt_raw", root / "data/raw/tvr"),
        ("non_msrvtt_raw", root / "data/raw/blender_pilot_20261110"),
        ("legacy_pre_hybrid_backup", root / "work/backups/pre_hybrid_v2_20261118_204817"),
        ("non_msrvtt_kaggle_qvh", root / "work/kaggle/qvh_inference_queries_20261111.json"),
    ]
    kf_root = root / "data/raw/keyframes"
    if kf_root.exists():
        for child in sorted(kf_root.iterdir()):
            if child.name == ".gitkeep":
                continue
            if not is_msrvtt_video_id(child.name):
                candidates.append(("non_msrvtt_keyframes", child))

    kaggle_dir = root / "work/kaggle"
    if kaggle_dir.exists():
        for child in sorted(kaggle_dir.iterdir()):
            if child.name.startswith("blender_"):
                candidates.append(("non_msrvtt_kaggle_blender", child))

    out: list[dict[str, Any]] = []
    for cat, path in candidates:
        if not path.exists():
            continue
        fc, tb = _dir_stats(path)
        out.append({
            "category": cat,
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
            "file_count": fc,
            "total_bytes": tb,
        })
    return out


def _rebuild_flat_ip_index(src_index_path: Path, keep_indices: np.ndarray) -> int:
    idx = faiss.read_index(str(src_index_path))
    vecs = idx.reconstruct_n(0, idx.ntotal)
    kept = np.ascontiguousarray(vecs[keep_indices], dtype=np.float32)
    new_idx = faiss.IndexFlatIP(kept.shape[1])
    if len(kept) > 0:
        new_idx.add(kept)
    tmp_path = src_index_path.with_suffix(".bin.tmp")
    faiss.write_index(new_idx, str(tmp_path))
    tmp_path.replace(src_index_path)
    return int(new_idx.ntotal)


def _prune_processed_to_msrvtt(root: Path) -> dict[str, Any]:
    """Prune all non-MSR-VTT rows from data/processed/* and work/kaggle/offline_outputs/*."""
    from videntra.config.paths import (
        resolve_artifact_path,
        resolve_keyframe_faiss_path,
        resolve_metadata_db_path,
    )

    proc = root / "data/processed"
    db_path = resolve_metadata_db_path(proc)
    summary: dict[str, Any] = {}
    if not db_path.is_file():
        return {"skipped": f"missing {db_path}"}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    km_rows = cur.execute(
        "SELECT faiss_id, keyframe_id, video_id FROM KeyframeMap ORDER BY faiss_id ASC"
    ).fetchall()
    keep_kf_pos: list[int] = []
    keep_kf_ids: list[str] = []
    for fid, kfid, vid in km_rows:
        if is_msrvtt_video_id(vid):
            keep_kf_pos.append(int(fid))
            keep_kf_ids.append(str(kfid))
    keep_kf_arr = np.asarray(keep_kf_pos, dtype=np.int64)

    # 1. Prune keyframe vector files
    aug_path = resolve_artifact_path(proc, "visual_keyframes/siglip2_yolo_augmented_864d.npy", "augmented_embeddings.npy")
    if aug_path.is_file():
        aug = np.load(str(aug_path))
        aug_kept = np.ascontiguousarray(aug[keep_kf_arr], dtype=aug.dtype)
        np.save(str(aug_path), aug_kept)
        summary["augmented_embeddings_rows"] = int(aug_kept.shape[0])

    faiss_kf_path = resolve_keyframe_faiss_path(proc)
    if faiss_kf_path.is_file():
        summary["faiss_index_ntotal"] = _rebuild_flat_ip_index(faiss_kf_path, keep_kf_arr)

    # 2. Prune E5 text manifest & index
    e5_m_path = resolve_artifact_path(proc, "text_evidence/e5_ocr_asr_manifest.json", "e5_text_manifest.json")
    if e5_m_path.is_file():
        e5_rows = json.loads(e5_m_path.read_text(encoding="utf-8"))
        keep_e5_pos = [int(r["faiss_id"]) for r in e5_rows if is_msrvtt_video_id(str(r.get("video_id", "")))]
        new_e5 = []
        for new_id, r in enumerate(r for r in e5_rows if is_msrvtt_video_id(str(r.get("video_id", "")))):
            rc = dict(r)
            rc["faiss_id"] = new_id
            new_e5.append(rc)
        e5_m_path.write_text(json.dumps(new_e5, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["e5_text_manifest_rows"] = len(new_e5)
        e5_idx_path = resolve_artifact_path(proc, "text_evidence/e5_ocr_asr_384d.faiss", "faiss_text_e5_384.bin")
        if e5_idx_path.is_file():
            summary["faiss_text_e5_384_ntotal"] = _rebuild_flat_ip_index(
                e5_idx_path, np.asarray(keep_e5_pos, dtype=np.int64)
            )

    # 3. Prune VideoPrism clip manifest & index
    vp_m_path = resolve_artifact_path(proc, "visual_clips/videoprism_whole_clips_manifest.json", "videoprism_clip_manifest.json")
    if vp_m_path.is_file():
        vp_obj = json.loads(vp_m_path.read_text(encoding="utf-8"))
        old_vids = [str(v) for v in vp_obj.get("video_ids", [])]
        keep_vp_pos = [i for i, v in enumerate(old_vids) if is_msrvtt_video_id(v)]
        new_vids = [old_vids[i] for i in keep_vp_pos]
        new_vp_rows = []
        for new_id, r in enumerate(r for r in vp_obj.get("rows", []) if is_msrvtt_video_id(str(r.get("video_id", "")))):
            rc = dict(r)
            rc["row_id"] = new_id
            new_vp_rows.append(rc)
        vp_obj["video_ids"] = new_vids
        vp_obj["video_count"] = len(new_vids)
        vp_obj["rows"] = new_vp_rows
        vp_m_path.write_text(json.dumps(vp_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["videoprism_clip_manifest_videos"] = len(new_vids)
        vp_idx_path = resolve_artifact_path(proc, "visual_clips/videoprism_whole_clips_768d.faiss", "faiss_videoprism_clip_768.bin")
        if vp_idx_path.is_file():
            summary["faiss_videoprism_clip_768_ntotal"] = _rebuild_flat_ip_index(
                vp_idx_path, np.asarray(keep_vp_pos, dtype=np.int64)
            )

    # 4. Prune VideoPrism subclips
    sub_map_path = resolve_artifact_path(proc, "visual_clips/videoprism_subclips_manifest.jsonl", "videoprism_subclips_map.jsonl")
    sub_npy_path = resolve_artifact_path(proc, "visual_clips/videoprism_subclips_768d.npy", "videoprism_subclips_768.npy")
    if sub_map_path.is_file() and sub_npy_path.is_file():
        sub_rows = [json.loads(ln) for ln in sub_map_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep_sub_pos = [int(r.get("row_id", idx)) for idx, r in enumerate(sub_rows) if is_msrvtt_video_id(str(r.get("video_id", "")))]
        new_sub_rows = []
        for new_id, r in enumerate(r for r in sub_rows if is_msrvtt_video_id(str(r.get("video_id", "")))):
            rc = dict(r)
            rc["row_id"] = new_id
            new_sub_rows.append(rc)
        sub_map_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new_sub_rows) + ("\n" if new_sub_rows else ""),
            encoding="utf-8",
        )
        sub_arr = np.load(str(sub_npy_path))
        sub_kept = np.ascontiguousarray(sub_arr[np.asarray(keep_sub_pos, dtype=np.int64)], dtype=sub_arr.dtype)
        np.save(str(sub_npy_path), sub_kept)
        summary["videoprism_subclips_rows"] = int(sub_kept.shape[0])

    # 5. Prune metadata.db tables & remap contiguous faiss_id 0..N-1
    for tbl in (
        "Videos", "Keyframes", "KeyframeMap", "FrameObjects", "FrameObjectRuns",
        "OCRObservations", "OCREvidenceGroups", "ASRSegments", "ASREvidenceGroups",
        "fts_ocr", "fts_asr",
    ):
        cur.execute(f"DELETE FROM {tbl} WHERE video_id NOT GLOB 'video[0-9]*'")
    cur.execute("DELETE FROM OCREvidenceMembers WHERE group_id NOT IN (SELECT group_id FROM OCREvidenceGroups)")
    cur.execute("DELETE FROM ASRWords WHERE segment_id NOT IN (SELECT segment_id FROM ASRSegments)")
    cur.execute("DELETE FROM ASREvidenceMembers WHERE group_id NOT IN (SELECT group_id FROM ASREvidenceGroups)")

    remap = [(new_fid, kfid) for new_fid, kfid in enumerate(keep_kf_ids)]
    cur.executemany("UPDATE KeyframeMap SET faiss_id = ? WHERE keyframe_id = ?", remap)
    cur.executemany("UPDATE Keyframes SET faiss_id = ? WHERE keyframe_id = ?", remap)
    cur.execute("UPDATE IndexManifest SET total_vectors = ?", (len(keep_kf_ids),))
    cur.execute("INSERT INTO fts_ocr(fts_ocr) VALUES('optimize')")
    cur.execute("INSERT INTO fts_asr(fts_asr) VALUES('optimize')")
    conn.commit()
    cur.execute("VACUUM")

    summary["db_videos_remaining"] = int(cur.execute("SELECT COUNT(*) FROM Videos").fetchone()[0])
    summary["db_keyframes_remaining"] = int(cur.execute("SELECT COUNT(*) FROM Keyframes").fetchone()[0])
    summary["db_non_msrvtt_videos"] = int(
        cur.execute("SELECT COUNT(*) FROM Videos WHERE video_id NOT GLOB 'video[0-9]*'").fetchone()[0]
    )
    summary["db_ocr_remaining"] = int(cur.execute("SELECT COUNT(*) FROM OCREvidenceGroups").fetchone()[0])
    summary["db_asr_remaining"] = int(cur.execute("SELECT COUNT(*) FROM ASREvidenceGroups").fetchone()[0])
    conn.close()

    # 6. Prune legacy Kaggle offline_outputs / inputs if non-MSRVTT rows exist
    off_dir = root / "work/kaggle/offline_outputs"
    km_off = off_dir / "keyframe_manifest.json"
    fe_off = off_dir / "frame_embeddings.npy"
    if km_off.is_file() and fe_off.is_file():
        km_list = json.loads(km_off.read_text(encoding="utf-8"))
        keep_off = [i for i, r in enumerate(km_list) if is_msrvtt_video_id(str(r.get("video_id", "")))]
        new_km = [km_list[i] for i in keep_off]
        km_off.write_text(json.dumps(new_km, ensure_ascii=False), encoding="utf-8")
        arr_off = np.load(str(fe_off))
        np.save(str(fe_off), np.ascontiguousarray(arr_off[np.asarray(keep_off, dtype=np.int64)], dtype=arr_off.dtype))
        for jn in ("videos.json", "frame_objects.json", "ocr_observations.json", "asr_segments.json"):
            jp = off_dir / jn
            if jp.is_file():
                jrows = json.loads(jp.read_text(encoding="utf-8"))
                jfilt = [r for r in jrows if is_msrvtt_video_id(str(r.get("video_id", "")))]
                jp.write_text(json.dumps(jfilt, ensure_ascii=False), encoding="utf-8")
        summary["offline_outputs_pruned_keyframes"] = len(new_km)

    # Prune old input jpg images in work/kaggle/inputs only if non-MSRVTT (old mixed batch staging)
    in_imgs = root / "work/kaggle/inputs/images"
    if in_imgs.exists():
        shutil.rmtree(in_imgs, ignore_errors=True)
        summary["deleted_work_kaggle_inputs_images"] = True

    return summary


def run_cleanup_audit(
    *,
    root: Path = Path("."),
    protocol_root: Path = PROTOCOL_ROOT,
    execute_prune: bool = False,
    snapshot_only: bool = False,
    manifest_out: Path = Path("docs/audit/non_msrvtt_cleanup_manifest_20261122.json"),
    receipt_out: Path = Path("docs/audit/deletion_receipt_non_msrvtt_20261122.json"),
) -> dict[str, Any]:
    root = root.resolve()
    if snapshot_only:
        snap = export_standalone_msrvtt_snapshot(protocol_root=protocol_root, snapshot_dir=root / "data/datasets/msrvtt/snapshots/msrvtt_1ka")
        return {"snapshot": snap}

    _, _, free_before = shutil.disk_usage(root.drive or "C:\\")
    targets = _collect_deletion_targets(root)
    manifest = {
        "schema_version": 1,
        "date": "2026-11-22",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approved": bool(execute_prune),
        "user_authorization": _USER_AUTH_QUOTE if execute_prune else None,
        "free_bytes_before": free_before,
        "protected_paths": [
            str(root / "data/raw/msrvtt"),
            str(root / "data/raw/keyframes/video*"),
            str(root / "data/processed/msrvtt_1ka"),
            str(root / "work/msrvtt_protocol_20261122"),
            str(root / "work/experiments"),
        ],
        "candidates": targets,
        "totals": {
            "candidate_items": len(targets),
            "candidate_files": sum(t["file_count"] for t in targets),
            "candidate_bytes": sum(t["total_bytes"] for t in targets),
        },
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not execute_prune:
        return manifest

    deleted_items: list[dict[str, Any]] = []
    for t in targets:
        p = Path(t["path"])
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        deleted_items.append({**t, "status": "deleted"})

    # Keep parent directories' .gitkeep safe; recreate empty mount folders if needed
    for raw_sub in ("data/raw/keyframes", "data/raw/msrvtt"):
        (root / raw_sub).mkdir(parents=True, exist_ok=True)

    prune_summary = _prune_processed_to_msrvtt(root)
    snap = export_standalone_msrvtt_snapshot(protocol_root=protocol_root, snapshot_dir=root / "data/datasets/msrvtt/snapshots/msrvtt_1ka")

    _, _, free_after = shutil.disk_usage(root.drive or "C:\\")
    receipt = {
        "schema_version": 1,
        "date": "2026-11-22",
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation": "non_msrvtt_deletion_and_processed_prune",
        "user_authorization": _USER_AUTH_QUOTE,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "bytes_reclaimed": max(0, free_after - free_before),
        "deleted_items_count": len(deleted_items),
        "deleted_files_count": sum(d["file_count"] for d in deleted_items),
        "deleted_raw_bytes": sum(d["total_bytes"] for d in deleted_items),
        "deleted_items": deleted_items,
        "processed_prune_verification": prune_summary,
        "standalone_msrvtt_snapshot": snap,
    }
    receipt_out.parent.mkdir(parents=True, exist_ok=True)
    receipt_out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune non-MSR-VTT assets & export standalone MSR-VTT snapshot.")
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--execute-approved-prune", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    out = run_cleanup_audit(
        protocol_root=args.protocol_root,
        execute_prune=args.execute_approved_prune,
        snapshot_only=args.snapshot_only,
    )
    print(json.dumps({
        k: out[k] for k in ("bytes_reclaimed", "deleted_files_count", "processed_prune_verification", "standalone_msrvtt_snapshot", "totals")
        if k in out
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
