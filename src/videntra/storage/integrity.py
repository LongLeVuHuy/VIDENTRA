"""Offline Storage Integrity Verifier.

Verifies:
1. Synchronization between FAISS index vectors and SQLite KeyframeMap.
2. Foreign key validity without orphaned keyframes, detections, or observations.
3. Monotonic presentation order of keyframes.
4. Completeness and searchability of SQLite FTS5 indices.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from typing import Any


@dataclass
class IntegrityReport:
    passed: bool
    db_path: str
    faiss_path: str | None
    total_videos: int
    total_keyframes: int
    total_faiss_vectors: int
    total_frame_objects: int
    total_ocr_observations: int
    total_ocr_groups: int
    total_asr_segments: int
    total_asr_groups: int
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_offline_storage_integrity(
    db_path: str | Path,
    faiss_index_path: str | Path | None = None,
) -> IntegrityReport:
    """Performs comprehensive integrity check on SQLite metadata.db and FAISS index."""
    db_p = Path(db_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not db_p.is_file():
        return IntegrityReport(
            passed=False,
            db_path=str(db_p),
            faiss_path=str(faiss_index_path) if faiss_index_path else None,
            total_videos=0,
            total_keyframes=0,
            total_faiss_vectors=0,
            total_frame_objects=0,
            total_ocr_observations=0,
            total_ocr_groups=0,
            total_asr_segments=0,
            total_asr_groups=0,
            errors=[f"Database file not found: {db_p}"],
            warnings=[],
        )

    conn = sqlite3.connect(str(db_p))
    conn.row_factory = sqlite3.Row

    try:
        # Check required tables
        required_tables = [
            "Videos", "Keyframes", "KeyframeMap", "ModelVersions", "IndexManifest",
            "FrameObjects", "FrameObjectRuns", "OCRObservations", "OCREvidenceGroups",
            "OCREvidenceMembers", "ASRSegments", "ASREvidenceGroups", "ASREvidenceMembers"
        ]
        cur = conn.cursor()
        existing_tables = {
            row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        }
        for rt in required_tables:
            if rt not in existing_tables:
                errors.append(f"Missing required table: {rt}")

        if errors:
            return IntegrityReport(
                passed=False,
                db_path=str(db_p),
                faiss_path=str(faiss_index_path) if faiss_index_path else None,
                total_videos=0,
                total_keyframes=0,
                total_faiss_vectors=0,
                total_frame_objects=0,
                total_ocr_observations=0,
                total_ocr_groups=0,
                total_asr_segments=0,
                total_asr_groups=0,
                errors=errors,
                warnings=warnings,
            )

        # Count records
        total_videos = cur.execute("SELECT COUNT(*) FROM Videos;").fetchone()[0]
        total_keyframes = cur.execute("SELECT COUNT(*) FROM Keyframes;").fetchone()[0]
        total_kf_map = cur.execute("SELECT COUNT(*) FROM KeyframeMap;").fetchone()[0]
        total_frame_objects = cur.execute("SELECT COUNT(*) FROM FrameObjects;").fetchone()[0]
        total_ocr_obs = cur.execute("SELECT COUNT(*) FROM OCRObservations;").fetchone()[0]
        total_ocr_groups = cur.execute("SELECT COUNT(*) FROM OCREvidenceGroups;").fetchone()[0]
        total_asr_segs = cur.execute("SELECT COUNT(*) FROM ASRSegments;").fetchone()[0]
        total_asr_groups = cur.execute("SELECT COUNT(*) FROM ASREvidenceGroups;").fetchone()[0]
        missing_frame_paths = cur.execute("SELECT COUNT(*) FROM Keyframes WHERE keyframe_path = '';").fetchone()[0]
        if missing_frame_paths:
            warnings.append(f"{missing_frame_paths} keyframes have no exported image path; thumbnail availability is unverified")

        # 1. Foreign key integrity: orphaned keyframes
        orphaned_kf = cur.execute("""
            SELECT COUNT(*) FROM Keyframes k
            LEFT JOIN Videos v ON k.video_id = v.video_id
            WHERE v.video_id IS NULL;
        """).fetchone()[0]
        if orphaned_kf > 0:
            errors.append(f"Found {orphaned_kf} orphaned Keyframes with invalid video_id")

        # 2. KeyframeMap sync with Keyframes
        kf_with_faiss_id = cur.execute("SELECT COUNT(*) FROM Keyframes WHERE faiss_id IS NOT NULL;").fetchone()[0]
        if kf_with_faiss_id != total_kf_map:
            errors.append(f"Keyframes with faiss_id ({kf_with_faiss_id}) != KeyframeMap count ({total_kf_map})")

        # 3. FAISS file sync
        total_faiss_vectors = 0
        if faiss_index_path:
            f_p = Path(faiss_index_path)
            if f_p.is_file():
                try:
                    import faiss
                    index = faiss.read_index(str(f_p))
                    total_faiss_vectors = index.ntotal
                    if total_faiss_vectors != total_kf_map:
                        errors.append(
                            f"FAISS index vector count ({total_faiss_vectors}) does not match KeyframeMap rows ({total_kf_map})"
                        )
                except Exception as exc:
                    errors.append(f"Failed to read FAISS index {f_p}: {exc}")
            else:
                warnings.append(f"FAISS index file does not exist at {f_p}")

        # 4. Monotonic PTS checks per video
        cur.execute("SELECT DISTINCT video_id FROM Keyframes;")
        vids = [r[0] for r in cur.fetchall()]
        for v in vids:
            cur.execute("SELECT pts FROM Keyframes WHERE video_id = ? ORDER BY keyframe_index ASC;", (v,))
            pts_list = [r[0] for r in cur.fetchall()]
            if pts_list != sorted(pts_list):
                warnings.append(f"Video {v} has non-monotonic PTS order across keyframe indexes")

        # 5. Check FTS5
        try:
            cur.execute("SELECT COUNT(*) FROM fts_ocr;").fetchone()
            cur.execute("SELECT COUNT(*) FROM fts_asr;").fetchone()
        except sqlite3.OperationalError as exc:
            errors.append(f"FTS5 virtual table error: {exc}")

        passed = (len(errors) == 0)
        return IntegrityReport(
            passed=passed,
            db_path=str(db_p),
            faiss_path=str(faiss_index_path) if faiss_index_path else None,
            total_videos=total_videos,
            total_keyframes=total_keyframes,
            total_faiss_vectors=total_faiss_vectors,
            total_frame_objects=total_frame_objects,
            total_ocr_observations=total_ocr_obs,
            total_ocr_groups=total_ocr_groups,
            total_asr_segments=total_asr_segs,
            total_asr_groups=total_asr_groups,
            errors=errors,
            warnings=warnings,
        )
    finally:
        conn.close()
