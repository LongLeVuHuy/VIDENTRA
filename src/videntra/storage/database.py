"""VIDENTRA Offline Metadata Storage.

SQLite-backed persistent metadata store managing videos, keyframes, FAISS mappings,
object detections, multi-model OCR observations & evidence groups, and multi-model ASR segments.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Sequence

from videntra.evidence.normalize import normalize_text, fold_diacritics


def current_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PayloadConflictError(ValueError):
    """Raised when an existing record ID is re-ingested with a different payload."""


def compute_recipe_hash(recipe_spec: dict[str, Any]) -> str:
    """Compute a deterministic 12-char hex digest from model/config/timeline provenance."""
    canonical = json.dumps(recipe_spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def make_deterministic_record_id(
    record_type: str,
    target_id: str,
    recipe_hash: str,
    ordinal: int,
) -> str:
    """Build collision-free deterministic ID: <target_id>__<record_type>_<recipe_hash>_<ordinal:05d>."""
    return f"{target_id}__{record_type}_{recipe_hash}_{ordinal:05d}"


class OfflineMetadataStore:
    """Encapsulates SQLite metadata.db operations with transactional integrity."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _open_raw_connection(self, *, isolation_level: str | None = "") -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=isolation_level)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def _get_connection(self, *, isolation_level: str | None = "") -> Iterator[sqlite3.Connection]:
        conn = self._open_raw_connection(isolation_level=isolation_level)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run multi-step writes + FTS rebuild in a single atomic transaction with FK enforcement."""
        conn = self._open_raw_connection(isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            # 1. Videos
            conn.execute("""
            CREATE TABLE IF NOT EXISTS Videos (
                video_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                file_path TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                fps REAL NOT NULL,
                time_base_num INTEGER NOT NULL,
                time_base_den INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                has_audio INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'ingested',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 2. Keyframes
            conn.execute("""
            CREATE TABLE IF NOT EXISTS Keyframes (
                keyframe_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                keyframe_index INTEGER NOT NULL,
                source_frame_index INTEGER NOT NULL,
                pts INTEGER NOT NULL,
                time_base TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                keyframe_path TEXT NOT NULL,
                thumbnail_path TEXT,
                faiss_id INTEGER UNIQUE,
                phash TEXT,
                sampling_reason TEXT,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_video ON Keyframes(video_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_faiss ON Keyframes(faiss_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_time ON Keyframes(video_id, timestamp_ms);")

            # 3. KeyframeMap (AIC compliant mapping)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS KeyframeMap (
                faiss_id INTEGER PRIMARY KEY,
                keyframe_id TEXT NOT NULL UNIQUE REFERENCES Keyframes(keyframe_id),
                video_id TEXT NOT NULL REFERENCES Videos(video_id),
                pts_time REAL NOT NULL,
                source_frame_index INTEGER NOT NULL,
                keyframe_path TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kfmap_video ON KeyframeMap(video_id);")

            # 4. ModelVersions
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ModelVersions (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                modality TEXT NOT NULL,
                weights_hash TEXT,
                config_json TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 5. IndexManifest
            conn.execute("""
            CREATE TABLE IF NOT EXISTS IndexManifest (
                manifest_id TEXT PRIMARY KEY,
                index_type TEXT NOT NULL,
                vector_dim INTEGER NOT NULL DEFAULT 768,
                metric TEXT NOT NULL DEFAULT 'inner_product',
                total_vectors INTEGER NOT NULL DEFAULT 0,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                weights_hash TEXT,
                index_file_path TEXT NOT NULL,
                checksum TEXT,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """)

            # 6. ProcessingJobs
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ProcessingJobs (
                job_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                progress_pct REAL NOT NULL DEFAULT 0.0,
                error_message TEXT,
                worker TEXT NOT NULL DEFAULT 'local_cpu',
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            """)

            # 7. FrameObjects (YOLO / DETR Object Detection)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS FrameObjects (
                detection_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                keyframe_id TEXT NOT NULL REFERENCES Keyframes(keyframe_id) ON DELETE CASCADE,
                timestamp_ms INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                confidence REAL NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_keyframe ON FrameObjects(keyframe_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_class ON FrameObjects(class_name);")

            # 8. FrameObjectRuns
            conn.execute("""
            CREATE TABLE IF NOT EXISTS FrameObjectRuns (
                keyframe_id TEXT PRIMARY KEY REFERENCES Keyframes(keyframe_id) ON DELETE CASCADE,
                video_id TEXT NOT NULL,
                object_count INTEGER NOT NULL DEFAULT 0,
                model_id TEXT NOT NULL,
                processed_at TEXT NOT NULL
            );
            """)

            # 9. OCRObservations (Original observations from each model)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS OCRObservations (
                observation_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                keyframe_id TEXT NOT NULL REFERENCES Keyframes(keyframe_id) ON DELETE CASCADE,
                timestamp_ms INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                confidence REAL,
                language TEXT DEFAULT 'vi',
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_obs_kf ON OCRObservations(keyframe_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_obs_model ON OCRObservations(model_id);")

            # 10. OCREvidenceGroups
            conn.execute("""
            CREATE TABLE IF NOT EXISTS OCREvidenceGroups (
                group_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                keyframe_id TEXT NOT NULL REFERENCES Keyframes(keyframe_id) ON DELETE CASCADE,
                timestamp_ms INTEGER NOT NULL,
                consensus_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                consensus_bbox TEXT NOT NULL,
                status TEXT NOT NULL, -- agreed, partial, disputed, single_model
                track_id TEXT,
                estimated_start_ms INTEGER,
                estimated_end_ms INTEGER,
                models_involved TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_group_kf ON OCREvidenceGroups(keyframe_id);")

            # 11. OCREvidenceMembers
            conn.execute("""
            CREATE TABLE IF NOT EXISTS OCREvidenceMembers (
                group_id TEXT NOT NULL REFERENCES OCREvidenceGroups(group_id) ON DELETE CASCADE,
                observation_id TEXT NOT NULL REFERENCES OCRObservations(observation_id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                PRIMARY KEY (group_id, observation_id)
            );
            """)

            # 12. ASRSegments (Original transcripts from each model)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ASRSegments (
                segment_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                language TEXT DEFAULT 'vi',
                confidence REAL,
                speaker_id TEXT,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_seg_video ON ASRSegments(video_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_seg_time ON ASRSegments(video_id, start_ms, end_ms);")

            # 13. ASRWords
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ASRWords (
                word_id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL REFERENCES ASRSegments(segment_id) ON DELETE CASCADE,
                word TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                confidence REAL,
                alignment_source TEXT
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_words_seg ON ASRWords(segment_id);")

            # 14. ASREvidenceGroups
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ASREvidenceGroups (
                group_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                consensus_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                status TEXT NOT NULL, -- agreed, partial, disputed, single_model
                models_involved TEXT NOT NULL,
                temporal_overlap_ratio REAL NOT NULL DEFAULT 1.0,
                text_similarity REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_group_time ON ASREvidenceGroups(video_id, start_ms, end_ms);")

            # 15. ASREvidenceMembers
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ASREvidenceMembers (
                group_id TEXT NOT NULL REFERENCES ASREvidenceGroups(group_id) ON DELETE CASCADE,
                segment_id TEXT NOT NULL REFERENCES ASRSegments(segment_id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                PRIMARY KEY (group_id, segment_id)
            );
            """)

            # 16. FTS5 Virtual Tables for OCR and ASR Search
            try:
                conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_ocr USING fts5(
                    normalized_text,
                    unaccented_text,
                    video_id UNINDEXED,
                    group_id UNINDEXED,
                    keyframe_id UNINDEXED,
                    timestamp_ms UNINDEXED,
                    tokenize='unicode61'
                );
                """)
                conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_asr USING fts5(
                    normalized_text,
                    unaccented_text,
                    video_id UNINDEXED,
                    group_id UNINDEXED,
                    segment_id UNINDEXED,
                    start_ms UNINDEXED,
                    end_ms UNINDEXED,
                    model_id UNINDEXED,
                    tokenize='unicode61'
                );
                """)
            except sqlite3.OperationalError:
                # Fallback if fts5 without unicode61
                conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_ocr USING fts5(
                    normalized_text,
                    unaccented_text,
                    video_id UNINDEXED,
                    group_id UNINDEXED,
                    keyframe_id UNINDEXED,
                    timestamp_ms UNINDEXED
                );
                """)
                conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_asr USING fts5(
                    normalized_text,
                    unaccented_text,
                    video_id UNINDEXED,
                    group_id UNINDEXED,
                    segment_id UNINDEXED,
                    start_ms UNINDEXED,
                    end_ms UNINDEXED,
                    model_id UNINDEXED
                );
                """)

    # -------------------------------------------------------------------------
    # Ingestion & Video Operations
    # -------------------------------------------------------------------------
    def register_video(self, record: dict[str, Any]) -> None:
        """Register or update a video record."""
        now = current_iso()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO Videos (
                video_id, dataset, file_path, duration_ms, fps, time_base_num, time_base_den,
                width, height, total_frames, has_audio, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                dataset=excluded.dataset,
                file_path=excluded.file_path,
                duration_ms=excluded.duration_ms,
                fps=excluded.fps,
                time_base_num=excluded.time_base_num,
                time_base_den=excluded.time_base_den,
                width=excluded.width,
                height=excluded.height,
                total_frames=excluded.total_frames,
                has_audio=excluded.has_audio,
                status=excluded.status,
                updated_at=excluded.updated_at;
            """, (
                record["video_id"],
                record.get("dataset", "unknown"),
                str(record["file_path"]),
                int(record.get("duration_ms", 0)),
                float(record.get("fps", 0.0)),
                int(record.get("time_base_num", 1)),
                int(record.get("time_base_den", 1000)),
                int(record.get("width", 0)),
                int(record.get("height", 0)),
                int(record.get("total_frames", 0)),
                int(record.get("has_audio", 0)),
                record.get("status", "ingested"),
                record.get("created_at", now),
                now,
            ))

    def insert_keyframes(self, keyframes: Sequence[dict[str, Any]]) -> None:
        """Batch insert keyframe records."""
        now = current_iso()
        rows = [
            (
                k["keyframe_id"],
                k["video_id"],
                int(k["keyframe_index"]),
                int(k["source_frame_index"]),
                int(k["pts"]),
                str(k["time_base"]),
                int(k["timestamp_ms"]),
                str(k["keyframe_path"]),
                str(k.get("thumbnail_path", "")),
                k.get("faiss_id"),
                k.get("phash"),
                k.get("sampling_reason", "adaptive"),
                k.get("created_at", now),
            )
            for k in keyframes
        ]
        with self._get_connection() as conn:
            conn.executemany("""
            INSERT OR REPLACE INTO Keyframes (
                keyframe_id, video_id, keyframe_index, source_frame_index, pts, time_base,
                timestamp_ms, keyframe_path, thumbnail_path, faiss_id, phash, sampling_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows)

    # -------------------------------------------------------------------------
    # FAISS Mapping & Index Manifest Operations
    # -------------------------------------------------------------------------
    def update_faiss_mappings(self, mappings: Sequence[dict[str, Any]]) -> None:
        """Map faiss_id to keyframe_id and update KeyframeMap."""
        with self._get_connection() as conn:
            # 1. Update Keyframes.faiss_id
            conn.executemany("""
            UPDATE Keyframes SET faiss_id = ? WHERE keyframe_id = ?;
            """, [(m["faiss_id"], m["keyframe_id"]) for m in mappings])

            # 2. Insert into KeyframeMap
            rows = [
                (
                    int(m["faiss_id"]),
                    m["keyframe_id"],
                    m["video_id"],
                    float(m["pts_time"]),
                    int(m["source_frame_index"]),
                    str(m["keyframe_path"]),
                )
                for m in mappings
            ]
            conn.executemany("""
            INSERT OR REPLACE INTO KeyframeMap (
                faiss_id, keyframe_id, video_id, pts_time, source_frame_index, keyframe_path
            ) VALUES (?, ?, ?, ?, ?, ?);
            """, rows)

    def record_index_manifest(self, manifest: dict[str, Any]) -> None:
        now = current_iso()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO IndexManifest (
                manifest_id, index_type, vector_dim, metric, total_vectors,
                model_id, model_version, weights_hash, index_file_path, checksum, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                manifest["manifest_id"],
                manifest.get("index_type", "IndexFlatIP"),
                manifest.get("vector_dim", 768),
                manifest.get("metric", "inner_product"),
                manifest.get("total_vectors", 0),
                manifest.get("model_id", "google/siglip2-base-patch16-224"),
                manifest.get("model_version", "v1.0"),
                manifest.get("weights_hash", ""),
                str(manifest["index_file_path"]),
                manifest.get("checksum", ""),
                int(manifest.get("is_active", 1)),
                manifest.get("created_at", now),
            ))

    # -------------------------------------------------------------------------
    # Object Detection Operations
    # -------------------------------------------------------------------------
    def record_frame_objects(self, keyframe_id: str, video_id: str, objects: Sequence[dict[str, Any]], model_id: str = "yolov8x", model_version: str = "v1") -> None:
        now = current_iso()
        rows = [
            (
                obj["detection_id"],
                video_id,
                keyframe_id,
                int(obj["timestamp_ms"]),
                int(obj["class_id"]),
                str(obj["class_name"]),
                float(obj["bbox_x1"]),
                float(obj["bbox_y1"]),
                float(obj["bbox_x2"]),
                float(obj["bbox_y2"]),
                float(obj["confidence"]),
                int(obj.get("image_width", 1280)),
                int(obj.get("image_height", 720)),
                model_id,
                model_version,
                now,
            )
            for obj in objects
        ]
        with self._get_connection() as conn:
            if rows:
                conn.executemany("""
                INSERT OR REPLACE INTO FrameObjects (
                    detection_id, video_id, keyframe_id, timestamp_ms, class_id, class_name,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, image_width, image_height,
                    model_id, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, rows)

            # Record run status even if 0 objects
            conn.execute("""
            INSERT OR REPLACE INTO FrameObjectRuns (
                keyframe_id, video_id, object_count, model_id, processed_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (keyframe_id, video_id, len(objects), model_id, now))

    # -------------------------------------------------------------------------
    # OCR Multi-Model Operations
    # -------------------------------------------------------------------------
    @staticmethod
    def _values_match(existing: Sequence[Any], incoming: Sequence[Any]) -> bool:
        if len(existing) != len(incoming):
            return False
        for a, b in zip(existing, incoming):
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6):
                    return False
            elif a != b:
                return False
        return True

    def insert_ocr_observations(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Insert OCR observations, treating identical retries as no-ops and raising on payload conflicts."""
        now = current_iso()

        def _apply(c: sqlite3.Connection) -> int:
            new_rows: list[tuple[Any, ...]] = []
            seen_batch: dict[str, tuple[Any, ...]] = {}
            for obs in observations:
                obs_id = obs["observation_id"]
                payload = (
                    obs["video_id"],
                    obs["keyframe_id"],
                    int(obs["timestamp_ms"]),
                    obs["model_id"],
                    obs["raw_text"],
                    normalize_text(obs["raw_text"]),
                    float(obs["bbox_x1"]),
                    float(obs["bbox_y1"]),
                    float(obs["bbox_x2"]),
                    float(obs["bbox_y2"]),
                    float(obs.get("confidence", 1.0)),
                    obs.get("language", "vi"),
                )
                if obs_id in seen_batch:
                    if not self._values_match(seen_batch[obs_id], payload):
                        raise PayloadConflictError(f"Conflicting payload for observation_id={obs_id}")
                    continue
                seen_batch[obs_id] = payload

                existing = c.execute(
                    """
                    SELECT video_id, keyframe_id, timestamp_ms, model_id, raw_text,
                           normalized_text, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language
                    FROM OCRObservations WHERE observation_id = ?;
                    """,
                    (obs_id,),
                ).fetchone()
                if existing is not None:
                    if not self._values_match(tuple(existing), payload):
                        raise PayloadConflictError(f"Conflicting payload for observation_id={obs_id}")
                    continue
                new_rows.append((obs_id, *payload, obs.get("created_at", now)))

            if new_rows:
                c.executemany(
                    """
                    INSERT INTO OCRObservations (
                        observation_id, video_id, keyframe_id, timestamp_ms, model_id, raw_text,
                        normalized_text, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    new_rows,
                )
            return len(new_rows)

        if conn is not None:
            return _apply(conn)
        with self._get_connection() as own_conn:
            return _apply(own_conn)

    def insert_ocr_evidence_groups(
        self,
        groups: Sequence[dict[str, Any]],
        member_mappings: Sequence[tuple[str, str, str]],
        *,
        conn: sqlite3.Connection | None = None,
        update_fts: bool = True,
    ) -> None:
        now = current_iso()
        group_rows = [
            (
                g["group_id"],
                g["video_id"],
                g["keyframe_id"],
                int(g["timestamp_ms"]),
                g["consensus_text"],
                normalize_text(g["consensus_text"]),
                str(g["consensus_bbox"]),
                g["status"],
                g.get("track_id"),
                g.get("estimated_start_ms"),
                g.get("estimated_end_ms"),
                g["models_involved"],
                g.get("created_at", now),
            )
            for g in groups
        ]

        def _apply(c: sqlite3.Connection) -> None:
            if group_rows:
                c.executemany(
                    """
                    INSERT INTO OCREvidenceGroups (
                        group_id, video_id, keyframe_id, timestamp_ms, consensus_text,
                        normalized_text, consensus_bbox, status, track_id,
                        estimated_start_ms, estimated_end_ms, models_involved, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(group_id) DO UPDATE SET
                        video_id=excluded.video_id,
                        keyframe_id=excluded.keyframe_id,
                        timestamp_ms=excluded.timestamp_ms,
                        consensus_text=excluded.consensus_text,
                        normalized_text=excluded.normalized_text,
                        consensus_bbox=excluded.consensus_bbox,
                        status=excluded.status,
                        track_id=excluded.track_id,
                        estimated_start_ms=excluded.estimated_start_ms,
                        estimated_end_ms=excluded.estimated_end_ms,
                        models_involved=excluded.models_involved;
                    """,
                    group_rows,
                )

            if member_mappings:
                c.executemany(
                    """
                    INSERT INTO OCREvidenceMembers (group_id, observation_id, model_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(group_id, observation_id) DO UPDATE SET
                        model_id=excluded.model_id;
                    """,
                    member_mappings,
                )

            if update_fts:
                fts_rows = [
                    (
                        normalize_text(g["consensus_text"]),
                        fold_diacritics(normalize_text(g["consensus_text"])),
                        g["video_id"],
                        g["group_id"],
                        g["keyframe_id"],
                        int(g["timestamp_ms"]),
                    )
                    for g in groups
                    if g["consensus_text"].strip()
                ]
                if fts_rows:
                    c.executemany(
                        """
                        INSERT INTO fts_ocr (normalized_text, unaccented_text, video_id, group_id, keyframe_id, timestamp_ms)
                        VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        fts_rows,
                    )

        if conn is not None:
            _apply(conn)
            return
        with self._get_connection() as own_conn:
            _apply(own_conn)

    # -------------------------------------------------------------------------
    # ASR Multi-Model Operations
    # -------------------------------------------------------------------------
    def insert_asr_segments(
        self,
        segments: Sequence[dict[str, Any]],
        words: Sequence[dict[str, Any]] | None = None,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Insert ASR segments & words, treating identical retries as no-ops and raising on payload conflicts."""
        now = current_iso()

        def _apply(c: sqlite3.Connection) -> int:
            new_seg_rows: list[tuple[Any, ...]] = []
            seen_segs: dict[str, tuple[Any, ...]] = {}
            for s in segments:
                seg_id = s["segment_id"]
                payload = (
                    s["video_id"],
                    s["model_id"],
                    int(s["start_ms"]),
                    int(s["end_ms"]),
                    s["raw_text"],
                    normalize_text(s["raw_text"]),
                    s.get("language", "vi"),
                    float(s.get("confidence", 1.0)),
                    s.get("speaker_id"),
                )
                if seg_id in seen_segs:
                    if not self._values_match(seen_segs[seg_id], payload):
                        raise PayloadConflictError(f"Conflicting payload for segment_id={seg_id}")
                    continue
                seen_segs[seg_id] = payload

                existing = c.execute(
                    """
                    SELECT video_id, model_id, start_ms, end_ms, raw_text,
                           normalized_text, language, confidence, speaker_id
                    FROM ASRSegments WHERE segment_id = ?;
                    """,
                    (seg_id,),
                ).fetchone()
                if existing is not None:
                    if not self._values_match(tuple(existing), payload):
                        raise PayloadConflictError(f"Conflicting payload for segment_id={seg_id}")
                    continue
                new_seg_rows.append((seg_id, *payload, s.get("created_at", now)))

            if new_seg_rows:
                c.executemany(
                    """
                    INSERT INTO ASRSegments (
                        segment_id, video_id, model_id, start_ms, end_ms, raw_text,
                        normalized_text, language, confidence, speaker_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    new_seg_rows,
                )

            new_word_rows: list[tuple[Any, ...]] = []
            if words:
                seen_words: dict[str, tuple[Any, ...]] = {}
                for w in words:
                    word_id = w["word_id"]
                    w_payload = (
                        w["segment_id"],
                        w["word"],
                        int(w["start_ms"]),
                        int(w["end_ms"]),
                        float(w.get("confidence", 1.0)),
                        w.get("alignment_source", "ctc"),
                    )
                    if word_id in seen_words:
                        if not self._values_match(seen_words[word_id], w_payload):
                            raise PayloadConflictError(f"Conflicting payload for word_id={word_id}")
                        continue
                    seen_words[word_id] = w_payload

                    existing_w = c.execute(
                        """
                        SELECT segment_id, word, start_ms, end_ms, confidence, alignment_source
                        FROM ASRWords WHERE word_id = ?;
                        """,
                        (word_id,),
                    ).fetchone()
                    if existing_w is not None:
                        if not self._values_match(tuple(existing_w), w_payload):
                            raise PayloadConflictError(f"Conflicting payload for word_id={word_id}")
                        continue
                    new_word_rows.append((word_id, *w_payload))

                if new_word_rows:
                    c.executemany(
                        """
                        INSERT INTO ASRWords (
                            word_id, segment_id, word, start_ms, end_ms, confidence, alignment_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        new_word_rows,
                    )
            return len(new_seg_rows) + len(new_word_rows)

        if conn is not None:
            return _apply(conn)
        with self._get_connection() as own_conn:
            return _apply(own_conn)

    def insert_asr_evidence_groups(
        self,
        groups: Sequence[dict[str, Any]],
        member_mappings: Sequence[tuple[str, str, str]],
        *,
        conn: sqlite3.Connection | None = None,
        update_fts: bool = True,
    ) -> None:
        now = current_iso()
        group_rows = [
            (
                g["group_id"],
                g["video_id"],
                int(g["start_ms"]),
                int(g["end_ms"]),
                g["consensus_text"],
                normalize_text(g["consensus_text"]),
                g["status"],
                g["models_involved"],
                float(g.get("temporal_overlap_ratio", 1.0)),
                float(g.get("text_similarity", 1.0)),
                g.get("created_at", now),
            )
            for g in groups
        ]

        def _apply(c: sqlite3.Connection) -> None:
            if group_rows:
                c.executemany(
                    """
                    INSERT INTO ASREvidenceGroups (
                        group_id, video_id, start_ms, end_ms, consensus_text, normalized_text,
                        status, models_involved, temporal_overlap_ratio, text_similarity, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(group_id) DO UPDATE SET
                        video_id=excluded.video_id,
                        start_ms=excluded.start_ms,
                        end_ms=excluded.end_ms,
                        consensus_text=excluded.consensus_text,
                        normalized_text=excluded.normalized_text,
                        status=excluded.status,
                        models_involved=excluded.models_involved,
                        temporal_overlap_ratio=excluded.temporal_overlap_ratio,
                        text_similarity=excluded.text_similarity;
                    """,
                    group_rows,
                )

            if member_mappings:
                c.executemany(
                    """
                    INSERT INTO ASREvidenceMembers (group_id, segment_id, model_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(group_id, segment_id) DO UPDATE SET
                        model_id=excluded.model_id;
                    """,
                    member_mappings,
                )

            if update_fts:
                fts_rows = [
                    (
                        normalize_text(g["consensus_text"]),
                        fold_diacritics(normalize_text(g["consensus_text"])),
                        g["video_id"],
                        g["group_id"],
                        f"grp_{g['group_id']}",
                        int(g["start_ms"]),
                        int(g["end_ms"]),
                        g["models_involved"],
                    )
                    for g in groups
                    if g["consensus_text"].strip()
                ]
                if fts_rows:
                    c.executemany(
                        """
                        INSERT INTO fts_asr (normalized_text, unaccented_text, video_id, group_id, segment_id, start_ms, end_ms, model_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        fts_rows,
                    )

        if conn is not None:
            _apply(conn)
            return
        with self._get_connection() as own_conn:
            _apply(own_conn)

    # -------------------------------------------------------------------------
    # Progress Jobs & Model Registry
    # -------------------------------------------------------------------------
    def update_job_progress(self, job_id: str, video_id: str, stage: str, status: str, progress_pct: float, worker: str = "local_cpu", error_msg: str | None = None) -> None:
        now = current_iso()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO ProcessingJobs (
                job_id, video_id, stage, status, progress_pct, error_message, worker, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                progress_pct=excluded.progress_pct,
                error_message=excluded.error_message,
                completed_at=CASE WHEN excluded.status IN ('completed', 'failed') THEN excluded.completed_at ELSE ProcessingJobs.completed_at END;
            """, (
                job_id,
                video_id,
                stage,
                status,
                progress_pct,
                error_msg,
                worker,
                now,
                now if status in ("completed", "failed") else None,
            ))

    def register_model_version(self, model_id: str, model_name: str, version: str, modality: str, weights_hash: str = "", config_dict: dict | None = None) -> None:
        now = current_iso()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO ModelVersions (
                model_id, model_name, version, modality, weights_hash, config_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                model_id,
                model_name,
                version,
                modality,
                weights_hash,
                json.dumps(config_dict or {}),
                now,
            ))
