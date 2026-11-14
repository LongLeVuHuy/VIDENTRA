"""Evidence storage, Parquet schemas, and interval lookup operations."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import normalize_text


OCR_TRACK_SCHEMA = pa.schema([
    ("track_id", pa.string()),
    ("video_id", pa.string()),
    ("start_ordinal", pa.int64()),
    ("end_ordinal", pa.int64()),
    ("start_seconds", pa.float64()),
    ("end_seconds", pa.float64()),
    ("text", pa.string()),
    ("confidence", pa.float64()),
    ("bbox_x1", pa.float64()),
    ("bbox_y1", pa.float64()),
    ("bbox_x2", pa.float64()),
    ("bbox_y2", pa.float64()),
    ("frame_count", pa.int64()),
])

ASR_SEGMENT_SCHEMA = pa.schema([
    ("segment_id", pa.string()),
    ("video_id", pa.string()),
    ("start_us", pa.int64()),
    ("end_us", pa.int64()),
    ("start_seconds", pa.float64()),
    ("end_seconds", pa.float64()),
    ("start_ordinal", pa.int64()),
    ("end_ordinal", pa.int64()),
    ("text", pa.string()),
    ("language", pa.string()),
    ("confidence", pa.float64()),
])


@dataclass(frozen=True)
class OCRTrackRecord:
    track_id: str
    video_id: str
    start_ordinal: int
    end_ordinal: int
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    frame_count: int = 1

    def to_evidence_dict(self) -> dict[str, Any]:
        """Convert to canonical evidence item matching Result contract schema."""
        return {
            "evidence_id": f"ev_ocr_{self.track_id}",
            "kind": "ocr_track",
            "track_id": self.track_id,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "start_seconds": round(float(self.start_seconds), 4),
            "end_seconds": round(float(self.end_seconds), 4),
            "start_original_frame_index": self.start_ordinal,
            "end_original_frame_index": self.end_ordinal,
            "bbox": [round(float(v), 4) for v in self.bbox],
        }


@dataclass(frozen=True)
class ASRSegmentRecord:
    segment_id: str
    video_id: str
    start_us: int
    end_us: int
    start_seconds: float
    end_seconds: float
    start_ordinal: int | None
    end_ordinal: int | None
    text: str
    language: str = "vi"
    confidence: float = 1.0

    def to_evidence_dict(self) -> dict[str, Any]:
        """Convert to canonical evidence item matching Result contract schema."""
        return {
            "evidence_id": f"ev_asr_{self.segment_id}",
            "kind": "asr_segment",
            "segment_id": self.segment_id,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "start_seconds": round(float(self.start_seconds), 4),
            "end_seconds": round(float(self.end_seconds), 4),
            "start_us": self.start_us,
            "end_us": self.end_us,
            "language": self.language,
        }


class EvidenceStore:
    """Manages serialization, loading, and interval lookup of multimodal evidence."""

    @staticmethod
    def save_ocr_tracks(tracks: Sequence[OCRTrackRecord], path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "track_id": [t.track_id for t in tracks],
            "video_id": [t.video_id for t in tracks],
            "start_ordinal": [t.start_ordinal for t in tracks],
            "end_ordinal": [t.end_ordinal for t in tracks],
            "start_seconds": [t.start_seconds for t in tracks],
            "end_seconds": [t.end_seconds for t in tracks],
            "text": [normalize_text(t.text) for t in tracks],
            "confidence": [t.confidence for t in tracks],
            "bbox_x1": [t.bbox[0] for t in tracks],
            "bbox_y1": [t.bbox[1] for t in tracks],
            "bbox_x2": [t.bbox[2] for t in tracks],
            "bbox_y2": [t.bbox[3] for t in tracks],
            "frame_count": [t.frame_count for t in tracks],
        }
        table = pa.Table.from_pydict(data, schema=OCR_TRACK_SCHEMA)
        pq.write_table(table, str(p), compression="zstd")

    @staticmethod
    def load_ocr_tracks(path: Path | str) -> list[OCRTrackRecord]:
        p = Path(path)
        if not p.is_file():
            return []
        table = pq.read_table(str(p), schema=OCR_TRACK_SCHEMA)
        records = []
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            records.append(
                OCRTrackRecord(
                    track_id=row["track_id"],
                    video_id=row["video_id"],
                    start_ordinal=row["start_ordinal"],
                    end_ordinal=row["end_ordinal"],
                    start_seconds=row["start_seconds"],
                    end_seconds=row["end_seconds"],
                    text=row["text"],
                    confidence=row["confidence"],
                    bbox=(row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
                    frame_count=row["frame_count"],
                )
            )
        return records

    @staticmethod
    def save_asr_segments(segments: Sequence[ASRSegmentRecord], path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "segment_id": [s.segment_id for s in segments],
            "video_id": [s.video_id for s in segments],
            "start_us": [s.start_us for s in segments],
            "end_us": [s.end_us for s in segments],
            "start_seconds": [s.start_seconds for s in segments],
            "end_seconds": [s.end_seconds for s in segments],
            "start_ordinal": [s.start_ordinal if s.start_ordinal is not None else -1 for s in segments],
            "end_ordinal": [s.end_ordinal if s.end_ordinal is not None else -1 for s in segments],
            "text": [normalize_text(s.text) for s in segments],
            "language": [s.language for s in segments],
            "confidence": [s.confidence for s in segments],
        }
        table = pa.Table.from_pydict(data, schema=ASR_SEGMENT_SCHEMA)
        pq.write_table(table, str(p), compression="zstd")

    @staticmethod
    def load_asr_segments(path: Path | str) -> list[ASRSegmentRecord]:
        p = Path(path)
        if not p.is_file():
            return []
        table = pq.read_table(str(p), schema=ASR_SEGMENT_SCHEMA)
        records = []
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            st_ord = row["start_ordinal"] if row["start_ordinal"] >= 0 else None
            end_ord = row["end_ordinal"] if row["end_ordinal"] >= 0 else None
            records.append(
                ASRSegmentRecord(
                    segment_id=row["segment_id"],
                    video_id=row["video_id"],
                    start_us=row["start_us"],
                    end_us=row["end_us"],
                    start_seconds=row["start_seconds"],
                    end_seconds=row["end_seconds"],
                    start_ordinal=st_ord,
                    end_ordinal=end_ord,
                    text=row["text"],
                    language=row["language"],
                    confidence=row["confidence"],
                )
            )
        return records

    @staticmethod
    def lookup_evidence(
        video_id: str,
        start_seconds: float,
        end_seconds: float,
        ocr_tracks: Sequence[OCRTrackRecord] = (),
        asr_segments: Sequence[ASRSegmentRecord] = (),
    ) -> list[dict[str, Any]]:
        """Find overlapping OCR tracks and ASR segments within [start_seconds, end_seconds)."""
        evidence_items = []
        for track in ocr_tracks:
            if track.video_id != video_id:
                continue
            # Overlap check
            if max(start_seconds, track.start_seconds) < min(end_seconds, track.end_seconds):
                evidence_items.append(track.to_evidence_dict())
            elif track.start_seconds == track.end_seconds and start_seconds <= track.start_seconds <= end_seconds:
                # Instant singleton track
                evidence_items.append(track.to_evidence_dict())

        for seg in asr_segments:
            if seg.video_id != video_id:
                continue
            if max(start_seconds, seg.start_seconds) < min(end_seconds, seg.end_seconds):
                evidence_items.append(seg.to_evidence_dict())

        return evidence_items
