"""Temporal OCR tracking across video presentation timeline."""
from dataclasses import dataclass
import difflib
from typing import Sequence
import uuid

from .detection import box_2d_iou
from videntra.evidence.normalize import normalize_text
from videntra.evidence.store import OCRTrackRecord


@dataclass(frozen=True)
class FrameOCRObservation:
    ordinal: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    text: str
    confidence: float


def text_similarity(s1: str, s2: str) -> float:
    """Levenshtein-based string similarity ratio in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


class OCRTracker:
    """Assembles instantaneous frame-level OCR observations into persistent temporal tracks."""

    def __init__(
        self,
        max_gap_seconds: float = 1.5,
        min_spatial_iou: float = 0.30,
        min_text_similarity: float = 0.70,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.min_spatial_iou = min_spatial_iou
        self.min_text_similarity = min_text_similarity

    def track(self, video_id: str, observations: Sequence[FrameOCRObservation]) -> list[OCRTrackRecord]:
        """Group observations into tracks enforcing singleton instant and gap separation rules."""
        if not observations:
            return []

        # Sort observations by temporal ordinal
        sorted_obs = sorted(observations, key=lambda o: (o.ordinal, o.timestamp_seconds))

        active_tracks: list[dict] = []
        completed_tracks: list[OCRTrackRecord] = []

        for obs in sorted_obs:
            matched_idx = None
            best_match_score = -1.0

            clean_obs_text = normalize_text(obs.text)

            for idx, tr in enumerate(active_tracks):
                dt = obs.timestamp_seconds - tr["last_seconds"]
                # Hard rule: If gap > max_gap_seconds, strictly do NOT merge
                if dt > self.max_gap_seconds:
                    continue

                iou = box_2d_iou(obs.bbox, tr["last_bbox"])
                sim = text_similarity(clean_obs_text, tr["text"])

                if iou >= self.min_spatial_iou and sim >= self.min_text_similarity:
                    combined_score = 0.5 * iou + 0.5 * sim
                    if combined_score > best_match_score:
                        best_match_score = combined_score
                        matched_idx = idx

            if matched_idx is not None:
                # Extend existing active track
                tr = active_tracks[matched_idx]
                tr["end_ordinal"] = obs.ordinal
                tr["end_seconds"] = obs.timestamp_seconds
                tr["last_seconds"] = obs.timestamp_seconds
                tr["last_bbox"] = obs.bbox
                tr["frame_count"] += 1
                # Update text to highest-confidence reading
                if obs.confidence > tr["confidence"]:
                    tr["text"] = clean_obs_text
                    tr["confidence"] = obs.confidence
                # Update running average bounding box
                n = tr["frame_count"]
                tr["bbox"] = tuple(
                    (tr["bbox"][i] * (n - 1) + obs.bbox[i]) / n for i in range(4)
                )
            else:
                # Start a new active track (instantaneous singleton until extended)
                new_track = {
                    "track_id": f"ocr_tr_{uuid.uuid4().hex[:8]}",
                    "start_ordinal": obs.ordinal,
                    "end_ordinal": obs.ordinal,
                    "start_seconds": obs.timestamp_seconds,
                    "end_seconds": obs.timestamp_seconds,
                    "last_seconds": obs.timestamp_seconds,
                    "last_bbox": obs.bbox,
                    "bbox": obs.bbox,
                    "text": clean_obs_text,
                    "confidence": obs.confidence,
                    "frame_count": 1,
                }
                active_tracks.append(new_track)

        # Convert all tracks to immutable OCRTrackRecord
        for tr in active_tracks:
            completed_tracks.append(
                OCRTrackRecord(
                    track_id=tr["track_id"],
                    video_id=video_id,
                    start_ordinal=tr["start_ordinal"],
                    end_ordinal=tr["end_ordinal"],
                    start_seconds=tr["start_seconds"],
                    end_seconds=tr["end_seconds"],
                    text=tr["text"],
                    confidence=tr["confidence"],
                    bbox=tr["bbox"],
                    frame_count=tr["frame_count"],
                )
            )

        return completed_tracks
