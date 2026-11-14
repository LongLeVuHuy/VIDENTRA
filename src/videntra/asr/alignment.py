"""Reversible timeline alignment between audio time and video presentation frames."""
import bisect
from pathlib import Path
from typing import Sequence
import uuid

import pyarrow.parquet as pq

from .vad import VADSegment
from .transcription import TranscriptResult
from videntra.evidence.store import ASRSegmentRecord


class TimelineAligner:
    """Binds speech intervals to video timeline ledger ordinals and vice versa."""

    def __init__(self, ledger_path: Path | str | None = None):
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self._timestamps_us: list[int] = []
        self._ordinals: list[int] = []
        if self.ledger_path and self.ledger_path.is_file():
            self._load_ledger()

    def _load_ledger(self) -> None:
        table = pq.read_table(str(self.ledger_path), columns=["original_decoded_frame_index", "timestamp_us"])
        ords = table.column("original_decoded_frame_index").to_pylist()
        ts_us = table.column("timestamp_us").to_pylist()

        valid_pairs = [(ts, ord_idx) for ts, ord_idx in zip(ts_us, ords) if ts is not None]
        valid_pairs.sort(key=lambda p: p[0])

        self._timestamps_us = [p[0] for p in valid_pairs]
        self._ordinals = [p[1] for p in valid_pairs]

    def us_to_ordinal(self, target_us: int) -> int | None:
        """Find the presentation frame ordinal closest in time to target_us."""
        if not self._timestamps_us:
            return None

        idx = bisect.bisect_left(self._timestamps_us, target_us)
        if idx == 0:
            return self._ordinals[0]
        if idx >= len(self._timestamps_us):
            return self._ordinals[-1]

        # Choose closest between idx-1 and idx
        before_diff = abs(self._timestamps_us[idx - 1] - target_us)
        after_diff = abs(self._timestamps_us[idx] - target_us)
        return self._ordinals[idx - 1] if before_diff <= after_diff else self._ordinals[idx]

    def create_aligned_segment(
        self,
        video_id: str,
        vad_segment: VADSegment,
        transcript: TranscriptResult,
    ) -> ASRSegmentRecord:
        """Construct ASRSegmentRecord aligned to video ledger ordinals."""
        start_ord = self.us_to_ordinal(vad_segment.start_us)
        end_ord = self.us_to_ordinal(vad_segment.end_us)

        seg_id = f"asr_seg_{uuid.uuid4().hex[:8]}"
        st_sec = round(vad_segment.start_us / 1_000_000.0, 4)
        end_sec = round(vad_segment.end_us / 1_000_000.0, 4)

        return ASRSegmentRecord(
            segment_id=seg_id,
            video_id=video_id,
            start_us=vad_segment.start_us,
            end_us=vad_segment.end_us,
            start_seconds=st_sec,
            end_seconds=end_sec,
            start_ordinal=start_ord,
            end_ordinal=end_ord,
            text=transcript.text,
            language=transcript.language,
            confidence=transcript.confidence,
        )

    @staticmethod
    def reverse_lookup(ordinal: int, segments: Sequence[ASRSegmentRecord]) -> list[ASRSegmentRecord]:
        """Find all ASR segments that cover the given video presentation frame ordinal."""
        matched = []
        for seg in segments:
            if seg.start_ordinal is not None and seg.end_ordinal is not None:
                if seg.start_ordinal <= ordinal <= seg.end_ordinal:
                    matched.append(seg)
        return matched
