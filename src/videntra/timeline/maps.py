"""Rational clip, source, and audio mapping contracts."""
from dataclasses import dataclass, field, asdict
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import av
from videntra.contracts.timing import microseconds, interval
from videntra.ingestion.probe import ProbeResult


@dataclass(frozen=True)
class RationalClipMap:
    """Exact rational conversion from raw PTS to clip presentation time."""
    origin: Fraction
    time_base_num: int
    time_base_den: int

    @property
    def time_base(self) -> Fraction:
        return Fraction(self.time_base_num, self.time_base_den)

    def pts_to_clip_time(self, pts: int) -> Fraction:
        return Fraction(pts * self.time_base_num, self.time_base_den) - self.origin

    def pts_to_us(self, pts: int) -> int:
        return microseconds(self.pts_to_clip_time(pts))

    def pts_to_seconds(self, pts: int) -> float:
        return float(self.pts_to_clip_time(pts))


@dataclass(frozen=True)
class SourceMapSegment:
    clip_start_us: int
    clip_end_us: int
    source_start_us: int
    rate_num: int = 1
    rate_den: int = 1
    source_epoch_id: str = "source_epoch_0"
    mapping_provenance: str = "verified"

    def __post_init__(self):
        interval(self.clip_start_us, self.clip_end_us)
        if self.rate_num <= 0 or self.rate_den <= 0:
            raise ValueError("Rate must be strictly positive")

    @property
    def rate(self) -> Fraction:
        return Fraction(self.rate_num, self.rate_den)


@dataclass(frozen=True)
class PiecewiseSourceMap:
    """Mapping from clip-relative timestamps to upstream source timestamps."""
    source_video_id: str | None = None
    mapping_kind: str = "unknown"  # "unknown", "constant_offset", "piecewise"
    source_offset_us: int | None = None
    segments: tuple[SourceMapSegment, ...] = ()

    def validate_segments(self) -> None:
        """Validate that piecewise segments are ordered and non-overlapping."""
        sorted_segs = sorted(self.segments, key=lambda s: s.clip_start_us)
        for i in range(len(sorted_segs) - 1):
            if sorted_segs[i].clip_end_us > sorted_segs[i + 1].clip_start_us:
                raise ValueError("Piecewise source map segments must not overlap")

    def clip_to_source_us(self, clip_us: int | None) -> int | None:
        if clip_us is None:
            return None
        if self.mapping_kind == "constant_offset":
            if self.source_offset_us is None:
                return None
            return clip_us + self.source_offset_us
        elif self.mapping_kind == "piecewise":
            for seg in self.segments:
                if seg.clip_start_us <= clip_us < seg.clip_end_us:
                    delta = clip_us - seg.clip_start_us
                    return seg.source_start_us + round(delta * seg.rate)
            return None
        return None

    def clip_to_source_seconds(self, clip_seconds: float | None) -> float | None:
        us = self.clip_to_source_us(round(clip_seconds * 1_000_000)) if clip_seconds is not None else None
        return us / 1_000_000.0 if us is not None else None


@dataclass(frozen=True)
class AudioMapEntry:
    chunk_id: str
    sample_rate: int
    channels: int
    pts: int | None
    start_us: int | None
    duration_us: int
    end_us: int | None
    sample_count: int


@dataclass(frozen=True)
class AudioMap:
    """Authoritative audio chunk mapping with origin shared with video."""
    audio_stream_index: int | None
    sample_rate: int | None
    channels: int | None
    audio_delay_us: int | None
    shared_origin: Fraction
    entries: tuple[AudioMapEntry, ...] = ()

    def to_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            header = {
                "header": True,
                "audio_stream_index": self.audio_stream_index,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "audio_delay_us": self.audio_delay_us,
                "shared_origin": str(self.shared_origin),
                "entry_count": len(self.entries),
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for entry in self.entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "AudioMap":
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Empty audio map file")
        header = json.loads(lines[0])
        entries = []
        for line in lines[1:]:
            d = json.loads(line)
            entries.append(AudioMapEntry(**d))
        return cls(
            audio_stream_index=header.get("audio_stream_index"),
            sample_rate=header.get("sample_rate"),
            channels=header.get("channels"),
            audio_delay_us=header.get("audio_delay_us"),
            shared_origin=Fraction(header["shared_origin"]),
            entries=tuple(entries),
        )


def build_audio_map(
    media_path: Path | str,
    probe_result: ProbeResult,
    shared_origin: Fraction,
) -> AudioMap:
    """Construct audio chunk mapping using shared video origin."""
    if not probe_result.has_audio or probe_result.selected_audio_stream_index is None:
        return AudioMap(
            audio_stream_index=None,
            sample_rate=None,
            channels=None,
            audio_delay_us=None,
            shared_origin=shared_origin,
            entries=(),
        )

    p = Path(media_path)
    container = av.open(str(p))
    audio_stream_idx = probe_result.selected_audio_stream_index
    streams = [s for s in container.streams if s.index == audio_stream_idx]
    if not streams:
        container.close()
        raise ValueError(f"Audio stream {audio_stream_idx} not found")

    stream = streams[0]
    tb = stream.time_base or Fraction(probe_result.audio_stream.time_base_num, probe_result.audio_stream.time_base_den)
    sample_rate = stream.rate or probe_result.audio_stream.sample_rate or 16000
    channels = stream.channels or probe_result.audio_stream.channels or 1

    entries = []
    first_pts: int | None = None
    chunk_idx = 0

    try:
        for frame in container.decode(stream):
            if first_pts is None and frame.pts is not None:
                first_pts = frame.pts

            dur_sec = Fraction(frame.samples, sample_rate)
            dur_us = microseconds(dur_sec)

            # Presentation time relative to shared origin (keep None if missing)
            if frame.pts is not None:
                rational_time = Fraction(frame.pts * tb.numerator, tb.denominator) - shared_origin
                start_us = microseconds(rational_time)
                end_us = start_us + dur_us
            else:
                start_us = None
                end_us = None

            entries.append(
                AudioMapEntry(
                    chunk_id=f"audio_chunk_{chunk_idx:06d}",
                    sample_rate=sample_rate,
                    channels=channels,
                    pts=frame.pts,
                    start_us=start_us,
                    duration_us=dur_us,
                    end_us=end_us,
                    sample_count=frame.samples,
                )
            )
            chunk_idx += 1
    finally:
        container.close()

    audio_delay_us = None
    if first_pts is not None:
        first_audio_time = Fraction(first_pts * tb.numerator, tb.denominator)
        audio_delay_us = microseconds(first_audio_time - shared_origin)

    return AudioMap(
        audio_stream_index=audio_stream_idx,
        sample_rate=sample_rate,
        channels=channels,
        audio_delay_us=audio_delay_us,
        shared_origin=shared_origin,
        entries=tuple(entries),
    )
