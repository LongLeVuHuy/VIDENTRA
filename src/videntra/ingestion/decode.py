"""Sequential presentation-order frame decoding with exact PTS tracking."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any, Iterator
import numpy as np

import av
from videntra.config.ingestion import DecodeConfig
from videntra.contracts.timing import microseconds
from videntra.ingestion.probe import probe_media, ProbeResult, NoVideoStreamError


class DecodeError(Exception):
    """Raised when media decoding fails."""
    pass


_PICT_TYPE_MAP = {
    1: "I",
    2: "P",
    3: "B",
    4: "S",
    5: "SI",
    6: "SP",
    7: "BI",
}


@dataclass
class DecodedFrame:
    """Transient record representing one presentation-order decoded frame."""
    ordinal: int
    pts: int | None
    dts: int | None
    time_base_num: int
    time_base_den: int
    origin: Fraction
    pts_occurrence: int = 0
    epoch_id: str = "epoch_0"
    is_keyframe: bool = False
    pict_type: str = "unknown"
    width: int = 0
    height: int = 0
    pixels: np.ndarray | None = None
    error_context: dict[str, Any] | None = None
    best_effort_timestamp: int | None = None
    timestamp_quality: str = "exact_pts"

    @property
    def time_base(self) -> Fraction:
        return Fraction(self.time_base_num, self.time_base_den)

    @property
    def rational_pts(self) -> Fraction | None:
        if self.pts is None:
            return None
        return Fraction(self.pts * self.time_base_num, self.time_base_den)

    @property
    def clip_time(self) -> Fraction | None:
        if self.pts is None:
            return None
        return self.rational_pts - self.origin

    @property
    def timestamp_us(self) -> int | None:
        c = self.clip_time
        if c is None:
            return None
        return microseconds(c)

    @property
    def timestamp_seconds(self) -> float | None:
        c = self.clip_time
        if c is None:
            return None
        return float(c)

    def pixel_hash(self) -> str | None:
        if self.pixels is None:
            return None
        return hashlib.sha256(self.pixels.tobytes()).hexdigest()


def _format_pict_type(raw_pict_type: Any) -> str:
    if isinstance(raw_pict_type, int):
        return _PICT_TYPE_MAP.get(raw_pict_type, "unknown")
    if hasattr(raw_pict_type, "name"):
        return str(raw_pict_type.name)
    s = str(raw_pict_type).strip()
    return s if s else "unknown"


def decode_frames(
    asset: Path | str | dict[str, Any],
    probe_result: ProbeResult | None = None,
    recipe: DecodeConfig | dict[str, Any] | None = None,
    include_pixels: bool = True,
) -> Iterator[DecodedFrame]:
    """Decode presentation-order video frames sequentially from media container.
    
    Ordinal is zero-based presentation-order, never inferred from nominal FPS.
    Raw PTS and rational time_base are authoritative.
    """
    if probe_result is None:
        probe_result = probe_media(asset)

    if not probe_result.has_video or probe_result.selected_video_stream_index is None:
        raise NoVideoStreamError(f"No video stream available in {probe_result.video_path.name}")

    if recipe is None:
        cfg = DecodeConfig()
    elif isinstance(recipe, dict):
        cfg = DecodeConfig(**recipe)
    elif isinstance(recipe, DecodeConfig):
        cfg = recipe
    else:
        raise ValueError("Invalid decode recipe")

    video_stream_idx = probe_result.selected_video_stream_index
    media_path = probe_result.video_path

    try:
        container = av.open(str(media_path))
    except Exception as exc:
        raise DecodeError(f"Failed to open media container {media_path.name}: {exc}") from exc

    streams = [s for s in container.streams if s.index == video_stream_idx]
    if not streams:
        container.close()
        raise DecodeError(f"Video stream {video_stream_idx} not found in container")

    stream = streams[0]
    if cfg.thread_count > 0:
        stream.thread_count = cfg.thread_count
        stream.thread_type = "AUTO"
    tb = stream.time_base or Fraction(probe_result.video_stream.time_base_num, probe_result.video_stream.time_base_den)
    tb_num, tb_den = tb.numerator, tb.denominator

    ordinal = 0
    origin: Fraction | None = None
    epoch_idx = 0
    pts_occurrence = 0
    prev_pts: int | None = None
    max_frames = cfg.max_frames

    try:
        for frame in container.decode(stream):
            raw_pts = frame.pts
            raw_dts = frame.dts
            best_effort = getattr(frame, "best_effort_timestamp", None)

            # Determine canonical origin from first decoded frame with presentation time
            if origin is None:
                if raw_pts is not None:
                    origin = Fraction(raw_pts * tb_num, tb_den)
                elif best_effort is not None:
                    origin = Fraction(best_effort * tb_num, tb_den)
                else:
                    origin = Fraction(0, 1)

            # Sequence and epoch tracking
            if raw_pts is not None and prev_pts is not None:
                if raw_pts == prev_pts:
                    pts_occurrence += 1
                elif raw_pts < prev_pts:
                    # Backward PTS indicates container discontinuity / new epoch
                    epoch_idx += 1
                    pts_occurrence = 0
                else:
                    pts_occurrence = 0
            elif raw_pts is not None and prev_pts is None:
                pts_occurrence = 0

            prev_pts = raw_pts

            if raw_pts is not None:
                quality = "exact_pts"
            elif best_effort is not None:
                quality = "best_effort"
            else:
                quality = "missing"

            pixels = None
            if include_pixels:
                pixels = frame.to_ndarray(format="rgb24")

            decoded = DecodedFrame(
                ordinal=ordinal,
                pts=raw_pts,
                dts=raw_dts,
                time_base_num=tb_num,
                time_base_den=tb_den,
                origin=origin,
                pts_occurrence=pts_occurrence,
                epoch_id=f"epoch_{epoch_idx}",
                is_keyframe=bool(frame.key_frame),
                pict_type=_format_pict_type(frame.pict_type),
                width=frame.width,
                height=frame.height,
                pixels=pixels,
                best_effort_timestamp=best_effort,
                timestamp_quality=quality,
            )

            yield decoded
            ordinal += 1

            if max_frames is not None and ordinal >= max_frames:
                break
    except av.error.FFmpegError as exc:
        if cfg.error_policy == "strict":
            raise DecodeError(f"Decoder error at frame ordinal {ordinal}: {exc}") from exc
        # quarantine or log error context
    finally:
        container.close()
