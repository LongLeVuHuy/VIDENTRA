"""Typed frame timeline ledger with Parquet streaming and exact-seek lookup."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any, Iterator, Sequence
import numpy as np

import av
import pyarrow as pa
import pyarrow.parquet as pq

from videntra.contracts.timing import interval, frame_endpoints
from videntra.ingestion.decode import DecodedFrame, _format_pict_type
from videntra.timeline.maps import PiecewiseSourceMap


LEDGER_PARQUET_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("timeline_revision", pa.string()),
    ("original_decoded_frame_index", pa.int64()),
    ("presentation_timestamp", pa.int64()),
    ("best_effort_timestamp", pa.int64()),
    ("pts_occurrence", pa.int32()),
    ("epoch_id", pa.string()),
    ("time_base_num", pa.int32()),
    ("time_base_den", pa.int32()),
    ("origin_num", pa.int64()),
    ("origin_den", pa.int64()),
    ("timestamp_us", pa.int64()),
    ("timestamp_seconds", pa.float64()),
    ("is_keyframe", pa.bool_()),
    ("pict_type", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("source_offset_seconds", pa.float64()),
    ("source_timestamp_us", pa.int64()),
    ("source_original_frame_index", pa.int64()),
    ("timestamp_quality", pa.string()),
    ("sampling_reason", pa.string()),
    ("sampled_frame_id", pa.string()),
    ("pixel_hash", pa.string()),
])


@dataclass(frozen=True)
class LedgerSummary:
    video_id: str
    timeline_revision: str
    total_frames: int
    keyframe_count: int
    epochs: tuple[str, ...]
    missing_pts_count: int
    repeated_pts_count: int
    duration_us: int
    parquet_path: Path


def write_timeline_ledger(
    destination_path: Path | str,
    frames: Iterator[DecodedFrame] | Sequence[DecodedFrame],
    video_id: str,
    timeline_revision: str,
    row_group_size: int = 1000,
    source_map: PiecewiseSourceMap | None = None,
    store_pixel_hashes: bool = False,
) -> LedgerSummary:
    """Stream-writes decoded frame records into a Parquet ledger with bounded memory."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(str(dest), LEDGER_PARQUET_SCHEMA, compression="zstd")

    buffer: list[dict[str, Any]] = []
    total_frames = 0
    keyframe_count = 0
    epochs_set = set()
    missing_pts_count = 0
    repeated_pts_count = 0
    last_us = 0

    def _flush_buffer():
        if not buffer:
            return
        arrays = []
        for field_name in LEDGER_PARQUET_SCHEMA.names:
            vals = [row[field_name] for row in buffer]
            arrays.append(pa.array(vals, type=LEDGER_PARQUET_SCHEMA.field(field_name).type))
        table = pa.Table.from_arrays(arrays, schema=LEDGER_PARQUET_SCHEMA)
        writer.write_table(table)
        buffer.clear()

    try:
        for f in frames:
            total_frames += 1
            if f.is_keyframe:
                keyframe_count += 1
            epochs_set.add(f.epoch_id)
            if f.pts is None:
                missing_pts_count += 1
            if f.pts_occurrence > 0:
                repeated_pts_count += 1
            if f.timestamp_us is not None:
                last_us = max(last_us, f.timestamp_us)

            source_us = None
            source_sec = None
            if source_map is not None:
                source_us = source_map.clip_to_source_us(f.timestamp_us)
                source_sec = source_map.clip_to_source_seconds(f.timestamp_seconds)

            pix_hash = f.pixel_hash() if store_pixel_hashes else None

            row = {
                "video_id": video_id,
                "timeline_revision": timeline_revision,
                "original_decoded_frame_index": f.ordinal,
                "presentation_timestamp": f.pts,
                "best_effort_timestamp": f.best_effort_timestamp,
                "pts_occurrence": f.pts_occurrence,
                "epoch_id": f.epoch_id,
                "time_base_num": f.time_base_num,
                "time_base_den": f.time_base_den,
                "origin_num": f.origin.numerator,
                "origin_den": f.origin.denominator,
                "timestamp_us": f.timestamp_us,
                "timestamp_seconds": f.timestamp_seconds,
                "is_keyframe": f.is_keyframe,
                "pict_type": f.pict_type,
                "width": f.width,
                "height": f.height,
                "source_offset_seconds": source_sec,
                "source_timestamp_us": source_us,
                "source_original_frame_index": None,
                "timestamp_quality": f.timestamp_quality,
                "sampling_reason": None,
                "sampled_frame_id": None,
                "pixel_hash": pix_hash,
            }
            buffer.append(row)
            if len(buffer) >= row_group_size:
                _flush_buffer()

        _flush_buffer()
    finally:
        writer.close()

    return LedgerSummary(
        video_id=video_id,
        timeline_revision=timeline_revision,
        total_frames=total_frames,
        keyframe_count=keyframe_count,
        epochs=tuple(sorted(epochs_set)),
        missing_pts_count=missing_pts_count,
        repeated_pts_count=repeated_pts_count,
        duration_us=last_us,
        parquet_path=dest,
    )


def read_frame_record(parquet_path: Path | str, ordinal: int) -> dict[str, Any]:
    """Retrieve frame record metadata for a given presentation ordinal using Parquet row filter."""
    table = pq.read_table(
        str(parquet_path),
        filters=[("original_decoded_frame_index", "==", ordinal)],
    )
    if table.num_rows == 0:
        raise KeyError(f"Ordinal {ordinal} not found in ledger")
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def get_ledger_endpoints(parquet_path: Path | str, start_us: int, end_us: int) -> tuple[int, int]:
    """Calculate start and end ordinals for interval [start_us, end_us)."""
    interval(start_us, end_us)
    table = pq.read_table(str(parquet_path), columns=["timestamp_us", "original_decoded_frame_index"])
    timestamps = table.column("timestamp_us").to_pylist()
    ordinals = table.column("original_decoded_frame_index").to_pylist()
    valid_ordinals = [
        ordinals[i] for i, ts in enumerate(timestamps)
        if ts is not None and start_us <= ts < end_us
    ]
    if not valid_ordinals:
        raise ValueError(f"No observed frame in interval [{start_us}, {end_us})")
    return valid_ordinals[0], valid_ordinals[-1]


def exact_seek_frame(
    media_path: Path | str,
    parquet_path: Path | str,
    target_ordinal: int,
    video_stream_index: int = 0,
) -> DecodedFrame:
    """Exact-seek frame retrieval with anchor alignment and fallback to sequential decode.
    
    Guarantees bit-level parity with sequential decode output even with repeated PTS or epoch resets.
    """
    req_cols = [
        "original_decoded_frame_index",
        "presentation_timestamp",
        "pts_occurrence",
        "epoch_id",
        "is_keyframe",
        "time_base_num",
        "time_base_den",
        "origin_num",
        "origin_den",
        "timestamp_quality",
    ]
    table = pq.read_table(str(parquet_path), columns=req_cols)
    ordinals = table.column("original_decoded_frame_index").to_pylist()
    try:
        target_idx = ordinals.index(target_ordinal)
    except ValueError:
        raise KeyError(f"Target ordinal {target_ordinal} not found in ledger")

    target_epoch = table.column("epoch_id")[target_idx].as_py()
    target_pts = table.column("presentation_timestamp")[target_idx].as_py()
    target_occurrence = table.column("pts_occurrence")[target_idx].as_py()
    keyframes = table.column("is_keyframe").to_pylist()
    epochs = table.column("epoch_id").to_pylist()
    pts_list = table.column("presentation_timestamp").to_pylist()
    occurrences = table.column("pts_occurrence").to_pylist()

    # Find the nearest preceding keyframe within the same epoch
    key_idx = None
    for i in range(target_idx, -1, -1):
        if epochs[i] == target_epoch and keyframes[i]:
            key_idx = i
            break

    if key_idx is None:
        key_idx = 0

    key_pts = pts_list[key_idx]
    key_occurrence = occurrences[key_idx]
    steps_needed = target_idx - key_idx

    container = av.open(str(media_path))
    try:
        streams = [s for s in container.streams if s.index == video_stream_index]
        if not streams:
            raise ValueError(f"Stream index {video_stream_index} not found")
        stream = streams[0]

        seek_successful = False
        target_frame = None

        if key_pts is not None:
            container.seek(key_pts, stream=stream)

        # Track PTS occurrence to find the exact keyframe anchor
        curr_pts: int | None = None
        curr_occurrence = 0
        anchor_found = False
        frames_since_anchor = 0

        for frame in container.decode(stream):
            f_pts = frame.pts
            if f_pts == curr_pts:
                curr_occurrence += 1
            else:
                curr_pts = f_pts
                curr_occurrence = 0

            if not anchor_found:
                if f_pts == key_pts and curr_occurrence == key_occurrence:
                    anchor_found = True
                    frames_since_anchor = 0
                    if steps_needed == 0:
                        target_frame = frame
                        seek_successful = True
                        break
            else:
                frames_since_anchor += 1
                if frames_since_anchor == steps_needed:
                    if f_pts == target_pts and curr_occurrence == target_occurrence:
                        target_frame = frame
                        seek_successful = True
                    break

        if not seek_successful or target_frame is None:
            # Fallback to sequential decode to guarantee bit-level parity
            from videntra.ingestion.decode import decode_frames
            container.close()
            for dec in decode_frames(media_path, include_pixels=True):
                if dec.ordinal == target_ordinal:
                    return dec
            raise RuntimeError(f"Failed to reach frame at ordinal {target_ordinal}")

        tb = stream.time_base
        tb_num, tb_den = tb.numerator, tb.denominator
        origin_num = table.column("origin_num")[target_idx].as_py()
        origin_den = table.column("origin_den")[target_idx].as_py()
        origin = Fraction(origin_num, origin_den)

        return DecodedFrame(
            ordinal=target_ordinal,
            pts=target_frame.pts,
            dts=target_frame.dts,
            time_base_num=tb_num,
            time_base_den=tb_den,
            origin=origin,
            pts_occurrence=target_occurrence,
            epoch_id=target_epoch,
            is_keyframe=bool(target_frame.key_frame),
            pict_type=_format_pict_type(target_frame.pict_type),
            width=target_frame.width,
            height=target_frame.height,
            pixels=target_frame.to_ndarray(format="rgb24"),
            best_effort_timestamp=getattr(target_frame, "best_effort_timestamp", None),
            timestamp_quality=table.column("timestamp_quality")[target_idx].as_py(),
        )
    finally:
        try:
            container.close()
        except Exception:
            pass
