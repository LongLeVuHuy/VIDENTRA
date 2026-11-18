"""Independent OCR probe frame selection."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from videntra.ingestion.decode import DecodedFrame


@dataclass(frozen=True)
class OCRProbeRecord:
    probe_id: str
    video_id: str
    original_decoded_frame_index: int
    timestamp_us: int | None
    timestamp_seconds: float | None
    probe_reason: str
    epoch_id: str = "epoch_0"


@dataclass(frozen=True)
class OCRProbeManifest:
    video_id: str
    probes: tuple[OCRProbeRecord, ...]

    def to_parquet(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("probe_id", pa.string()),
            ("video_id", pa.string()),
            ("original_decoded_frame_index", pa.int64()),
            ("timestamp_us", pa.int64()),
            ("timestamp_seconds", pa.float64()),
            ("probe_reason", pa.string()),
            ("epoch_id", pa.string()),
        ])
        data = {name: [] for name in schema.names}
        for pr in self.probes:
            data["probe_id"].append(pr.probe_id)
            data["video_id"].append(pr.video_id)
            data["original_decoded_frame_index"].append(pr.original_decoded_frame_index)
            data["timestamp_us"].append(pr.timestamp_us)
            data["timestamp_seconds"].append(pr.timestamp_seconds)
            data["probe_reason"].append(pr.probe_reason)
            data["epoch_id"].append(pr.epoch_id)
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, str(p), compression="zstd")


def select_ocr_probes(
    frames: Iterator[DecodedFrame] | Sequence[DecodedFrame],
    video_id: str,
    probe_fps: float = 1.0,
    shot_cut_ordinals: Sequence[int] | None = None,
    burst_ordinals: Sequence[int] | None = None,
) -> OCRProbeManifest:
    """Select independent OCR text probe frames.

    Operates independently from visual sample filtering (1 fps base rate + cuts + optional bursts),
    resetting grid cadence cleanly across container epoch boundaries.
    """
    cuts_set = set(shot_cut_ordinals or ())
    burst_set = set(burst_ordinals or ())
    probes = []
    last_probe_sec: float | None = None
    current_epoch_id: str | None = None
    step_sec = 1.0 / probe_fps if probe_fps > 0 else 1.0

    for f in frames:
        if current_epoch_id is not None and f.epoch_id != current_epoch_id:
            last_probe_sec = None
        current_epoch_id = f.epoch_id

        t_sec = f.timestamp_seconds
        reasons = []

        if f.ordinal == 0:
            reasons.append("first")
        elif t_sec is not None and last_probe_sec is None:
            reasons.append(f"grid_{probe_fps:.1f}fps")
        elif t_sec is not None and (t_sec - last_probe_sec) >= step_sec:
            reasons.append(f"grid_{probe_fps:.1f}fps")
        elif t_sec is None and last_probe_sec is None:
            reasons.append("unindexed")

        if f.ordinal in cuts_set:
            reasons.append("shot_cut")
        if f.ordinal in burst_set:
            reasons.append("burst_probe")

        if reasons:
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}:ocr:{f.ordinal}"))
            record = OCRProbeRecord(
                probe_id=pid,
                video_id=video_id,
                original_decoded_frame_index=f.ordinal,
                timestamp_us=f.timestamp_us,
                timestamp_seconds=t_sec,
                probe_reason="+".join(reasons),
                epoch_id=f.epoch_id,
            )
            probes.append(record)
            if t_sec is not None:
                last_probe_sec = t_sec

    return OCRProbeManifest(video_id=video_id, probes=tuple(probes))
