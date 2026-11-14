"""Deterministic int64 identifier mapping for indexed frame samples."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


ID_MAP_SCHEMA = pa.schema([
    ("int64_id", pa.int64()),
    ("video_id", pa.string()),
    ("timeline_revision", pa.string()),
    ("ordinal", pa.int64()),
    ("timestamp_us", pa.int64()),
    ("timestamp_seconds", pa.float64()),
    ("sampled_frame_id", pa.string()),
])


@dataclass(frozen=True)
class SampleCoordinate:
    int64_id: int
    video_id: str
    timeline_revision: str
    ordinal: int
    timestamp_us: int | None
    timestamp_seconds: float | None
    sampled_frame_id: str


class IDMappingStore:
    """Bi-directional mapping between continuous int64 IDs and sample coordinates."""

    def __init__(self):
        self._by_int: list[SampleCoordinate] = []
        self._by_key: dict[tuple[str, int], int] = {}  # (video_id, ordinal) -> int64_id

    def __len__(self) -> int:
        return len(self._by_int)

    def register(
        self,
        video_id: str,
        timeline_revision: str,
        ordinal: int,
        sampled_frame_id: str,
        timestamp_us: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> int:
        key = (video_id, ordinal)
        if key in self._by_key:
            return self._by_key[key]

        int_id = len(self._by_int)
        coord = SampleCoordinate(
            int64_id=int_id,
            video_id=video_id,
            timeline_revision=timeline_revision,
            ordinal=ordinal,
            timestamp_us=timestamp_us,
            timestamp_seconds=timestamp_seconds,
            sampled_frame_id=sampled_frame_id,
        )
        self._by_int.append(coord)
        self._by_key[key] = int_id
        return int_id

    def get_by_int_id(self, int64_id: int) -> SampleCoordinate | None:
        if 0 <= int64_id < len(self._by_int):
            return self._by_int[int64_id]
        return None

    def get_by_key(self, video_id: str, ordinal: int) -> int | None:
        return self._by_key.get((video_id, ordinal))

    def to_parquet(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "int64_id": [c.int64_id for c in self._by_int],
            "video_id": [c.video_id for c in self._by_int],
            "timeline_revision": [c.timeline_revision for c in self._by_int],
            "ordinal": [c.ordinal for c in self._by_int],
            "timestamp_us": [c.timestamp_us for c in self._by_int],
            "timestamp_seconds": [c.timestamp_seconds for c in self._by_int],
            "sampled_frame_id": [c.sampled_frame_id for c in self._by_int],
        }
        table = pa.Table.from_pydict(data, schema=ID_MAP_SCHEMA)
        pq.write_table(table, str(p), compression="zstd")

    @classmethod
    def from_parquet(cls, path: Path | str) -> "IDMappingStore":
        p = Path(path)
        table = pq.read_table(str(p), schema=ID_MAP_SCHEMA)
        store = cls()
        int_ids = table.column("int64_id").to_pylist()
        vids = table.column("video_id").to_pylist()
        tl_revs = table.column("timeline_revision").to_pylist()
        ords = table.column("ordinal").to_pylist()
        ts_us = table.column("timestamp_us").to_pylist()
        ts_sec = table.column("timestamp_seconds").to_pylist()
        sample_ids = table.column("sampled_frame_id").to_pylist()

        for i in range(len(int_ids)):
            coord = SampleCoordinate(
                int64_id=int_ids[i],
                video_id=vids[i],
                timeline_revision=tl_revs[i],
                ordinal=ords[i],
                timestamp_us=ts_us[i],
                timestamp_seconds=ts_sec[i],
                sampled_frame_id=sample_ids[i],
            )
            store._by_int.append(coord)
            store._by_key[(vids[i], ords[i])] = int_ids[i]
        return store
