"""Unit tests for deterministic int64 identifier mapping."""
from pathlib import Path
import pytest

from videntra.indexing.ids import IDMappingStore, SampleCoordinate


def test_id_mapping_store_crud(tmp_path):
    store = IDMappingStore()
    assert len(store) == 0

    id0 = store.register(
        video_id="video_01",
        timeline_revision="rev1",
        ordinal=0,
        sampled_frame_id="sample_0",
        timestamp_us=0,
        timestamp_seconds=0.0,
    )
    assert id0 == 0
    assert len(store) == 1

    # Idempotent registration of same video + ordinal
    id0_again = store.register(
        video_id="video_01",
        timeline_revision="rev1",
        ordinal=0,
        sampled_frame_id="sample_0",
    )
    assert id0_again == 0
    assert len(store) == 1

    id1 = store.register(
        video_id="video_01",
        timeline_revision="rev1",
        ordinal=10,
        sampled_frame_id="sample_10",
        timestamp_us=333_333,
        timestamp_seconds=0.333333,
    )
    assert id1 == 1
    assert len(store) == 2

    coord = store.get_by_int_id(1)
    assert coord is not None
    assert coord.ordinal == 10
    assert coord.video_id == "video_01"

    assert store.get_by_key("video_01", 10) == 1
    assert store.get_by_key("video_01", 999) is None

    # Parquet persistence and reload
    out_file = tmp_path / "id_map.parquet"
    store.to_parquet(out_file)
    assert out_file.exists()

    loaded = IDMappingStore.from_parquet(out_file)
    assert len(loaded) == 2
    assert loaded.get_by_int_id(0).sampled_frame_id == "sample_0"
    assert loaded.get_by_int_id(1).ordinal == 10
