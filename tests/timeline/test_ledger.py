"""Unit tests for TimelineLedger Parquet persistence and exact seek parity."""
from fractions import Fraction
from pathlib import Path
import pytest
import numpy as np

from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import decode_frames, DecodedFrame
from videntra.ingestion.probe import probe_media
from videntra.timeline.ledger import (
    write_timeline_ledger,
    read_frame_record,
    get_ledger_endpoints,
    exact_seek_frame,
)
from videntra.timeline.maps import PiecewiseSourceMap


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("timeline_fixtures")
    build_fixtures(out, include_extended=True)
    return out


def test_sequential_decode_and_ledger_cfr(fixture_dir, tmp_path):
    media_path = fixture_dir / "cfr.mkv"
    probe_res = probe_media(media_path)
    frames = list(decode_frames(media_path, probe_res, include_pixels=True))
    assert len(frames) == 5

    parquet_path = tmp_path / "cfr_frames.parquet"
    summary = write_timeline_ledger(
        parquet_path,
        frames,
        video_id="video_cfr",
        timeline_revision="rev_1",
        store_pixel_hashes=True,
    )

    assert summary.total_frames == 5
    assert summary.keyframe_count >= 1
    assert summary.epochs == ("epoch_0",)
    assert summary.missing_pts_count == 0
    assert summary.repeated_pts_count == 0

    row0 = read_frame_record(parquet_path, 0)
    assert row0["video_id"] == "video_cfr"
    assert row0["original_decoded_frame_index"] == 0
    assert row0["timestamp_us"] == 0
    assert row0["pixel_hash"] is not None

    row4 = read_frame_record(parquet_path, 4)
    assert row4["original_decoded_frame_index"] == 4
    assert row4["timestamp_us"] == 800_000


def test_exact_seek_parity_on_b_frames(fixture_dir, tmp_path):
    media_path = fixture_dir / "b_frames.mkv"
    probe_res = probe_media(media_path)
    frames = list(decode_frames(media_path, probe_res, include_pixels=True))
    assert len(frames) == 10

    parquet_path = tmp_path / "bf_frames.parquet"
    write_timeline_ledger(
        parquet_path,
        frames,
        video_id="video_bf",
        timeline_revision="rev_1",
        store_pixel_hashes=True,
    )

    # Test seek parity for multiple frame ordinals including B and P frames
    for target_ord in [0, 2, 4, 7, 9]:
        seeked = exact_seek_frame(media_path, parquet_path, target_ord, video_stream_index=0)
        expected = frames[target_ord]
        assert seeked.ordinal == expected.ordinal
        assert seeked.pts == expected.pts
        assert seeked.pixel_hash() == expected.pixel_hash()
        assert np.array_equal(seeked.pixels, expected.pixels)


def test_ledger_endpoints_query(fixture_dir, tmp_path):
    media_path = fixture_dir / "vfr.mkv"
    probe_res = probe_media(media_path)
    frames = list(decode_frames(media_path, probe_res, include_pixels=False))
    parquet_path = tmp_path / "vfr_frames.parquet"
    write_timeline_ledger(parquet_path, frames, video_id="video_vfr", timeline_revision="rev_1")

    # Observed clip times in fixture are 0.0s, 0.2s, 0.6s, 0.8s, 1.2s
    # Range [0.1s, 0.9s) -> frames at 0.2s, 0.6s, 0.8s -> ordinals (1, 3)
    start_ord, end_ord = get_ledger_endpoints(parquet_path, 100_000, 900_000)
    assert start_ord == 1
    assert end_ord == 3

    # Empty range raises ValueError
    with pytest.raises(ValueError, match="No observed frame in interval"):
        get_ledger_endpoints(parquet_path, 300_000, 500_000)


def test_repeated_pts_and_epoch_split_synthetic(tmp_path):
    # Pure record fixture to test semantic mapping without container normalization
    tb = Fraction(1, 1000)
    origin = Fraction(0)
    mock_frames = [
        DecodedFrame(ordinal=0, pts=0, dts=0, time_base_num=1, time_base_den=1000, origin=origin, pts_occurrence=0, epoch_id="epoch_0"),
        # Repeated PTS at ordinal 1
        DecodedFrame(ordinal=1, pts=0, dts=0, time_base_num=1, time_base_den=1000, origin=origin, pts_occurrence=1, epoch_id="epoch_0"),
        DecodedFrame(ordinal=2, pts=200, dts=200, time_base_num=1, time_base_den=1000, origin=origin, pts_occurrence=0, epoch_id="epoch_0"),
        # Backward PTS at ordinal 3 -> epoch_1
        DecodedFrame(ordinal=3, pts=100, dts=100, time_base_num=1, time_base_den=1000, origin=origin, pts_occurrence=0, epoch_id="epoch_1"),
    ]

    parquet_path = tmp_path / "mock_frames.parquet"
    summary = write_timeline_ledger(
        parquet_path,
        mock_frames,
        video_id="video_mock",
        timeline_revision="rev_mock",
    )

    assert summary.total_frames == 4
    assert summary.repeated_pts_count == 1
    assert summary.epochs == ("epoch_0", "epoch_1")

    r1 = read_frame_record(parquet_path, 1)
    assert r1["pts_occurrence"] == 1
    assert r1["epoch_id"] == "epoch_0"

    r3 = read_frame_record(parquet_path, 3)
    assert r3["pts_occurrence"] == 0
    assert r3["epoch_id"] == "epoch_1"
