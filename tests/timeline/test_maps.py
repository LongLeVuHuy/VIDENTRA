"""Unit tests for timeline mapping contracts."""
from fractions import Fraction
import pytest

from videntra.contracts.timing import microseconds
from videntra.timeline.maps import (
    RationalClipMap,
    PiecewiseSourceMap,
    SourceMapSegment,
    AudioMap,
    AudioMapEntry,
    build_audio_map,
)
from videntra.fixtures import build_fixtures
from videntra.ingestion.probe import probe_media


def test_rational_clip_map():
    origin = Fraction(1200, 1000)
    cmap = RationalClipMap(origin=origin, time_base_num=1, time_base_den=1000)
    # PTS = 1200 -> clip time 0
    assert cmap.pts_to_clip_time(1200) == Fraction(0)
    assert cmap.pts_to_us(1200) == 0
    assert cmap.pts_to_seconds(1200) == 0.0

    # PTS = 1550 -> clip time 0.35s = 350,000 us
    assert cmap.pts_to_clip_time(1550) == Fraction(350, 1000)
    assert cmap.pts_to_us(1550) == 350_000
    assert cmap.pts_to_seconds(1550) == 0.35


def test_piecewise_source_map_unknown():
    smap = PiecewiseSourceMap()
    assert smap.clip_to_source_us(100_000) is None
    assert smap.clip_to_source_seconds(0.1) is None


def test_piecewise_source_map_constant_offset():
    # offset = 210s = 210,000,000 us
    smap = PiecewiseSourceMap(
        source_video_id="source_v1",
        mapping_kind="constant_offset",
        source_offset_us=210_000_000,
    )
    assert smap.clip_to_source_us(14_400_000) == 224_400_000
    assert smap.clip_to_source_seconds(14.4) == 224.4


def test_piecewise_source_map_segments():
    seg1 = SourceMapSegment(
        clip_start_us=0,
        clip_end_us=10_000_000,
        source_start_us=50_000_000,
        rate_num=1,
        rate_den=1,
    )
    seg2 = SourceMapSegment(
        clip_start_us=10_000_000,
        clip_end_us=20_000_000,
        source_start_us=100_000_000,
        rate_num=2,
        rate_den=1,
    )
    smap = PiecewiseSourceMap(
        source_video_id="source_v2",
        mapping_kind="piecewise",
        segments=(seg1, seg2),
    )
    # in seg1: clip 5s -> source 55s
    assert smap.clip_to_source_us(5_000_000) == 55_000_000
    # in seg2: clip 15s (delta 5s * 2) -> source 100s + 10s = 110s
    assert smap.clip_to_source_us(15_000_000) == 110_000_000
    # outside segments -> None
    assert smap.clip_to_source_us(25_000_000) is None


def test_audio_map_build_and_jsonl_roundtrip(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    build_fixtures(fixtures_dir)
    audio_path = fixtures_dir / "audio_delay.mkv"
    probe_res = probe_media(audio_path)
    shared_origin = Fraction(0, 1)

    amap = build_audio_map(audio_path, probe_res, shared_origin)
    assert amap.audio_delay_us == 350_000
    assert len(amap.entries) > 0

    jsonl_path = tmp_path / "audio_map.jsonl"
    amap.to_jsonl(jsonl_path)

    loaded = AudioMap.from_jsonl(jsonl_path)
    assert loaded.audio_delay_us == amap.audio_delay_us
    assert len(loaded.entries) == len(amap.entries)
    assert loaded.entries[0].chunk_id == amap.entries[0].chunk_id
