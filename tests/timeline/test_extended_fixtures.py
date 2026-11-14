"""Unit tests for extended media fixtures."""
from pathlib import Path
import pytest

from videntra.fixtures import build_fixtures
from videntra.ingestion.probe import probe_media
from videntra.ingestion.decode import decode_frames
from videntra.timeline.maps import build_audio_map
from fractions import Fraction


@pytest.fixture(scope="module")
def extended_fixtures(tmp_path_factory):
    dest = tmp_path_factory.mktemp("ext_fixtures")
    receipt = build_fixtures(dest, include_extended=True)
    return dest, receipt


def test_extended_fixtures_build(extended_fixtures):
    dest, receipt = extended_fixtures
    assert len(receipt["fixtures"]) == 9
    names = {f["name"] for f in receipt["fixtures"]}
    assert "b_frames" in names
    assert "audio_preroll" in names
    assert "no_audio" in names
    assert "corrupt_tail" in names
    assert "repeated_pts" in names


def test_audio_preroll_negative_origin(extended_fixtures):
    dest, _ = extended_fixtures
    path = dest / "audio_preroll.mkv"
    probe_res = probe_media(path)
    assert probe_res.has_audio is True

    shared_origin = Fraction(0, 1)
    amap = build_audio_map(path, probe_res, shared_origin)
    # Preroll is -0.2s = -200,000 us
    assert amap.audio_delay_us == -200_000
    assert amap.entries[0].start_us == -200_000


def test_no_audio_fixture(extended_fixtures):
    dest, _ = extended_fixtures
    path = dest / "no_audio.mkv"
    probe_res = probe_media(path)
    assert probe_res.has_video is True
    assert probe_res.has_audio is False
    amap = build_audio_map(path, probe_res, Fraction(0))
    assert amap.audio_stream_index is None
    assert len(amap.entries) == 0


def test_b_frames_presentation_order(extended_fixtures):
    dest, _ = extended_fixtures
    path = dest / "b_frames.mkv"
    probe_res = probe_media(path)
    frames = list(decode_frames(path, probe_res, include_pixels=False))
    # Check that ordinals are monotonically increasing zero-based
    ordinals = [f.ordinal for f in frames]
    assert ordinals == list(range(len(frames)))
    # Check that PTS is monotonically increasing in presentation order
    pts_list = [f.pts for f in frames]
    assert pts_list == sorted(pts_list)
    # Check that B-frames are present
    pict_types = {f.pict_type for f in frames}
    assert "B" in pict_types
