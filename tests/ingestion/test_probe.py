"""Unit tests for M1-A probe module."""
import json
from pathlib import Path
import subprocess
import pytest

from videntra.fixtures import build_fixtures
from videntra.ingestion.probe import (
    probe_media,
    ProbeError,
    ProbeTimeoutError,
    StreamSelectionError,
    HashMismatchError,
)
from videntra.config.ingestion import StreamSelectionConfig


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures")
    build_fixtures(out)
    return out


def test_probe_media_cfr_no_audio(fixture_dir):
    cfr_path = fixture_dir / "cfr.mkv"
    res = probe_media(cfr_path)
    assert res.has_video is True
    assert res.has_audio is False
    assert res.video_stream is not None
    assert res.video_stream.width == 64
    assert res.video_stream.height == 48
    assert res.video_stream.codec_name == "ffv1"
    assert res.selected_video_stream_index == 0
    assert res.selected_audio_stream_index is None
    assert res.sha256 is not None
    assert res.duration_us is not None


def test_probe_media_audio_delay(fixture_dir):
    path = fixture_dir / "audio_delay.mkv"
    res = probe_media(path)
    assert res.has_video is True
    assert res.has_audio is True
    assert res.video_stream is not None
    assert res.audio_stream is not None
    assert res.audio_stream.sample_rate == 16000
    assert res.selected_video_stream_index == 0
    assert res.selected_audio_stream_index == 1


def test_probe_media_dict_input(fixture_dir):
    path = fixture_dir / "cfr.mkv"
    res1 = probe_media(path)
    asset_dict = {
        "local_path": str(path),
        "sha256": res1.sha256,
    }
    res2 = probe_media(asset_dict)
    assert res2.sha256 == res1.sha256


def test_probe_media_hash_mismatch(fixture_dir):
    path = fixture_dir / "cfr.mkv"
    asset_dict = {
        "local_path": str(path),
        "sha256": "0" * 64,
    }
    with pytest.raises(HashMismatchError, match="Media SHA-256 mismatch"):
        probe_media(asset_dict)


def test_probe_media_file_not_found(tmp_path):
    non_existent = tmp_path / "does_not_exist.mkv"
    with pytest.raises(FileNotFoundError):
        probe_media(non_existent)


def test_probe_explicit_stream_selection_valid(fixture_dir):
    path = fixture_dir / "audio_delay.mkv"
    sel = StreamSelectionConfig(video_stream_index=0, audio_stream_index=1)
    res = probe_media(path, stream_selection=sel)
    assert res.selected_video_stream_index == 0
    assert res.selected_audio_stream_index == 1


def test_probe_explicit_stream_selection_invalid_video(fixture_dir):
    path = fixture_dir / "cfr.mkv"
    sel = StreamSelectionConfig(video_stream_index=99)
    with pytest.raises(StreamSelectionError, match="Selected video stream index 99 not found"):
        probe_media(path, stream_selection=sel)


def test_probe_explicit_stream_selection_invalid_audio_type(fixture_dir):
    # Stream 0 is video, passing it as audio stream must fail
    path = fixture_dir / "cfr.mkv"
    sel = StreamSelectionConfig(audio_stream_index=0)
    with pytest.raises(StreamSelectionError, match="Selected audio stream index 0 not found or is not an audio stream"):
        probe_media(path, stream_selection=sel)


def test_probe_allow_missing_audio_false(fixture_dir):
    path = fixture_dir / "cfr.mkv"
    sel = StreamSelectionConfig(allow_missing_audio=False)
    with pytest.raises(StreamSelectionError, match="Audio stream required but container contains no audio"):
        probe_media(path, stream_selection=sel)


def test_probe_timeout(fixture_dir, monkeypatch):
    path = fixture_dir / "cfr.mkv"
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=0.01)
    monkeypatch.setattr(subprocess, "run", mock_run)
    with pytest.raises(ProbeTimeoutError, match="ffprobe timed out"):
        probe_media(path, timeout_seconds=0.01)


def test_probe_malformed_json(fixture_dir, monkeypatch):
    path = fixture_dir / "cfr.mkv"
    class MockProc:
        returncode = 0
        stdout = "NOT_JSON"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockProc())
    with pytest.raises(ProbeError, match="Malformed ffprobe JSON"):
        probe_media(path)


def test_probe_process_error(fixture_dir, monkeypatch):
    path = fixture_dir / "cfr.mkv"
    class MockProc:
        returncode = 1
        stdout = ""
        stderr = "Unknown codec error"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockProc())
    with pytest.raises(ProbeError, match="ffprobe returned code 1"):
        probe_media(path)
