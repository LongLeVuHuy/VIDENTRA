"""Unit and API tests for improved Search/Playback UI and HTTP 206 media streaming."""
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from videntra.api.app import create_app
from videntra.api.media import DEFAULT_CHUNK_SIZE
from videntra.api.ui_html import UI_HTML, render_ui_html


class _TempMediaResolver:
    """Minimal resolver mapping video_id to temporary media files."""

    def __init__(self, files: dict[str, Path]):
        self._files = files

    def resolve(self, video_id: str, for_inference: bool = True) -> Path:
        if video_id not in self._files:
            raise KeyError(video_id)
        return self._files[video_id]


@pytest.fixture()
def media_client(tmp_path: Path):
    payload_mp4 = bytes((i % 251 for i in range(DEFAULT_CHUNK_SIZE * 2 + 12345)))
    payload_webm = b"WEBM_TEST_" * 300
    payload_mkv = b"MKV_TEST_" * 300

    mp4_path = tmp_path / "sample.mp4"
    webm_path = tmp_path / "sample.webm"
    mkv_path = tmp_path / "sample.mkv"
    mp4_path.write_bytes(payload_mp4)
    webm_path.write_bytes(payload_webm)
    mkv_path.write_bytes(payload_mkv)

    resolver = _TempMediaResolver({
        "vid_mp4": mp4_path,
        "vid_webm": webm_path,
        "vid_mkv": mkv_path,
    })
    app = create_app(resolver=resolver)
    client = TestClient(app)
    return client, {
        "vid_mp4": payload_mp4,
        "vid_webm": payload_webm,
        "vid_mkv": payload_mkv,
    }


def test_render_ui_html_and_root_endpoint(media_client):
    client, _ = media_client
    html_direct = render_ui_html()
    assert html_direct == UI_HTML

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Core video element & safe seek handler
    assert '<video id="player" controls preload="metadata"' in html
    assert "seekPlayerSafely" in html
    assert "loadedmetadata" in html
    assert "start_ms / 1000.0" in html

    # Playback & UX controls
    assert "Replay Moment" in html
    assert "momentJumpPills" in html
    assert "seekRelative(-5)" in html
    assert "seekRelative(5)" in html
    assert "playbackSpeedSelect" in html
    for rate in ("0.75x", "1x", "1.25x", "1.5x"):
        assert rate in html

    # Keyboard shortcuts & search/degraded/evidence UI
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert "degradedBanner" in html
    assert "verifyRelationsChk" in html
    assert "datasetInput" in html
    assert "backendSelect" in html
    assert "badge-ocr" in html
    assert "badge-asr" in html


def test_inspector_layout_prevents_video_shrinking_and_has_upgrades(media_client):
    client, _ = media_client
    html = client.get("/").text

    # Non-shrinking video container & aspect-ratio preservation
    assert "flex-shrink: 0" in html
    assert "aspect-ratio: 16 / 9" in html
    assert "object-fit: contain" in html
    assert "inspector-layout" in html

    # Theater mode, clickable timeline bar, and evidence modality tabs
    assert "theaterToggleBtn" in html
    assert "toggleTheaterMode" in html
    assert "timelineTrack" in html
    assert "timelineMomentWindow" in html
    assert "seekFromTimeline" in html
    assert "evidenceFilterTabs" in html
    assert "setEvidenceFilter" in html


def test_media_streaming_content_types_and_headers(media_client):
    client, payloads = media_client

    expected_types = {
        "vid_mp4": "video/mp4",
        "vid_webm": "video/webm",
        "vid_mkv": "video/x-matroska",
    }
    for vid, expected_mime in expected_types.items():
        resp = client.get(f"/v1/media/{vid}")
        assert resp.status_code == 200
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert "Cache-Control" in resp.headers
        assert resp.headers.get("Content-Type", "").startswith(expected_mime)
        assert int(resp.headers["Content-Length"]) == len(payloads[vid])
        assert resp.content == payloads[vid]

        head_resp = client.head(f"/v1/media/{vid}")
        assert head_resp.status_code == 200
        assert head_resp.headers.get("Accept-Ranges") == "bytes"
        assert int(head_resp.headers["Content-Length"]) == len(payloads[vid])
        assert head_resp.content == b""


def test_media_streaming_206_closed_open_and_suffix_ranges(media_client):
    client, payloads = media_client
    data = payloads["vid_mp4"]
    total = len(data)

    # Closed range: bytes=0-1023
    r1 = client.get("/v1/media/vid_mp4", headers={"Range": "bytes=0-1023"})
    assert r1.status_code == 206
    assert r1.headers.get("Accept-Ranges") == "bytes"
    assert r1.headers.get("Content-Range") == f"bytes 0-1023/{total}"
    assert int(r1.headers["Content-Length"]) == 1024
    assert r1.content == data[:1024]

    # Open range across multiple 512KB chunks: bytes=1024-
    r2 = client.get("/v1/media/vid_mp4", headers={"Range": "bytes=1024-"})
    assert r2.status_code == 206
    assert r2.headers.get("Content-Range") == f"bytes 1024-{total - 1}/{total}"
    assert int(r2.headers["Content-Length"]) == total - 1024
    assert r2.content == data[1024:]

    # Suffix range: bytes=-256
    r3 = client.get("/v1/media/vid_mp4", headers={"Range": "bytes=-256"})
    assert r3.status_code == 206
    assert r3.headers.get("Content-Range") == f"bytes {total - 256}-{total - 1}/{total}"
    assert int(r3.headers["Content-Length"]) == 256
    assert r3.content == data[-256:]


def test_media_streaming_416_invalid_ranges(media_client):
    client, payloads = media_client
    total = len(payloads["vid_mp4"])

    invalid_headers = [
        f"bytes={total}-",
        f"bytes={total + 100}-{total + 200}",
        "bytes=500-100",
        "bytes=invalid",
        "bytes=-0",
        "items=0-10",
    ]
    for hdr in invalid_headers:
        resp = client.get("/v1/media/vid_mp4", headers={"Range": hdr})
        assert resp.status_code == 416, f"Expected 416 for Range={hdr!r}, got {resp.status_code}"
        assert resp.headers.get("Content-Range") == f"bytes */{total}"
