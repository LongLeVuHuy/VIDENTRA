"""Media probe using ffprobe with argv lists, structured errors and explicit stream selection."""
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from videntra.config.ingestion import StreamSelectionConfig


class ProbeError(Exception):
    """Base exception for media probing failures."""
    pass


class ProbeTimeoutError(ProbeError):
    """Raised when probe subprocess times out."""
    pass


class StreamSelectionError(ProbeError):
    """Raised when explicit stream selection cannot be satisfied."""
    pass


class HashMismatchError(ProbeError):
    """Raised when media file content does not match expected SHA-256."""
    pass


class NoVideoStreamError(ProbeError):
    """Raised when video asset lacks any usable video stream."""
    pass


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec_name: str
    codec_type: str
    time_base_num: int
    time_base_den: int
    start_pts: int | None = None
    duration_pts: int | None = None
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    nb_frames: int | None = None
    sample_rate: int | None = None
    channels: int | None = None

    @property
    def time_base(self) -> Fraction:
        return Fraction(self.time_base_num, self.time_base_den)


@dataclass(frozen=True)
class ProbeResult:
    video_path: Path
    sha256: str
    container_format: str
    duration_seconds: float | None
    duration_us: int | None
    video_stream: StreamInfo | None
    audio_stream: StreamInfo | None
    has_video: bool
    has_audio: bool
    selected_video_stream_index: int | None
    selected_audio_stream_index: int | None
    raw_probe: dict[str, Any] = field(repr=False)


def _compute_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _parse_time_base(tb_str: str | None) -> tuple[int, int]:
    if not tb_str:
        return 1, 1000
    try:
        frac = Fraction(tb_str)
        return frac.numerator, frac.denominator
    except (ValueError, ZeroDivisionError):
        return 1, 1000


def probe_media(
    asset: Path | str | dict[str, Any],
    stream_selection: StreamSelectionConfig | dict[str, Any] | None = None,
    tools: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> ProbeResult:
    """Probe media container and streams using ffprobe."""
    if isinstance(asset, (Path, str)):
        media_path = Path(asset).resolve()
        expected_hash = None
    elif isinstance(asset, dict):
        if "local_path" in asset:
            media_path = Path(asset["local_path"]).resolve()
        elif "path" in asset:
            media_path = Path(asset["path"]).resolve()
        else:
            raise ProbeError("MediaAsset dict requires local_path or path key")
        expected_hash = asset.get("sha256")
    else:
        raise ProbeError(f"Unsupported asset type: {type(asset)}")

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    if not media_path.is_file():
        raise ProbeError(f"Media path is not a regular file: {media_path}")

    actual_hash = _compute_sha256(media_path)
    if expected_hash and actual_hash != expected_hash:
        raise HashMismatchError(
            f"Media SHA-256 mismatch for {media_path.name}: expected {expected_hash}, got {actual_hash}"
        )

    # Stream selection config
    if stream_selection is None:
        sel = StreamSelectionConfig()
    elif isinstance(stream_selection, dict):
        sel = StreamSelectionConfig(**stream_selection)
    elif isinstance(stream_selection, StreamSelectionConfig):
        sel = stream_selection
    else:
        raise ProbeError("Invalid stream_selection specification")

    ffprobe_bin = (tools or {}).get("ffprobe", "ffprobe")
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(media_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeTimeoutError(f"ffprobe timed out after {timeout_seconds}s for {media_path.name}") from exc
    except OSError as exc:
        raise ProbeError(f"Failed to execute ffprobe binary '{ffprobe_bin}': {exc}") from exc

    if proc.returncode != 0:
        raise ProbeError(f"ffprobe returned code {proc.returncode}: {proc.stderr.strip()}")

    try:
        raw_probe = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"Malformed ffprobe JSON output: {exc}") from exc

    streams = raw_probe.get("streams", [])
    format_info = raw_probe.get("format", {})
    container_format = format_info.get("format_name", "unknown")

    duration_sec = None
    if "duration" in format_info:
        try:
            duration_sec = float(format_info["duration"])
        except (ValueError, TypeError):
            pass
    duration_us = round(duration_sec * 1_000_000) if duration_sec is not None else None

    video_streams_by_idx = {}
    audio_streams_by_idx = {}

    for s in streams:
        idx = s.get("index")
        ctype = s.get("codec_type")
        tb_num, tb_den = _parse_time_base(s.get("time_base"))
        start_pts = int(s["start_pts"]) if "start_pts" in s and s["start_pts"] is not None else None
        duration_pts = int(s["duration_ts"]) if "duration_ts" in s and s["duration_ts"] is not None else None

        info = StreamInfo(
            index=idx,
            codec_name=s.get("codec_name", "unknown"),
            codec_type=ctype,
            time_base_num=tb_num,
            time_base_den=tb_den,
            start_pts=start_pts,
            duration_pts=duration_pts,
            width=s.get("width"),
            height=s.get("height"),
            avg_frame_rate=s.get("avg_frame_rate"),
            r_frame_rate=s.get("r_frame_rate"),
            nb_frames=int(s["nb_frames"]) if "nb_frames" in s and s["nb_frames"] is not None else None,
            sample_rate=int(s["sample_rate"]) if "sample_rate" in s and s["sample_rate"] is not None else None,
            channels=int(s["channels"]) if "channels" in s and s["channels"] is not None else None,
        )

        if ctype == "video":
            video_streams_by_idx[idx] = info
        elif ctype == "audio":
            audio_streams_by_idx[idx] = info

    # Video stream selection
    chosen_video: StreamInfo | None = None
    if sel.video_stream_index is not None:
        if sel.video_stream_index not in video_streams_by_idx:
            raise StreamSelectionError(
                f"Selected video stream index {sel.video_stream_index} not found or is not a video stream"
            )
        chosen_video = video_streams_by_idx[sel.video_stream_index]
    else:
        if video_streams_by_idx:
            # First video stream in order of appearance
            chosen_video = list(video_streams_by_idx.values())[0]

    # Audio stream selection
    chosen_audio: StreamInfo | None = None
    if sel.audio_stream_index is not None:
        if sel.audio_stream_index not in audio_streams_by_idx:
            raise StreamSelectionError(
                f"Selected audio stream index {sel.audio_stream_index} not found or is not an audio stream"
            )
        chosen_audio = audio_streams_by_idx[sel.audio_stream_index]
    else:
        if audio_streams_by_idx:
            chosen_audio = list(audio_streams_by_idx.values())[0]
        elif not sel.allow_missing_audio:
            raise StreamSelectionError("Audio stream required but container contains no audio")

    return ProbeResult(
        video_path=media_path,
        sha256=actual_hash,
        container_format=container_format,
        duration_seconds=duration_sec,
        duration_us=duration_us,
        video_stream=chosen_video,
        audio_stream=chosen_audio,
        has_video=chosen_video is not None,
        has_audio=chosen_audio is not None,
        selected_video_stream_index=chosen_video.index if chosen_video else None,
        selected_audio_stream_index=chosen_audio.index if chosen_audio else None,
        raw_probe=raw_probe,
    )
