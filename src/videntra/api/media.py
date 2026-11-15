"""Media playback and streaming endpoints with HTTP 206 Partial Content support."""
import mimetypes
from pathlib import Path
import re
from typing import Generator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
import pyarrow.parquet as pq


router = APIRouter(prefix="/v1", tags=["media"])

DEFAULT_CHUNK_SIZE = 512 * 1024  # 512 KB chunks for smooth browser video scrubbing

_EXT_MIME_MAP: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


def _resolve_mime_type(media_path: Path) -> str:
    """Resolve accurate Content-Type for common video containers."""
    ext = media_path.suffix.lower()
    if ext in _EXT_MIME_MAP:
        return _EXT_MIME_MAP[ext]
    guessed, _ = mimetypes.guess_type(str(media_path))
    return guessed or "application/octet-stream"


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse HTTP Range header ('bytes=start-end', 'bytes=start-', 'bytes=-suffix')."""
    if file_size <= 0:
        raise ValueError("Cannot satisfy byte range on empty file")

    header = range_header.strip()
    if not header.lower().startswith("bytes="):
        raise ValueError("Invalid Range header unit")

    spec = header[6:].strip()
    if not spec or "," in spec:
        raise ValueError("Unsupported or empty byte range specification")

    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError("Malformed byte range specification")

    start_str, end_str = parts[0].strip(), parts[1].strip()
    if not start_str and not end_str:
        raise ValueError("Empty byte range bounds")

    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
        if start < 0 or end < start or start >= file_size:
            raise ValueError("Byte range out of bounds")
        end = min(end, file_size - 1)
    elif start_str:
        start = int(start_str)
        if start < 0 or start >= file_size:
            raise ValueError("Byte range start out of bounds")
        end = file_size - 1
    else:
        suffix_len = int(end_str)
        if suffix_len <= 0:
            raise ValueError("Suffix byte range must be positive")
        start = max(0, file_size - suffix_len)
        end = file_size - 1

    return start, end


def _iter_file_chunk(
    path: Path,
    start: int,
    length: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[bytes, None, None]:
    """Yield file byte chunks within [start, start + length)."""
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            chunk = f.read(read_size)
            if not chunk:
                break
            yield chunk
            remaining -= len(chunk)


@router.api_route("/media/{video_id}", methods=["GET", "HEAD"])
def stream_media(video_id: str, request: Request):
    """Serve media with HTTP Range 206 partial content for HTML5 video player."""
    resolver = getattr(request.app.state, "resolver", None)
    if resolver is None:
        raise HTTPException(status_code=503, detail="Media resolver not configured")

    try:
        media_path = resolver.resolve(video_id, for_inference=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Media asset not found: {video_id}")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="Media file missing on storage")

    file_size = media_path.stat().st_size
    mime_type = _resolve_mime_type(media_path)
    is_head = request.method.upper() == "HEAD"

    base_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
        "Content-Type": mime_type,
    }

    range_header = request.headers.get("Range") or request.headers.get("range")
    if range_header:
        try:
            start, end = _parse_byte_range(range_header, file_size)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="Requested range not satisfiable",
                headers={
                    "Content-Range": f"bytes */{file_size}",
                    "Accept-Ranges": "bytes",
                },
            )

        content_length = end - start + 1
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
        }
        if is_head:
            return Response(
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
                media_type=mime_type,
            )
        return StreamingResponse(
            _iter_file_chunk(media_path, start, content_length),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
            media_type=mime_type,
        )

    headers = {
        **base_headers,
        "Content-Length": str(file_size),
    }
    if is_head:
        return Response(
            status_code=status.HTTP_200_OK,
            headers=headers,
            media_type=mime_type,
        )
    return StreamingResponse(
        _iter_file_chunk(media_path, 0, file_size),
        status_code=status.HTTP_200_OK,
        headers=headers,
        media_type=mime_type,
    )


@router.get("/media/{video_id}/timeline")
def get_media_timeline(video_id: str, request: Request, around_us: int | None = None, window_us: int = 2_000_000):
    """Get neighboring timeline records around a specific timestamp."""
    ingest_dir = getattr(request.app.state, "ingest_dir", None)
    if not ingest_dir:
        raise HTTPException(status_code=503, detail="Ingest directory not configured")

    safe_vid = re.sub(r'[<>:"/\\|?*]', '_', video_id)
    candidates = list(Path(ingest_dir).glob(f"timeline/{safe_vid}/*/frames.parquet"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No timeline ledger for {video_id}")

    ledger_path = candidates[0]
    table = pq.read_table(str(ledger_path))

    ords = table.column("original_decoded_frame_index").to_pylist()
    pts_list = table.column("presentation_timestamp").to_pylist()
    ts_us_list = table.column("timestamp_us").to_pylist()
    keyflags = table.column("is_keyframe").to_pylist()

    records = []
    for i in range(len(ords)):
        us = ts_us_list[i]
        if around_us is not None and us is not None:
            if abs(us - around_us) > window_us:
                continue
        records.append({
            "ordinal": ords[i],
            "presentation_timestamp": pts_list[i],
            "timestamp_us": us,
            "timestamp_seconds": (us / 1_000_000.0) if us is not None else None,
            "is_keyframe": bool(keyflags[i]),
        })

    return {
        "video_id": video_id,
        "timeline_revision": "rev1",
        "total_records": len(records),
        "records": records,
    }
