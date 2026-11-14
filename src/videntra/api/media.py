"""Media playback and streaming endpoints with HTTP 206 Partial Content support."""
import mimetypes
import os
from pathlib import Path
import re
from typing import Generator

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import Response, StreamingResponse
import pyarrow.parquet as pq


router = APIRouter(prefix="/v1", tags=["media"])


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse 'bytes=start-end' into (start, end) offsets."""
    if not range_header.startswith("bytes="):
        raise ValueError("Invalid Range header format")
    parts = range_header[6:].split("-")
    start_str = parts[0].strip()
    end_str = parts[1].strip() if len(parts) > 1 else ""

    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
    elif start_str:
        start = int(start_str)
        end = file_size - 1
    elif end_str:
        start = max(0, file_size - int(end_str))
        end = file_size - 1
    else:
        start = 0
        end = file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    return start, end


def _iter_file_chunk(path: Path, start: int, length: int, chunk_size: int = 1024 * 64) -> Generator[bytes, None, None]:
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


@router.get("/media/{video_id}")
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
        raise HTTPException(status_code=404, detail=f"Media file missing on storage")

    file_size = media_path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(media_path))
    if not mime_type:
        mime_type = "video/mp4" if media_path.suffix.lower() == ".mp4" else "video/x-matroska"

    range_header = request.headers.get("Range") or request.headers.get("range")
    if range_header:
        try:
            start, end = _parse_byte_range(range_header, file_size)
        except Exception:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")

        content_length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": mime_type,
        }
        return StreamingResponse(
            _iter_file_chunk(media_path, start, content_length),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
            media_type=mime_type,
        )

    # Full file stream
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": mime_type,
    }
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
