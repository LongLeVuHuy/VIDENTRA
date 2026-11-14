"""Exact frame extraction and thumbnail inspection endpoints."""
from fractions import Fraction
import io
from pathlib import Path
import re
import sqlite3

import av
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from PIL import Image

from videntra.timeline.ledger import exact_seek_frame


router = APIRouter(prefix="/v1", tags=["frames"])
_KEYFRAMES_ROOT = Path("data/raw/keyframes")


def _encode_jpeg_response(
    img_bytes: bytes,
    *,
    frame_index: int,
    pts: int | str | None,
    ts_us: str,
    revision: str,
) -> Response:
    headers = {
        "X-Frame-Index": str(frame_index),
        "X-PTS": str(pts if pts is not None else ""),
        "X-Timestamp-Us": ts_us,
        "X-Timeline-Revision": revision,
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400, immutable",
    }
    return Response(content=img_bytes, media_type="image/jpeg", headers=headers)


def _lookup_unified_keyframe(pipeline: object, video_id: str, index: int) -> tuple[Path, str, str] | None:
    """Look up a pre-extracted keyframe JPEG from pipeline DB or data/raw/keyframes."""
    safe_vid = re.sub(r'[<>:"/\\|?*]', "_", video_id)
    if safe_vid != video_id:
        return None

    db_path = getattr(pipeline, "db_path", None) if pipeline is not None else None
    if db_path and Path(db_path).exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT keyframe_path, pts, timestamp_ms FROM Keyframes WHERE video_id = ? AND source_frame_index = ? LIMIT 1",
                    (video_id, index),
                ).fetchone()
            if row and row[0]:
                kf_path = Path(row[0])
                if kf_path.is_file():
                    pts_str = "" if row[1] is None else str(row[1])
                    ts_us = "" if row[2] is None else str(int(row[2]) * 1000)
                    return kf_path, pts_str, ts_us
        except sqlite3.Error:
            pass

    if _KEYFRAMES_ROOT.exists():
        root = _KEYFRAMES_ROOT.resolve()
        candidate = (root / safe_vid / f"{index:06d}.jpg").resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate, "", ""
    return None


def _decode_frame_from_media(media_path: Path, target_ordinal: int) -> tuple[bytes, int | None, str]:
    """Decode zero-based frame ordinal directly from media_path via PyAV."""
    container = av.open(str(media_path))
    try:
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise ValueError("No video stream found")
        stream = video_streams[0]
        tb = stream.time_base
        for ordinal, frame in enumerate(container.decode(video=0)):
            if ordinal == target_ordinal:
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                ts_us = ""
                if frame.pts is not None and tb is not None:
                    sec = float(Fraction(frame.pts, 1) * tb)
                    ts_us = str(int(round(sec * 1_000_000)))
                return buf.getvalue(), frame.pts, ts_us
    finally:
        container.close()
    raise IndexError(f"Frame ordinal {target_ordinal} out of range")


@router.get("/frames/{video_id}/{index}")
def get_exact_frame(video_id: str, index: int, request: Request):
    """Retrieve exact decoded pixels for ordinal index with precision headers."""
    if index < 0:
        raise HTTPException(status_code=400, detail="Frame index must be non-negative")

    resolver = getattr(request.app.state, "resolver", None)
    ingest_dir = getattr(request.app.state, "ingest_dir", None)
    pipeline = getattr(request.app.state, "pipeline", None)

    if not resolver and not pipeline:
        raise HTTPException(status_code=503, detail="Resolver or pipeline not configured")

    media_path: Path | None = None
    if resolver:
        try:
            media_path = resolver.resolve(video_id, for_inference=True)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Media asset not found: {video_id}")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

    safe_vid = re.sub(r'[<>:"/\\|?*]', "_", video_id)
    if ingest_dir and media_path is not None:
        candidates = list(Path(ingest_dir).glob(f"timeline/{safe_vid}/*/frames.parquet"))
        if candidates:
            ledger_path = candidates[0]
            try:
                decoded = exact_seek_frame(media_path, ledger_path, target_ordinal=index)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Exact seek failed: {exc}")

            if decoded.pixels is None:
                raise HTTPException(status_code=500, detail="Frame pixels could not be decoded")

            img = Image.fromarray(decoded.pixels)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)

            ts_us = ""
            if decoded.pts is not None:
                try:
                    sec = float(Fraction(decoded.pts, 1) * Fraction(decoded.time_base_num, decoded.time_base_den))
                    ts_us = str(int(round(sec * 1_000_000)))
                except Exception:
                    pass

            return _encode_jpeg_response(
                buf.getvalue(),
                frame_index=decoded.ordinal,
                pts=decoded.pts,
                ts_us=ts_us,
                revision="rev1",
            )

    keyframe_hit = _lookup_unified_keyframe(pipeline, video_id, index)
    if keyframe_hit is not None:
        kf_path, pts_str, ts_us = keyframe_hit
        return _encode_jpeg_response(
            kf_path.read_bytes(),
            frame_index=index,
            pts=pts_str,
            ts_us=ts_us,
            revision="unified_keyframe",
        )

    if media_path is None:
        raise HTTPException(status_code=404, detail=f"No timeline ledger or media for {video_id}")

    try:
        img_bytes, pts, ts_us = _decode_frame_from_media(media_path, index)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Frame decode failed: {exc}")

    return _encode_jpeg_response(
        img_bytes,
        frame_index=index,
        pts=pts,
        ts_us=ts_us,
        revision="direct_decode",
    )


@router.get("/thumbnails/{sampled_frame_id}")
def get_thumbnail(sampled_frame_id: str, request: Request):
    """Retrieve thumbnail image by sampled_frame_id."""
    # Placeholder or exact frame extraction
    engine = getattr(request.app.state, "engine", None)
    if engine and hasattr(engine, "id_mapping"):
        # Match by sampled_frame_id in IDMappingStore
        for coord in engine.id_mapping._by_int:
            if coord.sampled_frame_id == sampled_frame_id:
                return get_exact_frame(coord.video_id, coord.ordinal, request)

    raise HTTPException(status_code=404, detail=f"Thumbnail not found for {sampled_frame_id}")
