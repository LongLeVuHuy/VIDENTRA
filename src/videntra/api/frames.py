"""Exact frame extraction and thumbnail inspection endpoints."""
import io
from pathlib import Path
import re

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from PIL import Image

from videntra.timeline.ledger import exact_seek_frame


router = APIRouter(prefix="/v1", tags=["frames"])


@router.get("/frames/{video_id}/{index}")
def get_exact_frame(video_id: str, index: int, request: Request):
    """Retrieve exact decoded pixels for ordinal index with precision headers."""
    resolver = getattr(request.app.state, "resolver", None)
    ingest_dir = getattr(request.app.state, "ingest_dir", None)

    if not resolver or not ingest_dir:
        raise HTTPException(status_code=503, detail="Resolver or ingest directory not configured")

    try:
        media_path = resolver.resolve(video_id, for_inference=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Media asset not found: {video_id}")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    safe_vid = re.sub(r'[<>:"/\\|?*]', '_', video_id)
    candidates = list(Path(ingest_dir).glob(f"timeline/{safe_vid}/*/frames.parquet"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No timeline ledger for {video_id}")

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
    img_bytes = buf.getvalue()

    ts_us = ""
    if decoded.pts is not None:
        try:
            from fractions import Fraction
            sec = float(Fraction(decoded.pts, 1) * Fraction(decoded.time_base_num, decoded.time_base_den))
            ts_us = str(int(round(sec * 1_000_000)))
        except Exception:
            pass

    headers = {
        "X-Frame-Index": str(decoded.ordinal),
        "X-PTS": str(decoded.pts if decoded.pts is not None else ""),
        "X-Timestamp-Us": ts_us,
        "X-Timeline-Revision": "rev1",
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400, immutable",
    }
    return Response(content=img_bytes, media_type="image/jpeg", headers=headers)


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
