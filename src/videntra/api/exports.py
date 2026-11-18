"""Export endpoint supporting CSV and JSON with formula-safe cell escaping."""
import csv
import io
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field


router = APIRouter(prefix="/v1", tags=["exports"])


class ExportRequest(BaseModel):
    results: list[dict[str, Any]]
    format: str = Field(default="csv", description="csv or json")
    snapshot_id: str = Field(default="corpus_current")


def _safe_escape_csv_cell(value: Any) -> str:
    """Escape spreadsheet formula triggers (=, +, -, @) to prevent injection."""
    s = str(value) if value is not None else ""
    if s and s[0] in ("=", "+", "-", "@"):
        return "'" + s
    return s


@router.post("/exports")
def export_results(req: ExportRequest):
    """Export ranked moment results to generic CSV or full-fidelity JSON."""
    fmt = req.format.lower().strip()
    if fmt == "json":
        data = {
            "snapshot_id": req.snapshot_id,
            "count": len(req.results),
            "results": req.results,
        }
        return JSONResponse(content=data)

    if fmt not in ("csv", "aic_kis", "vbs_csv"):
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {req.format}")

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    if fmt == "aic_kis":
        for item in req.results:
            frame_idx = item.get("representative_original_frame_index")
            if frame_idx is None:
                frame_idx = item.get("start_original_frame_index", 0)
            writer.writerow([
                _safe_escape_csv_cell(item.get("video_id")),
                _safe_escape_csv_cell(frame_idx),
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="videntra_aic_kis.csv"'},
        )

    if fmt == "vbs_csv":
        writer.writerow(["video_id", "start_ms", "end_ms", "frame_index", "score"])
        for item in req.results:
            st_ms = item.get("start_ms")
            if st_ms is None:
                st_ms = int(round(float(item.get("moment_start_seconds", 0.0) or 0.0) * 1000.0))
            et_ms = item.get("end_ms")
            if et_ms is None:
                et_ms = int(round(float(item.get("moment_end_seconds", 0.0) or 0.0) * 1000.0))
            frame_idx = item.get("representative_original_frame_index")
            if frame_idx is None:
                frame_idx = item.get("start_original_frame_index", 0)
            writer.writerow([
                _safe_escape_csv_cell(item.get("video_id")),
                _safe_escape_csv_cell(st_ms),
                _safe_escape_csv_cell(et_ms),
                _safe_escape_csv_cell(frame_idx),
                _safe_escape_csv_cell(item.get("ranking_score")),
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="videntra_vbs.csv"'},
        )

    # Generic full CSV headers
    headers = [
        "rank",
        "video_id",
        "source_video_id",
        "start_seconds",
        "end_seconds",
        "start_original_frame_index",
        "end_original_frame_index",
        "representative_original_frame_index",
        "frame_index_base",
        "coordinate_system",
        "ranking_score",
        "score_type",
        "match_status",
        "snapshot_id",
    ]

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)

    for rank, item in enumerate(req.results, 1):
        timing = item.get("timing_metadata", {})
        row = [
            _safe_escape_csv_cell(rank),
            _safe_escape_csv_cell(item.get("video_id")),
            _safe_escape_csv_cell(item.get("source_video_id")),
            _safe_escape_csv_cell(item.get("moment_start_seconds")),
            _safe_escape_csv_cell(item.get("moment_end_seconds")),
            _safe_escape_csv_cell(item.get("start_original_frame_index")),
            _safe_escape_csv_cell(item.get("end_original_frame_index")),
            _safe_escape_csv_cell(item.get("representative_original_frame_index")),
            _safe_escape_csv_cell(timing.get("frame_index_base", 0)),
            _safe_escape_csv_cell(timing.get("coordinate_system", "clip_relative")),
            _safe_escape_csv_cell(item.get("ranking_score")),
            _safe_escape_csv_cell(item.get("score_type", "uncalibrated_ranking")),
            _safe_escape_csv_cell(item.get("match_status", "supported")),
            _safe_escape_csv_cell(req.snapshot_id),
        ]
        writer.writerow(row)

    csv_text = output.getvalue()
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="videntra_results.csv"'},
    )
