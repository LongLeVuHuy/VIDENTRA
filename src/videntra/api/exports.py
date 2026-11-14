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
    if req.format.lower() == "json":
        data = {
            "snapshot_id": req.snapshot_id,
            "count": len(req.results),
            "results": req.results,
        }
        return JSONResponse(content=data)

    if req.format.lower() != "csv":
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {req.format}")

    # CSV headers
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
