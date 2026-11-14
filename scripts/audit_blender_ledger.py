"""Build and independently verify sequential Blender pilot frame ledgers.

This is a media/timestamp audit, not an embedding or retrieval benchmark.
The input receipts identify bytes and licenses; no query labels are read.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

import av
import pyarrow as pa
import pyarrow.parquet as pq

from videntra.contracts.timing import microseconds
from videntra.ingestion.decode import decode_frames
from videntra.timeline.ledger import write_timeline_ledger
from videntra.evaluation.provenance import sha256_file


def verify_ledger(media: Path, ledger: Path) -> dict:
    """Re-decode in presentation order and compare ordinal, rational PTS and clip time."""
    fields = ["original_decoded_frame_index", "presentation_timestamp", "time_base_num",
              "time_base_den", "timestamp_us"]
    table = pq.read_table(ledger, columns=fields)
    rows = iter(table.to_pylist())
    seen = errors = missing = backwards = repeated = 0
    origin = previous = None
    with av.open(str(media)) as container:
        stream = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(stream)):
            row = next(rows, None)
            if row is None:
                errors += 1
                continue
            seen += 1
            if frame.pts is None or frame.time_base is None:
                missing += 1
                if row["presentation_timestamp"] is not None:
                    errors += 1
                continue
            pts = Fraction(frame.pts) * frame.time_base
            if origin is None:
                origin = pts
            if previous is not None:
                backwards += int(pts < previous)
                repeated += int(pts == previous)
            previous = pts
            ledger_pts = row["presentation_timestamp"]
            ledger_tb = Fraction(row["time_base_num"], row["time_base_den"])
            if (row["original_decoded_frame_index"] != ordinal or ledger_pts is None or
                    Fraction(ledger_pts) * ledger_tb != pts or row["timestamp_us"] != microseconds(pts - origin)):
                errors += 1
    remaining = sum(1 for _ in rows)
    return {"decoded_frames": seen, "ledger_rows": table.num_rows,
            "missing_pts": missing, "backward_pts": backwards, "repeated_pts": repeated,
            "mismatched_rows": errors + remaining,
            "first_pts_seconds": str(origin) if origin is not None else None}


def audit(media_dir: Path, output: Path) -> dict:
    receipts = [json.loads(line) for line in (media_dir / "source_receipts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not receipts or len({r["video_id"] for r in receipts}) != len(receipts):
        raise ValueError("Empty receipts or duplicate video IDs")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for receipt in receipts:
        media = media_dir / receipt["name"]
        actual = sha256_file(media)
        if actual != receipt["sha256"]:
            raise ValueError(f"Media hash mismatch: {media.name}")
        ledger = output / (receipt["video_id"] + ".parquet")
        summary = write_timeline_ledger(ledger, decode_frames(media, include_pixels=False),
            video_id=receipt["video_id"], timeline_revision="blender_sequential_audit_v1")
        verification = verify_ledger(media, ledger)
        if sha256_file(media) != actual:
            raise RuntimeError("Media changed during audit")
        results.append({"video_id": receipt["video_id"], "media_sha256": actual,
                        "ledger_sha256": sha256_file(ledger), "frames": summary.total_frames,
                        "missing_pts": summary.missing_pts_count, "repeated_pts": summary.repeated_pts_count,
                        "epochs": list(summary.epochs), "verification": verification})
    report = {"status": "verified" if all(not r["verification"]["mismatched_rows"] and
              not r["verification"]["missing_pts"] for r in results) else "incomplete",
              "assets": results, "python": sys.version, "platform": platform.platform(),
              "pyav": av.__version__, "pyarrow": pa.__version__,
              "limitations": ["First valid video PTS is the declared clip origin; source-relative mapping and audio synchronization remain unverified.",
                              "Metadata parity does not assess retrieval accuracy or bit-exact seek."]}
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.media_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
