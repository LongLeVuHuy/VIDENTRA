"""Read-only comparison of saved Kaggle timestamps against sequential PyAV decode.

Local filenames are used as a lookup, not proof that those bytes were used by
the historical Kaggle run. The audit never changes media or saved artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

import av


def audit(outputs: Path, media_root: Path, limit: int | None = None) -> dict:
    with (outputs / "videos.json").open(encoding="utf-8") as stream:
        videos = {row["video_id"]: row for row in json.load(stream)}
    with (outputs / "keyframe_manifest.json").open(encoding="utf-8") as stream:
        frames = json.load(stream)
    affected: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        video = videos.get(frame["video_id"], {})
        pts, num, den = frame.get("pts"), video.get("time_base_num"), video.get("time_base_den")
        if pts is not None and num and den:
            recorded_ms = Fraction(int(pts) * int(num) * 1000, int(den))
            if abs(recorded_ms - Fraction(str(frame["timestamp_ms"]))) > 1:
                affected[frame["video_id"]].append(frame)

    paths: dict[str, list[Path]] = defaultdict(list)
    for path in media_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}:
            paths[path.stem].append(path)
    report = {"affected_videos_in_artifact": len(affected), "video_limit": limit,
              "local_unique_files": 0, "missing_or_ambiguous_files": 0, "decode_errors": 0,
              "videos_decoded": 0, "videos_first_pts_nonzero_or_missing": 0,
              "videos_with_clip_timestamp_difference": 0,
              "selected_rows": 0, "missing_decoded_frames": 0, "missing_decoded_pts": 0,
              "artifact_pts_match_decoded": 0, "artifact_pts_differ_decoded": 0,
              "saved_timestamp_differs_clip_pts_gt_1ms": 0, "max_timestamp_difference_ms": 0.0}
    ordered = sorted(affected)
    if limit is not None:
        ordered = ordered[:limit]
    for video_id in ordered:
        matches = paths.get(video_id, [])
        if len(matches) != 1:
            report["missing_or_ambiguous_files"] += 1
            continue
        report["local_unique_files"] += 1
        wanted: dict[int, list[dict]] = defaultdict(list)
        for row in affected[video_id]:
            wanted[int(row["source_frame_index"])].append(row)
        report["selected_rows"] += sum(map(len, wanted.values()))
        found: set[int] = set()
        video_has_timestamp_difference = False
        try:
            with av.open(str(matches[0])) as container:
                stream = container.streams.video[0]
                first_pts_seconds = None
                for ordinal, frame in enumerate(container.decode(stream)):
                    if ordinal == 0 and frame.pts is not None and frame.time_base:
                        first_pts_seconds = Fraction(frame.pts) * frame.time_base
                    if ordinal not in wanted:
                        continue
                    found.add(ordinal)
                    for row in wanted[ordinal]:
                        if frame.pts is None or frame.time_base is None or first_pts_seconds is None:
                            report["missing_decoded_pts"] += 1
                            continue
                        current_seconds = Fraction(frame.pts) * frame.time_base
                        video = videos[video_id]
                        saved_seconds = Fraction(int(row["pts"]) * int(video["time_base_num"]), int(video["time_base_den"]))
                        key = "artifact_pts_match_decoded" if current_seconds == saved_seconds else "artifact_pts_differ_decoded"
                        report[key] += 1
                        error = abs((current_seconds - first_pts_seconds) * 1000 - Fraction(str(row["timestamp_ms"])))
                        if error > 1:
                            video_has_timestamp_difference = True
                            report["saved_timestamp_differs_clip_pts_gt_1ms"] += 1
                            report["max_timestamp_difference_ms"] = max(report["max_timestamp_difference_ms"], float(error))
                report["videos_decoded"] += 1
                if video_has_timestamp_difference:
                    report["videos_with_clip_timestamp_difference"] += 1
                if first_pts_seconds is None or first_pts_seconds != 0:
                    report["videos_first_pts_nonzero_or_missing"] += 1
        except (av.error.FFmpegError, IndexError, OSError):
            report["decode_errors"] += 1
        report["missing_decoded_frames"] += sum(len(wanted[index]) for index in wanted.keys() - found)
    report["max_timestamp_difference_ms"] = round(report["max_timestamp_difference_ms"], 3)
    report["limitations"] = ("Filename identity is not a verified byte hash. Only artifact rows already flagged "
                             "by PTS/time-base mismatch were decoded; this is not a whole-corpus ledger audit.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    print(json.dumps(audit(args.outputs, args.media_root, args.limit), indent=2))
