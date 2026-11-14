"""Produce a review-only timestamp proposal; never modify an existing index.

The historical input has no source-media hashes. Matching by filename and even
matching PTS do not prove that local bytes generated the historical artifact.
The first-frame origin is an explicit hypothesis, not a verified source mapping.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

import av

from videntra.evaluation.provenance import sha256_file


def propose(outputs: Path, media_root: Path) -> dict:
    video_file = outputs / "videos.json"
    frame_file = outputs / "keyframe_manifest.json"
    hashes = {"videos_sha256": sha256_file(video_file), "keyframes_sha256": sha256_file(frame_file)}
    videos_list = json.loads(video_file.read_text(encoding="utf-8"))
    frames = json.loads(frame_file.read_text(encoding="utf-8"))
    videos = {v["video_id"]: v for v in videos_list}
    if len(videos) != len(videos_list):
        raise ValueError("Duplicate video IDs in artifact")
    groups: dict[str, list[dict]] = defaultdict(list)
    flagged = set()
    for row in frames:
        vid = row["video_id"]
        groups[vid].append(row)
        video = videos.get(vid, {})
        pts, num, den = row.get("pts"), video.get("time_base_num"), video.get("time_base_den")
        if pts is None or not num or not den or row.get("timestamp_ms") is None:
            flagged.add(vid)
        elif abs(Fraction(int(pts) * int(num) * 1000, int(den)) - Fraction(str(row["timestamp_ms"]))) > 1:
            flagged.add(vid)

    paths: dict[str, list[Path]] = defaultdict(list)
    for path in media_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}:
            paths[path.stem].append(path)

    proposals = []
    blocked = []
    decoded = 0
    for vid in sorted(flagged):
        files = paths.get(vid, [])
        if len(files) != 1:
            blocked.append({"video_id": vid, "reason": "missing_or_ambiguous_media"})
            continue
        path = files[0]
        media_hash = sha256_file(path)
        wanted: dict[int, list[dict]] = defaultdict(list)
        for row in groups[vid]:
            ordinal = int(row["source_frame_index"])
            if ordinal < 0:
                raise ValueError("Negative presentation ordinal")
            wanted[ordinal].append(row)
        local = []
        found = set()
        origin = None
        previous = None
        invalid_reason = None
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                for ordinal, frame in enumerate(container.decode(stream)):
                    if frame.pts is None or frame.time_base is None:
                        invalid_reason = "missing_decoded_pts_or_timebase"
                        break
                    timestamp = Fraction(frame.pts) * frame.time_base
                    if origin is None:
                        origin = timestamp
                    if previous is not None and timestamp < previous:
                        invalid_reason = "backward_pts_or_discontinuity"
                        break
                    previous = timestamp
                    if ordinal not in wanted:
                        continue
                    found.add(ordinal)
                    for row in wanted[ordinal]:
                        video = videos.get(vid, {})
                        if row.get("pts") is None or not video.get("time_base_num") or not video.get("time_base_den"):
                            invalid_reason = "artifact_pts_or_timebase_missing"
                            break
                        historical_pts = Fraction(int(row["pts"]) * int(video["time_base_num"]), int(video["time_base_den"]))
                        if historical_pts != timestamp:
                            invalid_reason = "artifact_pts_differs_from_decoded"
                            break
                        relative_ms = (timestamp - origin) * 1000
                        old_ms = Fraction(str(row["timestamp_ms"]))
                        if abs(relative_ms - old_ms) > 1:
                            local.append({"video_id": vid, "keyframe_id": row["keyframe_id"],
                                "presentation_ordinal": ordinal, "old_timestamp_ms": row["timestamp_ms"],
                                "proposed_timestamp_ms": round(relative_ms),
                                "clip_time_ms_num": relative_ms.numerator, "clip_time_ms_den": relative_ms.denominator,
                                "first_pts_seconds_num": origin.numerator, "first_pts_seconds_den": origin.denominator,
                                "decoded_pts": frame.pts, "decoded_time_base_num": frame.time_base.numerator,
                                "decoded_time_base_den": frame.time_base.denominator,
                                "local_media_sha256": media_hash,
                                "status": "REVIEW_REQUIRED_NOT_APPLIED"})
                    if invalid_reason:
                        break
        except (av.error.FFmpegError, OSError, IndexError, ValueError) as exc:
            invalid_reason = f"decode_error:{type(exc).__name__}"
        if sha256_file(path) != media_hash:
            invalid_reason = "local_media_changed_during_audit"
        if set(wanted) != found:
            invalid_reason = invalid_reason or "missing_decoded_ordinal"
        if invalid_reason:
            blocked.append({"video_id": vid, "reason": invalid_reason})
            continue
        decoded += 1
        proposals.extend(local)
    if hashes != {"videos_sha256": sha256_file(video_file), "keyframes_sha256": sha256_file(frame_file)}:
        raise RuntimeError("Historical inputs changed during audit")
    return {"status": "REVIEW_REQUIRED_NOT_APPLIED", "basis": "first_decoded_frame_pts_hypothesis",
            "input_hashes": hashes, "flagged_videos": len(flagged), "videos_decoded": decoded,
            "blocked_videos": blocked, "proposed_rows": proposals,
            "limitations": ["Filename association and local hashes do not prove historical source bytes.",
                            "Only videos with raw artifact PTS/timestamp discrepancies were selected.",
                            "First-frame origin, source-relative/edit/audio mapping and rights require review.",
                            "This file is a proposal; no index, database, evidence or media was changed."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = propose(args.outputs, args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "flagged_videos": result["flagged_videos"],
                      "videos_decoded": result["videos_decoded"], "blocked_videos": len(result["blocked_videos"]),
                      "proposed_rows": len(result["proposed_rows"]), "output": str(args.output)}))
