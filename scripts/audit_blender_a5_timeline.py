"""Read-only audit of Blender A5 output against media and a sequential video ledger."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import av
import numpy as np
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def first_ffprobe_frame(path: Path, stream: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", stream, "-read_intervals", "%+#1",
           "-show_frames", "-show_entries", "frame=pts,pts_time,nb_samples", "-of", "json", str(path)]
    return json.loads(subprocess.check_output(cmd))["frames"][0]


def audit(bundle: Path, outputs: Path, ledger: Path) -> dict:
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = json.loads((outputs / "keyframe_manifest.json").read_text(encoding="utf-8"))
    segments = json.loads((outputs / "asr_segments.json").read_text(encoding="utf-8"))
    vectors = np.load(outputs / "frame_embeddings.npy", allow_pickle=False)
    video_results = []
    for asset in manifest["assets"]:
        name = asset["video_id"]
        path = bundle / asset["name"]
        actual_hash = sha256(path)
        if actual_hash != asset["sha256"]:
            raise ValueError(f"Media checksum mismatch: {name}")
        with av.open(str(path)) as container:
            video_stream = container.streams.video[0]
            video_first = next(container.decode(video_stream))
            video_first_time = Fraction(video_first.pts) * video_first.time_base
        first = end = None
        count = missing = gaps = 0
        rates: Counter[int] = Counter()
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                count += 1
                if frame.pts is None or frame.time_base is None or not frame.sample_rate:
                    missing += 1
                    continue
                start = Fraction(frame.pts) * frame.time_base
                if first is None:
                    first = start
                if end is not None and start != end:
                    gaps += 1
                end = start + Fraction(frame.samples, frame.sample_rate)
                rates[frame.sample_rate] += 1
        if first is None or end is None:
            raise ValueError(f"No decoded audio timestamps: {name}")
        table = pq.read_table(ledger / f"{name}.parquet", columns=[
            "original_decoded_frame_index", "presentation_timestamp", "timestamp_us"])
        by_index = {r["original_decoded_frame_index"]: r for r in table.to_pylist()}
        selected = [r for r in frames if r["video_id"] == name]
        missing_indices = sum(r["source_frame_index"] not in by_index for r in selected)
        mismatches = sum(
            r["pts"] != by_index[r["source_frame_index"]]["presentation_timestamp"]
            or abs(r["timestamp_ms"] * 1000 - by_index[r["source_frame_index"]]["timestamp_us"]) > 1000
            for r in selected if r["source_frame_index"] in by_index)
        spoken = [r for r in segments if r["video_id"] == name]
        bad_intervals = sum(not (0 <= r["start_ms"] < r["end_ms"] <= float(end) * 1000 + 1)
                            for r in spoken)
        video_results.append({
            "video_id": name, "media_sha256": actual_hash,
            "video_first_pts_seconds": float(video_first_time),
            "audio_first_pts_seconds": float(first), "audio_end_seconds": float(end),
            "audio_frames": count, "audio_sample_rates": dict(rates),
            "audio_missing_pts": missing, "audio_discontinuities": gaps,
            "ffprobe_first_video": first_ffprobe_frame(path, "v:0"),
            "ffprobe_first_audio": first_ffprobe_frame(path, "a:0"),
            "keyframes": len(selected), "missing_ledger_ordinals": missing_indices,
            "keyframe_pts_or_timestamp_mismatches": mismatches,
            "asr_segments": len(spoken),
            "asr_start_min_ms": min((r["start_ms"] for r in spoken), default=None),
            "asr_end_max_ms": max((r["end_ms"] for r in spoken), default=None),
            "asr_bad_intervals": bad_intervals,
        })
    return {
        "input_sha256": {"bundle_manifest": sha256(manifest_path),
                         "keyframe_manifest": sha256(outputs / "keyframe_manifest.json"),
                         "asr_segments": sha256(outputs / "asr_segments.json"),
                         "frame_embeddings": sha256(outputs / "frame_embeddings.npy")},
        "pyav_version": av.__version__,
        "ffprobe_version": subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0],
        "vector_shape": list(vectors.shape), "vectors_finite": bool(np.isfinite(vectors).all()),
        "global_indices_contiguous": sorted(r["global_index"] for r in frames) == list(range(len(frames))),
        "videos": video_results,
        "limitations": ["Range and PTS continuity do not prove ASR words or individual boundaries are correct.",
                        "VAD chunks were not recorded; independent restoration of each timestamp is not possible.",
                        "No source-video offset is established; this audit is limited to local clip time."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.outputs, args.ledger), indent=2))


if __name__ == "__main__":
    main()
