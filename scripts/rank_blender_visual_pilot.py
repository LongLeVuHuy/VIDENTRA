"""Rank a fixed pilot corpus from GPU embeddings without reading any gold labels."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from videntra.evaluation.provenance import sha256_file


def rank(gpu_dir: Path, ledger_dir: Path, output: Path, *, query_input: Path, run_config: Path,
         window_seconds: float = 10.0) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    manifest = json.loads((gpu_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads(run_config.read_text(encoding="utf-8"))
    bundle_path = run_config.parent / "bundle_manifest.json"
    if sha256_file(bundle_path) != config.get("manifest_sha256"):
        raise ValueError("Local bundle manifest checksum mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (manifest.get("model") != "google/siglip2-base-patch16-224" or
        manifest.get("revision") != "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2" or
        manifest.get("weights_sha256") != "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b" or
        manifest.get("device") != "cuda" or manifest.get("worker_sha256") != config.get("worker_sha256") or
        manifest.get("input_manifest_sha256") != config.get("manifest_sha256") or
        manifest.get("query_input_sha256") != config.get("queries_sha256") or
        sha256_file(query_input) != config.get("queries_sha256")):
        raise ValueError("GPU run provenance or query input mismatch")
    required = {"frames.npy", "queries.npy", "coordinates.json", "query_ids.json"}
    if set(manifest.get("output_sha256", {})) != required:
        raise ValueError("Incomplete GPU artifact manifest")
    for name, expected in manifest["output_sha256"].items():
        if sha256_file(gpu_dir / name) != expected:
            raise ValueError(f"GPU artifact hash mismatch: {name}")
    coords = json.loads((gpu_dir / "coordinates.json").read_text(encoding="utf-8"))
    qids = json.loads((gpu_dir / "query_ids.json").read_text(encoding="utf-8"))
    input_queries = json.loads(query_input.read_text(encoding="utf-8"))
    if any(set(q) != {"query_id", "text"} for q in input_queries) or qids != [q["query_id"] for q in input_queries]:
        raise ValueError("GPU query IDs differ from frozen query input")
    images = np.load(gpu_dir / "frames.npy", allow_pickle=False)
    queries = np.load(gpu_dir / "queries.npy", allow_pickle=False)
    if (manifest.get("frame_count") != len(coords) or manifest.get("query_count") != len(qids) or
            len(set(qids)) != len(qids) or images.shape != (len(coords), 768) or
            queries.shape != (len(qids), 768) or not np.isfinite(images).all() or not np.isfinite(queries).all()):
        raise ValueError("Invalid IDs or embedding matrices")
    if not np.allclose(np.linalg.norm(images, axis=1), 1.0, atol=1e-3) or not np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-3):
        raise ValueError("Embeddings are not normalized")
    audit = json.loads((ledger_dir / "audit.json").read_text(encoding="utf-8"))
    if ({row["video_id"]: row["sha256"] for row in bundle["assets"]} !=
            {row["video_id"]: row["media_sha256"] for row in audit["assets"]}):
        raise ValueError("GPU input media and local ledger audit refer to different assets")
    ledgers, durations = {}, {}
    for asset in audit["assets"]:
        path = ledger_dir / (asset["video_id"] + ".parquet")
        if sha256_file(path) != asset["ledger_sha256"]:
            raise ValueError("Local ledger hash mismatch")
        table = pq.read_table(path, columns=["original_decoded_frame_index", "presentation_timestamp",
                                            "time_base_num", "time_base_den", "timestamp_seconds"])
        records = table.to_pylist()
        ledgers[asset["video_id"]] = {r["original_decoded_frame_index"]: r for r in records}
        durations[asset["video_id"]] = max(r["timestamp_seconds"] for r in records)
    if set(c["video_id"] for c in coords) != set(ledgers):
        raise ValueError("GPU corpus and audited ledger differ")
    for c in coords:
        row = ledgers[c["video_id"]].get(c["ordinal"])
        if row is None or Fraction(c["pts"] * c["time_base_num"], c["time_base_den"]) != Fraction(
                row["presentation_timestamp"] * row["time_base_num"], row["time_base_den"]):
            raise ValueError("GPU sample PTS does not match the local presentation-order ledger")
        if abs(c["clip_seconds"] - row["timestamp_seconds"]) > 0.001:
            raise ValueError("GPU sample timestamp mismatch")
    scores = queries @ images.T
    result = []
    for qid, row in zip(qids, scores):
        best = {}
        for i, c in enumerate(coords):
            vid = c["video_id"]
            score = float(row[i])
            if vid not in best or score > best[vid][0]:
                best[vid] = (score, c["clip_seconds"])
        moments = [{"video_id": vid, "start_sec": max(0.0, ts - window_seconds / 2),
                    "end_sec": min(durations[vid], ts + window_seconds / 2), "ranking_score": score}
                   for vid, (score, ts) in best.items()]
        moments.sort(key=lambda m: (-m["ranking_score"], m["video_id"]))
        result.append({"query_id": qid, "results": moments})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"prediction_sha256": sha256_file(output), "queries": len(result), "frames": len(coords),
            "window_seconds": window_seconds,
            "limitations": ["Fixed windows around visual anchors; not full A5.",
                            "No human-reviewed labels used in inference."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-input", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(rank(args.gpu_dir, args.ledger_dir, args.output, query_input=args.query_input,
                          run_config=args.run_config, window_seconds=args.window_seconds), indent=2))
