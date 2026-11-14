import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.rank_blender_visual_pilot import rank
from videntra.evaluation.provenance import sha256_file


def test_rank_checks_ledger_and_never_uses_qrels(tmp_path):
    gpu, ledger = tmp_path / "gpu", tmp_path / "ledger"
    gpu.mkdir(); ledger.mkdir()
    assets = []
    for vid in ("v1", "v2"):
        path = ledger / f"{vid}.parquet"
        pq.write_table(pa.Table.from_pylist([{"original_decoded_frame_index": 0, "presentation_timestamp": 0,
            "time_base_num": 1, "time_base_den": 1000, "timestamp_seconds": 0.0},
            {"original_decoded_frame_index": 1, "presentation_timestamp": 10000,
             "time_base_num": 1, "time_base_den": 1000, "timestamp_seconds": 10.0}]), path)
        assets.append({"video_id": vid, "ledger_sha256": sha256_file(path), "media_sha256": vid + "-sha"})
    (ledger / "audit.json").write_text(json.dumps({"assets": assets}))
    coords = [{"video_id": v, "ordinal": 1, "pts": 10000, "time_base_num": 1,
               "time_base_den": 1000, "clip_seconds": 10.0} for v in ("v1", "v2")]
    (gpu / "coordinates.json").write_text(json.dumps(coords))
    (gpu / "query_ids.json").write_text('["q"]')
    matrix = np.zeros((2, 768), dtype=np.float32); matrix[0, :2] = [0.1, np.sqrt(0.99)]; matrix[1, :2] = [0.9, np.sqrt(0.19)]
    np.save(gpu / "frames.npy", matrix)
    query = np.zeros((1, 768), dtype=np.float32); query[0, 0] = 1
    np.save(gpu / "queries.npy", query)
    hashes = {name: sha256_file(gpu / name) for name in ("frames.npy", "queries.npy", "coordinates.json", "query_ids.json")}
    query_input = tmp_path / "input.json"
    query_input.write_text(json.dumps([{"query_id": "q", "text": "query"}]))
    bundle = tmp_path / "bundle_manifest.json"
    bundle.write_text(json.dumps({"assets": [{"video_id": v, "sha256": v + "-sha"} for v in ("v1", "v2")]}))
    config = {"worker_sha256": "worker", "manifest_sha256": sha256_file(bundle), "queries_sha256": sha256_file(query_input)}
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    (gpu / "manifest.json").write_text(json.dumps({"output_sha256": hashes, "frame_count": 2, "query_count": 1,
        "model": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "weights_sha256": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b", "device": "cuda",
        "worker_sha256": "worker", "input_manifest_sha256": config["manifest_sha256"], "query_input_sha256": config["queries_sha256"]}))
    out = tmp_path / "predictions.jsonl"
    result = rank(gpu, ledger, out, query_input=query_input, run_config=config_path)
    assert result["queries"] == 1
    top = json.loads(out.read_text())["results"][0]
    assert top["video_id"] == "v2"
    assert (top["start_sec"], top["end_sec"]) == (5.0, 10.0)
    alternate = tmp_path / "four.jsonl"
    rank(gpu, ledger, alternate, query_input=query_input, run_config=config_path, window_seconds=4.0)
    top = json.loads(alternate.read_text())["results"][0]
    assert (top["start_sec"], top["end_sec"]) == (8.0, 10.0)
    with pytest.raises(ValueError, match="positive and finite"):
        rank(gpu, ledger, tmp_path / "bad.jsonl", query_input=query_input, run_config=config_path, window_seconds=0)
    with pytest.raises(FileExistsError):
        rank(gpu, ledger, out, query_input=query_input, run_config=config_path)
