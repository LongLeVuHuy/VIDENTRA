"""Integration and CLI tests for Milestone 2 indexing, search, and evaluation."""
import json
from pathlib import Path
import pytest
from videntra.cli import main
from videntra.fixtures import build_fixtures
from videntra.ingestion.jobs import IngestionJobManager
from videntra.config.ingestion import IngestionConfig


def test_cli_index_search_evaluate_workflow(tmp_path):
    # 1. Build fixtures
    fixtures_dir = tmp_path / "fixtures"
    build_fixtures(fixtures_dir)

    # 2. Ingest 1 video
    ingest_out = tmp_path / "ingest"
    from videntra.config.ingestion import SamplingConfig
    cfg = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=0.2))
    mgr = IngestionJobManager(config=cfg, output_dir=ingest_out)

    sample_video = fixtures_dir / "cfr.mkv"
    from videntra.registry.store import sha256_file
    from videntra.registry.access import MediaResolver
    sample_hash = sha256_file(sample_video)
    video_id = f"sha256:{sample_hash}"

    records = [{
        "schema_version": "1.0",
        "video_id": video_id,
        "source_video_id": "test:clip:cfr",
        "sha256": sample_hash,
        "media_uri": "asset://fixtures/cfr.mkv",
        "video_stream_index": 0,
        "audio_stream_index": None,
        "source_offset_us": 0,
        "source_mapping_kind": "constant_offset",
        "timeline_origin": {"num": 0, "den": 1},
        "duration_us": 1000000,
        "rights_status": "allowed",
        "rights_reference": "license://open-access-test",
        "original_dataset_id": "TestDS",
        "official_split": "train",
        "exposure": "train",
        "decode_status": "pending",
        "timeline_revision": None,
        "modality_status": {
            "visual": "not_run",
            "ocr": "not_run",
            "asr": "not_run",
        },
    }]

    roots_cfg = tmp_path / "asset_roots.json"
    roots_cfg.write_text(json.dumps({
        "roots": {"fixtures": str(fixtures_dir)},
        "denied_roots": [],
    }), encoding="utf-8")

    resolver = MediaResolver({"fixtures": fixtures_dir}, records)

    report = mgr.execute_manifest(records, resolver=resolver, for_inference=True)
    assert report.committed_assets == 1

    # Create media manifest
    manifest_file = tmp_path / "manifest.json"
    manifest_doc = {
        "kind": "media_asset",
        "revision": "test_rev1",
        "records": records,
    }
    manifest_file.write_text(json.dumps(manifest_doc), encoding="utf-8")

    # 3. CLI index build (using mock encoder for fast testing)
    index_out = tmp_path / "index_dir"
    exit_code = main([
        "index", "build",
        "--manifest", str(manifest_file),
        "--ingest-dir", str(ingest_out),
        "--output", str(index_out),
        "--asset-roots", str(roots_cfg),
        "--mock",
    ])
    assert exit_code == 0
    assert (index_out / "id_mapping.parquet").exists()

    # 4. CLI search
    search_out = tmp_path / "search_results.jsonl"
    exit_code = main([
        "search",
        "--query", "người đàn ông đang đi bộ trên đường",
        "--index-dir", str(index_out),
        "--output", str(search_out),
        "--top-k", "5",
        "--mock",
    ])
    assert exit_code == 0
    assert search_out.exists()
    lines = search_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    first_res = json.loads(lines[0])
    assert first_res["video_id"] == video_id
    assert "ranking_score" in first_res

    # 5. CLI evaluate
    gt_file = tmp_path / "ground_truth.jsonl"
    gt_record = {
        "query_id": "q1",
        "video_id": video_id,
        "moment_start_seconds": first_res["moment_start_seconds"],
        "moment_end_seconds": first_res["moment_end_seconds"],
    }
    gt_file.write_text(json.dumps(gt_record) + "\n", encoding="utf-8")

    eval_out = tmp_path / "eval_report.json"
    exit_code = main([
        "evaluate",
        "--predictions", str(search_out),
        "--ground-truth", str(gt_file),
        "--output", str(eval_out),
    ])
    assert exit_code == 0
    assert eval_out.exists()
    eval_data = json.loads(eval_out.read_text(encoding="utf-8"))
    assert "video_hit_at_1" in eval_data
    assert eval_data["video_hit_at_1"] == 1.0
