"""CLI integration test for Milestone 3 evidence build."""
import json
from pathlib import Path
import pytest

from videntra.cli import main
from videntra.config.ingestion import IngestionConfig, SamplingConfig
from videntra.fixtures import build_fixtures
from videntra.ingestion.jobs import IngestionJobManager
from videntra.registry.access import MediaResolver
from videntra.registry.store import sha256_file


def test_cli_evidence_build_workflow(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    build_fixtures(fixtures_dir)

    ingest_out = tmp_path / "ingest"
    cfg = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=0.25))
    mgr = IngestionJobManager(config=cfg, output_dir=ingest_out)

    audio_vid = fixtures_dir / "audio_delay.mkv"
    a_hash = sha256_file(audio_vid)
    video_id = f"sha256:{a_hash}"

    records = [{
        "schema_version": "1.0",
        "video_id": video_id,
        "source_video_id": "fixture:audio_delay",
        "sha256": a_hash,
        "media_uri": "asset://fixtures/audio_delay.mkv",
        "video_stream_index": 0,
        "audio_stream_index": 1,
        "source_offset_us": 0,
        "source_mapping_kind": "constant_offset",
        "timeline_origin": {"num": 0, "den": 1},
        "duration_us": 1000000,
        "rights_status": "allowed",
        "rights_reference": "license://fixture-test",
        "original_dataset_id": "FixtureSuite",
        "official_split": "train",
        "exposure": "train",
        "decode_status": "pending",
        "timeline_revision": None,
        "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
    }]

    roots_cfg = tmp_path / "asset_roots.json"
    roots_cfg.write_text(json.dumps({
        "roots": {"fixtures": str(fixtures_dir)},
        "denied_roots": [],
    }), encoding="utf-8")

    resolver = MediaResolver({"fixtures": fixtures_dir}, records)
    mgr.execute_manifest(records, resolver=resolver, for_inference=True)

    manifest_file = tmp_path / "manifest.json"
    manifest_doc = {
        "kind": "media_asset",
        "revision": "test_m3",
        "records": records,
    }
    manifest_file.write_text(json.dumps(manifest_doc), encoding="utf-8")

    # Run CLI evidence build
    evidence_out = tmp_path / "evidence_out"
    exit_code = main([
        "evidence", "build",
        "--manifest", str(manifest_file),
        "--ingest-dir", str(ingest_out),
        "--output", str(evidence_out),
        "--asset-roots", str(roots_cfg),
        "--modalities", "ocr,asr",
    ])
    assert exit_code == 0
    safe_vid = video_id.replace(":", "_")
    assert (evidence_out / "evidence" / "ocr_tracks" / safe_vid / "ocr_tracks.parquet").exists()
    assert (evidence_out / "evidence" / "asr_segments" / safe_vid / "asr_segments.parquet").exists()
