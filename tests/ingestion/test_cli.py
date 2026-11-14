"""Unit tests for M1 CLI commands."""
import json
from pathlib import Path
import pytest

from videntra.cli import main
from videntra.fixtures import build_fixtures


@pytest.fixture(scope="module")
def fixture_env(tmp_path_factory):
    dest = tmp_path_factory.mktemp("cli_fixtures")
    receipt = build_fixtures(dest, include_extended=True)
    return dest, receipt


def test_cli_ingest_and_timeline_inspect(fixture_env, tmp_path, capsys):
    dest, receipt = fixture_env
    # Build a valid media_asset manifest envelope for fixture media
    roots_cfg = tmp_path / "test_roots.json"
    roots_cfg.write_text(json.dumps({
        "roots": {"fixtures": str(dest)},
        "denied_roots": [],
    }), encoding="utf-8")

    records = []
    for f in receipt["fixtures"][:2]:
        records.append({
            "schema_version": "1.0",
            "video_id": f["name"],
            "source_video_id": f"src_{f['name']}",
            "sha256": f["sha256"],
            "media_uri": f"asset://fixtures/{f['file']}",
            "video_stream_index": 0,
            "audio_stream_index": None,
            "source_offset_us": None,
            "source_mapping_kind": "unknown",
            "timeline_origin": {"num": 0, "den": 1},
            "duration_us": 1_000_000,
            "rights_status": "unknown",
            "rights_reference": None,
            "original_dataset_id": "test_dataset",
            "official_split": "val",
            "exposure": "development_previously_used",
            "decode_status": "pending",
            "timeline_revision": "rev_m1_1",
            "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
        })

    manifest_file = tmp_path / "test.media.json"
    manifest_doc = {
        "kind": "media_asset",
        "revision": "rev_manifest_1",
        "records": records,
    }
    manifest_file.write_text(json.dumps(manifest_doc), encoding="utf-8")

    out_dir = tmp_path / "m1_run"
    ret = main([
        "ingest",
        "--manifest", str(manifest_file),
        "--output", str(out_dir),
        "--asset-roots", str(roots_cfg),
        "--audit-mode",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["total_assets"] == 2
    assert res["committed"] == 2

    # Timeline inspect CLI
    ret_inspect = main([
        "timeline", "inspect",
        "--video-id", "cfr",
        "--revision", "rev_m1_1",
        "--frame-index", "0",
        "--ledger", str(out_dir / "timeline/cfr/rev_m1_1/frames.parquet"),
    ])
    assert ret_inspect == 0
    captured_inspect = capsys.readouterr()
    inspect_res = json.loads(captured_inspect.out)
    assert inspect_res["video_id"] == "cfr"
    assert inspect_res["ordinal"] == 0
    assert inspect_res["timestamp_us"] == 0


def test_cli_sampling_compare(tmp_path, capsys):
    cfg_file = tmp_path / "e02_test.json"
    cfg_file.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": "E02",
        "description": "test comparison",
    }), encoding="utf-8")

    out_dir = tmp_path / "compare_out"
    ret = main([
        "sampling", "compare",
        "--config", str(cfg_file),
        "--output", str(out_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["status"] == "complete"
    assert (out_dir / "comparison_report.json").exists()
