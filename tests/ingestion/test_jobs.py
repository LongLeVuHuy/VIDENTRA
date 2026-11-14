"""Unit tests for ingestion jobs, staging attempts, and resume caching."""
from pathlib import Path
import pytest
import json

from videntra.config.ingestion import IngestionConfig
from videntra.fixtures import build_fixtures
from videntra.ingestion.jobs import IngestionJobManager, compute_config_hash, compute_job_key


@pytest.fixture(scope="module")
def fixture_media(tmp_path_factory):
    dest = tmp_path_factory.mktemp("job_fixtures")
    receipt = build_fixtures(dest, include_extended=True)
    return dest, receipt


def test_job_execution_and_resume_idempotency(fixture_media, tmp_path):
    dest, receipt = fixture_media
    cfr_path = dest / "cfr.mkv"
    cfr_hash = [f["sha256"] for f in receipt["fixtures"] if f["name"] == "cfr"][0]

    record = {
        "video_id": "test_cfr",
        "sha256": cfr_hash,
        "local_path": str(cfr_path),
        "timeline_revision": "rev_test_1",
    }

    out_dir = tmp_path / "run_out"
    cfg = IngestionConfig()
    mgr = IngestionJobManager(config=cfg, output_dir=out_dir, resume=True)

    # First run: should be committed
    res1 = mgr.execute_asset(record)
    assert res1.status == "committed"
    assert res1.frames_count == 5
    assert res1.samples_count >= 2
    assert Path(res1.timeline_path).exists()
    assert Path(res1.samples_path).exists()
    assert Path(res1.audio_map_path).exists()
    assert Path(res1.ocr_probes_path).exists()

    # Second run with resume=True: should be cached
    res2 = mgr.execute_asset(record)
    assert res2.status == "cached"
    assert res2.attempts == 0

    # Third run with resume=False: should re-commit
    mgr_no_resume = IngestionJobManager(config=cfg, output_dir=out_dir, resume=False)
    res3 = mgr_no_resume.execute_asset(record)
    assert res3.status == "committed"


def test_manifest_batch_execution(fixture_media, tmp_path):
    dest, receipt = fixture_media
    records = []
    for f in receipt["fixtures"][:3]:  # cfr, vfr, nonzero_pts
        records.append({
            "video_id": f["name"],
            "sha256": f["sha256"],
            "local_path": str(dest / f["file"]),
            "timeline_revision": "rev_1",
        })

    out_dir = tmp_path / "batch_out"
    mgr = IngestionJobManager(config=IngestionConfig(), output_dir=out_dir)
    report = mgr.execute_manifest(records)

    assert report.total_assets == 3
    assert report.committed_assets == 3
    assert report.failed_assets == 0
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "metrics.json").exists()
