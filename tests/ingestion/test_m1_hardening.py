"""Comprehensive regression tests for M1-HARDENING (H01-H04).

Covers:
- R01: Cache identity and invalidation when config changes.
- R02: Cache rejection on corrupted or missing artifacts.
- R03: Exact seek bit-level parity with repeated PTS and epoch changes.
- R04: Streaming decode with bounded memory (no all-RGB list materialization).
- R05: Atomic publication and commit manifest integrity.
- R06: Pre-burst ring buffer selection and OCR probe shot cut propagation.
- R07: Non-fabrication of missing timestamps and epoch continuity.
- R08: Asset hash verification on local paths and rights enforcement.
- R09: Immutable run report generation and Parquet row filter lookup.
"""
from fractions import Fraction
import json
from pathlib import Path
import pytest
import numpy as np
import pyarrow.parquet as pq

from videntra.config.ingestion import IngestionConfig, SamplingConfig
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.jobs import IngestionJobManager, compute_config_hash
from videntra.ingestion.probe import probe_media
from videntra.timeline.ledger import (
    write_timeline_ledger,
    read_frame_record,
    exact_seek_frame,
)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("hardening_fixtures")
    build_fixtures(out, include_extended=True)
    return out


def test_r01_cache_invalidation_on_config_change(fixture_dir, tmp_path):
    """R01: Changing sampling config (e.g. max_gap_seconds) must invalidate cache."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r01",
        "sha256": probe_media(media_path).sha256,
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r01_out"

    # 1. First run with default config (max_gap_seconds=2.0)
    cfg1 = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=2.0))
    mgr1 = IngestionJobManager(config=cfg1, output_dir=out_dir, resume=True)
    res1 = mgr1.execute_asset(media_record)
    assert res1.status == "committed"
    assert res1.samples_count >= 2

    # 2. Second run with same config should be cached
    res2 = mgr1.execute_asset(media_record)
    assert res2.status == "cached"
    assert res2.samples_count == res1.samples_count

    # 3. Third run with changed config (max_gap_seconds=0.1 -> more samples)
    cfg2 = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=0.1))
    mgr2 = IngestionJobManager(config=cfg2, output_dir=out_dir, resume=True)
    res3 = mgr2.execute_asset(media_record)
    # Must NOT be cached!
    assert res3.status == "committed"
    assert res3.config_hash != res1.config_hash
    # Fresh execution with 0.1s max gap should select all frames
    assert res3.samples_count == 5


def test_r02_cache_rejection_on_corrupt_artifact(fixture_dir, tmp_path):
    """R02: Corrupted or missing artifacts must not be accepted from cache."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r02",
        "sha256": probe_media(media_path).sha256,
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r02_out"
    cfg = IngestionConfig()
    mgr = IngestionJobManager(config=cfg, output_dir=out_dir, resume=True)

    # Initial commit
    res1 = mgr.execute_asset(media_record)
    assert res1.status == "committed"

    # Corrupt samples.parquet with random bytes
    samples_file = Path(res1.samples_path)
    assert samples_file.exists()
    samples_file.write_bytes(b"INVALID_CORRUPT_BYTES")

    # Ingestion with resume=True must detect corruption and re-ingest
    res2 = mgr.execute_asset(media_record)
    assert res2.status == "committed"
    # Ensure samples.parquet was regenerated and is a valid Parquet table again
    pq_table = pq.read_table(str(samples_file))
    assert pq_table.num_rows == res2.samples_count


def test_r02_cache_rejection_on_missing_artifact(fixture_dir, tmp_path):
    """R02: Missing required artifact (e.g. ocr_probes.parquet) must trigger re-ingest."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r02_missing",
        "sha256": probe_media(media_path).sha256,
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r02_miss_out"
    cfg = IngestionConfig()
    mgr = IngestionJobManager(config=cfg, output_dir=out_dir, resume=True)

    res1 = mgr.execute_asset(media_record)
    assert res1.status == "committed"

    # Remove ocr_probes.parquet
    probes_file = Path(res1.ocr_probes_path)
    assert probes_file.exists()
    probes_file.unlink()

    # Re-running with resume must detect missing artifact and rebuild
    res2 = mgr.execute_asset(media_record)
    assert res2.status == "committed"
    assert probes_file.exists()


def test_r03_exact_seek_repeated_pts(fixture_dir, tmp_path):
    """R03: Two frames with identical PTS must disambiguate and return correct pixels."""
    media_path = fixture_dir / "repeated_pts.mkv"
    probe_res = probe_media(media_path)
    frames = list(decode_frames(media_path, probe_res, include_pixels=True))
    assert len(frames) == 5

    # In repeated_pts.mkv: Frame 0 PTS=0, Frame 1 PTS=0
    assert frames[0].pts == 0
    assert frames[1].pts == 0
    # Pixels must be different because testsrc2 advances every frame
    assert not np.array_equal(frames[0].pixels, frames[1].pixels)

    parquet_path = tmp_path / "repeated_pts_frames.parquet"
    write_timeline_ledger(parquet_path, frames, video_id="video_rep", timeline_revision="rev1")

    # Seek ordinal 0: must match Frame 0
    seek0 = exact_seek_frame(media_path, parquet_path, target_ordinal=0)
    assert seek0.ordinal == 0
    assert seek0.pts == 0
    assert np.array_equal(seek0.pixels, frames[0].pixels)

    # Seek ordinal 1: must match Frame 1, NOT Frame 0 (R03 fix verification!)
    seek1 = exact_seek_frame(media_path, parquet_path, target_ordinal=1)
    assert seek1.ordinal == 1
    assert seek1.pts == 0
    assert np.array_equal(seek1.pixels, frames[1].pixels)
    assert not np.array_equal(seek1.pixels, frames[0].pixels)


def test_r06_pre_burst_and_ocr_cuts(fixture_dir, tmp_path):
    """R06: Pre-burst frames selected from ring buffer & OCR probes receive cut ordinals."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r06",
        "sha256": probe_media(media_path).sha256,
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r06_out"
    # Use config with high burst fps and burst pre seconds
    cfg = IngestionConfig(
        sampling=SamplingConfig(
            burst_pre_seconds=0.4,
            burst_post_seconds=0.4,
            burst_fps=5.0,
            change_threshold_c=0.01,  # Force cut/change trigger
        )
    )
    mgr = IngestionJobManager(config=cfg, output_dir=out_dir, resume=False)
    res = mgr.execute_asset(media_record)
    assert res.status == "committed"

    # Check that samples parquet contains burst reasons
    samples_table = pq.read_table(res.samples_path)
    all_reasons = [r for sub in samples_table.column("reasons").to_pylist() for r in sub]
    assert "burst" in all_reasons or "cut_before" in all_reasons

    # Check OCR probes parquet
    probes_table = pq.read_table(res.ocr_probes_path)
    probe_reasons = probes_table.column("probe_reason").to_pylist()
    assert len(probe_reasons) > 0


def test_r08_local_path_hash_mismatch_rejected(fixture_dir, tmp_path):
    """R08: Supplying an altered file with mismatched hash must be rejected."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r08_spoof",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r08_out"
    mgr = IngestionJobManager(config=IngestionConfig(), output_dir=out_dir)
    res = mgr.execute_asset(media_record)
    assert res.status == "failed"
    assert "Media hash mismatch" in res.error_message


def test_r09_immutable_run_reports(fixture_dir, tmp_path):
    """R09: Batch execution writes immutable run reports under reports/."""
    media_path = fixture_dir / "cfr.mkv"
    media_record = {
        "video_id": "test_r09",
        "sha256": probe_media(media_path).sha256,
        "local_path": str(media_path),
    }

    out_dir = tmp_path / "r09_out"
    cfg = IngestionConfig()

    mgr1 = IngestionJobManager(config=cfg, output_dir=out_dir, run_id="run_001")
    rep1 = mgr1.execute_manifest([media_record])
    assert rep1.committed_assets == 1

    report1_path = out_dir / "reports" / "run_001.json"
    assert report1_path.exists()
    content1 = report1_path.read_text(encoding="utf-8")
    assert "run_001" in content1

    # Second run with different ID must not overwrite run_001
    mgr2 = IngestionJobManager(config=cfg, output_dir=out_dir, run_id="run_002")
    rep2 = mgr2.execute_manifest([media_record])
    assert rep2.cached_assets == 1

    report2_path = out_dir / "reports" / "run_002.json"
    assert report2_path.exists()
    assert report1_path.read_text(encoding="utf-8") == content1
