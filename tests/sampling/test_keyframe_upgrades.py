"""Tests for audited keyframe extraction fixes, 5 sampling strategies, PyAV atomic extraction, and contracts."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from scripts.extract_all_keyframes import _is_valid_jpeg, extract_video_keyframes
from scripts.offline_processor import OfflineProcessingPipeline
from videntra.api.frames import _decode_frame_from_media
from videntra.cli import main as cli_main
from videntra.config.ingestion import IngestionConfig, SamplingConfig, VALID_SAMPLING_STRATEGIES
from videntra.encoders.siglip2 import MockSigLIP2Encoder
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import DecodedFrame, decode_frames
from videntra.ingestion.probe import probe_media
from videntra.sampling.compare import compare_sampling_strategies, evaluate_keyframe_quality
from videntra.sampling.hybrid import (
    HybridAdaptiveSampler,
    compute_grid_hist_512,
    compute_phash_64,
    phash_hamming_distance,
    sample_frames_hybrid,
)
from videntra.sampling.probes import select_ocr_probes


def _make_frame(ordinal: int, t_sec: float, pixels: np.ndarray, *, epoch_id: str = "epoch_0") -> DecodedFrame:
    pts = int(round(t_sec * 1000))
    return DecodedFrame(
        ordinal=ordinal,
        pts=pts,
        dts=pts,
        time_base_num=1,
        time_base_den=1000,
        origin=Fraction(0),
        epoch_id=epoch_id,
        is_keyframe=(ordinal == 0),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        pixels=pixels,
    )


def _textured(
    seed: int,
    *,
    h: int = 80,
    w: int = 112,
    tint: tuple[int, int, int] = (40, 120, 200),
    shift: int = 0,
) -> np.ndarray:
    y_grad = np.linspace(35, 195, h, dtype=np.float32)[:, None]
    x_grad = np.linspace(30, 185, w, dtype=np.float32)[None, :]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip((y_grad + seed * 7) % 210 + 20, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip((x_grad + seed * 13) % 210 + 20, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip((tint[2] + (y_grad * 0.3)) % 220 + 20, 0, 255).astype(np.uint8)
    sx = shift % max(1, w // 3)
    cv2.rectangle(img, (12 + sx, 12), (48 + sx, 48), tint, -1)
    cv2.rectangle(img, (15 + sx, 15), (45 + sx, 45), (245, 245, 245), 2)
    cv2.line(img, (55, 10), (104, 70), (240, 230, 60), 2)
    cv2.rectangle(img, (60, 20), (100, 64), (255, 255, 255), 2)
    return img


def test_config_roundtrip_and_strategy_validation():
    for strat in VALID_SAMPLING_STRATEGIES:
        cfg = IngestionConfig(sampling=SamplingConfig(strategy=strat))
        restored = IngestionConfig.from_dict(cfg.to_dict())
        assert restored.sampling.strategy == strat

    with pytest.raises(ValueError, match="Invalid sampling strategy"):
        SamplingConfig(strategy="unknown_strat").validate()


def test_use_after_free_pixel_clearing_regression():
    """Clearing frame.pixels immediately after process_frame() must NOT corrupt pre-burst pHash or sharpness."""
    cfg = SamplingConfig(
        strategy="hybrid_v2_adaptive",
        max_gap_seconds=5.0,
        burst_fps=4.0,
        burst_pre_seconds=0.5,
        burst_post_seconds=0.5,
        phash_min_distance=0,
    )
    sampler = HybridAdaptiveSampler(cfg, video_id="vid_uaf", timeline_revision="rev_1")

    # Evolving frames in scene A, then abrupt cut to distinct scene B
    for i in range(12):
        if i < 8:
            pix = _textured(11 + i * 2, tint=(20, 40, 210), shift=i * 4)
        else:
            pix = _textured(200 + i * 2, tint=(230, 50, 30), shift=i * 4)
        frame = _make_frame(i, i * 0.1, pix)
        sampler.process_frame(frame)
        # Simulate IngestionJobManager memory-bounded release (R04)
        frame.pixels = None

    manifest = sampler.finalize()
    burst_records = [s for s in manifest.samples if "burst" in s.reasons]
    assert len(burst_records) >= 1
    for s in manifest.samples:
        assert s.phash != "0000000000000000"
        assert s.sharpness is not None and s.sharpness > 10.0


def test_phash_all_ac_and_vectorized_grid_hist_non_divisible():
    # Horizontal bar vs vertical bar must separate strongly when top-row & left-col AC terms are included
    h_stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    h_stripes[::4, :, :] = 255
    v_stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    v_stripes[:, ::4, :] = 255
    assert phash_hamming_distance(compute_phash_64(h_stripes), compute_phash_64(v_stripes)) >= 8

    # Non-divisible grid dimensions (47x63) still yield exact normalized (16, 512) rows
    odd_img = _textured(7, h=47, w=63)
    hist = compute_grid_hist_512(odd_img)
    assert hist.shape == (16, 512)
    assert np.allclose(hist.sum(axis=1), 1.0, atol=1e-5)


def test_lookahead_avoids_motion_blur_and_wins_sharpness():
    """At max_gap trigger t=2.0s where frame is blurred, lookahead selects the sharp frame right after."""
    frames: list[DecodedFrame] = []
    for i in range(45):  # 3.0s at 15fps
        t = i / 15.0
        base = _textured(100, tint=(50, 150, 210))
        pix = cv2.GaussianBlur(base, (9, 9), 3.0) if abs(t - 2.0) <= 0.05 else base
        frames.append(_make_frame(i, t, pix))

    cfg_v1 = SamplingConfig(strategy="hybrid_v1_legacy", max_gap_seconds=2.0)
    cfg_v2 = SamplingConfig(strategy="hybrid_v2_adaptive", max_gap_seconds=2.0, lookahead_window_seconds=0.15)
    m1 = sample_frames_hybrid(frames, cfg_v1, "v_blur", "r1")
    m2 = sample_frames_hybrid(frames, cfg_v2, "v_blur", "r1")

    ords1 = [s.original_decoded_frame_index for s in m1.samples]
    ords2 = [s.original_decoded_frame_index for s in m2.samples]
    # Frame 30 is exactly at t=2.0s (blurred); lookahead picks sharp frame ~32 instead
    assert 30 in ords1
    assert 30 not in ords2
    assert any(o in (31, 32) for o in ords2)
    assert m2.max_gap_violations == 0


def test_gradual_dissolve_flash_suppression_and_fade_tail():
    # 1. Gradual dissolve over 25 frames with small step diffs
    img_a = _textured(111, tint=(220, 20, 20)).astype(np.float32)
    img_b = _textured(222, tint=(20, 20, 220)).astype(np.float32)
    dissolve_frames: list[DecodedFrame] = []
    for i in range(60):  # 4.0s at 15fps
        alpha = 0.0 if i < 10 else (min(1.0, (i - 10) / 22.0))
        pix = np.clip((1.0 - alpha) * img_a + alpha * img_b, 0, 255).astype(np.uint8)
        dissolve_frames.append(_make_frame(i, i / 15.0, pix))

    m_v1 = sample_frames_hybrid(dissolve_frames, SamplingConfig(strategy="hybrid_v1_legacy", max_gap_seconds=5.0), "dis", "r1")
    m_v2 = sample_frames_hybrid(
        dissolve_frames,
        SamplingConfig(strategy="hybrid_v2_adaptive", max_gap_seconds=5.0, drift_threshold_c=0.35),
        "dis",
        "r1",
    )
    # v2 catches cumulative dissolve drift in the interior [10..40], while v1 sees only endpoints [0, 59]
    assert len(m_v1.samples) == 2
    assert any(12 <= s.original_decoded_frame_index <= 42 for s in m_v2.samples)

    # 2. Flash strobe: single white frame inside steady scene suppresses return cut flood
    steady = _textured(333, tint=(60, 180, 90))
    flash = np.full_like(steady, 255)
    flash_frames = [_make_frame(i, i / 15.0, flash if i == 20 else steady.copy()) for i in range(45)]
    m_flash_v1 = sample_frames_hybrid(flash_frames, SamplingConfig(strategy="hybrid_v1_legacy"), "fl", "r1")
    m_flash_v2 = sample_frames_hybrid(flash_frames, SamplingConfig(strategy="hybrid_v2_adaptive"), "fl", "r1")
    assert len(m_flash_v2.shot_cut_ordinals) < len(m_flash_v1.shot_cut_ordinals)

    # 3. Pre-fade tail content preservation
    fade_frames = [
        _make_frame(i, i * 0.2, np.zeros((64, 80, 3), dtype=np.uint8) if i >= 8 else _textured(444))
        for i in range(15)
    ]
    m_fade = sample_frames_hybrid(fade_frames, SamplingConfig(strategy="hybrid_v2_adaptive", max_gap_seconds=2.0), "ft", "r1")
    by_ord = {s.original_decoded_frame_index: s.reasons for s in m_fade.samples}
    assert 14 in by_ord and "last" in by_ord[14]
    assert any("last_content" in r or o == 7 for o, r in by_ord.items())


def test_compare_all_five_strategies_and_parquet_roundtrip(tmp_path: Path):
    frames = []
    for i in range(60):
        pix = _textured(500 + (i // 20), tint=(30 + (i // 20) * 80, 110, 190))
        frames.append(_make_frame(i, i * 0.1, pix))

    metrics = compare_sampling_strategies(frames, base_config=SamplingConfig(max_gap_seconds=2.0))
    names = {m.strategy for m in metrics}
    assert set(VALID_SAMPLING_STRATEGIES).issubset(names)
    assert "uniform_matched_budget" in names
    for m in metrics:
        assert m.max_gap_violations == 0
        assert 0.0 <= m.mean_representativeness <= 1.0
        assert m.selected_count >= 2

    man = sample_frames_hybrid(frames, SamplingConfig(strategy="shot_medoid_submodular"), "med_vid", "rev_1")
    pq_file = tmp_path / "med_samples.parquet"
    man.to_parquet(pq_file)
    table = pq.read_table(str(pq_file))
    assert "pts" in table.column_names
    assert "sharpness" in table.column_names


def test_extract_video_keyframes_pyav_atomic_and_self_heal(tmp_path: Path):
    fixtures = tmp_path / "fix"
    build_fixtures(fixtures, include_extended=True)
    b_path = fixtures / "b_frames.mkv"

    decoded = list(decode_frames(b_path, include_pixels=True))
    out_dir = tmp_path / "kfs"
    targets = {0: str(out_dir / "000000.jpg"), 4: str(out_dir / "000004.jpg"), 9: str(out_dir / "000009.jpg")}

    saved = extract_video_keyframes("b_frames", str(b_path), targets)
    assert saved == 3
    for ord_idx, p_str in targets.items():
        p = Path(p_str)
        assert _is_valid_jpeg(p)
        img = cv2.imread(str(p))
        assert img.shape[:2] == (decoded[ord_idx].height, decoded[ord_idx].width)

    assert not list(out_dir.glob("*.tmp.jpg"))

    # Simulate 0-byte corrupted file -> re-run self-heals only the corrupted entry
    corrupt_p = Path(targets[4])
    corrupt_p.write_bytes(b"")
    assert not _is_valid_jpeg(corrupt_p)
    repaired = extract_video_keyframes("b_frames", str(b_path), targets)
    assert repaired == 1
    assert _is_valid_jpeg(corrupt_p)


def test_ocr_probes_epoch_reset_and_burst():
    p_frames = [
        _make_frame(0, 0.0, _textured(1), epoch_id="epoch_0"),
        _make_frame(1, 1.0, _textured(1), epoch_id="epoch_0"),
        _make_frame(2, 0.2, _textured(1), epoch_id="epoch_1"),  # backward timestamp in new epoch
        _make_frame(3, 1.2, _textured(1), epoch_id="epoch_1"),
    ]
    manifest = select_ocr_probes(p_frames, "ep_vid", probe_fps=1.0, shot_cut_ordinals=[2], burst_ordinals=[3])
    ords = [p.original_decoded_frame_index for p in manifest.probes]
    assert ords == [0, 1, 2, 3]
    assert "shot_cut" in manifest.probes[2].probe_reason
    assert "burst_probe" in manifest.probes[3].probe_reason


def test_offline_processor_pts_preservation_cli_compare_and_api_ts(tmp_path: Path):
    fixtures = tmp_path / "fix"
    build_fixtures(fixtures, include_extended=True)
    nz_path = fixtures / "nonzero_pts.mkv"
    probe_res = probe_media(nz_path)
    decoded = list(decode_frames(nz_path, probe_res, include_pixels=True))
    assert decoded[0].pts is not None and decoded[0].pts > 0

    # 1. API direct decode header should report clip-relative ts_us == "0" while keeping raw frame.pts
    _jpg, pts, ts_us = _decode_frame_from_media(nz_path, 0)
    assert pts == decoded[0].pts
    assert ts_us == "0"

    # 2. OfflineProcessingPipeline streaming Step 2 preserves raw PTS in SQLite Keyframes
    raw_dir = tmp_path / "raw"
    media_dir = raw_dir / "synthetic_ds" / "media"
    media_dir.mkdir(parents=True)
    import subprocess
    mp4_path = media_dir / "vid_nz.mp4"
    # Re-mux or copy fixture as mp4 with non-zero initial PTS offset
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5:duration=1", "-output_ts_offset", "5", str(mp4_path)],
        capture_output=True,
        check=True,
    )
    dec_mp4 = list(decode_frames(mp4_path, include_pixels=False))
    db_p = tmp_path / "meta.db"
    pipe = OfflineProcessingPipeline(
        db_path=db_p,
        faiss_path=tmp_path / "faiss.bin",
        raw_data_dir=raw_dir,
        keyframes_dir=tmp_path / "kf_out",
        encoder=MockSigLIP2Encoder(),
        i_understand_overwrite=True,
    )
    pipe.run_all()
    with sqlite3.connect(str(db_p)) as conn:
        row0 = conn.execute(
            "SELECT source_frame_index, pts, timestamp_ms FROM Keyframes WHERE source_frame_index = 0"
        ).fetchone()
    assert row0 is not None
    assert row0[1] == dec_mp4[0].pts
    assert row0[2] == 0

    # 3. CLI `sampling compare` produces real comparison report when given a manifest
    man_file = tmp_path / "m.json"
    cfr_p = fixtures / "cfr.mkv"
    cfr_sha = probe_media(cfr_p).sha256
    man_file.write_text(json.dumps({
        "kind": "media_asset",
        "revision": "rev_cmp",
        "records": [{
            "schema_version": "1.0",
            "video_id": "cmp_cfr",
            "source_video_id": "src_cfr",
            "sha256": cfr_sha,
            "media_uri": "asset://fixtures/cfr.mkv",
            "video_stream_index": 0,
            "audio_stream_index": None,
            "source_offset_us": None,
            "source_mapping_kind": "unknown",
            "timeline_origin": {"num": 0, "den": 1},
            "duration_us": 1_000_000,
            "rights_status": "allowed",
            "rights_reference": "license://test",
            "original_dataset_id": "FixtureSuite",
            "official_split": "val",
            "exposure": "train",
            "decode_status": "pending",
            "timeline_revision": "rev_1",
            "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
        }],
    }), encoding="utf-8")
    roots_file = tmp_path / "roots.json"
    roots_file.write_text(json.dumps({"roots": {"fixtures": str(fixtures)}, "denied_roots": []}), encoding="utf-8")
    cfg_file = tmp_path / "e02.json"
    cfg_file.write_text(json.dumps({"experiment_id": "E02", "sampling": {"max_gap_seconds": 0.4}}), encoding="utf-8")
    cmp_out = tmp_path / "cmp_out"

    rc = cli_main([
        "sampling", "compare",
        "--config", str(cfg_file),
        "--manifest", str(man_file),
        "--asset-roots", str(roots_file),
        "--output", str(cmp_out),
    ])
    assert rc == 0
    rep = json.loads((cmp_out / "comparison_report.json").read_text("utf-8"))
    assert len(rep["comparisons"]) == 1
    assert "hybrid_v2_adaptive" in rep["aggregate_by_strategy"]
