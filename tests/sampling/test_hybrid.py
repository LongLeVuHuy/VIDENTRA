"""Unit tests for M1-C hybrid adaptive sampler."""
from fractions import Fraction
from pathlib import Path
import pytest
import numpy as np

from videntra.config.ingestion import SamplingConfig
from videntra.ingestion.decode import DecodedFrame
from videntra.sampling.hybrid import (
    sample_frames_hybrid,
    compute_phash_64,
    phash_hamming_distance,
    compute_luma_difference,
    compute_hist_distance,
)


def _make_dummy_frame(ordinal: int, t_sec: float, color: tuple[int, int, int]) -> DecodedFrame:
    pixels = np.full((48, 64, 3), color, dtype=np.uint8)
    origin = Fraction(0)
    pts = int(round(t_sec * 1000))
    return DecodedFrame(
        ordinal=ordinal,
        pts=pts,
        dts=pts,
        time_base_num=1,
        time_base_den=1000,
        origin=origin,
        is_keyframe=(ordinal == 0),
        width=64,
        height=48,
        pixels=pixels,
    )


def test_phash_exact_and_distance():
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    white = np.full((48, 64, 3), 255, dtype=np.uint8)
    h_black = compute_phash_64(black)
    h_white = compute_phash_64(white)
    # Identical image -> distance 0
    assert phash_hamming_distance(h_black, h_black) == 0
    # Different image -> distance > 0
    assert phash_hamming_distance(h_black, h_white) > 0


def test_static_video_preserves_hard_anchors():
    """Static video longer than gap must NOT be wiped clean by pHash dedup."""
    cfg = SamplingConfig(max_gap_seconds=2.0)
    # 30 frames at 5 fps -> 6 seconds of static black
    frames = [_make_dummy_frame(i, i * 0.2, (0, 0, 0)) for i in range(30)]
    manifest = sample_frames_hybrid(frames, cfg, "video_static", "rev_1")

    # Mandatory anchors: first (0.0s), 2.0s (ord 10), 4.0s (ord 20), 6.0s (ord 29 / last)
    assert len(manifest.samples) >= 4
    ordinals = [s.original_decoded_frame_index for s in manifest.samples]
    assert 0 in ordinals
    assert 29 in ordinals
    # Max gap must not be violated
    assert manifest.max_gap_violations == 0
    for s in manifest.samples:
        assert s.is_mandatory is True


def test_cut_brackets_preservation():
    """Shot cut must preserve both cut_before and cut_after brackets."""
    cfg = SamplingConfig(max_gap_seconds=2.0, change_threshold_c=0.35)
    # 5 black frames, then abrupt cut to white frames
    frames = []
    for i in range(5):
        frames.append(_make_dummy_frame(i, i * 0.1, (0, 0, 0)))
    for i in range(5, 10):
        frames.append(_make_dummy_frame(i, i * 0.1, (255, 255, 255)))

    manifest = sample_frames_hybrid(frames, cfg, "video_cut", "rev_1")
    reasons_by_ord = {s.original_decoded_frame_index: s.reasons for s in manifest.samples}

    # Ordinal 4 should be cut_before, ordinal 5 should be cut_after
    assert 4 in reasons_by_ord
    assert "cut_before" in reasons_by_ord[4]
    assert 5 in reasons_by_ord
    assert "cut_after" in reasons_by_ord[5]


def test_burst_targets_snapped_to_real_frames():
    cfg = SamplingConfig(max_gap_seconds=5.0, burst_fps=4.0, burst_pre_seconds=0.5, burst_post_seconds=1.0)
    # 50 frames at 10 fps (0.0s to 4.9s)
    # Trigger a color pulse at t=1.0s (ord 10)
    frames = []
    for i in range(50):
        color = (255, 0, 0) if i == 10 else (0, 0, 0)
        frames.append(_make_dummy_frame(i, i * 0.1, color))

    manifest = sample_frames_hybrid(frames, cfg, "video_burst", "rev_1")
    # All sampled frame ordinals must exist in original frames
    for s in manifest.samples:
        assert 0 <= s.original_decoded_frame_index < 50
    # Must have burst or change_trigger reasons
    all_reasons = {r for s in manifest.samples for r in s.reasons}
    assert "change_trigger" in all_reasons or "cut_after" in all_reasons


def test_samples_to_parquet_roundtrip(tmp_path):
    cfg = SamplingConfig()
    frames = [_make_dummy_frame(i, i * 0.5, (i * 20, i * 20, i * 20)) for i in range(10)]
    manifest = sample_frames_hybrid(frames, cfg, "video_pq", "rev_1")

    pq_path = tmp_path / "samples.parquet"
    manifest.to_parquet(pq_path)
    assert pq_path.exists()
    assert pq_path.stat().st_size > 0


def test_sampling_config_for_duration():
    short_cfg = SamplingConfig.for_duration(10.0)
    assert short_cfg.max_gap_seconds == 1.0
    assert short_cfg.soft_target_fps == 1.5

    med_cfg = SamplingConfig.for_duration(30.0)
    assert med_cfg.max_gap_seconds == 1.5
    assert med_cfg.soft_target_fps == 1.0

    long_cfg = SamplingConfig.for_duration(120.0)
    assert long_cfg.max_gap_seconds == 2.0
    assert long_cfg.max_static_gap_seconds == 4.5


def test_sharpness_and_blank_frame_detection():
    from videntra.sampling.hybrid import compute_sharpness_score, is_blank_or_fade_frame

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert is_blank_or_fade_frame(blank) is True
    assert compute_sharpness_score(blank) == 0.0

    # Textured image has high luma std and high sharpness
    rng = np.random.RandomState(42)
    textured = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    assert is_blank_or_fade_frame(textured) is False
    assert compute_sharpness_score(textured) > 10.0


def test_black_frame_deferral_to_content():
    cfg = SamplingConfig(max_gap_seconds=2.0)
    # First 2 frames are black (t=0.0, 0.1), then content appears at t=0.2 (ord 2)
    frames = [
        _make_dummy_frame(0, 0.0, (0, 0, 0)),
        _make_dummy_frame(1, 0.1, (0, 0, 0)),
    ]
    # Ordinal 2 has content
    rng = np.random.RandomState(42)
    for i in range(2, 10):
        f = _make_dummy_frame(i, i * 0.1, (100, 100, 100))
        f.pixels = rng.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        frames.append(f)

    manifest = sample_frames_hybrid(frames, cfg, "video_defer", "rev_1")
    reasons_by_ord = {s.original_decoded_frame_index: s.reasons for s in manifest.samples}
    # Blank frame 0 does not hold 'first' anchor; 'first' was transferred to content frame (ord 2)
    assert "first" not in reasons_by_ord.get(0, ())
    assert "first" in reasons_by_ord[2]

