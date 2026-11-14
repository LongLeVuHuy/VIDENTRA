"""Tests for ASR confidence calibration and temperature-scaled multimodal score calibration."""
import math
import pytest

from videntra.ranking.calibration import (
    CalibrationMetadata,
    calibrate_asr_confidence,
    calibrate_multimodal_scores,
)


def test_calibrate_asr_confidence_whisper_logprobs_and_clamping():
    # At exact midpoint -0.45, calibrated confidence equals 0.5
    assert calibrate_asr_confidence(-0.45) == pytest.approx(0.5, abs=1e-6)

    # Higher logprob yields higher calibrated probability, bounded within (0, 1)
    high_prob = calibrate_asr_confidence(-0.10)
    low_prob = calibrate_asr_confidence(-1.80)
    assert 0.0 < low_prob < 0.5 < high_prob < 1.0

    # Non-negative bounded values are preserved/clamped into [0, 1]
    assert calibrate_asr_confidence(0.88) == pytest.approx(0.88)
    assert calibrate_asr_confidence(1.50) == pytest.approx(1.0)
    assert calibrate_asr_confidence(None) == pytest.approx(0.85)
    assert calibrate_asr_confidence(math.nan) == pytest.approx(0.85)


def test_calibrate_multimodal_scores_monotonic_and_metadata():
    raw_scores = [0.21, 0.35, 0.52, 0.68]
    probs, meta = calibrate_multimodal_scores(raw_scores, temperature=0.07)

    assert isinstance(meta, CalibrationMetadata)
    assert meta.is_calibrated is True
    assert meta.method == "temperature_sigmoid"
    assert meta.parameters["temperature"] == pytest.approx(0.07)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs)
    assert probs[-1] > probs[0]

    empty_probs, empty_meta = calibrate_multimodal_scores([])
    assert empty_probs == []
    assert empty_meta.is_calibrated is True
