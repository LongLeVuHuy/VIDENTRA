"""Score calibration helpers distinguishing raw model outputs from calibrated confidences."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence


@dataclass(frozen=True)
class CalibrationMetadata:
    """Provenance record for calibrated scores."""

    is_calibrated: bool
    method: str
    parameters: dict[str, float] = field(default_factory=dict)


def _stable_sigmoid(z: float) -> float:
    """Compute logistic sigmoid without overflow for extreme logits."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def calibrate_asr_confidence(
    raw_score: float | None,
    *,
    midpoint: float = -0.45,
    temperature: float = 0.55,
    default: float = 0.85,
) -> float:
    """Calibrate Whisper avg_logprob or bounded confidence into [0.0, 1.0].

    Negative scores represent Whisper average log-probabilities (<= 0) and are
    transformed with temperature-scaled logistic calibration; non-negative scores
    are clamped into [0.0, 1.0]. Missing or non-finite values return ``default``.
    """
    if raw_score is None:
        return float(default)
    val = float(raw_score)
    if not math.isfinite(val):
        return float(default)
    if val < 0.0:
        temp = max(float(temperature), 1e-6)
        prob = _stable_sigmoid((val - float(midpoint)) / temp)
        return float(min(1.0, max(0.0, prob)))
    return float(min(1.0, max(0.0, val)))


def calibrate_multimodal_scores(
    scores: Sequence[float],
    *,
    temperature: float = 0.07,
    center: float | None = None,
) -> tuple[list[float], CalibrationMetadata]:
    """Calibrate raw retrieval fusion scores into (0, 1) monotonic probabilities."""
    vals = [float(s) for s in scores]
    if not vals:
        meta = CalibrationMetadata(
            is_calibrated=True,
            method="temperature_sigmoid",
            parameters={"temperature": float(temperature), "center": 0.0 if center is None else float(center)},
        )
        return [], meta

    temp = max(float(temperature), 1e-6)
    if center is None:
        ordered = sorted(vals)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            ref_center = ordered[mid]
        else:
            ref_center = 0.5 * (ordered[mid - 1] + ordered[mid])
    else:
        ref_center = float(center)

    calibrated = [
        round(min(1.0, max(0.0, _stable_sigmoid((v - ref_center) / temp))), 6)
        for v in vals
    ]
    meta = CalibrationMetadata(
        is_calibrated=True,
        method="temperature_sigmoid",
        parameters={"temperature": temp, "center": float(ref_center)},
    )
    return calibrated, meta
