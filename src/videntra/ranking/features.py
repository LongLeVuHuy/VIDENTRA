"""Feature extraction for multimodal candidate moments."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from videntra.relations.matcher import GraphMatchResult
    from videntra.retrieval.visual import RankedMoment


LTR_FEATURE_NAMES_18D: tuple[str, ...] = (
    "s_vis_blend_max",
    "z_vis_max",
    "delta_z_vis",
    "s_vis_top3_mean",
    "s_vis_top5_mean",
    "s_vis_ctx_coherence",
    "s_late_interaction",
    "spike_drop_ratio",
    "rrf_vis",
    "rrf_ocr",
    "rrf_asr",
    "bm25_log_score",
    "high_spec_token_coverage",
    "anchor_lexical_match",
    "number_consistency",
    "yolo_object_verification_boost",
    "graph_relation_score",
    "llm_modality_alignment",
)

LTR_FEATURE_NAMES_22D: tuple[str, ...] = LTR_FEATURE_NAMES_18D + (
    "vis_lexical_synergy",
    "unverified_lexical_hazard",
    "subclause_weakest_link",
    "window_energy_concentration",
)

LTR_FEATURE_NAMES: tuple[str, ...] = LTR_FEATURE_NAMES_22D + (
    "temporal_attention_coherence",
    "rrf_vis_caption",
)


@dataclass(frozen=True)
class QueryZScoreResult:
    """Query-conditioned Z-score statistics over top visual candidates."""

    z_scores: np.ndarray
    delta_z_scores: np.ndarray
    mean: float
    std: float
    z_max: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to finite float, falling back to default on None/NaN/Inf."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def compute_query_z_scores(
    scores: Sequence[float] | np.ndarray,
    top_k: int = 120,
    eps: float = 1e-6,
) -> QueryZScoreResult:
    """Compute query-adaptive Z-scores using mean/std over Top-K visual candidates.

    Formula (Plan §4.2):
        z_vis(c) = (s_vis(c) - mu_vis(Q)) / (sigma_vis(Q) + eps)
        delta_z_vis(c) = z_vis_max(Q) - z_vis(c)
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return QueryZScoreResult(
            z_scores=empty,
            delta_z_scores=empty,
            mean=0.0,
            std=0.0,
            z_max=0.0,
        )

    arr = np.where(np.isfinite(arr), arr, 0.0)
    k = max(1, min(int(top_k), arr.size))
    if k < arr.size:
        top_vals = np.partition(arr, arr.size - k)[-k:]
    else:
        top_vals = arr

    mu = float(np.mean(top_vals))
    sigma = float(np.std(top_vals))
    denom = sigma + float(eps)

    z_scores = (arr - mu) / denom
    z_max = float(np.max(z_scores))
    delta_z_scores = z_max - z_scores
    return QueryZScoreResult(
        z_scores=z_scores,
        delta_z_scores=delta_z_scores,
        mean=mu,
        std=sigma,
        z_max=z_max,
    )


def extract_ltr_features_18d(
    candidate: Mapping[str, Any],
    z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
    *,
    return_dict: bool = False,
) -> np.ndarray | dict[str, float]:
    """Extract the 18 continuous LTR features defined in Plan §4.2.

    Returns an ordered float32 NumPy array of shape (18,) by default, or an
    ordered dict[str, float] matching ``LTR_FEATURE_NAMES`` when ``return_dict=True``.
    """
    s_vis_blend_max = _safe_float(candidate.get("s_vis_blend_max"))

    if "z_vis_max" in candidate and candidate["z_vis_max"] is not None:
        z_vis_max = _safe_float(candidate["z_vis_max"])
    elif z_stats is not None:
        if isinstance(z_stats, QueryZScoreResult):
            mu, sigma = z_stats.mean, z_stats.std
        else:
            mu = _safe_float(z_stats.get("mean"))
            sigma = _safe_float(z_stats.get("std"))
        z_vis_max = (s_vis_blend_max - mu) / (sigma + 1e-6)
    else:
        z_vis_max = 0.0

    if "delta_z_vis" in candidate and candidate["delta_z_vis"] is not None:
        delta_z_vis = _safe_float(candidate["delta_z_vis"])
    elif z_stats is not None:
        z_max = z_stats.z_max if isinstance(z_stats, QueryZScoreResult) else _safe_float(z_stats.get("z_max"))
        delta_z_vis = z_max - z_vis_max
    else:
        delta_z_vis = 0.0

    if "spike_drop_ratio" in candidate and candidate["spike_drop_ratio"] is not None:
        spike_drop_ratio = _safe_float(candidate["spike_drop_ratio"])
    elif "kf1_score" in candidate and "kf2_score" in candidate:
        spike_drop_ratio = _safe_float(candidate.get("kf1_score")) - _safe_float(candidate.get("kf2_score"))
    else:
        spike_drop_ratio = 0.0

    if "bm25_log_score" in candidate and candidate["bm25_log_score"] is not None:
        bm25_log_score = _safe_float(candidate["bm25_log_score"])
    else:
        fts_score = max(0.0, _safe_float(candidate.get("fts_score")))
        bm25_log_score = math.log1p(fts_score)

    if "high_spec_token_coverage" in candidate and candidate["high_spec_token_coverage"] is not None:
        high_spec_token_coverage = _safe_float(candidate["high_spec_token_coverage"])
    else:
        m_high = max(0.0, _safe_float(candidate.get("m_high")))
        k_high = max(1.0, _safe_float(candidate.get("k_high"), 1.0))
        high_spec_token_coverage = m_high / k_high

    raw_anchor = candidate.get("anchor_lexical_match", 0.0)
    anchor_lexical_match = 1.0 if raw_anchor is True else (0.0 if raw_anchor is False else _safe_float(raw_anchor))

    values: dict[str, float] = {
        "s_vis_blend_max": s_vis_blend_max,
        "z_vis_max": z_vis_max,
        "delta_z_vis": delta_z_vis,
        "s_vis_top3_mean": _safe_float(candidate.get("s_vis_top3_mean")),
        "s_vis_top5_mean": _safe_float(candidate.get("s_vis_top5_mean")),
        "s_vis_ctx_coherence": _safe_float(candidate.get("s_vis_ctx_coherence")),
        "s_late_interaction": _safe_float(candidate.get("s_late_interaction")),
        "spike_drop_ratio": spike_drop_ratio,
        "rrf_vis": _safe_float(candidate.get("rrf_vis")),
        "rrf_ocr": _safe_float(candidate.get("rrf_ocr")),
        "rrf_asr": _safe_float(candidate.get("rrf_asr")),
        "bm25_log_score": bm25_log_score,
        "high_spec_token_coverage": high_spec_token_coverage,
        "anchor_lexical_match": anchor_lexical_match,
        "number_consistency": _safe_float(candidate.get("number_consistency")),
        "yolo_object_verification_boost": _safe_float(candidate.get("yolo_object_verification_boost")),
        "graph_relation_score": _safe_float(candidate.get("graph_relation_score")),
        "llm_modality_alignment": _safe_float(candidate.get("llm_modality_alignment")),
    }

    if return_dict:
        return values
    return np.array([values[name] for name in LTR_FEATURE_NAMES_18D], dtype=np.float32)


def extract_ltr_features_22d(
    candidate: Mapping[str, Any],
    z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
    *,
    return_dict: bool = False,
) -> np.ndarray | dict[str, float]:
    """Extract the 22 continuous Synergistic-CrossLTR features (Phase 8 P8-3)."""
    values = dict(extract_ltr_features_18d(candidate, z_stats=z_stats, return_dict=True))
    s_vis = values["s_vis_blend_max"]
    z_vis = values["z_vis_max"]
    lex_rrf_sum = values["rrf_ocr"] + values["rrf_asr"]

    if "vis_lexical_synergy" in candidate and candidate["vis_lexical_synergy"] is not None:
        vis_lexical_synergy = _safe_float(candidate["vis_lexical_synergy"])
    else:
        vis_lexical_synergy = max(0.0, s_vis - 0.12) * lex_rrf_sum

    if "unverified_lexical_hazard" in candidate and candidate["unverified_lexical_hazard"] is not None:
        unverified_lexical_hazard = _safe_float(candidate["unverified_lexical_hazard"])
    else:
        unverified_lexical_hazard = max(0.0, 0.50 - z_vis) * lex_rrf_sum

    if "subclause_weakest_link" in candidate and candidate["subclause_weakest_link"] is not None:
        subclause_weakest_link = _safe_float(candidate["subclause_weakest_link"])
    else:
        subclause_weakest_link = s_vis

    if "window_energy_concentration" in candidate and candidate["window_energy_concentration"] is not None:
        window_energy_concentration = _safe_float(candidate["window_energy_concentration"])
    else:
        s_top3 = values["s_vis_top3_mean"]
        window_energy_concentration = (s_top3 / max(s_vis, 0.05)) if s_vis > 0.0 else 0.0

    values["vis_lexical_synergy"] = vis_lexical_synergy
    values["unverified_lexical_hazard"] = unverified_lexical_hazard
    values["subclause_weakest_link"] = subclause_weakest_link
    values["window_energy_concentration"] = window_energy_concentration

    if return_dict:
        return values
    return np.array([values[name] for name in LTR_FEATURE_NAMES_22D], dtype=np.float32)


def extract_ltr_features_24d(
    candidate: Mapping[str, Any],
    z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
    *,
    return_dict: bool = False,
) -> np.ndarray | dict[str, float]:
    """Extract the 24 continuous LTR features (22D + temporal_attention_coherence + rrf_vis_caption)."""
    values = dict(extract_ltr_features_22d(candidate, z_stats=z_stats, return_dict=True))
    values["temporal_attention_coherence"] = _safe_float(candidate.get("temporal_attention_coherence"))
    values["rrf_vis_caption"] = _safe_float(candidate.get("rrf_vis_caption"))

    if return_dict:
        return values
    return np.array([values[name] for name in LTR_FEATURE_NAMES], dtype=np.float32)


def extract_ltr_features_batch(
    candidates: Sequence[Mapping[str, Any]],
    z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Extract an (N, 24) float32 feature matrix for a batch of candidates."""
    if not candidates:
        return np.zeros((0, len(LTR_FEATURE_NAMES)), dtype=np.float32)
    rows = [extract_ltr_features_24d(cand, z_stats=z_stats, return_dict=False) for cand in candidates]
    return np.vstack(rows).astype(np.float32, copy=False)


@dataclass(frozen=True)
class MomentFeatures:
    moment_id: str
    video_id: str
    rrf_score: float
    duration_seconds: float
    structural_score: float
    is_contradicted: bool
    is_fully_supported: bool
    unresolved_count: int

    def to_feature_vector(self) -> list[float]:
        return [
            self.rrf_score,
            self.duration_seconds,
            self.structural_score,
            1.0 if self.is_contradicted else 0.0,
            1.0 if self.is_fully_supported else 0.0,
            float(self.unresolved_count),
        ]


def extract_moment_features(
    moment: RankedMoment,
    graph_result: GraphMatchResult | None = None,
) -> MomentFeatures:
    """Extract tabular features from moment and graph match results."""
    duration = max(0.0, moment.moment_end_seconds - moment.moment_start_seconds)
    if graph_result is None:
        return MomentFeatures(
            moment_id=moment.moment_id,
            video_id=moment.video_id,
            rrf_score=float(moment.ranking_score),
            duration_seconds=round(duration, 4),
            structural_score=0.0,
            is_contradicted=False,
            is_fully_supported=False,
            unresolved_count=0,
        )

    is_contra = graph_result.overall_status == "contradicted"
    is_supp = graph_result.overall_status == "supported"
    return MomentFeatures(
        moment_id=moment.moment_id,
        video_id=moment.video_id,
        rrf_score=float(moment.ranking_score),
        duration_seconds=round(duration, 4),
        structural_score=float(graph_result.structural_score),
        is_contradicted=is_contra,
        is_fully_supported=is_supp,
        unresolved_count=len(graph_result.unresolved_predicates),
    )
