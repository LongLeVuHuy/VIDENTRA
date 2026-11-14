"""Feature extraction for multimodal candidate moments."""
from dataclasses import dataclass
from typing import Any

from videntra.retrieval.visual import RankedMoment
from videntra.relations.matcher import GraphMatchResult


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
