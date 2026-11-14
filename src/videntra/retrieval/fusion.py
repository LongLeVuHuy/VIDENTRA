"""Reciprocal Rank Fusion (RRF) and multimodal temporal clustering."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence
import math
import uuid

from .branches import BranchCandidate
from .visual import RankedMoment, temporal_iou


@dataclass
class MultimodalCluster:
    video_id: str
    candidates: list[BranchCandidate]
    start_seconds: float
    end_seconds: float
    rrf_score: float = 0.0

    @property
    def branches(self) -> set[str]:
        return set(c.branch_name for c in self.candidates)


class MultimodalMomentFusion:
    """Fuses multi-branch retrieval candidates into unified ranked moments using RRF."""

    def __init__(
        self,
        k_rrf: int = 60,
        temporal_iou_threshold: float = 0.3,
        branch_weights: dict[str, float] | None = None,
    ):
        self.k_rrf = k_rrf
        self.temporal_iou_threshold = temporal_iou_threshold
        self.branch_weights = branch_weights or {"visual": 1.0, "ocr": 1.2, "asr": 1.1}

    def fuse(
        self,
        visual_candidates: Sequence[BranchCandidate],
        ocr_candidates: Sequence[BranchCandidate],
        asr_candidates: Sequence[BranchCandidate],
        top_k: int = 10,
    ) -> list[RankedMoment]:
        """Execute reciprocal rank fusion and temporal boundary clustering."""
        # 1. Compute intra-branch 1-based ranks
        branch_ranks: dict[str, dict[str, int]] = {
            "visual": {c.candidate_id: idx for idx, c in enumerate(visual_candidates, 1)},
            "ocr": {c.candidate_id: idx for idx, c in enumerate(ocr_candidates, 1)},
            "asr": {c.candidate_id: idx for idx, c in enumerate(asr_candidates, 1)},
        }

        all_candidates: list[BranchCandidate] = []
        all_candidates.extend(visual_candidates)
        all_candidates.extend(ocr_candidates)
        all_candidates.extend(asr_candidates)

        if not all_candidates:
            return []

        # 2. Group candidates by video_id
        by_video: dict[str, list[BranchCandidate]] = defaultdict(list)
        for c in all_candidates:
            by_video[c.video_id].append(c)

        fused_clusters: list[MultimodalCluster] = []

        # 3. Temporal clustering within each video
        for vid, cands in by_video.items():
            # Sort chronologically by start_seconds
            cands.sort(key=lambda c: (c.start_seconds, c.end_seconds))

            clusters: list[MultimodalCluster] = []
            for c in cands:
                merged = False
                for cluster in clusters:
                    iou = temporal_iou(c.start_seconds, c.end_seconds, cluster.start_seconds, cluster.end_seconds)
                    overlap = max(0.0, min(c.end_seconds, cluster.end_seconds) - max(c.start_seconds, cluster.start_seconds))
                    if iou >= self.temporal_iou_threshold or overlap > 0.5:
                        cluster.candidates.append(c)
                        cluster.start_seconds = min(cluster.start_seconds, c.start_seconds)
                        cluster.end_seconds = max(cluster.end_seconds, c.end_seconds)
                        merged = True
                        break

                if not merged:
                    clusters.append(
                        MultimodalCluster(
                            video_id=vid,
                            candidates=[c],
                            start_seconds=c.start_seconds,
                            end_seconds=c.end_seconds,
                        )
                    )

            # 4. Compute cluster RRF scores with anti-double-counting (max per modality)
            for cl in clusters:
                total_rrf = 0.0
                cands_by_modality: dict[str, list[BranchCandidate]] = defaultdict(list)
                for c in cl.candidates:
                    cands_by_modality[c.branch_name].append(c)

                for mod, mod_cands in cands_by_modality.items():
                    weight = self.branch_weights.get(mod, 1.0)
                    best_mod_rrf = max(
                        1.0 / (self.k_rrf + branch_ranks.get(mod, {}).get(c.candidate_id, 100))
                        for c in mod_cands
                    )
                    total_rrf += weight * best_mod_rrf
                cl.rrf_score = total_rrf

            fused_clusters.extend(clusters)

        # 5. Global rank across all videos by RRF score
        fused_clusters.sort(key=lambda cl: cl.rrf_score, reverse=True)

        ranked_moments: list[RankedMoment] = []
        for cl in fused_clusters[:top_k]:
            m_id = f"fused_{uuid.uuid4().hex[:8]}"

            # Determine representative ordinal & timestamp
            vis_cands = [c for c in cl.candidates if c.branch_name == "visual"]
            if vis_cands:
                rep_ord = vis_cands[0].start_ordinal or 0
                rep_ts = vis_cands[0].start_seconds
            else:
                rep_ord = cl.candidates[0].start_ordinal or 0
                rep_ts = cl.start_seconds

            # Collect unique evidence items
            evidence_items = []
            seen_ev_ids = set()
            evidence_ordinals = []
            for c in cl.candidates:
                if c.evidence_item:
                    ev_id = c.evidence_item.get("evidence_id")
                    if ev_id not in seen_ev_ids:
                        seen_ev_ids.add(ev_id)
                        evidence_items.append(c.evidence_item)
                if c.start_ordinal is not None:
                    evidence_ordinals.append(c.start_ordinal)

            st_ord = min((c.start_ordinal for c in cl.candidates if c.start_ordinal is not None), default=rep_ord)
            end_ord = max((c.end_ordinal for c in cl.candidates if c.end_ordinal is not None), default=rep_ord)

            match_status = "supported" if len(cl.branches) >= 2 else "partial"

            moment = RankedMoment(
                moment_id=m_id,
                video_id=cl.video_id,
                moment_start_seconds=round(cl.start_seconds, 4),
                moment_end_seconds=round(cl.end_seconds, 4),
                start_original_frame_index=st_ord,
                end_original_frame_index=end_ord,
                representative_original_frame_index=rep_ord,
                representative_timestamp_seconds=round(rep_ts, 4),
                ranking_score=round(cl.rrf_score, 4),
                match_status=match_status,
                evidence_frame_indices=tuple(evidence_ordinals) if evidence_ordinals else (rep_ord,),
            )
            ranked_moments.append(moment)

        return ranked_moments


def aggregate_video_moments(
    moments: Sequence[RankedMoment],
    top_k: int = 10,
    alpha: float = 0.60,
    density_bonus_factor: float = 0.015,
    video_keyframe_counts: dict[str, int] | None = None,
    length_damping_baseline: float = 12.0,
    multi_scene_penalty_coeff: float = 0.006,
) -> list[RankedMoment]:
    """Video-Level Max-Mean Aggregation with Temporal Density Bonus, Length Damping, and Deduplication.
    
    When video_keyframe_counts is provided:
      length_damping = 1.0 / max(1.0, sqrt(N_kf / length_damping_baseline))
      factor = 0.008 * length_damping
      relative_density_bonus = (factor * effective_s_max) * min(adjacent_pairs, 3)
      multi_scene_penalty = multi_scene_penalty_coeff * ln(N_kf / 15.0) if N_kf > 30 and no text/audio else 0.0
      S_video = 0.75 * s_max + 0.18 * top3_mean + 0.07 * top5_mean + relative_density_bonus - multi_scene_penalty
    When video_keyframe_counts is None:
      Legacy formula (0.65, 0.25, 0.10, undamped bonus) for backward compatibility.
    """
    if not moments:
        return []

    by_video: dict[str, list[RankedMoment]] = defaultdict(list)
    for m in moments:
        by_video[m.video_id].append(m)

    aggregated_candidates = []
    for vid, vid_moments in by_video.items():
        # Sort moments descending by score
        vid_moments.sort(key=lambda m: float(m.ranking_score), reverse=True)
        scores = [float(m.ranking_score) for m in vid_moments]
        m_count = len(scores)

        s_max = scores[0]
        top3_mean = sum(scores[: min(m_count, 3)]) / min(m_count, 3)
        top5_mean = sum(scores[: min(m_count, 5)]) / min(m_count, 5)

        # Count adjacent moment pairs within 6.0s
        adjacent_pairs = 0
        sorted_by_time = sorted(vid_moments, key=lambda m: m.moment_start_seconds)
        for i in range(len(sorted_by_time) - 1):
            if sorted_by_time[i + 1].moment_start_seconds - sorted_by_time[i].moment_end_seconds <= 6.0:
                adjacent_pairs += 1

        effective_s_max = max(0.0, s_max)

        if video_keyframe_counts is not None:
            n_kf = float(video_keyframe_counts.get(vid, 12))
            length_damping = 1.0 / max(1.0, math.sqrt(n_kf / max(1.0, length_damping_baseline)))
            factor = 0.008 * length_damping
            relative_density_bonus = (factor * effective_s_max) * min(adjacent_pairs, 3)

            has_text_or_audio = any(
                m.match_status and any(b in m.match_status.lower() for b in ("ocr", "asr", "text"))
                for m in vid_moments
            )

            if not has_text_or_audio and n_kf > 30.0:
                multi_scene_penalty = multi_scene_penalty_coeff * math.log(n_kf / 15.0)
            else:
                multi_scene_penalty = 0.0

            video_score = (
                0.75 * s_max
                + 0.18 * top3_mean
                + 0.07 * top5_mean
                + relative_density_bonus
                - multi_scene_penalty
            )
        else:
            # Relative density bonus scaled proportionally to s_max (up to 6% of s_max)
            factor = density_bonus_factor if density_bonus_factor != 0.015 else 0.02
            relative_density_bonus = (factor * effective_s_max) * min(adjacent_pairs, 3)
            video_score = 0.65 * s_max + 0.25 * top3_mean + 0.10 * top5_mean + relative_density_bonus

        best_moment = vid_moments[0]

        # Combine modality attribution across all moments belonging to this video
        all_branches_in_video: set[str] = set()
        for m in vid_moments:
            if m.match_status and m.match_status not in ("supported", "partial", "unknown", "contradicted"):
                for b in m.match_status.split("+"):
                    if b.strip():
                        all_branches_in_video.add(b.strip())
        combined_status = "+".join(sorted(all_branches_in_video)) if all_branches_in_video else best_moment.match_status

        # Return new RankedMoment with aggregated video score
        aggregated_moment = RankedMoment(
            moment_id=best_moment.moment_id,
            video_id=vid,
            moment_start_seconds=best_moment.moment_start_seconds,
            moment_end_seconds=best_moment.moment_end_seconds,
            start_original_frame_index=best_moment.start_original_frame_index,
            end_original_frame_index=best_moment.end_original_frame_index,
            representative_original_frame_index=best_moment.representative_original_frame_index,
            representative_timestamp_seconds=best_moment.representative_timestamp_seconds,
            ranking_score=round(float(video_score), 5),
            score_type="aggregated_video_score",
            match_status=combined_status,
            evidence_frame_indices=best_moment.evidence_frame_indices,
            unresolved_predicates=best_moment.unresolved_predicates,
            structural_score=best_moment.structural_score,
        )
        aggregated_candidates.append(aggregated_moment)

    # Sort descending by video score
    aggregated_candidates.sort(key=lambda m: float(m.ranking_score), reverse=True)
    return aggregated_candidates[:top_k]
