"""Relation-aware reranker applying GraphMatcher results and structural verification."""
from typing import Any, Sequence

from videntra.query.schemas import QueryGraph
from videntra.relations.matcher import GraphMatcher, GraphMatchResult
from videntra.retrieval.visual import RankedMoment


class RelationAwareRanker:
    """Reranks candidate moments by combining initial multimodal fusion scores with graph match verification."""

    def __init__(
        self,
        structural_weight: float = 0.4,
        contradiction_penalty: float = 0.85,
        time_margin_seconds: float = 0.0,
    ):
        self.structural_weight = structural_weight
        self.contradiction_penalty = contradiction_penalty
        self.time_margin_seconds = time_margin_seconds
        self.matcher = GraphMatcher()

    def rank(
        self,
        moments: Sequence[RankedMoment],
        query_graph: QueryGraph | None = None,
        evidence_by_video: dict[str, Sequence[dict[str, Any]]] | None = None,
        event_intervals_by_video: dict[str, dict[str, tuple[float, float]]] | None = None,
    ) -> list[RankedMoment]:
        """Rerank candidate moments."""
        if not moments:
            return []

        if query_graph is None or not query_graph.predicates:
            # Fallback to pure sorting by initial score
            return sorted(moments, key=lambda m: m.ranking_score, reverse=True)

        ev_map = evidence_by_video or {}
        event_map = event_intervals_by_video or {}
        reranked: list[RankedMoment] = []

        for m in moments:
            video_ev = ev_map.get(m.video_id, [])

            # Filter evidence near candidate window
            t_min = m.moment_start_seconds - self.time_margin_seconds
            t_max = m.moment_end_seconds + self.time_margin_seconds

            def within_window(ev: dict[str, Any]) -> bool:
                timestamp = ev.get("timestamp_seconds")
                if isinstance(timestamp, (int, float)):
                    return t_min <= timestamp < t_max
                start, end = ev.get("start_seconds"), ev.get("end_seconds")
                return (isinstance(start, (int, float)) and isinstance(end, (int, float))
                        and start < end and start >= t_min and end <= t_max)

            relevant_ev = [ev for ev in video_ev if within_window(ev)]

            ev_intervals = event_map.get(m.video_id, {})

            match_res: GraphMatchResult = self.matcher.match(
                query_graph=query_graph,
                evidence_items=relevant_ev,
                event_intervals=ev_intervals,
            )

            # Score calculation
            base_score = float(m.ranking_score)
            if match_res.overall_status == "contradicted":
                final_score = max(0.0001, base_score * (1.0 - self.contradiction_penalty))
            elif match_res.overall_status == "unknown":
                # Missing observations do not establish a contradiction.
                final_score = base_score
            else:
                weighted_score = (
                    base_score * (1.0 - self.structural_weight)
                    + match_res.structural_score * self.structural_weight
                )
                final_score = max(base_score, weighted_score) if match_res.overall_status == "partial" else weighted_score

            new_moment = RankedMoment(
                moment_id=m.moment_id,
                video_id=m.video_id,
                moment_start_seconds=m.moment_start_seconds,
                moment_end_seconds=m.moment_end_seconds,
                start_original_frame_index=m.start_original_frame_index,
                end_original_frame_index=m.end_original_frame_index,
                representative_original_frame_index=m.representative_original_frame_index,
                representative_timestamp_seconds=m.representative_timestamp_seconds,
                ranking_score=round(final_score, 4),
                score_type="relation_aware_ranking",
                match_status=match_res.overall_status,
                evidence_frame_indices=m.evidence_frame_indices,
                unresolved_predicates=tuple(match_res.unresolved_predicates),
                structural_score=round(match_res.structural_score, 4),
            )
            reranked.append(new_moment)

        # Sort descending by updated ranking_score
        reranked.sort(key=lambda x: x.ranking_score, reverse=True)
        return reranked
