"""Structural graph matcher aggregating multi-branch evaluations under three-valued logic."""
from dataclasses import dataclass
import math
from typing import Any, Sequence

from videntra.query.schemas import QueryGraph
from .entities import ground_entities, GroundedEntity
from .predicates import evaluate_predicate, PredicateEvaluation, MatchState
from .temporal import evaluate_temporal_order


def _extract_evidence_interval(item: dict[str, Any]) -> tuple[float, float] | None:
    """Extract (start_sec, end_sec) interval from an evidence item."""
    start_sec: float | None = None
    end_sec: float | None = None

    for key in ("start_sec", "start_seconds"):
        if item.get(key) is not None:
            start_sec = float(item[key])
            break
    if start_sec is None and item.get("start_ms") is not None:
        start_sec = float(item["start_ms"]) / 1000.0

    for key in ("end_sec", "end_seconds"):
        if item.get(key) is not None:
            end_sec = float(item[key])
            break
    if end_sec is None and item.get("end_ms") is not None:
        end_sec = float(item["end_ms"]) / 1000.0

    if start_sec is not None and end_sec is not None:
        if math.isfinite(start_sec) and math.isfinite(end_sec) and start_sec <= end_sec:
            return (start_sec, end_sec)

    point_sec: float | None = None
    for key in ("pts_sec", "timestamp_seconds"):
        if item.get(key) is not None:
            point_sec = float(item[key])
            break
    if point_sec is None and item.get("timestamp_ms") is not None:
        point_sec = float(item["timestamp_ms"]) / 1000.0

    if point_sec is not None and math.isfinite(point_sec):
        return (point_sec, point_sec)
    return None


def _interval_in_window(
    span: tuple[float, float],
    window: tuple[float, float],
    eps: float = 1e-6,
) -> bool:
    """Return True if ``span`` is contained within ``window``."""
    return (window[0] - eps) <= span[0] and span[1] <= (window[1] + eps)


def _scope_evidence_items(
    evidence_items: Sequence[dict[str, Any]],
    *,
    video_id: str | None = None,
    moment_window: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Filter evidence items to the requested video_id and moment_window."""
    if video_id is None and moment_window is None:
        return list(evidence_items)
    scoped: list[dict[str, Any]] = []
    for item in evidence_items:
        item_vid = item.get("video_id")
        if video_id is not None and item_vid not in (None, "") and str(item_vid) != str(video_id):
            continue
        if moment_window is not None:
            span = _extract_evidence_interval(item)
            if span is None or not _interval_in_window(span, moment_window):
                continue
        scoped.append(item)
    return scoped


@dataclass(frozen=True)
class GraphMatchResult:
    overall_status: str  # "supported", "partial", "contradicted", "unknown"
    structural_score: float
    predicate_evaluations: dict[str, PredicateEvaluation]
    unresolved_predicates: list[str]


class GraphMatcher:
    """Evaluates QueryGraph structure against candidate moment evidence graph."""

    def match(
        self,
        query_graph: QueryGraph,
        evidence_items: Sequence[dict[str, Any]],
        event_intervals: dict[str, tuple[float, float]] | None = None,
        *,
        video_id: str | None = None,
        timeline_revision: str | None = None,
        moment_window: tuple[float, float] | None = None,
    ) -> GraphMatchResult:
        """Execute full relation and logic verification."""
        scoped_evidence = _scope_evidence_items(
            evidence_items,
            video_id=video_id,
            moment_window=moment_window,
        )

        # 1. Ground entities
        grounded = ground_entities(query_graph, scoped_evidence)

        # 2. Evaluate each predicate
        pred_evals: dict[str, PredicateEvaluation] = {}
        caller_intervals = dict(event_intervals) if event_intervals else {}

        # First pass: evaluate non-temporal predicates
        for p in query_graph.predicates:
            if p.kind != "temporal":
                if "negation_scope_unresolved" in p.uncertainty:
                    ev = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
                else:
                    ev = evaluate_predicate(p, scoped_evidence, grounded)
                pred_evals[p.id] = ev

        # Build verified grounded intervals & refs only from grounded non-temporal events
        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in scoped_evidence
            if item.get("evidence_id") not in (None, "")
        }
        grounded_intervals: dict[str, tuple[float, float]] = {}
        grounded_refs: dict[str, list[str]] = {}
        grounded_revs: dict[str, set[str]] = {}
        grounded_vids: dict[str, set[str]] = {}

        for p_id, ev in pred_evals.items():
            if not p_id or ev.state not in (MatchState.SUPPORTED, MatchState.PARTIAL):
                continue
            valid_refs = [str(ref) for ref in ev.evidence_refs if str(ref) in evidence_by_id]
            if not valid_refs:
                continue

            item_vids: set[str] = set()
            item_revs: set[str] = set()
            spans: list[tuple[float, float]] = []
            for ref in valid_refs:
                item = evidence_by_id[ref]
                if item.get("video_id") not in (None, ""):
                    item_vids.add(str(item["video_id"]))
                if item.get("timeline_revision") not in (None, ""):
                    item_revs.add(str(item["timeline_revision"]))
                span = _extract_evidence_interval(item)
                if span is not None:
                    spans.append(span)

            if video_id is not None and any(v != str(video_id) for v in item_vids):
                continue
            if len(item_vids) > 1:
                continue
            if timeline_revision is not None and any(r != str(timeline_revision) for r in item_revs):
                continue
            if len(item_revs) > 1:
                continue

            chosen_span: tuple[float, float] | None = None
            if p_id in caller_intervals:
                cand_span = caller_intervals[p_id]
                if (
                    len(cand_span) == 2
                    and math.isfinite(cand_span[0])
                    and math.isfinite(cand_span[1])
                    and cand_span[0] <= cand_span[1]
                    and (moment_window is None or _interval_in_window(cand_span, moment_window))
                ):
                    chosen_span = (float(cand_span[0]), float(cand_span[1]))
            elif spans:
                chosen_span = (min(s for s, _ in spans), max(e for _, e in spans))

            if chosen_span is None:
                continue

            grounded_intervals[p_id] = chosen_span
            grounded_refs[p_id] = list(dict.fromkeys(valid_refs))
            grounded_revs[p_id] = item_revs
            grounded_vids[p_id] = item_vids

        # Second pass: evaluate temporal predicates referencing prior grounded events
        for p in query_graph.predicates:
            if p.kind == "temporal":
                if "negation_scope_unresolved" in p.uncertainty:
                    ev = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
                    pred_evals[p.id] = ev
                    continue

                event_a_id = p.roles.get("event_a")
                event_b_id = p.roles.get("event_b")
                if (
                    not event_a_id
                    or not event_b_id
                    or event_a_id not in grounded_intervals
                    or event_b_id not in grounded_intervals
                ):
                    pred_evals[p.id] = PredicateEvaluation(
                        p.id,
                        MatchState.UNKNOWN,
                        0.0,
                        f"Missing grounded anchors for events '{event_a_id}' or '{event_b_id}'",
                        [],
                    )
                    continue

                revs_a = grounded_revs.get(event_a_id, set())
                revs_b = grounded_revs.get(event_b_id, set())
                if revs_a and revs_b and revs_a != revs_b:
                    pred_evals[p.id] = PredicateEvaluation(
                        p.id,
                        MatchState.UNKNOWN,
                        0.0,
                        "Time basis unaligned between events (timeline_revision mismatch)",
                        [],
                    )
                    continue

                vids_a = grounded_vids.get(event_a_id, set())
                vids_b = grounded_vids.get(event_b_id, set())
                if vids_a and vids_b and vids_a != vids_b:
                    pred_evals[p.id] = PredicateEvaluation(
                        p.id,
                        MatchState.UNKNOWN,
                        0.0,
                        "Events belong to different videos",
                        [],
                    )
                    continue

                refs_a = grounded_refs.get(event_a_id, [])
                refs_b = grounded_refs.get(event_b_id, [])
                combined_refs = list(dict.fromkeys([*refs_a, *refs_b]))

                st_a, end_a = grounded_intervals[event_a_id]
                st_b, end_b = grounded_intervals[event_b_id]
                if set(refs_a) == set(refs_b) and st_a == st_b and end_a == end_b:
                    pred_evals[p.id] = PredicateEvaluation(
                        p.id,
                        MatchState.PARTIAL,
                        0.0,
                        "Events share the same anchor/timestamp; temporal order is ambiguous",
                        combined_refs,
                    )
                    continue

                ev = evaluate_temporal_order(p, grounded_intervals)
                if ev.state in (MatchState.SUPPORTED, MatchState.PARTIAL, MatchState.CONTRADICTED):
                    ev = PredicateEvaluation(
                        ev.predicate_id,
                        ev.state,
                        ev.score,
                        ev.reason,
                        combined_refs,
                    )
                pred_evals[p.id] = ev

        # 3. Evaluate logic root node
        # Default root
        root_node = next((l for l in query_graph.logic if l.id == query_graph.root_id), None)
        child_ids = root_node.children if root_node else list(pred_evals.keys())

        states = [pred_evals[c_id].state for c_id in child_ids if c_id in pred_evals]

        # Three-valued logic aggregation
        operator = root_node.op.upper() if root_node else "AND"
        if len(states) != len(child_ids) or operator not in ("AND", "OR"):
            overall = "unknown"
            score = 0.0
        elif operator == "OR" and any(s == MatchState.SUPPORTED for s in states):
            overall = "supported"
            score = 1.0
        elif operator == "OR" and states and all(s == MatchState.CONTRADICTED for s in states):
            overall = "contradicted"
            score = 0.0
        elif operator == "AND" and any(s == MatchState.CONTRADICTED for s in states):
            overall = "contradicted"
            score = 0.0
        elif all(s == MatchState.SUPPORTED for s in states) and len(states) > 0:
            overall = "supported"
            score = 1.0
        elif not states or all(s == MatchState.UNKNOWN for s in states) or (operator == "OR" and MatchState.UNKNOWN in states and MatchState.PARTIAL not in states):
            overall = "unknown"
            score = 0.0
        else:
            overall = "partial"
            supported_count = sum(1 for s in states if s == MatchState.SUPPORTED)
            score = round(supported_count / max(1, len(states)), 4)

        unresolved = [
            p_id for p_id, ev in pred_evals.items()
            if ev.state in (MatchState.UNKNOWN, MatchState.PARTIAL, MatchState.CONTRADICTED)
        ]

        return GraphMatchResult(
            overall_status=overall,
            structural_score=score,
            predicate_evaluations=pred_evals,
            unresolved_predicates=unresolved,
        )
