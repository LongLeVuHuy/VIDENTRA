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
    ) -> GraphMatchResult:
        """Execute full relation and logic verification."""
        # 1. Ground entities
        grounded = ground_entities(query_graph, evidence_items)

        # 2. Evaluate each predicate
        pred_evals: dict[str, PredicateEvaluation] = {}
        intervals: dict[str, tuple[float, float]] = dict(event_intervals) if event_intervals else {}

        # First pass: evaluate non-temporal predicates
        for p in query_graph.predicates:
            if p.kind != "temporal":
                if "negation_scope_unresolved" in p.uncertainty:
                    ev = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
                else:
                    ev = evaluate_predicate(p, evidence_items, grounded)
                pred_evals[p.id] = ev

        # Infer missing event intervals from grounded evidence_refs before temporal pass
        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in evidence_items
            if item.get("evidence_id") not in (None, "")
        }
        for p_id, ev in pred_evals.items():
            if p_id in intervals:
                continue
            if ev.state not in (MatchState.SUPPORTED, MatchState.PARTIAL):
                continue
            spans: list[tuple[float, float]] = []
            for ref in ev.evidence_refs:
                item = evidence_by_id.get(str(ref))
                if item is None:
                    continue
                span = _extract_evidence_interval(item)
                if span is not None:
                    spans.append(span)
            if spans:
                intervals[p_id] = (min(s for s, _ in spans), max(e for _, e in spans))

        # Second pass: evaluate temporal predicates referencing prior events
        for p in query_graph.predicates:
            if p.kind == "temporal":
                if "negation_scope_unresolved" in p.uncertainty:
                    ev = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
                else:
                    ev = evaluate_temporal_order(p, intervals)
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
