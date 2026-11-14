"""Structural graph matcher aggregating multi-branch evaluations under three-valued logic."""
from dataclasses import dataclass
from typing import Any, Sequence

from videntra.query.schemas import QueryGraph
from .entities import ground_entities, GroundedEntity
from .predicates import evaluate_predicate, PredicateEvaluation, MatchState
from .temporal import evaluate_temporal_order


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
        intervals = event_intervals or {}

        # First pass: evaluate non-temporal predicates
        for p in query_graph.predicates:
            if p.kind != "temporal":
                if "negation_scope_unresolved" in p.uncertainty:
                    ev = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
                else:
                    ev = evaluate_predicate(p, evidence_items, grounded)
                pred_evals[p.id] = ev

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
