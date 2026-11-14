"""Temporal order evaluation between events enforcing interval boundaries and tolerances."""
from typing import Any
import math
from videntra.query.schemas import Predicate
from .predicates import PredicateEvaluation, MatchState


def evaluate_temporal_order(
    pred: Predicate,
    event_intervals: dict[str, tuple[float, float]],  # pred_id -> (start_sec, end_sec)
    tolerance_seconds: float = 0.3,
) -> PredicateEvaluation:
    """Evaluate 'before' or 'after' temporal ordering between two grounded events."""
    if "negation_scope_unresolved" in pred.uncertainty:
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
    if pred.kind != "temporal":
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Not a temporal predicate", [])

    event_a_id = pred.roles.get("event_a")
    event_b_id = pred.roles.get("event_b")

    if not event_a_id or not event_b_id:
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Missing event roles", [])

    if event_a_id not in event_intervals or event_b_id not in event_intervals:
        return PredicateEvaluation(
            pred.id,
            MatchState.UNKNOWN,
            0.0,
            f"Missing intervals for events '{event_a_id}' or '{event_b_id}'",
            [],
        )

    st_a, end_a = event_intervals[event_a_id]
    st_b, end_b = event_intervals[event_b_id]

    op = pred.operator.lower()
    if not all(math.isfinite(x) for x in (st_a, end_a, st_b, end_b)) or st_a > end_a or st_b > end_b:
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Invalid event intervals", [])
    if pred.polarity == "negative":
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Negative temporal scope is not verified", [])
    if op == "after":
        st_a, end_a, st_b, end_b = st_b, end_b, st_a, end_a
        event_a_id, event_b_id = event_b_id, event_a_id
        op = "before"
    if op == "before":
        # Event A must occur before Event B (allowing slight tolerance)
        if end_a <= st_b + tolerance_seconds:
            if end_a > st_b:
                return PredicateEvaluation(pred.id, MatchState.PARTIAL, 0.0,
                    "Events overlap within temporal tolerance; order is ambiguous", [])
            gap_ms = int(round((st_b - end_a) * 1000))
            # Check scope min/max gap bounds if present
            if pred.scope:
                if pred.scope.min_gap_ms is not None and gap_ms < pred.scope.min_gap_ms:
                    return PredicateEvaluation(
                        pred.id,
                        MatchState.CONTRADICTED,
                        0.0,
                        f"Temporal gap {gap_ms}ms < min_gap_ms {pred.scope.min_gap_ms}ms",
                        [],
                    )
                if pred.scope.max_gap_ms is not None and gap_ms > pred.scope.max_gap_ms:
                    return PredicateEvaluation(
                        pred.id,
                        MatchState.CONTRADICTED,
                        0.0,
                        f"Temporal gap {gap_ms}ms > max_gap_ms {pred.scope.max_gap_ms}ms",
                        [],
                    )

            return PredicateEvaluation(
                pred.id,
                MatchState.SUPPORTED,
                0.95,
                f"Event '{event_a_id}' [{st_a:.2f}s, {end_a:.2f}s] occurred before '{event_b_id}' [{st_b:.2f}s, {end_b:.2f}s]",
                [],
            )
        else:
            return PredicateEvaluation(
                pred.id,
                MatchState.CONTRADICTED,
                0.0,
                f"Order violated: '{event_a_id}' ended at {end_a:.2f}s after '{event_b_id}' started at {st_b:.2f}s",
                [],
            )

    return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, f"Unsupported temporal operator '{op}'", [])
