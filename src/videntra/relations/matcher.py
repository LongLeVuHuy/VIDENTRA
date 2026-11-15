"""Structural graph matcher aggregating multi-branch evaluations under three-valued logic."""
from dataclasses import dataclass
import math
from typing import Any, Sequence

from videntra.evidence.normalize import contains_text_phrase
from videntra.query.schemas import Predicate, QueryGraph
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


def _collect_predicate_refs(
    pred: Predicate,
    scoped_evidence: Sequence[dict[str, Any]],
    ev: PredicateEvaluation,
) -> list[str]:
    """Collect all matching evidence_ids for a grounded predicate (beyond first-match)."""
    refs: list[str] = [str(r) for r in ev.evidence_refs if r not in (None, "")]
    kind = pred.kind.lower()
    if kind == "ocr":
        for item in scoped_evidence:
            if item.get("kind") in ("ocr_track", "ocr_observation") and item.get("evidence_id") not in (None, ""):
                if contains_text_phrase(item.get("text"), pred.value):
                    refs.append(str(item["evidence_id"]))
    elif kind == "asr":
        for item in scoped_evidence:
            if item.get("kind") == "asr_segment" and item.get("evidence_id") not in (None, ""):
                if contains_text_phrase(item.get("text"), pred.value):
                    refs.append(str(item["evidence_id"]))
    return list(dict.fromkeys(refs))


@dataclass(frozen=True)
class _GroundedOccurrence:
    span: tuple[float, float]
    refs: list[str]
    video_ids: set[str]
    timeline_revisions: set[str]


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

        # Build verified grounded occurrences only from grounded non-temporal events
        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in scoped_evidence
            if item.get("evidence_id") not in (None, "")
        }
        pred_by_id = {p.id: p for p in query_graph.predicates if p.id}
        grounded_occurrences: dict[str, list[_GroundedOccurrence]] = {}
        anchor_failure_reasons: dict[str, str] = {}

        for p_id, ev in pred_evals.items():
            if not p_id or ev.state not in (MatchState.SUPPORTED, MatchState.PARTIAL):
                continue
            pred_obj = pred_by_id.get(p_id)
            candidate_refs = (
                _collect_predicate_refs(pred_obj, scoped_evidence, ev)
                if pred_obj is not None
                else [str(ref) for ref in ev.evidence_refs]
            )
            valid_refs = [str(ref) for ref in candidate_refs if str(ref) in evidence_by_id]
            if not valid_refs:
                anchor_failure_reasons[p_id] = f"No evidence refs for '{p_id}'"
                continue

            timed_refs: list[tuple[str, dict[str, Any], tuple[float, float]]] = []
            for ref in valid_refs:
                item = evidence_by_id[ref]
                span = _extract_evidence_interval(item)
                if span is not None:
                    timed_refs.append((ref, item, span))

            if not timed_refs:
                anchor_failure_reasons[p_id] = f"No timed observations for '{p_id}'"
                continue

            # Caller event_intervals acts strictly as a candidate occurrence filter window
            if p_id in caller_intervals:
                cand_window = caller_intervals[p_id]
                if (
                    len(cand_window) != 2
                    or not math.isfinite(cand_window[0])
                    or not math.isfinite(cand_window[1])
                    or cand_window[0] > cand_window[1]
                ):
                    anchor_failure_reasons[p_id] = f"Invalid candidate window for '{p_id}'"
                    continue
                cand_win_f = (float(cand_window[0]), float(cand_window[1]))
                timed_refs = [
                    (ref, item, span)
                    for ref, item, span in timed_refs
                    if _interval_in_window(span, cand_win_f)
                ]
                if not timed_refs:
                    anchor_failure_reasons[p_id] = (
                        f"No timed evidence refs for '{p_id}' fall within candidate window "
                        f"[{cand_win_f[0]:.2f}s, {cand_win_f[1]:.2f}s]"
                    )
                    continue

            # Group occurrences separately by track_id / segment_id (or per-ref when untrack-grouped)
            grouped: dict[str, list[tuple[str, dict[str, Any], tuple[float, float]]]] = {}
            for ref, item, span in timed_refs:
                if item.get("track_id") not in (None, ""):
                    occ_key = f"track:{item['track_id']}"
                elif item.get("segment_id") not in (None, ""):
                    occ_key = f"segment:{item['segment_id']}"
                else:
                    occ_key = f"ref:{ref}"
                grouped.setdefault(occ_key, []).append((ref, item, span))

            valid_occs: list[_GroundedOccurrence] = []
            for entries in grouped.values():
                occ_vids = {
                    str(it["video_id"])
                    for _, it, _ in entries
                    if it.get("video_id") not in (None, "")
                }
                occ_revs = {
                    str(it["timeline_revision"])
                    for _, it, _ in entries
                    if it.get("timeline_revision") not in (None, "")
                }
                if video_id is not None and any(v != str(video_id) for v in occ_vids):
                    continue
                if len(occ_vids) > 1:
                    continue
                if timeline_revision is not None and any(r != str(timeline_revision) for r in occ_revs):
                    continue
                if len(occ_revs) > 1:
                    continue
                occ_span = (
                    min(s for _, _, (s, _) in entries),
                    max(e for _, _, (_, e) in entries),
                )
                occ_refs = list(dict.fromkeys(r for r, _, _ in entries))
                valid_occs.append(
                    _GroundedOccurrence(
                        span=occ_span,
                        refs=occ_refs,
                        video_ids=occ_vids,
                        timeline_revisions=occ_revs,
                    )
                )

            if not valid_occs:
                anchor_failure_reasons[p_id] = (
                    f"No timeline/video-consistent occurrences for '{p_id}'"
                )
                continue

            grounded_occurrences[p_id] = valid_occs

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
                    or event_a_id not in grounded_occurrences
                    or event_b_id not in grounded_occurrences
                ):
                    reason = (
                        anchor_failure_reasons.get(event_a_id or "")
                        or anchor_failure_reasons.get(event_b_id or "")
                        or f"Missing grounded anchors for events '{event_a_id}' or '{event_b_id}'"
                    )
                    pred_evals[p.id] = PredicateEvaluation(
                        p.id,
                        MatchState.UNKNOWN,
                        0.0,
                        reason,
                        [],
                    )
                    continue

                occs_a = grounded_occurrences[event_a_id]
                occs_b = grounded_occurrences[event_b_id]
                pair_evals: list[PredicateEvaluation] = []
                saw_timeline_mismatch = False
                saw_video_mismatch = False

                for occ_a in occs_a:
                    for occ_b in occs_b:
                        if occ_a.video_ids and occ_b.video_ids and occ_a.video_ids != occ_b.video_ids:
                            saw_video_mismatch = True
                            continue
                        revs_a = occ_a.timeline_revisions
                        revs_b = occ_b.timeline_revisions
                        if (revs_a and revs_b and revs_a != revs_b) or (bool(revs_a) != bool(revs_b)):
                            saw_timeline_mismatch = True
                            continue

                        combined_refs = list(dict.fromkeys([*occ_a.refs, *occ_b.refs]))
                        st_a, end_a = occ_a.span
                        st_b, end_b = occ_b.span
                        if set(occ_a.refs) == set(occ_b.refs) and st_a == st_b and end_a == end_b:
                            pair_evals.append(
                                PredicateEvaluation(
                                    p.id,
                                    MatchState.PARTIAL,
                                    0.0,
                                    "Events share the same anchor/timestamp; temporal order is ambiguous",
                                    combined_refs,
                                )
                            )
                            continue

                        pair_ev = evaluate_temporal_order(
                            p,
                            {event_a_id: occ_a.span, event_b_id: occ_b.span},
                        )
                        if pair_ev.state in (MatchState.SUPPORTED, MatchState.PARTIAL, MatchState.CONTRADICTED):
                            pair_ev = PredicateEvaluation(
                                pair_ev.predicate_id,
                                pair_ev.state,
                                pair_ev.score,
                                pair_ev.reason,
                                combined_refs,
                            )
                        pair_evals.append(pair_ev)

                if not pair_evals:
                    if saw_timeline_mismatch:
                        reason = "Time basis unaligned between events (timeline_revision mismatch)"
                    elif saw_video_mismatch:
                        reason = "Events belong to different videos"
                    else:
                        reason = "No valid occurrence pairs for temporal comparison"
                    pred_evals[p.id] = PredicateEvaluation(p.id, MatchState.UNKNOWN, 0.0, reason, [])
                    continue

                if len(pair_evals) == 1:
                    ev = pair_evals[0]
                else:
                    pair_states = {e.state for e in pair_evals}
                    all_refs = list(dict.fromkeys(r for e in pair_evals for r in e.evidence_refs))
                    if pair_states == {MatchState.SUPPORTED}:
                        best = max(pair_evals, key=lambda e: e.score)
                        ev = PredicateEvaluation(p.id, MatchState.SUPPORTED, best.score, best.reason, all_refs)
                    elif pair_states == {MatchState.CONTRADICTED}:
                        ev = PredicateEvaluation(p.id, MatchState.CONTRADICTED, 0.0, pair_evals[0].reason, all_refs)
                    elif pair_states == {MatchState.UNKNOWN}:
                        ev = pair_evals[0]
                    else:
                        ev = PredicateEvaluation(
                            p.id,
                            MatchState.PARTIAL,
                            0.0,
                            "Multiple occurrences yield conflicting or ambiguous temporal orders",
                            all_refs,
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
