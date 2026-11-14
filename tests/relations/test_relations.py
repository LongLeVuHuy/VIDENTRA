"""Unit tests for relation grounding, predicates, temporal order, and graph matcher."""
import pytest

from videntra.query.schemas import (
    Entity,
    EntityAttribute,
    Predicate,
    PredicateScope,
    LogicNode,
    QueryGraph,
)
from videntra.relations.entities import ground_entities
from videntra.relations.predicates import evaluate_predicate, MatchState
from videntra.relations.temporal import evaluate_temporal_order
from videntra.relations.matcher import GraphMatcher


def test_entity_grounding_matches_labels():
    query_graph = QueryGraph(
        original_query="người áo đỏ",
        language="vi",
        entities=[Entity(id="e1", type="person", mention="người", attributes=[EntityAttribute("color", "đỏ", "đỏ")])],
        predicates=[],
        logic=[],
        root_id="",
    )
    evidence = [
        {"kind": "visual_detection", "label": "person", "attributes": {"color": "đỏ"}},
        {"kind": "visual_detection", "label": "car"},
    ]
    grounded = ground_entities(query_graph, evidence)
    assert "e1" in grounded
    assert grounded["e1"].matched is True
    assert grounded["e1"].attributes_matched.get("color") is True


def test_predicate_visual_text_and_three_valued_logic():
    # OCR predicate
    pred_ocr = Predicate(
        id="p1",
        kind="ocr",
        operator="contains",
        roles={"target": "e1"},
        value="dừng lại",
        source_span="dừng lại",
        polarity="positive",
    )
    ev_with_ocr = [{"kind": "ocr_track", "text": "Biển báo DỪNG LẠI phía trước", "evidence_id": "ev_ocr_1"}]
    res = evaluate_predicate(pred_ocr, ev_with_ocr, {})
    assert res.state == MatchState.SUPPORTED

    ev_without_ocr = [{"kind": "ocr_track", "text": "Tốc độ tối đa 50", "evidence_id": "ev_ocr_2"}]
    res_miss = evaluate_predicate(pred_ocr, ev_without_ocr, {})
    # Missing evidence is unknown, not automatically false
    assert res_miss.state == MatchState.UNKNOWN

    # Negative polarity
    pred_not_ocr = Predicate(
        id="p2",
        kind="ocr",
        operator="contains",
        roles={"target": "e1"},
        value="cấm",
        source_span="không có cấm",
        polarity="negative",
    )
    # If text has "cấm", it contradicts the negative polarity
    ev_contra = [{"kind": "ocr_track", "text": "Biển cấm đỗ xe", "evidence_id": "ev_ocr_3"}]
    res_contra = evaluate_predicate(pred_not_ocr, ev_contra, {})
    assert res_contra.state == MatchState.CONTRADICTED
    assert evaluate_predicate(pred_not_ocr, [], {}).state == MatchState.UNKNOWN
    assert evaluate_predicate(pred_ocr, [{"kind": "ocr_track", "text": ""}], {}).state == MatchState.UNKNOWN


def test_entity_detection_does_not_prove_action():
    pred = Predicate(id="p", kind="action", operator="give", roles={"subject": "e1"},
                     source_span="gives", polarity="positive")
    graph = QueryGraph(original_query="person gives", language="en",
        entities=[Entity(id="e1", type="person", mention="person")], predicates=[pred],
        logic=[LogicNode(id="root", op="AND", children=["p"])], root_id="root")
    result = GraphMatcher().match(graph, [{"kind": "visual_detection", "label": "person"}])
    assert result.overall_status == "unknown"


def test_transcript_speaker_identity_invariant():
    """Rule check: transcript does not automatically prove speaker identity without evidence."""
    pred_speaking = Predicate(
        id="p_spk",
        kind="asr",
        operator="speaks",
        roles={"speaker": "e_speaker"},
        value="xin chào",
        source_span="người nói",
        polarity="positive",
    )
    ev = [{"kind": "asr_segment", "text": "Xin chào quý vị", "timestamp_seconds": 2.5, "evidence_id": "ev_spk"}]
    res = evaluate_predicate(pred_speaking, ev, {})
    # ASR alone doesn't prove who spoke unless speaker is explicitly linked
    assert res.state == MatchState.PARTIAL


def test_temporal_ordering_with_gap_bounds():
    pred_before = Predicate(
        id="p_temp",
        kind="temporal",
        operator="before",
        roles={"event_a": "ev_a", "event_b": "ev_b"},
        polarity="positive",
        source_span="trước khi",
        scope=PredicateScope(min_gap_ms=100, max_gap_ms=2000),
    )
    # Valid interval: ev_a ends at 2.0s, ev_b starts at 2.5s (gap = 500ms -> valid)
    intervals_valid = {
        "ev_a": (1.0, 2.0),
        "ev_b": (2.5, 4.0),
    }
    res_valid = evaluate_temporal_order(pred_before, intervals_valid)
    assert res_valid.state == MatchState.SUPPORTED

    # Gap too large: 5000ms > max_gap_ms (2000ms)
    intervals_too_far = {
        "ev_a": (1.0, 2.0),
        "ev_b": (7.0, 9.0),
    }
    res_far = evaluate_temporal_order(pred_before, intervals_too_far)
    assert res_far.state == MatchState.CONTRADICTED


def test_graph_matcher_full_integration():
    qg = QueryGraph(
        original_query="người nói xin chào",
        language="vi",
        entities=[Entity(id="e1", type="person", mention="người")],
        predicates=[
            Predicate(
                id="p1",
                kind="asr",
                operator="contains",
                roles={"target": "e1"},
                value="xin chào",
                source_span="xin chào",
                polarity="positive",
            )
        ],
        logic=[LogicNode(id="root", op="AND", children=["p1"])],
        root_id="root",
    )

    matcher = GraphMatcher()
    evidence_match = [{"kind": "asr_segment", "text": "Tôi xin chào mọi người", "evidence_id": "ev_asr_1"}]
    res = matcher.match(qg, evidence_match)
    assert res.overall_status == "supported"
    assert res.structural_score == 1.0
    assert len(res.unresolved_predicates) == 0
