"""Unit tests for Phase 7.1 Bug 3, Bug 4, and Bug 5 fixes in query & relations."""
from videntra.contracts import validate
from videntra.query.canonicalize import canonicalize_spatial
from videntra.query.rules import RuleBasedQueryParser
from videntra.query.schemas import Entity, EntityAttribute, LogicNode, Predicate, QueryGraph
from videntra.relations.entities import (
    _object_frame_key,
    compute_object_verification_boost,
    ground_entities,
)
from videntra.relations.matcher import GraphMatcher
from videntra.relations.predicates import MatchState, evaluate_predicate


def test_bug3_spatial_reference_role_supported_and_contradicted():
    pred = Predicate(
        id="sp_1",
        kind="spatial",
        operator="left_of",
        roles={"subject": "obj_1", "reference": "person_1"},
        source_span="left of",
        polarity="positive",
    )
    evidence_ok = [
        {
            "kind": "visual_detection",
            "evidence_id": "ev_dog",
            "class_name": "dog",
            "keyframe_id": "kf_1",
            "bbox": (10.0, 20.0, 50.0, 80.0),
            "confidence": 0.9,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "ev_person",
            "class_name": "person",
            "keyframe_id": "kf_1",
            "bbox": (100.0, 20.0, 160.0, 120.0),
            "confidence": 0.92,
        },
    ]
    grounded = {
        "obj_1": type("G", (), {"entity_id": "obj_1", "target_classes": ("dog",)})(),
        "person_1": type("G", (), {"entity_id": "person_1", "target_classes": ("person",)})(),
    }
    res_ok = evaluate_predicate(pred, evidence_ok, grounded)
    assert res_ok.state == MatchState.SUPPORTED
    assert set(res_ok.evidence_refs) == {"ev_dog", "ev_person"}

    evidence_bad = [
        {
            "kind": "visual_detection",
            "evidence_id": "ev_dog",
            "class_name": "dog",
            "keyframe_id": "kf_1",
            "bbox": (200.0, 20.0, 250.0, 80.0),
            "confidence": 0.9,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "ev_person",
            "class_name": "person",
            "keyframe_id": "kf_1",
            "bbox": (100.0, 20.0, 160.0, 120.0),
            "confidence": 0.92,
        },
    ]
    res_bad = evaluate_predicate(pred, evidence_bad, grounded)
    assert res_bad.state == MatchState.CONTRADICTED


def test_bug3_bilingual_2d_spatial_canonicalize_and_parser_order():
    expected = {
        "left of": "left_of",
        "right of": "right_of",
        "near": "near",
        "next to": "beside",
        "beside": "beside",
        "above": "above",
        "below": "below",
        "inside": "inside",
        "bên trái": "left_of",
        "bên phải": "right_of",
        "gần": "near",
        "bên cạnh": "beside",
        "ở trên": "above",
        "ở dưới": "below",
        "bên trong": "inside",
    }
    for phrase, canonical in expected.items():
        assert canonicalize_spatial(phrase) == canonical

    parser = RuleBasedQueryParser()

    # English: subject before preposition, object after preposition
    g_en = parser.parse("a person next to a car")
    validate("query_graph", g_en.to_dict())
    sp_en = next(p for p in g_en.predicates if p.kind == "spatial")
    assert sp_en.operator == "beside"
    assert sp_en.roles["subject"].startswith("person")
    assert sp_en.roles["object"].startswith("obj_")

    # Vietnamese: object before preposition, person after preposition
    g_vi = parser.parse("xe hơi bên trái người áo đỏ")
    validate("query_graph", g_vi.to_dict())
    sp_vi = next(p for p in g_vi.predicates if p.kind == "spatial")
    assert sp_vi.operator == "left_of"
    assert sp_vi.roles["subject"].startswith("obj_")
    assert sp_vi.roles["object"].startswith("person")

    # End-to-end matcher on parsed Vietnamese 2D spatial query
    ev_vi = [
        {
            "kind": "visual_detection",
            "evidence_id": "ev_car",
            "class_name": "car",
            "timestamp_ms": 1000.0,
            "bbox": (10.0, 50.0, 80.0, 120.0),
            "confidence": 0.9,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "ev_p",
            "class_name": "person",
            "timestamp_ms": 1000.0,
            "bbox": (140.0, 40.0, 200.0, 160.0),
            "confidence": 0.9,
        },
    ]
    match_vi = GraphMatcher().match(g_vi, ev_vi)
    assert match_vi.predicate_evaluations["spatial_rel"].state == MatchState.SUPPORTED


def test_bug4_graph_matcher_infers_temporal_intervals_from_evidence():
    graph = QueryGraph(
        original_query="sign SALE then person says thank you",
        language="en",
        entities=[
            Entity(id="sign", type="text_region", mention="SALE"),
            Entity(id="person_1", type="person", mention="person"),
        ],
        predicates=[
            Predicate(
                id="p_ocr",
                kind="ocr",
                operator="exact_text",
                roles={"text_entity": "sign"},
                value="SALE",
                source_span="SALE",
                polarity="positive",
            ),
            Predicate(
                id="p_asr",
                kind="asr",
                operator="semantic_phrase",
                roles={"speaker": "person_1"},
                value="thank you",
                source_span="says thank you",
                polarity="positive",
            ),
            Predicate(
                id="p_temp",
                kind="temporal",
                operator="before",
                roles={"event_a": "p_ocr", "event_b": "p_asr"},
                source_span="then",
                polarity="positive",
            ),
        ],
        logic=[LogicNode(id="root", op="AND", children=["p_ocr", "p_asr", "p_temp"])],
        root_id="root",
    )

    # p_ocr is SUPPORTED at 2.0s; p_asr is PARTIAL (speaker unverified) at [5.0s, 7.0s]
    evidence_ordered = [
        {"kind": "ocr_observation", "evidence_id": "ev_ocr_1", "text": "BIG SALE", "timestamp_ms": 2000.0},
        {
            "kind": "asr_segment",
            "evidence_id": "ev_asr_1",
            "text": "thank you very much",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_ordered = GraphMatcher().match(graph, evidence_ordered)
    assert res_ordered.predicate_evaluations["p_ocr"].state == MatchState.SUPPORTED
    assert res_ordered.predicate_evaluations["p_asr"].state == MatchState.PARTIAL
    assert res_ordered.predicate_evaluations["p_temp"].state == MatchState.SUPPORTED

    # Reversed order -> CONTRADICTED
    evidence_reversed = [
        {"kind": "ocr_observation", "evidence_id": "ev_ocr_2", "text": "BIG SALE", "timestamp_seconds": 9.0},
        {
            "kind": "asr_segment",
            "evidence_id": "ev_asr_2",
            "text": "thank you very much",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_reversed = GraphMatcher().match(graph, evidence_reversed)
    assert res_reversed.predicate_evaluations["p_temp"].state == MatchState.CONTRADICTED
    assert res_reversed.overall_status == "contradicted"

    # Caller-provided event_intervals are respected and not mutated
    caller_intervals = {"p_ocr": (1.0, 2.0), "p_asr": (4.0, 6.0)}
    snapshot = dict(caller_intervals)
    res_explicit = GraphMatcher().match(graph, evidence_reversed, event_intervals=caller_intervals)
    assert res_explicit.predicate_evaluations["p_temp"].state == MatchState.SUPPORTED
    assert caller_intervals == snapshot


def test_bug5_object_frame_key_consistency():
    assert _object_frame_key({"keyframe_id": "kf1", "pts_sec": 1.0, "timestamp_ms": 1000.0}) == "kf1"
    assert _object_frame_key({"pts_sec": 2.5, "timestamp_ms": 2500.0}) == 2.5
    assert _object_frame_key({"timestamp_ms": 3000.0, "timestamp_seconds": 3.0}) == 3000.0
    assert _object_frame_key({"timestamp_seconds": 4.0}) == 4.0
    assert _object_frame_key({}) is None

    graph = QueryGraph(
        original_query="two cars",
        language="en",
        entities=[
            Entity(
                id="obj_1",
                type="object",
                mention="two cars",
                attributes=[
                    EntityAttribute("object_class", "car", "cars"),
                    EntityAttribute("cardinality", "2", "two"),
                ],
            )
        ],
        predicates=[],
        logic=[],
        root_id="",
    )

    # Same frame via timestamp_ms -> cardinality 2 satisfied
    same_frame = [
        {"kind": "visual_detection", "class_name": "car", "timestamp_ms": 1500.0, "confidence": 0.9},
        {"kind": "visual_detection", "class_name": "car", "timestamp_ms": 1500.0, "confidence": 0.85},
    ]
    grounded_same = ground_entities(graph, same_frame)
    assert grounded_same["obj_1"].attributes_matched["cardinality"] is True
    assert grounded_same["obj_1"].state == "supported"

    # Different frames via timestamp_ms -> cardinality 2 NOT satisfied in a single frame
    diff_frame = [
        {"kind": "visual_detection", "class_name": "car", "timestamp_ms": 1500.0, "confidence": 0.9},
        {"kind": "visual_detection", "class_name": "car", "timestamp_ms": 2500.0, "confidence": 0.85},
    ]
    grounded_diff = ground_entities(graph, diff_frame)
    assert grounded_diff["obj_1"].attributes_matched["cardinality"] is False

    # compute_object_verification_boost with canonical_class already set and keyframe_id / timestamp_ms
    reqs = [{"class_name": "car", "canonical_class": "car", "cardinality": 2}]
    same_kf_canonical = [
        {"class_name": "car", "canonical_class": "car", "keyframe_id": "kf_10", "confidence": 1.0},
        {"class_name": "car", "canonical_class": "car", "keyframe_id": "kf_10", "confidence": 0.9},
    ]
    diff_kf_canonical = [
        {"class_name": "car", "canonical_class": "car", "keyframe_id": "kf_10", "confidence": 1.0},
        {"class_name": "car", "canonical_class": "car", "keyframe_id": "kf_20", "confidence": 0.9},
    ]
    boost_same = compute_object_verification_boost(reqs, same_kf_canonical)
    boost_diff = compute_object_verification_boost(reqs, diff_kf_canonical)
    assert boost_same > boost_diff
    assert round(boost_same - boost_diff, 4) == 0.10
