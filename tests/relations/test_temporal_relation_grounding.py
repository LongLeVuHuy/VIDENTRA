"""Regression tests for Section 2.3 temporal relation grounding, window scoping, and parser canonicalization."""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pytest

from videntra.query.rules import RuleBasedQueryParser
from videntra.query.schemas import Entity, LogicNode, Predicate, PredicateScope, QueryGraph
from videntra.relations.matcher import GraphMatcher
from videntra.relations.predicates import MatchState
from videntra.retrieval.alignment import viterbi_subevent_localization
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
from videntra.retrieval.visual import RankedMoment


def _ocr_asr_temporal_graph(
    *,
    operator: str = "before",
    scope: PredicateScope | None = None,
) -> QueryGraph:
    return QueryGraph(
        original_query="sign BIG SALE before speaker says thank you",
        language="en",
        entities=[
            Entity(id="sign_1", type="text_region", mention="sign BIG SALE"),
            Entity(id="person_1", type="person", mention="person"),
        ],
        predicates=[
            Predicate(
                id="p_ocr",
                kind="ocr",
                operator="exact_text",
                roles={"text_entity": "sign_1"},
                value="BIG SALE",
                source_span="sign BIG SALE",
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
                operator=operator,
                roles={"event_a": "p_ocr", "event_b": "p_asr"},
                source_span=operator,
                polarity="positive",
                scope=scope,
            ),
        ],
        logic=[LogicNode(id="root", op="AND", children=["p_ocr", "p_asr", "p_temp"])],
        root_id="root",
    )


def test_temporal_grounding_supported_contradicted_and_refs_attached() -> None:
    graph = _ocr_asr_temporal_graph()
    ev_supported = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_1",
            "video_id": "v1",
            "timeline_revision": "tl_v1",
            "text": "BIG SALE",
            "timestamp_ms": 2000.0,
        },
        {
            "kind": "asr_segment",
            "evidence_id": "asr_1",
            "video_id": "v1",
            "timeline_revision": "tl_v1",
            "text": "thank you",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_sup = GraphMatcher().match(graph, ev_supported, video_id="v1", timeline_revision="tl_v1")
    p_temp_sup = res_sup.predicate_evaluations["p_temp"]
    assert p_temp_sup.state == MatchState.SUPPORTED
    assert set(p_temp_sup.evidence_refs) == {"ocr_1", "asr_1"}

    # Swapped anchors -> CONTRADICTED with both refs attached
    ev_swapped = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_2",
            "video_id": "v1",
            "timeline_revision": "tl_v1",
            "text": "BIG SALE",
            "timestamp_ms": 9000.0,
        },
        {
            "kind": "asr_segment",
            "evidence_id": "asr_2",
            "video_id": "v1",
            "timeline_revision": "tl_v1",
            "text": "thank you",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_con = GraphMatcher().match(graph, ev_swapped, video_id="v1", timeline_revision="tl_v1")
    p_temp_con = res_con.predicate_evaluations["p_temp"]
    assert p_temp_con.state == MatchState.CONTRADICTED
    assert set(p_temp_con.evidence_refs) == {"ocr_2", "asr_2"}


def test_temporal_grounding_missing_anchor_and_empty_evidence_with_caller_intervals() -> None:
    graph = _ocr_asr_temporal_graph()

    # Single anchor only -> UNKNOWN
    single_ev = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_only",
            "video_id": "v1",
            "text": "BIG SALE",
            "timestamp_ms": 2000.0,
        }
    ]
    res_single = GraphMatcher().match(graph, single_ev)
    assert res_single.predicate_evaluations["p_temp"].state == MatchState.UNKNOWN
    assert res_single.predicate_evaluations["p_temp"].evidence_refs == []

    # Empty evidence CANNOT be bypassed by caller event_intervals
    res_empty = GraphMatcher().match(
        graph,
        [],
        event_intervals={"p_ocr": (1.0, 2.0), "p_asr": (4.0, 6.0)},
    )
    assert res_empty.predicate_evaluations["p_temp"].state == MatchState.UNKNOWN
    assert res_empty.predicate_evaluations["p_temp"].evidence_refs == []


def test_temporal_grounding_window_scoping_win2_without_evidence() -> None:
    graph = _ocr_asr_temporal_graph()
    ev_in_win1 = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_w1",
            "video_id": "v1",
            "text": "BIG SALE",
            "timestamp_seconds": 2.0,
        },
        {
            "kind": "asr_segment",
            "evidence_id": "asr_w1",
            "video_id": "v1",
            "text": "thank you",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_win1 = GraphMatcher().match(graph, ev_in_win1, video_id="v1", moment_window=(0.0, 10.0))
    assert res_win1.predicate_evaluations["p_temp"].state == MatchState.SUPPORTED

    # _win2 [20.0, 30.0] has no evidence inside its window -> UNKNOWN
    res_win2 = GraphMatcher().match(graph, ev_in_win1, video_id="v1", moment_window=(20.0, 30.0))
    assert res_win2.predicate_evaluations["p_temp"].state == MatchState.UNKNOWN
    assert res_win2.predicate_evaluations["p_temp"].evidence_refs == []


def test_temporal_grounding_shared_anchor_and_unaligned_timeline() -> None:
    # Shared anchor: both events resolve to the exact same evidence item & timestamp
    shared_graph = QueryGraph(
        original_query="sign A before sign B",
        language="en",
        entities=[Entity(id="s1", type="text_region", mention="sign")],
        predicates=[
            Predicate(id="p_a", kind="ocr", operator="exact_text", roles={"text_entity": "s1"}, value="SALE"),
            Predicate(id="p_b", kind="ocr", operator="exact_text", roles={"text_entity": "s1"}, value="OPEN"),
            Predicate(id="p_t", kind="temporal", operator="before", roles={"event_a": "p_a", "event_b": "p_b"}),
        ],
        logic=[LogicNode(id="root", op="AND", children=["p_a", "p_b", "p_t"])],
        root_id="root",
    )
    shared_ev = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_shared",
            "video_id": "v1",
            "timeline_revision": "tl_1",
            "text": "SALE OPEN",
            "timestamp_seconds": 4.0,
        }
    ]
    res_shared = GraphMatcher().match(shared_graph, shared_ev, video_id="v1")
    assert res_shared.predicate_evaluations["p_t"].state == MatchState.PARTIAL
    assert res_shared.predicate_evaluations["p_t"].evidence_refs == ["ocr_shared"]

    # Unaligned timeline_revision between event_a and event_b -> UNKNOWN
    graph = _ocr_asr_temporal_graph()
    unaligned_ev = [
        {
            "kind": "ocr_observation",
            "evidence_id": "ocr_1",
            "video_id": "v1",
            "timeline_revision": "tl_1",
            "text": "BIG SALE",
            "timestamp_seconds": 2.0,
        },
        {
            "kind": "asr_segment",
            "evidence_id": "asr_1",
            "video_id": "v1",
            "timeline_revision": "tl_2",
            "text": "thank you",
            "start_seconds": 5.0,
            "end_seconds": 7.0,
        },
    ]
    res_unaligned = GraphMatcher().match(graph, unaligned_ev, video_id="v1")
    assert res_unaligned.predicate_evaluations["p_temp"].state == MatchState.UNKNOWN


def test_temporal_grounding_min_max_gap_and_after_operator() -> None:
    graph_gap = _ocr_asr_temporal_graph(
        operator="after",
        scope=PredicateScope(min_gap_ms=500, max_gap_ms=3000),
    )
    # p_ocr after p_asr: p_asr=[1.0, 2.0], p_ocr=4.0 -> gap = 2000ms -> SUPPORTED
    ev_ok = [
        {"kind": "asr_segment", "evidence_id": "asr_1", "text": "thank you", "start_seconds": 1.0, "end_seconds": 2.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_1", "text": "BIG SALE", "timestamp_seconds": 4.0},
    ]
    res_ok = GraphMatcher().match(graph_gap, ev_ok)
    assert res_ok.predicate_evaluations["p_temp"].state == MatchState.SUPPORTED
    assert set(res_ok.predicate_evaluations["p_temp"].evidence_refs) == {"ocr_1", "asr_1"}

    # Gap too large (8000ms > 3000ms) -> CONTRADICTED
    ev_far = [
        {"kind": "asr_segment", "evidence_id": "asr_1", "text": "thank you", "start_seconds": 1.0, "end_seconds": 2.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_1", "text": "BIG SALE", "timestamp_seconds": 10.0},
    ]
    res_far = GraphMatcher().match(graph_gap, ev_far)
    assert res_far.predicate_evaluations["p_temp"].state == MatchState.CONTRADICTED


def test_vfr_timestamps_and_se_viterbi_multimodal_curves() -> None:
    # Non-uniform VFR PTS timestamps
    vfr_ts = [0.0, 0.35, 1.85, 5.2, 12.7]
    vis_curves = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]
    ocr_curves = [
        [0.0, 0.95, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    asr_curves = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.95, 0.0],
    ]
    vit = viterbi_subevent_localization(
        timestamps_sec=vfr_ts,
        event_vis_curves=vis_curves,
        event_ocr_curves=ocr_curves,
        event_asr_curves=asr_curves,
        max_gap_sec=10.0,
    )
    assert vit is not None
    assert vit.selected_indices == (1, 3)
    assert vit.start_sec == pytest.approx(0.35)
    assert vit.end_sec == pytest.approx(5.2)


def test_rule_parser_canonicalizes_after_to_before() -> None:
    parser = RuleBasedQueryParser()
    qg_en = parser.parse("a person runs after saying thank you")
    t_en = [p for p in qg_en.predicates if p.kind == "temporal"]
    assert len(t_en) == 1
    assert t_en[0].operator == "before"
    assert t_en[0].roles["event_a"] == "speak_asr"
    assert t_en[0].roles["event_b"] == "action_main"

    qg_vi = parser.parse("người chạy sau khi nói cảm ơn")
    t_vi = [p for p in qg_vi.predicates if p.kind == "temporal"]
    assert len(t_vi) == 1
    assert t_vi[0].operator == "before"
    assert t_vi[0].roles["event_a"] == "speak_asr"
    assert t_vi[0].roles["event_b"] == "action_main"


def test_pipeline_verify_relations_never_fabricates_midpoint_intervals() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE OCRObservations (observation_id TEXT, video_id TEXT, raw_text TEXT, model_id TEXT, timestamp_ms REAL)"
    )
    conn.execute(
        "CREATE TABLE ASRSegments (segment_id TEXT, video_id TEXT, raw_text TEXT, model_id TEXT, start_ms REAL, end_ms REAL)"
    )
    conn.execute(
        "CREATE TABLE FrameObjects (detection_id TEXT, video_id TEXT, class_name TEXT, timestamp_ms REAL)"
    )
    pipe = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipe.db_conn = conn
    moments = [
        RankedMoment(
            moment_id="v1_0.0_10.0",
            video_id="v1",
            moment_start_seconds=0.0,
            moment_end_seconds=10.0,
            start_original_frame_index=0,
            end_original_frame_index=10,
            representative_original_frame_index=5,
            representative_timestamp_seconds=5.0,
            ranking_score=0.9,
        )
    ]
    assessments = pipe._verify_relations("person runs then says thank you", moments)
    assert len(assessments) == 1
    assert assessments[0]["predicates"]["temporal_order"]["status"] == "unknown"
    conn.close()
