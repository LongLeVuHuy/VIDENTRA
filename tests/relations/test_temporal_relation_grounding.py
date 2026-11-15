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

    # Gap too small (200ms < min_gap_ms 500ms) -> CONTRADICTED
    ev_close = [
        {"kind": "asr_segment", "evidence_id": "asr_1", "text": "thank you", "start_seconds": 1.0, "end_seconds": 2.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_1", "text": "BIG SALE", "timestamp_seconds": 2.2},
    ]
    res_close = GraphMatcher().match(graph_gap, ev_close)
    assert res_close.predicate_evaluations["p_temp"].state == MatchState.CONTRADICTED


def test_temporal_grounding_case_a_wrong_caller_window_vs_refs_only() -> None:
    graph = QueryGraph(
        original_query="sign A before sign B",
        language="en",
        entities=[
            Entity(id="s1", type="text_region", mention="ALPHA"),
            Entity(id="s2", type="text_region", mention="BETA"),
        ],
        predicates=[
            Predicate(id="p_a", kind="ocr", operator="exact_text", roles={"text_entity": "s1"}, value="ALPHA"),
            Predicate(id="p_b", kind="ocr", operator="exact_text", roles={"text_entity": "s2"}, value="BETA"),
            Predicate(id="p_t", kind="temporal", operator="before", roles={"event_a": "p_a", "event_b": "p_b"}),
        ],
        logic=[LogicNode(id="root", op="AND", children=["p_a", "p_b", "p_t"])],
        root_id="root",
    )
    ev = [
        {"kind": "ocr_observation", "evidence_id": "ocr_a", "video_id": "v1", "timeline_revision": "tl1", "text": "ALPHA", "timestamp_ms": 6000.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_b", "video_id": "v1", "timeline_revision": "tl1", "text": "BETA", "timestamp_ms": 2000.0},
    ]
    # Refs only: A=6s, B=2s -> CONTRADICTED
    res_refs = GraphMatcher().match(graph, ev, video_id="v1", timeline_revision="tl1")
    assert res_refs.predicate_evaluations["p_t"].state == MatchState.CONTRADICTED
    assert set(res_refs.predicate_evaluations["p_t"].evidence_refs) == {"ocr_a", "ocr_b"}

    # Wrong caller event_intervals (1,2) / (4,5) filters out refs -> UNKNOWN (not SUPPORTED!)
    wrong_win = {"p_a": (1.0, 2.0), "p_b": (4.0, 5.0)}
    snapshot = dict(wrong_win)
    res_filtered = GraphMatcher().match(graph, ev, event_intervals=wrong_win, video_id="v1", timeline_revision="tl1")
    assert res_filtered.predicate_evaluations["p_t"].state == MatchState.UNKNOWN
    assert res_filtered.predicate_evaluations["p_t"].evidence_refs == []
    assert wrong_win == snapshot

    # Valid filter window still uses real ref timestamps (6s vs 2s) -> CONTRADICTED
    valid_win = {"p_a": (5.0, 7.0), "p_b": (1.0, 3.0)}
    res_valid_win = GraphMatcher().match(graph, ev, event_intervals=valid_win, video_id="v1", timeline_revision="tl1")
    assert res_valid_win.predicate_evaluations["p_t"].state == MatchState.CONTRADICTED


def test_temporal_grounding_case_b_identical_point_timestamps_and_touching_intervals() -> None:
    from videntra.relations.temporal import evaluate_temporal_order

    graph = QueryGraph(
        original_query="sign ALPHA before sign BETA",
        language="en",
        entities=[
            Entity(id="s1", type="text_region", mention="ALPHA"),
            Entity(id="s2", type="text_region", mention="BETA"),
        ],
        predicates=[
            Predicate(id="p_a", kind="ocr", operator="exact_text", roles={"text_entity": "s1"}, value="ALPHA"),
            Predicate(id="p_b", kind="ocr", operator="exact_text", roles={"text_entity": "s2"}, value="BETA"),
            Predicate(id="p_t", kind="temporal", operator="before", roles={"event_a": "p_a", "event_b": "p_b"}),
        ],
        logic=[LogicNode(id="root", op="AND", children=["p_a", "p_b", "p_t"])],
        root_id="root",
    )
    ev_same_ts = [
        {"kind": "ocr_observation", "evidence_id": "ocr_a", "video_id": "v1", "timeline_revision": "tl1", "text": "ALPHA", "timestamp_ms": 5000.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_b", "video_id": "v1", "timeline_revision": "tl1", "text": "BETA", "timestamp_ms": 5000.0},
    ]
    res_same = GraphMatcher().match(graph, ev_same_ts, video_id="v1", timeline_revision="tl1")
    assert res_same.predicate_evaluations["p_t"].state == MatchState.PARTIAL
    assert set(res_same.predicate_evaluations["p_t"].evidence_refs) == {"ocr_a", "ocr_b"}

    # Touching-boundary non-point intervals (0, 1000) & (1000, 2000) remain SUPPORTED
    p_touch = Predicate(id="p_t", kind="temporal", operator="before", roles={"event_a": "a", "event_b": "b"})
    res_touch = evaluate_temporal_order(p_touch, {"a": (0.0, 1000.0), "b": (1000.0, 2000.0)})
    assert res_touch.state == MatchState.SUPPORTED


def test_temporal_grounding_multi_occurrence_conflicting_and_filtered() -> None:
    graph = _ocr_asr_temporal_graph()
    ev_multi = [
        {"kind": "ocr_observation", "evidence_id": "ocr_1", "track_id": "trk_1", "video_id": "v1", "text": "BIG SALE", "timestamp_seconds": 2.0},
        {"kind": "ocr_observation", "evidence_id": "ocr_2", "track_id": "trk_2", "video_id": "v1", "text": "BIG SALE", "timestamp_seconds": 8.0},
        {"kind": "asr_segment", "evidence_id": "asr_1", "segment_id": "seg_1", "video_id": "v1", "text": "thank you", "start_seconds": 5.0, "end_seconds": 6.0},
    ]
    # Unfiltered: trk_1 (2s) < seg_1 (5-6s) SUPPORTED, trk_2 (8s) > seg_1 (5-6s) CONTRADICTED -> PARTIAL
    res_multi = GraphMatcher().match(graph, ev_multi, video_id="v1")
    assert res_multi.predicate_evaluations["p_temp"].state == MatchState.PARTIAL
    assert set(res_multi.predicate_evaluations["p_temp"].evidence_refs) == {"ocr_1", "ocr_2", "asr_1"}

    # Filtered to first occurrence window -> SUPPORTED
    res_first = GraphMatcher().match(graph, ev_multi, event_intervals={"p_ocr": (0.0, 4.0)}, video_id="v1")
    assert res_first.predicate_evaluations["p_temp"].state == MatchState.SUPPORTED
    assert set(res_first.predicate_evaluations["p_temp"].evidence_refs) == {"ocr_1", "asr_1"}

    # Filtered to second occurrence window -> CONTRADICTED
    res_second = GraphMatcher().match(graph, ev_multi, event_intervals={"p_ocr": (7.0, 10.0)}, video_id="v1")
    assert res_second.predicate_evaluations["p_temp"].state == MatchState.CONTRADICTED
    assert set(res_second.predicate_evaluations["p_temp"].evidence_refs) == {"ocr_2", "asr_1"}


def test_temporal_grounding_timeline_revision_mismatch_preserves_standalone() -> None:
    graph = _ocr_asr_temporal_graph()
    ev = [
        {"kind": "ocr_observation", "evidence_id": "ocr_1", "video_id": "v1", "timeline_revision": "tl_v1", "text": "BIG SALE", "timestamp_seconds": 2.0},
        {"kind": "asr_segment", "evidence_id": "asr_1", "video_id": "v1", "timeline_revision": "tl_v2", "text": "thank you", "start_seconds": 5.0, "end_seconds": 6.0},
    ]
    res = GraphMatcher().match(graph, ev, video_id="v1", timeline_revision="tl_v1")
    assert res.predicate_evaluations["p_ocr"].state == MatchState.SUPPORTED
    assert res.predicate_evaluations["p_temp"].state == MatchState.UNKNOWN


def test_search_with_metadata_verify_relations_window_scoping() -> None:
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
    conn.execute(
        "INSERT INTO OCRObservations VALUES ('ocr_w1', 'v1', 'BIG SALE', 'rapidocr', 2000.0)"
    )
    conn.execute(
        "INSERT INTO ASRSegments VALUES ('asr_w1', 'v1', 'thank you', 'whisper', 5000.0, 7000.0)"
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
        ),
        RankedMoment(
            moment_id="v1_20.0_30.0",
            video_id="v1",
            moment_start_seconds=20.0,
            moment_end_seconds=30.0,
            start_original_frame_index=20,
            end_original_frame_index=30,
            representative_original_frame_index=25,
            representative_timestamp_seconds=25.0,
            ranking_score=0.8,
        ),
    ]
    pipe.search = lambda **kwargs: moments  # type: ignore[method-assign]
    out = pipe.search_with_metadata(
        snapshot_id="snap_test",
        verify_relations=True,
        query="sign BIG SALE before speaker says thank you",
    )
    assessments = out["relation_assessments"]
    assert len(assessments) == 2
    assert assessments[0]["predicates"]["temporal_order"]["status"] == "supported"
    assert set(assessments[0]["predicates"]["temporal_order"]["evidence_refs"]) == {"ocr_w1", "asr_w1"}
    assert assessments[1]["predicates"]["temporal_order"]["status"] == "unknown"
    assert assessments[1]["predicates"]["temporal_order"]["evidence_refs"] == []
    conn.close()


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
