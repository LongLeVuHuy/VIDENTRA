"""Tests for 2D same-frame spatial grounding, action observations, and pipeline verification."""
import sqlite3

import faiss
import numpy as np
import pytest

from videntra.query.schemas import Entity, LogicNode, Predicate, QueryGraph
from videntra.relations.matcher import GraphMatcher
from videntra.relations.predicates import MatchState, evaluate_predicate
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
from videntra.retrieval.visual import RankedMoment


def _two_entity_graph(pred: Predicate) -> QueryGraph:
    return QueryGraph(
        original_query="test spatial or action",
        language="en",
        entities=[
            Entity(id="e1", type="person", mention="person"),
            Entity(id="e2", type="car", mention="car"),
        ],
        predicates=[pred],
        logic=[LogicNode(id="root", op="AND", children=[pred.id])],
        root_id="root",
    )


@pytest.mark.parametrize(
    "operator,box_s,box_o,expected",
    [
        ("left_of", (10, 10, 50, 90), (120, 10, 200, 90), MatchState.SUPPORTED),
        ("left_of", (150, 10, 210, 90), (20, 10, 90, 90), MatchState.CONTRADICTED),
        ("right_of", (150, 10, 210, 90), (20, 10, 90, 90), MatchState.SUPPORTED),
        ("above", (20, 10, 80, 50), (20, 120, 80, 190), MatchState.SUPPORTED),
        ("below", (20, 120, 80, 190), (20, 10, 80, 50), MatchState.SUPPORTED),
        ("inside", (30, 30, 70, 70), (10, 10, 100, 100), MatchState.SUPPORTED),
        ("inside", (200, 200, 260, 260), (10, 10, 100, 100), MatchState.CONTRADICTED),
        ("near", (10, 10, 60, 60), (80, 10, 130, 60), MatchState.SUPPORTED),
        ("near", (10, 10, 60, 60), (400, 400, 450, 450), MatchState.CONTRADICTED),
    ],
)
def test_2d_spatial_operators_same_frame(operator, box_s, box_o, expected):
    pred = Predicate(
        id="p_sp",
        kind="spatial",
        operator=operator,
        roles={"subject": "e1", "object": "e2"},
        polarity="positive",
        source_span=operator,
    )
    ev = [
        {
            "kind": "visual_detection",
            "evidence_id": "d1",
            "class_name": "person",
            "keyframe_id": "kf_01",
            "bbox": box_s,
            "confidence": 0.92,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "d2",
            "class_name": "car",
            "keyframe_id": "kf_01",
            "bbox": box_o,
            "confidence": 0.88,
        },
    ]
    res = GraphMatcher().match(_two_entity_graph(pred), ev)
    assert res.predicate_evaluations["p_sp"].state == expected


def test_spatial_negative_polarity_and_3d_depth_invariant():
    pred_neg = Predicate(
        id="p_neg",
        kind="spatial",
        operator="left_of",
        roles={"subject": "e1", "object": "e2"},
        polarity="negative",
        source_span="not left of",
    )
    ev_right = [
        {
            "kind": "visual_detection",
            "evidence_id": "d1",
            "class_name": "person",
            "keyframe_id": "kf_01",
            "bbox": (150, 10, 210, 90),
            "confidence": 0.9,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "d2",
            "class_name": "car",
            "keyframe_id": "kf_01",
            "bbox": (10, 10, 70, 90),
            "confidence": 0.9,
        },
    ]
    assert GraphMatcher().match(_two_entity_graph(pred_neg), ev_right).overall_status == "supported"

    pred_3d = Predicate(
        id="p_3d",
        kind="spatial",
        operator="behind",
        roles={"subject": "e1", "object": "e2"},
        polarity="positive",
        source_span="behind",
    )
    assert GraphMatcher().match(_two_entity_graph(pred_3d), ev_right).overall_status == "unknown"


def test_spatial_cross_frame_remains_unknown():
    pred = Predicate(
        id="p_sp",
        kind="spatial",
        operator="left_of",
        roles={"subject": "e1", "object": "e2"},
        polarity="positive",
        source_span="left of",
    )
    ev = [
        {
            "kind": "visual_detection",
            "evidence_id": "d1",
            "class_name": "person",
            "keyframe_id": "kf_01",
            "bbox": (10, 10, 50, 90),
            "confidence": 0.95,
        },
        {
            "kind": "visual_detection",
            "evidence_id": "d2",
            "class_name": "car",
            "keyframe_id": "kf_02",
            "bbox": (150, 10, 220, 90),
            "confidence": 0.95,
        },
    ]
    res = GraphMatcher().match(_two_entity_graph(pred), ev)
    assert res.predicate_evaluations["p_sp"].state == MatchState.UNKNOWN


def test_action_observation_supported_and_reversed_contradicted():
    pred = Predicate(
        id="p_act",
        kind="action",
        operator="give",
        roles={"subject": "p_red", "object": "p_blue"},
        polarity="positive",
        source_span="give",
    )
    # Bare visual detections never prove actions
    assert (
        evaluate_predicate(
            pred,
            [{"kind": "visual_detection", "class_name": "person", "bbox": (0, 0, 50, 50)}],
            {},
        ).state
        == MatchState.UNKNOWN
    )

    # Matching explicit action observation supports predicate
    ev_support = [
        {
            "kind": "action_observation",
            "evidence_id": "act_1",
            "operator": "give",
            "roles": {"subject": "p_red", "object": "p_blue"},
            "confidence": 0.87,
        }
    ]
    eval_sup = evaluate_predicate(pred, ev_support, {})
    assert eval_sup.state == MatchState.SUPPORTED
    assert eval_sup.evidence_refs == ["act_1"]

    # Reversed explicit action observation contradicts directed positive action
    ev_rev = [
        {
            "kind": "action_observation",
            "evidence_id": "act_2",
            "operator": "give",
            "roles": {"subject": "p_blue", "object": "p_red"},
            "confidence": 0.89,
        }
    ]
    eval_rev = evaluate_predicate(pred, ev_rev, {})
    assert eval_rev.state == MatchState.CONTRADICTED


def test_pipeline_verify_relations_collects_frame_objects_and_calibrates_asr():
    pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    pipeline.db_conn = sqlite3.connect(":memory:")
    pipeline.db_conn.execute(
        "CREATE TABLE OCRObservations (observation_id TEXT, video_id TEXT, keyframe_id TEXT, "
        "timestamp_ms INTEGER, model_id TEXT, raw_text TEXT, confidence REAL, "
        "bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL)"
    )
    pipeline.db_conn.execute(
        "CREATE TABLE ASRSegments (segment_id TEXT, video_id TEXT, model_id TEXT, "
        "start_ms INTEGER, end_ms INTEGER, raw_text TEXT, confidence REAL)"
    )
    pipeline.db_conn.execute(
        "CREATE TABLE FrameObjects (detection_id TEXT, video_id TEXT, keyframe_id TEXT, "
        "timestamp_ms INTEGER, class_name TEXT, bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, "
        "bbox_y2 REAL, confidence REAL, model_id TEXT)"
    )
    pipeline.db_conn.execute(
        "INSERT INTO ASRSegments VALUES ('asr_1', 'vid_1', 'whisper', 1000, 4000, 'hello world', -0.45)"
    )
    pipeline.db_conn.execute(
        "INSERT INTO FrameObjects VALUES ('fo_1', 'vid_1', 'kf_1', 2000, 'person', 10, 10, 50, 90, 0.91, 'yolo')"
    )

    ev = pipeline._collect_moment_evidence("vid_1", 0.0, 5000.0)
    asr_ev = next(item for item in ev if item["kind"] == "asr_segment")
    fo_ev = next(item for item in ev if item["kind"] == "visual_detection")
    assert abs(asr_ev["confidence"] - 0.5) < 1e-6
    assert asr_ev["raw_confidence"] == pytest.approx(-0.45)
    assert fo_ev["bbox"] == (10.0, 10.0, 50.0, 90.0)
    assert fo_ev["keyframe_id"] == "kf_1"

    # Optional relation-aware rerank prioritizes corroborated moment over contradicted moment
    def fake_search(**_kwargs):
        return [
            RankedMoment("m_bad", "vid_bad", 0.0, 5.0, 0, 0, 0, 2.0, ranking_score=0.90),
            RankedMoment("m_good", "vid_good", 0.0, 5.0, 0, 0, 0, 2.0, ranking_score=0.80),
        ]

    pipeline.db_conn.execute(
        "INSERT INTO OCRObservations VALUES ('o_bad', 'vid_bad', 'kf_b', 2000, 'paddle', 'EXIT', 0.95, 10, 10, 80, 40)"
    )
    pipeline.db_conn.execute(
        "INSERT INTO OCRObservations VALUES ('o_ok', 'vid_good', 'kf_g', 2000, 'paddle', 'ENTRANCE', 0.95, 10, 10, 80, 40)"
    )
    pipeline.search = fake_search  # type: ignore[method-assign]

    res = pipeline.search_with_metadata(
        snapshot_id="s_test",
        verify_relations=True,
        rerank_with_relations=True,
        query='no sign "EXIT"',
    )
    assert [m.video_id for m in res["moments"]] == ["vid_good", "vid_bad"]
    assert res["relation_assessments"][1]["status"] == "contradicted"
    pipeline.close()
