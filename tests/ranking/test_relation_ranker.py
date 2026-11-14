"""Tests for feature extraction and relation-aware reranker."""
import pytest

from videntra.query.schemas import QueryGraph, Predicate, LogicNode
from videntra.ranking.features import extract_moment_features
from videntra.ranking.ranker import RelationAwareRanker
from videntra.retrieval.visual import RankedMoment


def test_relation_aware_ranker_boosts_supported_moment():
    qg = QueryGraph(
        original_query="biển báo dừng lại",
        language="vi",
        entities=[],
        predicates=[
            Predicate(
                id="p1",
                kind="ocr",
                operator="contains",
                roles={},
                value="dừng lại",
                source_span="dừng lại",
                polarity="positive",
            )
        ],
        logic=[LogicNode(id="root", op="AND", children=["p1"])],
        root_id="root",
    )

    moments = [
        RankedMoment(
            moment_id="m1",
            video_id="v1",
            moment_start_seconds=1.0,
            moment_end_seconds=4.0,
            start_original_frame_index=30,
            end_original_frame_index=120,
            representative_original_frame_index=60,
            representative_timestamp_seconds=2.0,
            ranking_score=0.6,
        ),
        RankedMoment(
            moment_id="m2",
            video_id="v2",
            moment_start_seconds=5.0,
            moment_end_seconds=8.0,
            start_original_frame_index=150,
            end_original_frame_index=240,
            representative_original_frame_index=180,
            representative_timestamp_seconds=6.0,
            ranking_score=0.7,
        ),
    ]

    # Video 1 has evidence matching "dừng lại"
    evidence_by_video = {
        "v1": [{"kind": "ocr_track", "text": "Biển báo DỪNG LẠI", "timestamp_seconds": 2.0, "evidence_id": "ev_ocr_1"}],
        "v2": [],  # no matching evidence
    }

    ranker = RelationAwareRanker(structural_weight=0.5, contradiction_penalty=0.8)
    reranked = ranker.rank(moments, qg, evidence_by_video)

    assert len(reranked) == 2
    # m1 should be boosted because predicate p1 was supported
    assert reranked[0].video_id == "v1"
    assert reranked[0].match_status == "supported"
    assert reranked[0].ranking_score > reranked[1].ranking_score
    assert reranked[1].match_status == "unknown"
    assert reranked[1].ranking_score == 0.7


def test_partial_missing_predicate_does_not_reduce_base_score():
    graph = QueryGraph(original_query='text "STOP" and action', language="en", entities=[],
        predicates=[Predicate(id="ocr", kind="ocr", operator="contains", roles={}, value="STOP"),
                    Predicate(id="action", kind="action", operator="give", roles={})],
        logic=[LogicNode(id="root", op="AND", children=["ocr", "action"])], root_id="root")
    moment = RankedMoment("m", "v", 0, 3, 0, 0, 0, 1, 0.9)
    results = RelationAwareRanker().rank([moment], graph, {"v": [
        {"kind": "ocr_observation", "text": "STOP", "timestamp_seconds": 1}]})
    assert results[0].match_status == "partial"
    assert results[0].ranking_score == 0.9


def test_feature_extraction():
    moment = RankedMoment(
        moment_id="m_feat",
        video_id="v_feat",
        moment_start_seconds=1.0,
        moment_end_seconds=3.0,
        start_original_frame_index=10,
        end_original_frame_index=30,
        representative_original_frame_index=20,
        representative_timestamp_seconds=2.0,
        ranking_score=0.85,
    )
    feats = extract_moment_features(moment)
    vec = feats.to_feature_vector()
    assert len(vec) == 6
    assert vec[0] == 0.85
    assert vec[1] == 2.0  # duration
