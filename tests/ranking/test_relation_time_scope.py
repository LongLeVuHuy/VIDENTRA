from videntra.query.rules import RuleBasedQueryParser
from videntra.query.schemas import LogicNode, Predicate, QueryGraph
from videntra.ranking.ranker import RelationAwareRanker
from videntra.relations.matcher import GraphMatcher
from videntra.retrieval.visual import RankedMoment


def test_untimed_ocr_does_not_support_a_moment():
    graph = QueryGraph(original_query='text "STOP"', language="en", entities=[],
        predicates=[Predicate(id="p", kind="ocr", operator="contains", roles={}, value="STOP",
                              source_span="STOP", polarity="positive")],
        logic=[LogicNode(id="root", op="AND", children=["p"])], root_id="root")
    moment = RankedMoment("m", "v", 1, 3, 0, 0, 0, 2, 0.5)
    result = RelationAwareRanker().rank([moment], graph,
        {"v": [{"kind": "ocr_observation", "text": "STOP", "evidence_id": "e"}]})
    assert result[0].match_status == "unknown"


def test_ocr_just_after_half_open_interval_is_not_support():
    graph = QueryGraph(original_query='text "STOP"', language="en", entities=[],
        predicates=[Predicate(id="p", kind="ocr", operator="contains", roles={}, value="STOP",
                              source_span="STOP", polarity="positive")],
        logic=[LogicNode(id="root", op="AND", children=["p"])], root_id="root")
    moment = RankedMoment("m", "v", 1, 3, 0, 0, 0, 2, 0.5)
    for timestamp in (3.0, 3.1):
        result = RelationAwareRanker().rank([moment], graph,
            {"v": [{"kind": "ocr_observation", "text": "STOP", "timestamp_seconds": timestamp}]})
        assert result[0].match_status == "unknown"


def test_partially_overlapping_asr_does_not_prove_words_in_interval():
    graph = QueryGraph(original_query="says hello", language="en", entities=[],
        predicates=[Predicate(id="p", kind="asr", operator="contains", roles={}, value="hello",
                              source_span="says hello", polarity="positive")],
        logic=[LogicNode(id="root", op="AND", children=["p"])], root_id="root")
    moment = RankedMoment("m", "v", 1, 3, 0, 0, 0, 2, 0.5)
    evidence = {"v": [{"kind": "asr_segment", "text": "hello", "start_seconds": 0.5,
                       "end_seconds": 2.0, "evidence_id": "e"}]}
    assert RelationAwareRanker().rank([moment], graph, evidence)[0].match_status == "unknown"


def test_ambiguous_negation_does_not_become_contradiction():
    graph = RuleBasedQueryParser().parse('a person does not walk next to a sign "STOP"')
    assert "negation_scope_unresolved" in graph.ambiguities
    result = GraphMatcher().match(graph, [{"kind": "ocr_observation", "text": "STOP", "evidence_id": "e"}])
    assert result.overall_status == "unknown"
