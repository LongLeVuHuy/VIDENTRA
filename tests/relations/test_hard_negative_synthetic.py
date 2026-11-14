"""Synthetic contrast cases: these do not constitute a human-labeled evaluation."""
import pytest

from videntra.query.schemas import Predicate
from videntra.relations.predicates import MatchState, evaluate_predicate
from videntra.relations.temporal import evaluate_temporal_order


def test_unresolved_scope_cannot_support_or_contradict_directly():
    ocr = Predicate(id="p", kind="ocr", operator="contains", roles={}, value="STOP",
                    polarity="negative", source_span="not walking, sign STOP",
                    uncertainty=["negation_scope_unresolved"])
    evidence = [{"kind": "ocr_observation", "text": "STOP", "evidence_id": "e"}]
    assert evaluate_predicate(ocr, evidence, {}).state == MatchState.UNKNOWN
    temporal = Predicate(id="t", kind="temporal", operator="before", roles={"event_a": "a", "event_b": "b"},
                         polarity="positive", source_span="not before", uncertainty=["negation_scope_unresolved"])
    assert evaluate_temporal_order(temporal, {"a": (1, 2), "b": (3, 4)}).state == MatchState.UNKNOWN


@pytest.mark.parametrize("operator,expected", [("before", MatchState.SUPPORTED), ("after", MatchState.CONTRADICTED)])
def test_temporal_reversal(operator, expected):
    p = Predicate(id="p", kind="temporal", operator=operator, roles={"event_a": "a", "event_b": "b"}, polarity="positive", source_span="before")
    assert evaluate_temporal_order(p, {"a": (1, 2), "b": (4, 5)}).state == expected


def test_overlapping_events_do_not_prove_order():
    p = Predicate(id="p", kind="temporal", operator="before", roles={"event_a": "a", "event_b": "b"}, polarity="positive", source_span="before")
    assert evaluate_temporal_order(p, {"a": (1, 2.1), "b": (2, 3)}).state == MatchState.PARTIAL
    assert evaluate_temporal_order(p, {"a": (2, 1), "b": (2, 3)}).state == MatchState.UNKNOWN


def test_transcript_does_not_prove_negated_speaker_role():
    p = Predicate(id="p", kind="asr", operator="contains", roles={"speaker": "person_a"}, value="hello", polarity="negative", source_span="person does not say hello")
    ev = [{"kind": "asr_segment", "text": "hello", "evidence_id": "s1"}]
    assert evaluate_predicate(p, ev, {}).state == MatchState.UNKNOWN


@pytest.mark.parametrize("roles", [{"subject": "red", "object": "blue"}, {"subject": "blue", "object": "red"}])
def test_reversed_roles_do_not_prove_give_without_action_evidence(roles):
    p = Predicate(id="p", kind="action", operator="give", roles=roles, polarity="positive", source_span="give")
    assert evaluate_predicate(p, [{"kind": "visual_detection", "label": "person"}], {}).state == MatchState.UNKNOWN
