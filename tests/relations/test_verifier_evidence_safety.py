"""Synthetic counterexamples for false support; no human labels are implied."""
import pytest

from videntra.query.schemas import Entity, EntityAttribute, Predicate, QueryGraph
from videntra.relations.entities import ground_entities
from videntra.relations.predicates import MatchState, evaluate_predicate


def _graph(entity):
    return QueryGraph(original_query=entity.mention, language="en", entities=[entity],
                      predicates=[], logic=[], root_id="")


def test_no_observation_does_not_ground_an_entity_or_attributes():
    entity = Entity(id="p", type="person", mention="person",
                    attributes=[EntityAttribute("color", "red", "red")])
    result = ground_entities(_graph(entity), [])["p"]
    assert not result.matched
    assert result.state == "unknown"
    assert result.attributes_matched["color"] is None


def test_missing_attribute_is_partial_and_unmatched_class_is_unknown():
    entity = Entity(id="p", type="person", mention="person",
                    attributes=[EntityAttribute("color", "red", "red")])
    detected = [{"kind": "visual_detection", "label": "person"}]
    result = ground_entities(_graph(entity), detected)["p"]
    assert result.matched and result.state == "partial"
    assert result.attributes_matched["color"] is None
    result = ground_entities(_graph(entity), [{"kind": "visual_detection", "label": "car"}])["p"]
    assert not result.matched and result.state == "unknown"


def test_attributes_on_different_objects_do_not_support_one_entity():
    entity = Entity(id="p", type="person", mention="person", attributes=[
        EntityAttribute("color", "red", "red"), EntityAttribute("hat", "black", "black hat")])
    evidence = [{"kind": "visual_detection", "label": "person", "attributes": {"color": "red"}},
                {"kind": "visual_detection", "label": "person", "attributes": {"hat": "black"}}]
    assert ground_entities(_graph(entity), evidence)["p"].state == "partial"


def test_empty_ocr_cannot_ground_text_region():
    entity = Entity(id="t", type="text_region", mention="STOP")
    assert ground_entities(_graph(entity), [{"kind": "ocr_observation", "text": ""}])["t"].state == "unknown"


@pytest.mark.parametrize("kind,observed", [("ocr", "NONSTOP"), ("asr", "stopping")])
def test_text_phrase_does_not_match_inside_another_word(kind, observed):
    p = Predicate(id="p", kind=kind, operator="contains", roles={}, value="stop", polarity="positive")
    ev = [{"kind": "ocr_observation" if kind == "ocr" else "asr_segment", "text": observed}]
    assert evaluate_predicate(p, ev, {}).state == MatchState.UNKNOWN


def test_spatial_relation_without_binding_evidence_is_unknown():
    p = Predicate(id="p", kind="spatial", operator="left_of", roles={}, polarity="positive")
    assert evaluate_predicate(p, [], {}).state == MatchState.UNKNOWN
