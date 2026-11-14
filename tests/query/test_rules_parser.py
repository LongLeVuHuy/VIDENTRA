"""Tests for deterministic rule-based query parsing into valid QueryGraph."""
import pytest
from videntra.contracts import validate
from videntra.query.rules import RuleBasedQueryParser


def test_complex_vietnamese_query():
    parser = RuleBasedQueryParser()
    query = "Tìm người áo đỏ đưa túi cho người áo xanh, sau đó người áo xanh nói cảm ơn, phía sau có chữ SALE."

    graph = parser.parse(query)
    graph_dict = graph.to_dict()

    # 1. Official schema adherence must pass 100%
    validate("query_graph", graph_dict)

    # 2. Assert entities
    assert len(graph.entities) >= 2
    types = [e.type for e in graph.entities]
    assert "person" in types
    assert "text_region" in types

    # 3. Assert predicates
    kinds = [p.kind for p in graph.predicates]
    assert "action" in kinds
    assert "ocr" in kinds
    assert "asr" in kinds
    assert "spatial" in kinds
    assert "temporal" in kinds

    # 4. Assert OCR text value
    ocr_preds = [p for p in graph.predicates if p.kind == "ocr"]
    assert len(ocr_preds) == 1
    assert ocr_preds[0].value == "SALE"

    # 5. Assert ASR speech value
    asr_preds = [p for p in graph.predicates if p.kind == "asr"]
    assert len(asr_preds) == 1
    assert "cảm ơn" in asr_preds[0].value

    # 6. Assert semantic uncertainty for 'xanh'
    assert any("blue" in amb and "green" in amb for amb in graph.ambiguities)


def test_quoted_text_ocr_parsing():
    parser = RuleBasedQueryParser()
    query = 'Tìm chiếc xe có dán chữ "DISCOUNT 50%"'

    graph = parser.parse(query)
    validate("query_graph", graph.to_dict())

    ocr_preds = [p for p in graph.predicates if p.kind == "ocr"]
    assert len(ocr_preds) == 1
    assert ocr_preds[0].value == "DISCOUNT 50%"


def test_negation_polarity():
    parser = RuleBasedQueryParser()
    query = "người không đi bộ trên đường"

    graph = parser.parse(query)
    validate("query_graph", graph.to_dict())

    assert len(graph.predicates) >= 1
    assert graph.predicates[0].polarity == "negative"


def test_english_query():
    parser = RuleBasedQueryParser()
    query = "A person in red gives a bag to a person in blue, then someone says thank you, with sign EXIT"

    graph = parser.parse(query)
    validate("query_graph", graph.to_dict())
    assert graph.language == "en"

    kinds = [p.kind for p in graph.predicates]
    assert "action" in kinds
    assert "ocr" in kinds


def test_query_decomposition_and_coco_objects():
    from videntra.query.rules import decompose_query_clauses, extract_object_requirements

    query = "A woman is cooking in the kitchen and then talks on phone with two cups on the table"
    clauses = decompose_query_clauses(query)
    assert len(clauses) >= 2
    assert clauses[0] == query

    objs = extract_object_requirements(query)
    class_names = [o["class_name"] for o in objs]
    assert "person" in class_names
    assert "cell phone" in class_names
    assert "cup" in class_names
    cup_req = next(o for o in objs if o["class_name"] == "cup")
    assert cup_req["cardinality"] == 2


def test_b8_plural_normalization_irregular_and_regular():
    """Verifies B8: Plural nouns (-s, -es, -ies, -ves) normalize to singular COCO classes."""
    from videntra.query.rules import extract_object_requirements

    # 1. -s plural
    res1 = extract_object_requirements("there are two laptops on the desk")
    laptop = next(r for r in res1 if r["class_name"] == "laptop")
    assert laptop["cardinality"] == 2

    # 2. -s plural with 3 count
    res2 = extract_object_requirements("he prepares three cups of coffee")
    cup = next(r for r in res2 if r["class_name"] == "cup")
    assert cup["cardinality"] == 3

    # 3. -ies plural (puppies -> puppy -> dog)
    res3 = extract_object_requirements("several puppies are playing in the garden")
    dog = next(r for r in res3 if r["class_name"] == "dog")
    assert dog["cardinality"] == 2

    # 4. -es plural (buses -> bus)
    res4 = extract_object_requirements("many buses waiting at the station")
    bus = next(r for r in res4 if r["class_name"] == "bus")
    assert bus["cardinality"] == 2

    # 5. -ves plural (knives -> knife)
    res5 = extract_object_requirements("chef holds two knives")
    knife = next(r for r in res5 if r["class_name"] == "knife")
    assert knife["cardinality"] == 2

    # 6. -es plural (couches -> couch)
    res6 = extract_object_requirements("a room with three couches")
    couch = next(r for r in res6 if r["class_name"] == "couch")
    assert couch["cardinality"] == 3

    # 7. Irregular plural: people -> person, default cardinality = 2
    res7 = extract_object_requirements("people walking in the park")
    person7 = next(r for r in res7 if r["class_name"] == "person")
    assert person7["cardinality"] == 2

    # 8. Irregular plural with count: three people -> person, cardinality = 3
    res8 = extract_object_requirements("three people in the meeting room")
    person8 = next(r for r in res8 if r["class_name"] == "person")
    assert person8["cardinality"] == 3

    # 9. Irregular plural: children -> person, default cardinality = 2
    res9 = extract_object_requirements("children playing soccer")
    person9 = next(r for r in res9 if r["class_name"] == "person")
    assert person9["cardinality"] == 2

    # 10. Irregular plural with count: five children -> person, cardinality = 5
    res10 = extract_object_requirements("five children eating snacks")
    person10 = next(r for r in res10 if r["class_name"] == "person")
    assert person10["cardinality"] == 5

    # 11. Irregular plural: men -> person, default cardinality = 2
    res11 = extract_object_requirements("two men running in marathon")
    person11 = next(r for r in res11 if r["class_name"] == "person")
    assert person11["cardinality"] == 2

    res11_bare = extract_object_requirements("men working at construction site")
    person11_bare = next(r for r in res11_bare if r["class_name"] == "person")
    assert person11_bare["cardinality"] == 2

    # 12. Irregular plural: women -> person, default cardinality = 2
    res12 = extract_object_requirements("four women discussing project")
    person12 = next(r for r in res12 if r["class_name"] == "person")
    assert person12["cardinality"] == 4

    res12_bare = extract_object_requirements("women talking together")
    person12_bare = next(r for r in res12_bare if r["class_name"] == "person")
    assert person12_bare["cardinality"] == 2

    # 13. Irregular plural: mice -> mouse, default cardinality = 2
    res13 = extract_object_requirements("mice hiding in the pantry")
    mouse13 = next(r for r in res13 if r["class_name"] == "mouse")
    assert mouse13["cardinality"] == 2

    res13_count = extract_object_requirements("two mice running under table")
    mouse13_count = next(r for r in res13_count if r["class_name"] == "mouse")
    assert mouse13_count["cardinality"] == 2

    # 14. Singular controls: a person -> 1, a mouse -> 1
    res14_p = extract_object_requirements("a person walking alone")
    p14 = next(r for r in res14_p if r["class_name"] == "person")
    assert p14["cardinality"] == 1

    res14_m = extract_object_requirements("a mouse on the floor")
    m14 = next(r for r in res14_m if r["class_name"] == "mouse")
    assert m14["cardinality"] == 1


def test_b8_query_864d_cardinality_subspace():
    """Verifies that 'people' activates q_meta[1] (2 persons) instead of q_meta[0] (single person)."""
    import numpy as np
    from videntra.indexing.augmented_vector import build_query_864d_vector

    dummy_768 = np.ones(768, dtype=np.float32)

    # 1. Bare "people" -> q_meta[1] (2 persons), NOT q_meta[0]
    v_people = build_query_864d_vector(dummy_768, "people in the park", {})
    assert v_people[848] == 0.0  # q_meta[0] (single person) must be 0
    assert v_people[849] > 0.0   # q_meta[1] (2 persons) must be active

    # 2. "three people" -> q_meta[2] (>= 3 persons)
    v_three = build_query_864d_vector(dummy_768, "three people in the park", {})
    assert v_three[848] == 0.0
    assert v_three[849] == 0.0
    assert v_three[850] > 0.0   # q_meta[2] (>= 3 persons) must be active

    # 3. "a person" -> q_meta[0] (single person)
    v_single = build_query_864d_vector(dummy_768, "a person in the park", {})
    assert v_single[848] > 0.0   # q_meta[0] must be active
    assert v_single[849] == 0.0
    assert v_single[850] == 0.0


def test_b11_temporal_conjunction_after_sau_khi_reversal():
    """Verifies B11: Conjunction 'after' and 'sau khi' reverses decomposed clause order
    to reflect true chronological sequence (t1 <= t2), while 'before', 'trước khi', and 'then'
    preserve forward order. Short clauses (>= 2 words) are preserved.
    """
    from videntra.query.rules import decompose_query_clauses

    # 1. English "after": Event 1 in time: walking to podium; Event 2: presents slide
    q_en = "A speaker presents slide after walking to the podium"
    clauses_en = decompose_query_clauses(q_en)
    assert len(clauses_en) == 3
    assert clauses_en[0] == q_en
    assert "walking to the podium" in clauses_en[1]
    assert "presents slide" in clauses_en[2]

    # 2. Vietnamese "sau khi": Event 1 in time: bước lên bục giảng; Event 2: trình bày slide
    q_vi = "Diễn giả trình bày slide sau khi bước lên bục giảng"
    clauses_vi = decompose_query_clauses(q_vi)
    assert len(clauses_vi) == 3
    assert clauses_vi[0] == q_vi
    assert "bước lên bục giảng" in clauses_vi[1]
    assert "trình bày slide" in clauses_vi[2]

    # 3. English "before": Event 1 in time: walks to podium; Event 2: presents slides (forward order preserved)
    q_before = "The speaker walks to the podium before he presents the slides"
    clauses_before = decompose_query_clauses(q_before)
    assert len(clauses_before) == 3
    assert clauses_before[0] == q_before
    assert "The speaker walks to the podium" in clauses_before[1]
    assert "he presents the slides" in clauses_before[2]

    # 4. Vietnamese "trước khi": forward order preserved
    q_vi_before = "Diễn giả bước lên bục giảng trước khi trình bày slide"
    clauses_vi_before = decompose_query_clauses(q_vi_before)
    assert len(clauses_vi_before) == 3
    assert clauses_vi_before[0] == q_vi_before
    assert "bước lên bục giảng" in clauses_vi_before[1]
    assert "trình bày slide" in clauses_vi_before[2]

    # 5. Standalone "then": forward order preserved
    q_then = "The speaker walks to the podium then he presents the slides"
    clauses_then = decompose_query_clauses(q_then)
    assert len(clauses_then) == 3
    assert clauses_then[0] == q_then
    assert "The speaker walks to the podium" in clauses_then[1]
    assert "he presents the slides" in clauses_then[2]

    # 6. Short 3-word clause with "after": "closing the door" (3 words) must not be filtered out
    q_short_after = "The man leaves the house after closing the door"
    clauses_short_after = decompose_query_clauses(q_short_after)
    assert len(clauses_short_after) == 3
    assert "closing the door" in clauses_short_after[1]
    assert "leaves the house" in clauses_short_after[2]

    # 7. Short 3-word clause with "sau khi": "bước vào phòng" (3 words) must not be filtered out
    q_short_vi = "Người phụ nữ mở chiếc hộp sau khi bước vào phòng"
    clauses_short_vi = decompose_query_clauses(q_short_vi)
    assert len(clauses_short_vi) == 3
    assert "bước vào phòng" in clauses_short_vi[1]
    assert "mở chiếc hộp" in clauses_short_vi[2]

    # 8. Short 3-word clause with "and then": "presents the slides" (3 words) must not be filtered out
    q_short_then = "The speaker walks to the podium and then presents the slides"
    clauses_short_then = decompose_query_clauses(q_short_then)
    assert len(clauses_short_then) == 3
    assert "The speaker walks to the podium" in clauses_short_then[1]
    assert "presents the slides" in clauses_short_then[2]
