"""Unit tests for entity grounding and object verification boosting."""
from videntra.relations.entities import compute_object_verification_boost, ground_entities
from videntra.query.schemas import Entity, EntityAttribute, QueryGraph


def test_object_verification_boost_cardinality():
    requirements = [{"class_name": "cup", "count": 2}]
    detected = [
        {"class_name": "cup", "confidence": 0.9},
        {"class_name": "cup", "confidence": 0.8},
        {"class_name": "person", "confidence": 0.95},
    ]

    boost = compute_object_verification_boost(requirements, detected)
    # Expected: 0.15 * 0.9 + 0.10 = 0.135 + 0.10 = 0.235
    expected = 0.15 * 0.9 + 0.10
    assert abs(boost - expected) < 1e-4


def test_object_verification_penalty_no_objects():
    requirements = [{"class_name": "laptop", "count": 1}]
    detected = []

    boost = compute_object_verification_boost(requirements, detected)
    assert boost == -0.10


def test_object_verification_penalty_wrong_objects():
    requirements = [{"class_name": "car", "count": 1}]
    detected = [
        {"class_name": "cup", "confidence": 0.9},
        {"class_name": "bottle", "confidence": 0.8},
    ]

    boost = compute_object_verification_boost(requirements, detected)
    assert boost == -0.10


def test_object_verification_no_requirements():
    requirements = []
    detected = [{"class_name": "dog", "confidence": 0.95}]

    boost = compute_object_verification_boost(requirements, detected)
    assert boost == 0.0


def test_ground_entities_with_detections():
    graph = QueryGraph(
        original_query="A person holding a cup",
        language="en",
        entities=[
            Entity(id="ent_1", type="person", mention="person"),
            Entity(
                id="ent_2",
                type="object",
                mention="cup",
                attributes=[
                    EntityAttribute(name="object_class", value="cup", source_span="cup"),
                    EntityAttribute(name="cardinality", value="1", source_span="a cup"),
                ],
            ),
        ],
        predicates=[],
        logic=[],
        root_id="root",
    )
    evidence = [
        {"kind": "visual_detection", "class_name": "person", "confidence": 0.92},
        {"kind": "visual_detection", "class_name": "cup", "confidence": 0.88},
    ]

    grounded = ground_entities(graph, evidence)
    assert grounded["ent_1"].matched is True
    assert grounded["ent_1"].confidence == 0.92
    assert grounded["ent_2"].matched is True
    assert grounded["ent_2"].confidence == 0.88
    assert grounded["ent_2"].attributes_matched.get("cardinality") is True


def test_ground_entities_cardinality_per_frame():
    graph = QueryGraph(
        original_query="Two cups on the table",
        language="en",
        entities=[
            Entity(
                id="ent_cups",
                type="object",
                mention="two cups",
                attributes=[
                    EntityAttribute(name="object_class", value="cup", source_span="cup"),
                    EntityAttribute(name="cardinality", value="2", source_span="two cups"),
                ],
            ),
        ],
        predicates=[],
        logic=[],
        root_id="root",
    )

    # Case A: Two cups detected across different frames (1 cup in kf_1, 1 cup in kf_2)
    evidence_different_frames = [
        {"kind": "visual_detection", "class_name": "cup", "confidence": 0.9, "keyframe_id": "kf_1"},
        {"kind": "visual_detection", "class_name": "cup", "confidence": 0.85, "keyframe_id": "kf_2"},
    ]
    grounded_a = ground_entities(graph, evidence_different_frames)
    assert grounded_a["ent_cups"].matched is True
    assert grounded_a["ent_cups"].attributes_matched.get("cardinality") is False

    # Case B: Two cups detected in the SAME keyframe
    evidence_same_frame = [
        {"kind": "visual_detection", "class_name": "cup", "confidence": 0.9, "keyframe_id": "kf_3"},
        {"kind": "visual_detection", "class_name": "cup", "confidence": 0.85, "keyframe_id": "kf_3"},
    ]
    grounded_b = ground_entities(graph, evidence_same_frame)
    assert grounded_b["ent_cups"].matched is True
    assert grounded_b["ent_cups"].attributes_matched.get("cardinality") is True

