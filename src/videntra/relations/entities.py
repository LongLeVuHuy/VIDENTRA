"""Entity grounding and attribute matching against extracted multimodal observations and YOLOv8 detections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from videntra.query.schemas import Entity, QueryGraph
from videntra.evidence.normalize import contains_text_phrase
from videntra.query.rules import COCO_SYNONYM_MAP


@dataclass(frozen=True)
class GroundedEntity:
    entity_id: str
    matched: bool
    confidence: float
    attributes_matched: dict[str, bool | None]
    evidence_ref: str | None = None
    state: str = "supported"
    target_classes: tuple[str, ...] = ()


def _object_frame_key(obj: dict[str, Any]) -> Any:
    """Return the first non-None frame identifier for per-frame cardinality counting."""
    for key in ("keyframe_id", "pts_sec", "timestamp_ms", "timestamp_seconds"):
        val = obj.get(key)
        if val is not None:
            return val
    return None


def ground_entities(
    query_graph: QueryGraph,
    evidence_items: Sequence[dict[str, Any]],
) -> dict[str, GroundedEntity]:
    """Map QueryGraph entities to extracted visual, OCR, and YOLOv8 object evidence."""
    grounded = {}

    ocr_texts = [
        ev.get("text", "")
        for ev in evidence_items
        if ev.get("kind") in ("ocr_track", "ocr_observation", "ocr")
    ]

    detected_objects = [
        ev for ev in evidence_items
        if ev.get("kind") in ("visual_detection", "frame_object", "detection", "object")
    ]

    for ent in query_graph.entities:
        ent_id = ent.id
        if ent.type == "text_region":
            matched = any(contains_text_phrase(ocr_t, ent.mention) for ocr_t in ocr_texts)
            text_aliases = tuple(sorted({
                v for v in ("text_region", "ocr_observation", "ocr_track", ent.mention.lower().strip()) if v
            }))
            grounded[ent_id] = GroundedEntity(
                entity_id=ent_id,
                matched=matched,
                confidence=0.95 if matched else 0.0,
                attributes_matched={},
                evidence_ref="ocr_track" if matched else None,
                state="supported" if matched else "unknown",
                target_classes=text_aliases,
            )
        else:
            # Person / Object grounding
            ent_type_coco = COCO_SYNONYM_MAP.get(ent.type.lower(), ent.type.lower())
            ent_mention_coco = COCO_SYNONYM_MAP.get(ent.mention.lower(), ent_type_coco)
            obj_class_attr = next((a.value.lower() for a in ent.attributes if a.name == "object_class"), "")
            obj_class_coco = COCO_SYNONYM_MAP.get(obj_class_attr, obj_class_attr) if obj_class_attr else ""

            target_classes = {
                ent.type.lower(),
                ent.mention.lower(),
                ent_type_coco,
                ent_mention_coco,
            }
            if obj_class_attr:
                target_classes.add(obj_class_attr)
            if obj_class_coco:
                target_classes.add(obj_class_coco)
            target_classes.discard("")
            sorted_targets = tuple(sorted(target_classes))

            # If evidence contains detection records, match against them
            if detected_objects:
                matched_objs = [
                    obj for obj in detected_objects
                    if (obj.get("class_name", "").lower() in target_classes)
                    or (obj.get("label", "").lower() in target_classes)
                    or (str(obj.get("canonical_class", "")).lower() in target_classes)
                ]

                attr_matches = {}
                for attr in ent.attributes:
                    if attr.name == "object_class":
                        attr_matches[attr.name] = bool(matched_objs)
                    elif attr.name == "cardinality":
                        try:
                            req_count = int(attr.value)
                            counts_by_frame = {}
                            for obj in matched_objs:
                                f_key = _object_frame_key(obj)
                                if f_key is not None:
                                    counts_by_frame[f_key] = counts_by_frame.get(f_key, 0) + 1
                            if counts_by_frame:
                                attr_matches["cardinality"] = max(counts_by_frame.values()) >= req_count
                            elif req_count == 1 and matched_objs:
                                attr_matches["cardinality"] = True
                            else:
                                attr_matches["cardinality"] = None
                        except (ValueError, TypeError):
                            attr_matches["cardinality"] = None
                    else:
                        # Match attribute against obj attributes if present
                        attr_val = attr.value.lower()
                        observed = [obj.get("attributes", {}).get(attr.name) for obj in matched_objs]
                        observed = [value for value in observed if value is not None]
                        attr_matches[attr.name] = (any(attr_val == str(value).lower() for value in observed)
                                                   if observed else None)

                if matched_objs:
                    best_conf = max(float(obj.get("confidence", 0.85)) for obj in matched_objs)
                    required_attrs = [a for a in ent.attributes if a.name not in ("object_class", "cardinality")]
                    coherent = not required_attrs or any(
                        all(str(obj.get("attributes", {}).get(a.name, "")).lower() == a.value.lower()
                            for a in required_attrs) for obj in matched_objs
                    )
                    grounded[ent_id] = GroundedEntity(
                        entity_id=ent_id,
                        matched=True,
                        confidence=best_conf,
                        attributes_matched=attr_matches,
                        evidence_ref="visual_detection",
                        state="supported" if coherent and all(value is True for value in attr_matches.values()) else "partial",
                        target_classes=sorted_targets,
                    )
                else:
                    grounded[ent_id] = GroundedEntity(
                        entity_id=ent_id,
                        matched=False,
                        confidence=0.0,
                        attributes_matched=attr_matches,
                        evidence_ref=None,
                        state="unknown",
                        target_classes=sorted_targets,
                    )
            else:
                # A missing detector observation cannot establish an entity or its attributes.
                attr_matches = {attr.name: None for attr in ent.attributes}
                grounded[ent_id] = GroundedEntity(
                    entity_id=ent_id,
                    matched=False,
                    confidence=0.0,
                    attributes_matched=attr_matches,
                    evidence_ref=None,
                    state="unknown",
                    target_classes=sorted_targets,
                )

    return grounded


def compute_object_verification_boost(
    object_requirements: Sequence[dict[str, Any]],
    detected_objects: Sequence[dict[str, Any]],
    base_match_boost: float = 0.15,
    cardinality_boost: float = 0.10,
    no_object_penalty: float = 0.10,
) -> float:
    """Calculate ranking score adjustment based on detected YOLO objects and query requirements.

    Rules:
    - If no object requirements: return 0.0
    - If query requires objects but frame has 0 detected objects: return -no_object_penalty
    - Non-person objects receive base_match_boost * conf (+0.15 * conf) and cardinality_boost (+0.10)
    - Person class receives mild adjustment (0.04 * conf) to prevent disrupting visual semantics
    - Multi-object cardinality is evaluated within a single frame if keyframe_id/pts_sec is available
    """
    if not object_requirements:
        return 0.0

    if not detected_objects:
        # If requirements only contain common "person", do not heavily penalize
        only_person = all(
            (r.get("canonical_class") or COCO_SYNONYM_MAP.get(str(r.get("class_name", "")).lower(), str(r.get("class_name", "")).lower())) == "person"
            for r in object_requirements
        )
        return -0.02 if only_person else -float(no_object_penalty)

    total_boost = 0.0
    has_any_target_match = False

    for req in object_requirements:
        raw_class = str(req.get("class_name", "")).lower()
        req_count = int(req.get("cardinality", req.get("count", 1)))
        canonical_class = req.get("canonical_class") or COCO_SYNONYM_MAP.get(raw_class, raw_class)

        # Match objects
        matched = []
        counts_by_frame: dict[Any, int] = {}
        for obj in detected_objects:
            obj_canonical = obj.get("canonical_class")
            if obj_canonical is None:
                obj_name = str(obj.get("class_name", obj.get("label", ""))).lower()
                obj_canonical = COCO_SYNONYM_MAP.get(obj_name, obj_name)
                if obj_name == raw_class or obj_canonical == canonical_class:
                    matched.append(obj)
                    f_key = _object_frame_key(obj)
                    if f_key is not None:
                        counts_by_frame[f_key] = counts_by_frame.get(f_key, 0) + 1
            elif obj_canonical == canonical_class:
                matched.append(obj)
                f_key = _object_frame_key(obj)
                if f_key is not None:
                    counts_by_frame[f_key] = counts_by_frame.get(f_key, 0) + 1

        if matched:
            has_any_target_match = True
            best_conf = max(float(obj.get("confidence", 1.0)) for obj in matched)

            if counts_by_frame:
                is_satisfied = max(counts_by_frame.values()) >= req_count
            else:
                is_satisfied = len(matched) >= req_count

            if canonical_class == "person":
                match_boost = 0.04 * best_conf
                card_boost = (0.14 if req_count >= 2 else 0.02) if is_satisfied else (-0.04 if req_count >= 2 else 0.0)
            else:
                match_boost = base_match_boost * best_conf
                card_boost = cardinality_boost if is_satisfied else 0.0

            total_boost += (match_boost + card_boost)

    if not has_any_target_match:
        only_person = all(
            (r.get("canonical_class") or COCO_SYNONYM_MAP.get(str(r.get("class_name", "")).lower(), str(r.get("class_name", "")).lower())) == "person"
            for r in object_requirements
        )
        return -0.02 if only_person else -float(no_object_penalty)

    return float(round(total_boost, 4))

