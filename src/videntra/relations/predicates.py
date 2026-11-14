"""Three-valued logic predicate evaluation preserving unknown states for missing evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence

from videntra.evidence.normalize import contains_text_phrase
from videntra.query.rules import COCO_SYNONYM_MAP
from videntra.query.schemas import Predicate
from .entities import GroundedEntity

_DEPTH_3D_OPERATORS = frozenset({"behind", "in_front_of", "front_of"})
_SPATIAL_ALIASES: dict[str, str] = {
    "left": "left_of",
    "left_of": "left_of",
    "to_the_left_of": "left_of",
    "right": "right_of",
    "right_of": "right_of",
    "to_the_right_of": "right_of",
    "above": "above",
    "over": "above",
    "on_top_of": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "inside": "inside",
    "within": "inside",
    "in": "inside",
    "near": "near",
    "next_to": "near",
    "beside": "near",
    "close_to": "near",
}


class MatchState(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"  # Missing evidence is unknown, not automatically false


@dataclass(frozen=True)
class PredicateEvaluation:
    predicate_id: str
    state: MatchState
    score: float
    reason: str
    evidence_refs: list[str]


def _extract_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Extract validated 2D bounding box (x1, y1, x2, y2) if present and positive-area."""
    raw = item.get("bbox")
    coords: tuple[Any, Any, Any, Any] | None = None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        coords = (raw[0], raw[1], raw[2], raw[3])
    elif all(k in item for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        coords = (item["bbox_x1"], item["bbox_y1"], item["bbox_x2"], item["bbox_y2"])
    if coords is None:
        return None
    try:
        x1, y1, x2, y2 = (float(c) for c in coords)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _frame_anchor_key(item: dict[str, Any]) -> str | None:
    """Return canonical per-frame identifier so 2D geometry only compares same-frame boxes."""
    if item.get("keyframe_id") not in (None, ""):
        return f"kf:{item['keyframe_id']}"
    if item.get("frame_index") is not None:
        return f"fi:{item['frame_index']}"
    if item.get("original_decoded_frame_index") is not None:
        return f"ofi:{item['original_decoded_frame_index']}"
    if item.get("timestamp_ms") is not None:
        return f"ms:{int(round(float(item['timestamp_ms'])))}"
    if item.get("pts_sec") is not None:
        return f"ms:{int(round(float(item['pts_sec']) * 1000.0))}"
    if item.get("timestamp_seconds") is not None:
        return f"ms:{int(round(float(item['timestamp_seconds']) * 1000.0))}"
    return None


def _entity_target_tokens(
    role_ref: str,
    grounded_entities: dict[str, GroundedEntity],
) -> set[str]:
    """Collect lowercase class/entity tokens acceptable for a predicate role reference."""
    ref_low = role_ref.strip().lower()
    tokens: set[str] = {ref_low}
    if ref_low in COCO_SYNONYM_MAP:
        tokens.add(COCO_SYNONYM_MAP[ref_low])
    ge = grounded_entities.get(role_ref)
    if ge is not None:
        for t in ge.target_classes:
            t_low = str(t).strip().lower()
            if t_low:
                tokens.add(t_low)
        if ge.entity_id:
            tokens.add(ge.entity_id.strip().lower())
    tokens.discard("")
    return tokens


def _detection_matches_role(
    item: dict[str, Any],
    role_ref: str,
    grounded_entities: dict[str, GroundedEntity],
) -> bool:
    """Check whether a localized observation can ground the given predicate role."""
    if float(item.get("confidence", 0.9)) < 0.5:
        return False
    kind = str(item.get("kind", "")).lower()
    targets = _entity_target_tokens(role_ref, grounded_entities)
    if kind in ("visual_detection", "frame_object", "detection", "object"):
        obs_tokens = {
            str(item.get(k, "")).strip().lower()
            for k in ("entity_id", "class_name", "label", "canonical_class")
        }
        obs_tokens.discard("")
        expanded = set(obs_tokens)
        for tok in obs_tokens:
            if tok in COCO_SYNONYM_MAP:
                expanded.add(COCO_SYNONYM_MAP[tok])
        return bool(targets & expanded)
    if kind in ("ocr_observation", "ocr_track", "ocr") and "text_region" in targets:
        text_val = str(item.get("text", "")).strip()
        if not text_val:
            return False
        ge = grounded_entities.get(role_ref)
        if ge is not None:
            return any(
                t not in ("text_region", "ocr_observation", "ocr_track") and contains_text_phrase(text_val, t)
                for t in ge.target_classes
            )
        return True
    return False


def _check_2d_spatial_geometry(
    canonical_op: str,
    subj_box: tuple[float, float, float, float],
    obj_box: tuple[float, float, float, float],
) -> bool | None:
    """Evaluate 2D spatial relation between same-frame boxes; returns None on ambiguous ties."""
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box
    cx_s, cy_s = 0.5 * (sx1 + sx2), 0.5 * (sy1 + sy2)
    cx_o, cy_o = 0.5 * (ox1 + ox2), 0.5 * (oy1 + oy2)
    w_s, h_s = sx2 - sx1, sy2 - sy1
    w_o, h_o = ox2 - ox1, oy2 - oy1

    if canonical_op == "left_of":
        margin = 0.02 * max(w_s, w_o, 1.0)
        if cx_s < cx_o - margin:
            return True
        if cx_s > cx_o + margin:
            return False
        return None
    if canonical_op == "right_of":
        margin = 0.02 * max(w_s, w_o, 1.0)
        if cx_s > cx_o + margin:
            return True
        if cx_s < cx_o - margin:
            return False
        return None
    if canonical_op == "above":
        margin = 0.02 * max(h_s, h_o, 1.0)
        if cy_s < cy_o - margin:
            return True
        if cy_s > cy_o + margin:
            return False
        return None
    if canonical_op == "below":
        margin = 0.02 * max(h_s, h_o, 1.0)
        if cy_s > cy_o + margin:
            return True
        if cy_s < cy_o - margin:
            return False
        return None
    if canonical_op == "inside":
        ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
        ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ratio = inter / max(w_s * h_s, 1e-6)
        if ratio >= 0.70:
            return True
        if ratio <= 0.10:
            return False
        return None
    if canonical_op == "near":
        dx = max(0.0, ox1 - sx2, sx1 - ox2)
        dy = max(0.0, oy1 - sy2, sy1 - oy2)
        edge_dist = math.hypot(dx, dy)
        scale = max(w_s, h_s, w_o, h_o, 1.0)
        if edge_dist <= 0.75 * scale:
            return True
        if edge_dist >= 2.5 * scale:
            return False
        return None
    return None


def _evaluate_spatial_predicate(
    pred: Predicate,
    evidence_items: Sequence[dict[str, Any]],
    grounded_entities: dict[str, GroundedEntity],
) -> PredicateEvaluation:
    """Evaluate spatial relations using verified same-frame 2D boxes; keep 3D/unbound UNKNOWN."""
    op_low = pred.operator.strip().lower()
    if op_low in _DEPTH_3D_OPERATORS:
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason="3D depth cannot be decisively proven from 2D plane; marked unknown",
            evidence_refs=[],
        )

    canonical_op = _SPATIAL_ALIASES.get(op_low)
    subj_ref = pred.roles.get("subject") or pred.roles.get("source") or pred.roles.get("arg1")
    obj_ref = (
        pred.roles.get("object")
        or pred.roles.get("reference")
        or pred.roles.get("target")
        or pred.roles.get("arg2")
    )
    if canonical_op is None or not subj_ref or not obj_ref:
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"Spatial relation '{pred.operator}' has no verified binding evidence",
            evidence_refs=[],
        )

    subj_boxes: list[tuple[dict[str, Any], tuple[float, float, float, float], str]] = []
    obj_boxes: list[tuple[dict[str, Any], tuple[float, float, float, float], str]] = []
    for ev in evidence_items:
        bbox = _extract_bbox(ev)
        anchor = _frame_anchor_key(ev)
        if bbox is None or anchor is None:
            continue
        if _detection_matches_role(ev, subj_ref, grounded_entities):
            subj_boxes.append((ev, bbox, anchor))
        if _detection_matches_role(ev, obj_ref, grounded_entities):
            obj_boxes.append((ev, bbox, anchor))

    if not subj_boxes or not obj_boxes:
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"Spatial relation '{pred.operator}' has no verified binding evidence",
            evidence_refs=[],
        )

    is_neg = pred.polarity == "negative"
    true_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    false_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    has_ambiguous_pair = False

    for ev_s, box_s, anc_s in subj_boxes:
        for ev_o, box_o, anc_o in obj_boxes:
            if anc_s != anc_o or ev_s is ev_o:
                continue
            verdict = _check_2d_spatial_geometry(canonical_op, box_s, box_o)
            if verdict is True:
                true_pair = (ev_s, ev_o)
                break
            if verdict is False:
                false_pair = (ev_s, ev_o)
            else:
                has_ambiguous_pair = True
        if true_pair is not None:
            break

    if true_pair is not None:
        refs = [str(x.get("evidence_id", "")) for x in true_pair if x.get("evidence_id")]
        conf = min(float(true_pair[0].get("confidence", 0.9)), float(true_pair[1].get("confidence", 0.9)))
        if is_neg:
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.CONTRADICTED,
                score=0.0,
                reason=f"2D geometry confirmed negated relation '{pred.operator}'",
                evidence_refs=refs,
            )
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.SUPPORTED,
            score=conf,
            reason=f"Same-frame 2D bounding boxes satisfy '{pred.operator}'",
            evidence_refs=refs,
        )

    if false_pair is not None and not has_ambiguous_pair:
        refs = [str(x.get("evidence_id", "")) for x in false_pair if x.get("evidence_id")]
        if is_neg:
            conf = min(float(false_pair[0].get("confidence", 0.9)), float(false_pair[1].get("confidence", 0.9)))
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.SUPPORTED,
                score=conf,
                reason=f"Same-frame 2D geometry refutes '{pred.operator}', supporting negative predicate",
                evidence_refs=refs,
            )
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.CONTRADICTED,
            score=0.0,
            reason=f"Same-frame 2D bounding boxes contradict '{pred.operator}'",
            evidence_refs=refs,
        )

    return PredicateEvaluation(
        predicate_id=pred.id,
        state=MatchState.UNKNOWN,
        score=0.0,
        reason=f"Spatial relation '{pred.operator}' lacks same-frame unambiguous 2D boxes",
        evidence_refs=[],
    )


def _action_role_matches(
    expected_ref: str,
    observed_ref: Any,
    grounded_entities: dict[str, GroundedEntity],
) -> bool:
    """Match predicate role ID/entity class against an explicit action_observation role value."""
    if observed_ref is None:
        return False
    obs_low = str(observed_ref).strip().lower()
    if not obs_low:
        return False
    obs_tokens = {obs_low}
    if obs_low in COCO_SYNONYM_MAP:
        obs_tokens.add(COCO_SYNONYM_MAP[obs_low])
    return bool(_entity_target_tokens(expected_ref, grounded_entities) & obs_tokens)


def _evaluate_action_predicate(
    pred: Predicate,
    evidence_items: Sequence[dict[str, Any]],
    grounded_entities: dict[str, GroundedEntity],
) -> PredicateEvaluation:
    """Evaluate action predicates solely from explicit action_observation evidence.

    Static visual_detection boxes never prove actions on their own.
    """
    op_low = pred.operator.strip().lower()
    is_neg = pred.polarity == "negative"

    for ev in evidence_items:
        if str(ev.get("kind", "")).lower() != "action_observation":
            continue
        if float(ev.get("confidence", 0.9)) < 0.5:
            continue
        ev_op = str(ev.get("operator") or ev.get("action") or ev.get("predicate") or "").strip().lower()
        if ev_op != op_low:
            continue

        ev_roles = ev.get("roles") if isinstance(ev.get("roles"), dict) else {}
        ev_id = str(ev.get("evidence_id", ""))
        refs = [ev_id] if ev_id else []
        ev_is_neg = str(ev.get("polarity", "positive")).lower() == "negative"

        if not pred.roles:
            contradicts = is_neg ^ ev_is_neg
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.CONTRADICTED if contradicts else MatchState.SUPPORTED,
                score=0.0 if contradicts else float(ev.get("confidence", 0.9)),
                reason=f"Explicit action_observation for '{pred.operator}'",
                evidence_refs=refs,
            )

        pred_subj = pred.roles.get("subject") or pred.roles.get("agent")
        pred_obj = pred.roles.get("object") or pred.roles.get("target") or pred.roles.get("recipient")
        obs_subj = ev_roles.get("subject") or ev_roles.get("agent")
        obs_obj = ev_roles.get("object") or ev_roles.get("target") or ev_roles.get("recipient")

        # Exact/canonical matching over requested role slots
        all_roles_matched = True
        for rk, rv in pred.roles.items():
            obs_val = ev_roles.get(rk)
            if obs_val is None and rk in ("object", "target", "recipient"):
                obs_val = obs_obj
            elif obs_val is None and rk in ("subject", "agent"):
                obs_val = obs_subj
            if not _action_role_matches(rv, obs_val, grounded_entities):
                all_roles_matched = False
                break

        if all_roles_matched:
            contradicts = is_neg ^ ev_is_neg
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.CONTRADICTED if contradicts else MatchState.SUPPORTED,
                score=0.0 if contradicts else float(ev.get("confidence", 0.9)),
                reason=f"Explicit action_observation matched roles for '{pred.operator}'",
                evidence_refs=refs,
            )

        # Explicit reversed roles refute positive directed action when roles swap cleanly
        if (
            not is_neg
            and not ev_is_neg
            and pred_subj
            and pred_obj
            and _action_role_matches(pred_subj, obs_obj, grounded_entities)
            and _action_role_matches(pred_obj, obs_subj, grounded_entities)
        ):
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.CONTRADICTED,
                score=0.0,
                reason=f"Observed action '{pred.operator}' has reversed subject/object roles",
                evidence_refs=refs,
            )

    # Invariant: Bounding boxes alone do not prove hand-over / action predicates
    return PredicateEvaluation(
        predicate_id=pred.id,
        state=MatchState.UNKNOWN,
        score=0.0,
        reason=f"Action '{pred.operator}' lacks verified action evidence",
        evidence_refs=[],
    )


def evaluate_predicate(
    pred: Predicate,
    evidence_items: Sequence[dict[str, Any]],
    grounded_entities: dict[str, GroundedEntity],
) -> PredicateEvaluation:
    """Evaluate single predicate against evidence honoring three-valued logic and polarity."""
    if "negation_scope_unresolved" in pred.uncertainty:
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
    pred_kind = pred.kind.lower()
    is_neg = pred.polarity == "negative"

    # 1. OCR Predicate evaluation
    if pred_kind == "ocr":
        matched_ev = None
        for ev in evidence_items:
            if ev.get("kind") in ("ocr_track", "ocr_observation"):
                if contains_text_phrase(ev.get("text"), pred.value):
                    matched_ev = ev
                    break

        if matched_ev:
            if is_neg:
                # Contradiction: text was found when negated
                return PredicateEvaluation(
                    predicate_id=pred.id,
                    state=MatchState.CONTRADICTED,
                    score=0.0,
                    reason=f"Found negated text '{matched_ev.get('text')}'",
                    evidence_refs=[matched_ev.get("evidence_id", "")],
                )
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.SUPPORTED,
                score=float(matched_ev.get("confidence", 0.95)),
                reason=f"Found exact text match '{matched_ev.get('text')}'",
                evidence_refs=[matched_ev.get("evidence_id", "")],
            )
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"No OCR observation establishes '{pred.value}'",
            evidence_refs=[],
        )

    # 2. ASR Predicate evaluation
    if pred_kind == "asr":
        matched_asr = None
        for ev in evidence_items:
            if ev.get("kind") == "asr_segment":
                if contains_text_phrase(ev.get("text"), pred.value):
                    matched_asr = ev
                    break

        if matched_asr:
            if is_neg:
                if pred.roles.get("speaker"):
                    return PredicateEvaluation(
                        predicate_id=pred.id,
                        state=MatchState.UNKNOWN,
                        score=0.0,
                        reason="Speaker identity is not established by ASR",
                        evidence_refs=[matched_asr.get("evidence_id", "")],
                    )
                return PredicateEvaluation(
                    predicate_id=pred.id,
                    state=MatchState.CONTRADICTED,
                    score=0.0,
                    reason="Negated speech observed",
                    evidence_refs=[matched_asr.get("evidence_id", "")],
                )
            speaker_role = pred.roles.get("speaker")
            # Invariant: Transcript alone does NOT prove speaker identity without active biometric lip-sync
            if speaker_role:
                return PredicateEvaluation(
                    predicate_id=pred.id,
                    state=MatchState.PARTIAL,
                    score=0.75,
                    reason=f"Speech transcript matched '{matched_asr.get('text')}', but speaker binding '{speaker_role}' unverified",
                    evidence_refs=[matched_asr.get("evidence_id", "")],
                )
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.SUPPORTED,
                score=float(matched_asr.get("confidence", 0.90)),
                reason=f"Found speech transcript '{matched_asr.get('text')}'",
                evidence_refs=[matched_asr.get("evidence_id", "")],
            )
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"No speech segment matching '{pred.value}'",
            evidence_refs=[],
        )

    # 3. Action Predicate evaluation
    if pred_kind == "action":
        return _evaluate_action_predicate(pred, evidence_items, grounded_entities)

    # 4. Spatial Predicate evaluation
    if pred_kind == "spatial":
        return _evaluate_spatial_predicate(pred, evidence_items, grounded_entities)

    # Default fallback
    return PredicateEvaluation(
        predicate_id=pred.id,
        state=MatchState.UNKNOWN,
        score=0.0,
        reason=f"Unsupported predicate kind '{pred_kind}'",
        evidence_refs=[],
    )
