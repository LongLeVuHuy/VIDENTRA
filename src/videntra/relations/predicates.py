"""Three-valued logic predicate evaluation preserving unknown states for missing evidence."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from videntra.query.schemas import Predicate
from videntra.evidence.normalize import contains_text_phrase
from .entities import GroundedEntity


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


def evaluate_predicate(
    pred: Predicate,
    evidence_items: Sequence[dict[str, Any]],
    grounded_entities: dict[str, GroundedEntity],
) -> PredicateEvaluation:
    """Evaluate single predicate against evidence honoring three-valued logic and polarity."""
    if "negation_scope_unresolved" in pred.uncertainty:
        return PredicateEvaluation(pred.id, MatchState.UNKNOWN, 0.0, "Negation scope is unresolved", [])
    pred_kind = pred.kind.lower()
    is_neg = (pred.polarity == "negative")

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
            else:
                return PredicateEvaluation(
                    predicate_id=pred.id,
                    state=MatchState.SUPPORTED,
                    score=float(matched_ev.get("confidence", 0.95)),
                    reason=f"Found exact text match '{matched_ev.get('text')}'",
                    evidence_refs=[matched_ev.get("evidence_id", "")],
                )
        else:
            return PredicateEvaluation(predicate_id=pred.id, state=MatchState.UNKNOWN, score=0.0,
                reason=f"No OCR observation establishes '{pred.value}'", evidence_refs=[])

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
                    return PredicateEvaluation(predicate_id=pred.id, state=MatchState.UNKNOWN,
                        score=0.0, reason="Speaker identity is not established by ASR", evidence_refs=[matched_asr.get("evidence_id", "")])
                return PredicateEvaluation(predicate_id=pred.id, state=MatchState.CONTRADICTED,
                    score=0.0, reason="Negated speech observed", evidence_refs=[matched_asr.get("evidence_id", "")])
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
        else:
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.UNKNOWN,
                score=0.0,
                reason=f"No speech segment matching '{pred.value}'",
                evidence_refs=[],
            )

    # 3. Action Predicate evaluation
    if pred_kind == "action":
        # Invariant: Bounding boxes alone do not prove hand-over action
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"Action '{pred.operator}' lacks verified action evidence",
            evidence_refs=[],
        )

    # 4. Spatial Predicate evaluation
    if pred_kind == "spatial":
        # Invariant: 'behind' 3D is unknown without explicit depth
        if pred.operator == "behind":
            return PredicateEvaluation(
                predicate_id=pred.id,
                state=MatchState.UNKNOWN,
                score=0.0,
                reason="3D depth cannot be decisively proven from 2D plane; marked unknown",
                evidence_refs=[],
            )
        return PredicateEvaluation(
            predicate_id=pred.id,
            state=MatchState.UNKNOWN,
            score=0.0,
            reason=f"Spatial relation '{pred.operator}' has no verified binding evidence",
            evidence_refs=[],
        )

    # Default fallback
    return PredicateEvaluation(
        predicate_id=pred.id,
        state=MatchState.UNKNOWN,
        score=0.0,
        reason=f"Unsupported predicate kind '{pred_kind}'",
        evidence_refs=[],
    )
