"""Independent retrieval branches for Visual, OCR, and ASR modalities."""
from dataclasses import dataclass
from typing import Any, Sequence
import uuid

from videntra.retrieval.visual import VisualRetrievalEngine, RankedMoment
from videntra.indexing.text_store import TextIndexStore, TextHit


@dataclass(frozen=True)
class BranchCandidate:
    candidate_id: str
    video_id: str
    branch_name: str  # "visual", "ocr", "asr"
    start_seconds: float
    end_seconds: float
    start_ordinal: int | None
    end_ordinal: int | None
    score: float
    matched_text: str | None = None
    evidence_item: dict[str, Any] | None = None


class VisualBranch:
    """Executes visual similarity search via SigLIP 2 and FAISS index."""

    def __init__(self, engine: VisualRetrievalEngine):
        self.engine = engine

    def retrieve(self, query: str, top_k: int = 50) -> tuple[list[BranchCandidate], str, str | None]:
        try:
            moments = self.engine.search(query=query, top_k_moments=top_k)
            candidates = []
            for m in moments:
                c = BranchCandidate(
                    candidate_id=m.moment_id,
                    video_id=m.video_id,
                    branch_name="visual",
                    start_seconds=m.moment_start_seconds,
                    end_seconds=m.moment_end_seconds,
                    start_ordinal=m.start_original_frame_index,
                    end_ordinal=m.end_original_frame_index,
                    score=float(m.ranking_score),
                    evidence_item={
                        "evidence_id": f"ev_vis_{m.video_id}_{m.representative_original_frame_index}",
                        "kind": "frame",
                        "original_decoded_frame_index": m.representative_original_frame_index,
                        "timestamp_us": int(round(m.representative_timestamp_seconds * 1_000_000)),
                    },
                )
                candidates.append(c)
            return candidates, "complete", None
        except Exception as exc:
            return [], "failed", f"Visual branch error: {exc}"


class OCRBranch:
    """Executes lexical scene-text search via FTS5 index on OCR tracks."""

    def __init__(self, text_store: TextIndexStore, padding_seconds: float = 1.0):
        self.text_store = text_store
        self.padding_seconds = padding_seconds

    def retrieve(self, query: str, top_k: int = 50) -> tuple[list[BranchCandidate], str, str | None]:
        try:
            hits = self.text_store.search(query=query, source_type="ocr", top_k=top_k)
            candidates = []
            for h in hits:
                st = max(0.0, h.start_seconds - self.padding_seconds)
                et = h.end_seconds + self.padding_seconds
                c_id = f"cand_ocr_{h.hit_id}"
                ev = {
                    "evidence_id": f"ev_ocr_{h.hit_id}",
                    "kind": "ocr_track",
                    "text": h.text,
                    "confidence": h.confidence,
                    "start_seconds": h.start_seconds,
                    "end_seconds": h.end_seconds,
                }
                candidates.append(
                    BranchCandidate(
                        candidate_id=c_id,
                        video_id=h.video_id,
                        branch_name="ocr",
                        start_seconds=st,
                        end_seconds=et,
                        start_ordinal=h.start_ordinal,
                        end_ordinal=h.end_ordinal,
                        score=float(h.score),
                        matched_text=h.text,
                        evidence_item=ev,
                    )
                )
            return candidates, "complete", None
        except Exception as exc:
            return [], "failed", f"OCR branch error: {exc}"


class ASRBranch:
    """Executes speech transcript search via FTS5 index on ASR segments."""

    def __init__(self, text_store: TextIndexStore, padding_seconds: float = 1.0):
        self.text_store = text_store
        self.padding_seconds = padding_seconds

    def retrieve(self, query: str, top_k: int = 50) -> tuple[list[BranchCandidate], str, str | None]:
        try:
            hits = self.text_store.search(query=query, source_type="asr", top_k=top_k)
            candidates = []
            for h in hits:
                st = max(0.0, h.start_seconds - self.padding_seconds)
                et = h.end_seconds + self.padding_seconds
                c_id = f"cand_asr_{h.hit_id}"
                ev = {
                    "evidence_id": f"ev_asr_{h.hit_id}",
                    "kind": "asr_segment",
                    "text": h.text,
                    "confidence": h.confidence,
                    "start_seconds": h.start_seconds,
                    "end_seconds": h.end_seconds,
                }
                candidates.append(
                    BranchCandidate(
                        candidate_id=c_id,
                        video_id=h.video_id,
                        branch_name="asr",
                        start_seconds=st,
                        end_seconds=et,
                        start_ordinal=h.start_ordinal,
                        end_ordinal=h.end_ordinal,
                        score=float(h.score),
                        matched_text=h.text,
                        evidence_item=ev,
                    )
                )
            return candidates, "complete", None
        except Exception as exc:
            return [], "failed", f"ASR branch error: {exc}"
