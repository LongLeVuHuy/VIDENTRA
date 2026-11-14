"""Independent branch union preserving constituent seeds across modalities."""
from typing import Sequence
from .branches import BranchCandidate


class IndependentBranchUnion:
    """Combines candidates from visual, OCR, and ASR branches without early pruning."""

    @staticmethod
    def collect(
        visual_candidates: Sequence[BranchCandidate],
        ocr_candidates: Sequence[BranchCandidate],
        asr_candidates: Sequence[BranchCandidate],
    ) -> list[BranchCandidate]:
        """Aggregate candidates preserving non-visual seeds (rescue candidates)."""
        combined = []
        combined.extend(visual_candidates)
        combined.extend(ocr_candidates)
        combined.extend(asr_candidates)
        return combined

    @staticmethod
    def compute_rescue_rate(
        fused_moments_video_ids: Sequence[str],
        visual_candidates: Sequence[BranchCandidate],
    ) -> float:
        """Measure proportion of final returned videos that were NOT in top-5 visual candidates."""
        if not fused_moments_video_ids:
            return 0.0

        top_visual_vids = set(c.video_id for c in visual_candidates[:5])
        rescued_count = sum(1 for vid in fused_moments_video_ids if vid not in top_visual_vids)
        return round(rescued_count / len(fused_moments_video_ids), 4)
