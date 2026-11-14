"""Generation of candidate temporal intervals around multimodal evidence seeds."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TemporalProposal:
    proposal_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    start_ordinal: int
    end_ordinal: int
    seed_score: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def generate_proposals(
    video_id: str,
    seed_timestamps: Sequence[float],
    seed_ordinals: Sequence[int],
    window_duration: float = 4.0,
    stride: float = 2.0,
    max_video_duration: float = 300.0,
) -> list[TemporalProposal]:
    """Generate fixed-duration sliding window proposals anchored at seeds."""
    proposals = []
    seen_intervals = set()

    for idx, (ts, ord_idx) in enumerate(zip(seed_timestamps, seed_ordinals)):
        st = max(0.0, ts - window_duration / 2.0)
        et = min(max_video_duration, ts + window_duration / 2.0)
        key = (round(st, 1), round(et, 1))

        if key not in seen_intervals:
            seen_intervals.add(key)
            proposals.append(
                TemporalProposal(
                    proposal_id=f"prop_{video_id}_{idx}",
                    video_id=video_id,
                    start_seconds=round(st, 4),
                    end_seconds=round(et, 4),
                    start_ordinal=max(0, ord_idx - 15),
                    end_ordinal=ord_idx + 15,
                    seed_score=1.0,
                )
            )

    return proposals
