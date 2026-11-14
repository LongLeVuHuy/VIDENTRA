"""Evidence normalization, representation, and persistent storage."""
from .normalize import normalize_text, has_vietnamese_diacritics
from .store import EvidenceStore, OCRTrackRecord, ASRSegmentRecord

__all__ = [
    "normalize_text",
    "has_vietnamese_diacritics",
    "EvidenceStore",
    "OCRTrackRecord",
    "ASRSegmentRecord",
]
