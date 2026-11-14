"""Adaptive OCR routing dispatching crops to specialist recognizers."""
from pathlib import Path
from typing import Any
import numpy as np

from .recognition import OCRRecognizer, GeneralOCRRecognizer, VietOCRRecognizer, TextRecognitionResult
from videntra.evidence.normalize import has_vietnamese_diacritics


class OCRRouter:
    """Dispatches image crops between general and specialist Vietnamese recognizers."""

    def __init__(
        self,
        general_recognizer: OCRRecognizer | None = None,
        vietnamese_recognizer: OCRRecognizer | None = None,
        prefer_vietnamese: bool = True,
    ):
        self.general_recognizer = general_recognizer or GeneralOCRRecognizer()
        self.vietnamese_recognizer = vietnamese_recognizer or VietOCRRecognizer()
        self.prefer_vietnamese = prefer_vietnamese

    def route_and_recognize(self, crop: np.ndarray, hint_text: str | None = None) -> tuple[TextRecognitionResult, str]:
        """Recognize crop using appropriate specialist and return (result, routing_reason)."""
        # If hint has Vietnamese diacritics or preferred
        if hint_text and has_vietnamese_diacritics(hint_text):
            res = self.vietnamese_recognizer.recognize_crop(crop)
            return res, "vietnamese_diacritics_hint"

        # Try general recognizer first
        primary_res = self.general_recognizer.recognize_crop(crop)

        # If primary output reveals Vietnamese diacritics or low confidence
        if has_vietnamese_diacritics(primary_res.text) and self.vietnamese_recognizer:
            viet_res = self.vietnamese_recognizer.recognize_crop(crop)
            if viet_res.confidence >= primary_res.confidence:
                return viet_res, "vietnamese_specialist_promoted"

        if self.prefer_vietnamese and primary_res.confidence < 0.6 and self.vietnamese_recognizer:
            viet_res = self.vietnamese_recognizer.recognize_crop(crop)
            return viet_res, "low_confidence_vietnamese_fallback"

        return primary_res, "primary_general"
