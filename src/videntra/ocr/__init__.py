"""OCR detection, recognition, routing, and temporal tracking."""
from .detection import OCRDetector, TextBoundingBox
from .recognition import OCRRecognizer, MockOCRRecognizer, TextRecognitionResult
from .routing import OCRRouter
from .tracking import OCRTracker, FrameOCRObservation

__all__ = [
    "OCRDetector",
    "TextBoundingBox",
    "OCRRecognizer",
    "MockOCRRecognizer",
    "TextRecognitionResult",
    "OCRRouter",
    "OCRTracker",
    "FrameOCRObservation",
]
