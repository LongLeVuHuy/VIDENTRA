"""Tests for OCR detection, recognition routing, and temporal tracking rules."""
import numpy as np
import pytest

from videntra.ocr.detection import OCRDetector, TextBoundingBox, box_2d_iou
from videntra.ocr.recognition import MockOCRRecognizer, VietOCRRecognizer, GeneralOCRRecognizer
from videntra.ocr.routing import OCRRouter
from videntra.ocr.tracking import OCRTracker, FrameOCRObservation


def test_box_2d_iou():
    b1 = TextBoundingBox(0.0, 0.0, 10.0, 10.0)
    b2 = TextBoundingBox(5.0, 0.0, 15.0, 10.0)
    # inter = 5 * 10 = 50, union = 100 + 100 - 50 = 150 -> 1/3
    assert pytest.approx(box_2d_iou(b1, b2), rel=1e-3) == 1.0 / 3.0

    b3 = TextBoundingBox(20.0, 20.0, 30.0, 30.0)
    assert box_2d_iou(b1, b3) == 0.0


def test_ocr_detector_gradient_regions():
    detector = OCRDetector()
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Add high-contrast text bar in the middle
    img[40:60, 20:180] = 255

    boxes = detector.detect(img)
    assert len(boxes) > 0
    assert 0.0 <= boxes[0].x1 < boxes[0].x2 <= 1.0


def test_ocr_router_dispatches_vietnamese():
    general = MockOCRRecognizer(mock_text="STORE 101", confidence=0.85)
    viet = MockOCRRecognizer(mock_text="Cửa hàng 101", confidence=0.92)
    router = OCRRouter(general_recognizer=general, vietnamese_recognizer=viet)

    dummy_crop = np.zeros((32, 100, 3), dtype=np.uint8)

    # With Vietnamese hint
    res, reason = router.route_and_recognize(dummy_crop, hint_text="Cửa hàng giảm giá")
    assert reason == "vietnamese_diacritics_hint"
    assert res.text == "Cửa hàng 101"

    # Pure English
    general_en = MockOCRRecognizer(mock_text="SALE", confidence=0.95)
    router_en = OCRRouter(general_recognizer=general_en, vietnamese_recognizer=viet)
    res_en, reason_en = router_en.route_and_recognize(dummy_crop)
    assert reason_en == "primary_general"
    assert res_en.text == "SALE"


def test_ocr_tracker_singleton_is_instant():
    """Blueprint rule: Singleton OCR observation must produce an instant track."""
    tracker = OCRTracker()
    obs = [
        FrameOCRObservation(
            ordinal=42,
            timestamp_seconds=4.20,
            bbox=(0.1, 0.1, 0.3, 0.2),
            text="FLASH_SALE",
            confidence=0.95,
        )
    ]
    tracks = tracker.track(video_id="v1", observations=obs)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.start_ordinal == 42
    assert tr.end_ordinal == 42
    assert tr.start_seconds == 4.20
    assert tr.end_seconds == 4.20
    assert tr.frame_count == 1


def test_ocr_tracker_gap_separation_rule():
    """Blueprint rule: The same text appearing at two distant times must NOT be merged."""
    tracker = OCRTracker(max_gap_seconds=1.5, min_spatial_iou=0.3)

    # Event 1: frames at t = 1.0s and t = 1.2s
    # Event 2: frame at t = 5.0s (gap is 3.8s > 1.5s!)
    obs = [
        FrameOCRObservation(10, 1.0, (0.1, 0.1, 0.3, 0.2), "CAUTION", 0.9),
        FrameOCRObservation(12, 1.2, (0.1, 0.1, 0.3, 0.2), "CAUTION", 0.95),
        FrameOCRObservation(50, 5.0, (0.1, 0.1, 0.3, 0.2), "CAUTION", 0.92),
    ]

    tracks = tracker.track(video_id="v1", observations=obs)

    # Must be split into 2 separate tracks!
    assert len(tracks) == 2

    # Track 1: [1.0s, 1.2s]
    assert tracks[0].start_seconds == 1.0
    assert tracks[0].end_seconds == 1.2
    assert tracks[0].frame_count == 2

    # Track 2: instant [5.0s, 5.0s]
    assert tracks[1].start_seconds == 5.0
    assert tracks[1].end_seconds == 5.0
    assert tracks[1].frame_count == 1
