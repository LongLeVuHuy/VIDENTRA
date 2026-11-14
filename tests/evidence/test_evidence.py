"""Tests for evidence normalization, persistence, and interval queries."""
from pathlib import Path
import pytest

from videntra.evidence.normalize import normalize_text, has_vietnamese_diacritics
from videntra.evidence.store import (
    EvidenceStore,
    OCRTrackRecord,
    ASRSegmentRecord,
)


def test_unicode_normalization_and_nfc():
    import unicodedata
    target = "người"
    decomposed = unicodedata.normalize("NFD", target)
    assert decomposed != target  # Ensure it is actually decomposed
    normalized = normalize_text(decomposed)
    assert normalized == target

    # Whitespace collapse & control char strip
    messy = "  Tìm kiếm\t\x00\x08  người   áo đỏ \n "
    clean = normalize_text(messy)
    assert clean == "Tìm kiếm người áo đỏ"


def test_vietnamese_diacritics_detection():
    assert has_vietnamese_diacritics("Cửa hàng giảm giá") is True
    assert has_vietnamese_diacritics("Đang đi bộ") is True
    assert has_vietnamese_diacritics("SALE 50 PERCENT") is False
    assert has_vietnamese_diacritics("TEST PATTERN 123") is False
    assert has_vietnamese_diacritics("") is False


def test_evidence_store_parquet_roundtrip(tmp_path):
    tracks = [
        OCRTrackRecord(
            track_id="tr_01",
            video_id="vid_test",
            start_ordinal=10,
            end_ordinal=20,
            start_seconds=1.0,
            end_seconds=2.0,
            text="Siêu thị",
            confidence=0.92,
            bbox=(0.1, 0.2, 0.4, 0.5),
            frame_count=5,
        ),
        OCRTrackRecord(
            track_id="tr_02",
            video_id="vid_test",
            start_ordinal=50,
            end_ordinal=50,
            start_seconds=5.0,
            end_seconds=5.0,  # instant singleton
            text="SALE",
            confidence=0.98,
            bbox=(0.5, 0.5, 0.8, 0.7),
            frame_count=1,
        ),
    ]

    segments = [
        ASRSegmentRecord(
            segment_id="asr_01",
            video_id="vid_test",
            start_us=1000000,
            end_us=2500000,
            start_seconds=1.0,
            end_seconds=2.5,
            start_ordinal=10,
            end_ordinal=25,
            text="Cảm ơn bạn đã ghé thăm",
            language="vi",
            confidence=0.95,
        )
    ]

    ocr_path = tmp_path / "ocr_tracks.parquet"
    asr_path = tmp_path / "asr_segments.parquet"

    EvidenceStore.save_ocr_tracks(tracks, ocr_path)
    EvidenceStore.save_asr_segments(segments, asr_path)

    loaded_tracks = EvidenceStore.load_ocr_tracks(ocr_path)
    loaded_segs = EvidenceStore.load_asr_segments(asr_path)

    assert len(loaded_tracks) == 2
    assert loaded_tracks[0].text == "Siêu thị"
    assert loaded_tracks[1].start_seconds == loaded_tracks[1].end_seconds

    assert len(loaded_segs) == 1
    assert loaded_segs[0].text == "Cảm ơn bạn đã ghé thăm"
    assert loaded_segs[0].start_us == 1000000


def test_evidence_temporal_lookup():
    tracks = [
        OCRTrackRecord("tr1", "v1", 0, 10, 0.0, 1.0, "SALE", 0.9),
        OCRTrackRecord("tr2", "v1", 50, 50, 5.0, 5.0, "EXIT", 0.95),  # instant singleton
        OCRTrackRecord("tr3", "v2", 0, 10, 0.0, 1.0, "OTHER", 0.9),
    ]
    segments = [
        ASRSegmentRecord("s1", "v1", 500000, 1500000, 0.5, 1.5, 5, 15, "Xin chào", "vi", 0.9),
    ]

    # Query v1 in [0.2, 0.8]
    hits = EvidenceStore.lookup_evidence("v1", 0.2, 0.8, tracks, segments)
    kinds = [h["kind"] for h in hits]
    assert "ocr_track" in kinds
    assert "asr_segment" in kinds

    # Query v1 around instant singleton @ 5.0s
    instant_hits = EvidenceStore.lookup_evidence("v1", 4.8, 5.2, tracks, segments)
    assert len(instant_hits) == 1
    assert instant_hits[0]["text"] == "EXIT"
