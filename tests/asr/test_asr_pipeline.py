"""Tests for ASR voice activity detection, transcription, and timeline alignment."""
import numpy as np
import pytest

from videntra.asr.vad import VoiceActivityDetector, VADSegment, extract_audio_mono
from videntra.asr.transcription import MockASRTranscriber, TranscriptResult
from videntra.asr.alignment import TimelineAligner
from videntra.fixtures import build_fixtures


def test_vad_speech_intervals():
    vad = VoiceActivityDetector(sample_rate=16000, frame_ms=30, energy_threshold=0.02, min_speech_ms=200)

    # 3 seconds: 0.0-1.0s silence, 1.0-2.0s 440Hz tone (simulated speech), 2.0-3.0s silence
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False, dtype=np.float32)
    signal = np.zeros(3 * sr, dtype=np.float32)

    # 440Hz tone with amplitude 0.2
    speech_start_idx = 1 * sr
    speech_end_idx = 2 * sr
    signal[speech_start_idx:speech_end_idx] = 0.2 * np.sin(2 * np.pi * 440 * t[speech_start_idx:speech_end_idx])

    segments = vad.detect_segments(signal)
    assert len(segments) == 1
    seg = segments[0]
    # Check start is approximately 1.0s (1_000_000 us) and end ~ 2.0s
    assert 900_000 <= seg.start_us <= 1_100_000
    assert 1_900_000 <= seg.end_us <= 2_100_000


def test_vad_no_speech_vs_missing_audio(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    build_fixtures(fixtures_dir)

    # cfr.mkv has NO audio stream
    cfr_path = fixtures_dir / "cfr.mkv"
    cfr_samples, cfr_has_audio = extract_audio_mono(cfr_path)
    assert cfr_has_audio is False
    assert cfr_samples is None

    # audio_delay.mkv HAS audio stream
    audio_path = fixtures_dir / "audio_delay.mkv"
    audio_samples, audio_has_audio = extract_audio_mono(audio_path)
    assert audio_has_audio is True
    assert audio_samples is not None
    assert len(audio_samples) > 0


def test_transcription_mock_and_normalization():
    transcriber = MockASRTranscriber(default_text="  Xin  chào  quý khách   ")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    res = transcriber.transcribe_chunk(dummy_audio)
    assert res.text == "Xin chào quý khách"
    assert res.language == "vi"
    assert res.confidence == 0.94


def test_timeline_alignment_reversible():
    aligner = TimelineAligner()
    # Mock ledger timeline with 10 frames at 0.1s intervals (100_000 us)
    aligner._timestamps_us = [i * 100_000 for i in range(10)]
    aligner._ordinals = list(range(10))

    # Speech segment from 250_000 us to 650_000 us
    vad_seg = VADSegment(start_us=250_000, end_us=650_000, energy=0.1)
    trans = TranscriptResult(text="Cảm ơn", confidence=0.92, language="vi")

    aligned = aligner.create_aligned_segment(
        video_id="vid_test",
        vad_segment=vad_seg,
        transcript=trans,
    )

    # 250_000 us is closest to frame 2 (200k) or 3 (300k) -> 2
    # 650_000 us is closest to frame 6 (600k) or 7 (700k) -> 6
    assert aligned.start_ordinal in (2, 3)
    assert aligned.end_ordinal in (6, 7)
    assert aligned.start_seconds == 0.25
    assert aligned.end_seconds == 0.65

    # Reverse lookup from frame 4
    covered = aligner.reverse_lookup(4, [aligned])
    assert len(covered) == 1
    assert covered[0].text == "Cảm ơn"

    # Reverse lookup from frame 9 (outside segment)
    outside = aligner.reverse_lookup(9, [aligned])
    assert len(outside) == 0
