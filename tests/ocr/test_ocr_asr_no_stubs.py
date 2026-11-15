"""CPU-safe unit tests ensuring real OCR/ASR paths have no fake stubs and Kaggle ocr_asr roundtrip works."""
from __future__ import annotations

import builtins
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.ingest_kaggle_offline_results import ingest_ocr_asr_artifacts
from scripts.kaggle_multimodal_offline_worker import run_ocr_asr_extraction
from videntra.asr.transcription import PhoWhisperTranscriber, TranscriptResult
from videntra.ocr.recognition import (
    GeneralOCRRecognizer,
    TextRecognitionResult,
    VietOCRRecognizer,
)
from videntra.ocr.routing import OCRRouter
from videntra.storage.database import OfflineMetadataStore


def _block_imports(monkeypatch: pytest.MonkeyPatch, blocked_prefixes: tuple[str, ...]) -> None:
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == p or name.startswith(f"{p}.") for p in blocked_prefixes):
            raise ImportError(f"Blocked import for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def test_general_ocr_recognizer_raises_when_libs_missing(monkeypatch: pytest.MonkeyPatch):
    _block_imports(monkeypatch, ("paddleocr", "rapidocr_onnxruntime"))
    rec = GeneralOCRRecognizer()
    crop = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Neither PaddleOCR nor RapidOCR"):
        rec.recognize_crop(crop)


def test_vietocr_recognizer_raises_when_weights_or_lib_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    crop = np.zeros((32, 64, 3), dtype=np.uint8)
    missing_weights = tmp_path / "does_not_exist.pth"
    rec = VietOCRRecognizer(model_path=missing_weights)
    with pytest.raises(RuntimeError):
        rec.recognize_crop(crop)

    _block_imports(monkeypatch, ("vietocr",))
    with pytest.raises(RuntimeError):
        rec.recognize_crop(crop)


def test_no_fake_stub_strings_in_recognition_or_asr_source():
    rec_src = Path("src/videntra/ocr/recognition.py").read_text(encoding="utf-8")
    assert "VIDENTRA 2026" not in rec_src
    assert "Cửa hàng giảm giá" not in rec_src

    asr_src = Path("src/videntra/asr/transcription.py").read_text(encoding="utf-8")
    # MockASRTranscriber has "Cảm ơn bạn rất nhiều", but PhoWhisperTranscriber must never have "Cảm ơn" fallback
    phowhisper_section = asr_src.split("class PhoWhisperTranscriber", 1)[1]
    assert "Cảm ơn" not in phowhisper_section


def test_phowhisper_error_returns_empty_transcript_with_error_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    transcriber = PhoWhisperTranscriber(model_path=tmp_path / "missing_dir", local_files_only=True)

    def _fail_load():
        raise FileNotFoundError("PhoWhisper model checkpoint directory not found")

    monkeypatch.setattr(transcriber, "_ensure_loaded", _fail_load)
    audio = np.ones(1600, dtype=np.float32) * 0.01
    with caplog.at_level(logging.WARNING):
        res = transcriber.transcribe_chunk(audio, sample_rate=16000)

    assert isinstance(res, TranscriptResult)
    assert res.text == ""
    assert res.confidence == 0.0
    assert res.status == "error"
    assert "Cảm ơn" not in res.text
    assert any("PhoWhisper transcription failed" in rec.message for rec in caplog.records)


def test_kaggle_ocr_asr_worker_and_ingest_roundtrip(tmp_path: Path):
    # 1. Seed existing DB with 1 video, 1 keyframe, and 1 pre-existing OCR observation
    db_path = tmp_path / "metadata.db"
    store = OfflineMetadataStore(db_path)
    store.register_video({
        "video_id": "vid_001",
        "dataset": "test",
        "file_path": "vid_001.mp4",
        "duration_ms": 5000,
        "fps": 25.0,
        "time_base_num": 1,
        "time_base_den": 1000,
        "width": 320,
        "height": 240,
        "total_frames": 125,
        "has_audio": 1,
        "status": "completed",
    })
    store.insert_keyframes([{
        "keyframe_id": "vid_001__kf_00000",
        "video_id": "vid_001",
        "keyframe_index": 0,
        "source_frame_index": 0,
        "pts": 0,
        "time_base": "1/1000",
        "timestamp_ms": 0,
        "keyframe_path": "",
        "thumbnail_path": "",
        "faiss_id": 0,
        "sampling_reason": "uniform",
    }])
    store.insert_ocr_observations([{
        "observation_id": "existing_obs_0",
        "video_id": "vid_001",
        "keyframe_id": "vid_001__kf_00000",
        "timestamp_ms": 0,
        "model_id": "existing_model",
        "raw_text": "EXISTING SIGN",
        "bbox_x1": 0.05,
        "bbox_y1": 0.05,
        "bbox_x2": 0.25,
        "bbox_y2": 0.15,
        "confidence": 0.91,
        "language": "en",
    }])

    # 2. Prepare dummy keyframe image + manifest for worker
    from PIL import Image
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    kf_img_path = images_dir / "kf_00000.jpg"
    Image.fromarray(np.full((64, 128, 3), 255, dtype=np.uint8)).save(kf_img_path)

    manifest_path = tmp_path / "keyframes_manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "keyframe_id": "vid_001__kf_00000",
            "video_id": "vid_001",
            "timestamp_ms": 0,
            "image_file": "kf_00000.jpg",
        }) + "\n",
        encoding="utf-8",
    )

    class _FakeGen:
        model_id = "paddleocr_en"

        def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
            return TextRecognitionResult(text="SALE 50%", confidence=0.92, model_id="paddleocr_en", script="latin")

    class _FakeViet:
        model_id = "vietocr_vgg"

        def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
            return TextRecognitionResult(text="Khuyến mãi", confidence=0.95, model_id="vietocr_vgg", script="vietnamese")

    fake_router = OCRRouter(general_recognizer=_FakeGen(), vietnamese_recognizer=_FakeViet())

    def _fake_detector(img_rgb: np.ndarray):
        return [((0.1, 0.1, 0.9, 0.5), "Khuyến mãi", 0.85)]

    out_dir = tmp_path / "kaggle_ocr_asr_out"
    receipt = run_ocr_asr_extraction(
        keyframes_manifest=manifest_path,
        out_dir=out_dir,
        images_dir=images_dir,
        media_dir=None,
        ocr_backend="paddleocr_en",
        enable_vietocr=True,
        asr_backend="phowhisper_small",
        device="cpu",
        ocr_router=fake_router,
        detector_fn=_fake_detector,
    )
    assert receipt["status"] == "success"
    assert receipt["total_ocr_observations"] == 1
    assert (out_dir / "ocr_observations.jsonl").is_file()
    assert (out_dir / "asr_segments.jsonl").is_file()
    assert (out_dir / "ocr_asr_receipt.json").is_file()

    # Append a mock ASR segment to asr_segments.jsonl to test full OCR+ASR ingest + FTS5 rebuild
    (out_dir / "asr_segments.jsonl").write_text(
        json.dumps({
            "segment_id": "vid_001__asr_0",
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 100,
            "end_ms": 2100,
            "raw_text": "Xin chào các bạn",
            "language": "vi",
            "confidence": 0.90,
        }) + "\n",
        encoding="utf-8",
    )

    ingest_res = ingest_ocr_asr_artifacts(outputs_dir=out_dir, db_path=db_path)
    assert ingest_res["ocr_observations_ingested"] == 1
    assert ingest_res["asr_segments_ingested"] == 1
    assert ingest_res["fts_rebuilt"] is True

    with sqlite3.connect(str(db_path)) as conn:
        obs_ids = {r[0] for r in conn.execute("SELECT observation_id FROM OCRObservations").fetchall()}
        assert "existing_obs_0" in obs_ids
        assert "vid_001__kf_00000__ocr_0" in obs_ids

        ocr_fts = conn.execute("SELECT video_id FROM fts_ocr WHERE fts_ocr MATCH 'khuyen'").fetchall()
        asr_fts = conn.execute("SELECT video_id FROM fts_asr WHERE fts_asr MATCH 'chao'").fetchall()
        assert len(ocr_fts) == 1
        assert len(asr_fts) == 1
