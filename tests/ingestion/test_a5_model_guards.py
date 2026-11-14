import hashlib
from types import SimpleNamespace

import pytest

from scripts import kaggle_multimodal_offline_worker as worker


def test_a5_serializes_a_real_asr_segment_without_synthetic_confidence():
    segment = SimpleNamespace(start=1.25, end=2.5, text=" hello ", avg_logprob=-0.42)
    record = worker.asr_segment_record("film", 3, segment, SimpleNamespace(language="en"))
    assert record == {"segment_id": "film__asr_3", "video_id": "film",
                      "model_id": "faster_whisper_small", "start_ms": 1250,
                      "end_ms": 2500, "raw_text": "hello", "language": "en",
                      "confidence": -0.42}


def test_a5_bundle_rejects_missing_or_changed_model(tmp_path, monkeypatch):
    content = b"verified fixture"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(worker, "A5_MODEL_FILES", {"ocr/model.onnx": digest})
    with pytest.raises(FileNotFoundError, match="Missing pinned A5 model"):
        worker.verify_a5_model_bundle(tmp_path)
    model = tmp_path / "ocr" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(content)
    assert worker.verify_a5_model_bundle(tmp_path) == {"ocr/model.onnx": digest}
    model.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        worker.verify_a5_model_bundle(tmp_path)


def test_a5_rejects_unpinned_optional_whisper_preprocessor(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "A5_MODEL_FILES", {})
    model_dir = tmp_path / "whisper-small"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Unpinned Whisper preprocessor"):
        worker.verify_a5_model_bundle(tmp_path)


def test_a5_runtime_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr(worker, "A5_PACKAGE_VERSIONS", {"example": "1.0"})
    monkeypatch.setattr(worker.importlib_metadata, "version", lambda _: "1.0")
    assert worker.verify_a5_runtime() == {"example": "1.0"}
    monkeypatch.setattr(worker.importlib_metadata, "version", lambda _: "2.0")
    with pytest.raises(RuntimeError, match="Runtime mismatch"):
        worker.verify_a5_runtime()


def test_a5_main_requires_explicit_local_bundle(monkeypatch):
    monkeypatch.delenv("VIDENTRA_A5_MODEL_ROOT", raising=False)
    monkeypatch.setattr(worker, "ensure_output_files_absent", lambda _: None)
    with pytest.raises(RuntimeError, match="VIDENTRA_A5_MODEL_ROOT"):
        worker.main()
