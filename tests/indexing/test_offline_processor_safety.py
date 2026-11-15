"""CPU-safe safety tests for OfflineProcessingPipeline (B1 guards, defaults, no random embeddings)."""
from __future__ import annotations

from pathlib import Path
import pytest

from scripts.offline_processor import OfflineProcessingPipeline, _guard_and_backup_path


def test_offline_processor_default_faiss_path_is_scratch(tmp_path: Path):
    pipeline = OfflineProcessingPipeline(db_path=tmp_path / "metadata.db")
    assert pipeline.faiss_path == Path("work/offline_scratch/faiss_index.bin")
    assert "data/processed" not in pipeline.faiss_path.as_posix()


def test_offline_processor_run_all_raises_if_faiss_exists_without_overwrite(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    faiss_path = tmp_path / "faiss_index.bin"
    faiss_path.write_bytes(b"existing-index")

    pipeline = OfflineProcessingPipeline(
        db_path=db_path,
        faiss_path=faiss_path,
        raw_data_dir=tmp_path / "raw",
        keyframes_dir=tmp_path / "keyframes",
        i_understand_overwrite=False,
    )
    with pytest.raises(RuntimeError, match="i-understand-overwrite"):
        pipeline.run_all()


def test_guard_and_backup_path_creates_bak_when_overwrite_enabled(tmp_path: Path):
    target = tmp_path / "faiss_index.bin"
    target.write_bytes(b"original-bytes")

    with pytest.raises(RuntimeError, match="i-understand-overwrite"):
        _guard_and_backup_path(target, i_understand_overwrite=False, label="FAISS index")

    backup = _guard_and_backup_path(target, i_understand_overwrite=True, label="FAISS index")
    assert backup is not None
    assert backup.is_file()
    assert backup.name.startswith("faiss_index.bin.bak-")
    assert backup.read_bytes() == b"original-bytes"


def test_offline_processor_raises_not_implemented_when_siglip2_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from videntra.encoders.siglip2 import SigLIP2Config

    def _no_weights(self):
        raise FileNotFoundError("Cannot locate SigLIP 2 model weights directory")

    monkeypatch.setattr(SigLIP2Config, "resolve_model_path", _no_weights)
    pipeline = OfflineProcessingPipeline(
        db_path=tmp_path / "metadata.db",
        faiss_path=tmp_path / "faiss_index.bin",
        models_dir=tmp_path / "empty_models",
    )
    with pytest.raises(NotImplementedError, match="SigLIP2Encoder"):
        pipeline._resolve_encoder()

    src = Path("scripts/offline_processor.py").read_text(encoding="utf-8")
    assert "np.random.randn" not in src
