"""Unit tests for device auto-resolution, CUDA/FP16 safety, videntra doctor CLI, and /health telemetry."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from videntra.api.app import create_app
from videntra.cli import main
from videntra.config.device import get_device_info, preferred_cuda_dtype, resolve_device
from videntra.encoders.siglip2 import SigLIP2Config, SigLIP2Encoder
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


def test_resolve_device_auto_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDENTRA_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None, default="auto") == "cuda"
    assert resolve_device("auto", default="cpu") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None, default="auto") == "cpu"
    assert resolve_device("auto") == "cpu"

    monkeypatch.setenv("VIDENTRA_DEVICE", "cuda:0")
    assert resolve_device(None, default="auto") == "cuda:0"
    assert resolve_device("cpu") == "cpu"


def test_preferred_cuda_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDENTRA_DTYPE", raising=False)
    assert preferred_cuda_dtype("cpu") == torch.float32
    assert preferred_cuda_dtype("cuda") == torch.float16
    assert preferred_cuda_dtype("cuda:0") == torch.float16

    monkeypatch.setenv("VIDENTRA_DTYPE", "float32")
    assert preferred_cuda_dtype("cuda") == torch.float32
    monkeypatch.setenv("VIDENTRA_DTYPE", "bfloat16")
    assert preferred_cuda_dtype("cuda") == torch.bfloat16


def test_siglip2_encoder_fp16_cuda_casting_safety(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model_dir = tmp_path / "siglip2"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"stub")

    moved_calls: list[dict[str, object]] = []
    input_move_log: list[tuple[str, object, object]] = []

    class _TrackedTensor:
        def __init__(self, key: str, tensor: torch.Tensor):
            self.key = key
            self.tensor = tensor

        def is_floating_point(self) -> bool:
            return bool(self.tensor.is_floating_point())

        def to(self, device=None, dtype=None):
            input_move_log.append((self.key, str(device), dtype))
            return self.tensor.to(dtype=dtype) if dtype is not None else self.tensor

    class _FakeProcessor:
        def __call__(self, images=None, text=None, **_kwargs):
            if images is not None:
                return {"pixel_values": _TrackedTensor("pixel_values", torch.ones((len(images), 3, 4, 4), dtype=torch.float32))}
            return {
                "input_ids": _TrackedTensor("input_ids", torch.ones((len(text), 8), dtype=torch.long)),
                "attention_mask": _TrackedTensor("attention_mask", torch.ones((len(text), 8), dtype=torch.long)),
            }

    class _FakeModel:
        def to(self, device=None, dtype=None):
            moved_calls.append({"device": str(device), "dtype": dtype})
            return self

        def eval(self):
            return self

        def get_image_features(self, pixel_values):
            assert pixel_values.dtype == torch.float16
            out = torch.zeros((pixel_values.shape[0], 768), dtype=torch.float16)
            out[:, 0] = 3.0
            out[:, 1] = 4.0
            return SimpleNamespace(pooler_output=out)

        def get_text_features(self, input_ids, attention_mask=None):
            assert input_ids.dtype == torch.long
            out = torch.zeros((input_ids.shape[0], 768), dtype=torch.float16)
            out[:, 2] = 5.0
            out[:, 3] = 12.0
            return out

    import transformers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *_a, **_kw: _FakeProcessor()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(lambda *_a, **_kw: _FakeModel()))

    enc = SigLIP2Encoder(SigLIP2Config(model_path=model_dir, device="auto", dtype="auto", batch_size=2))
    assert enc.device == "cuda"

    img_vecs = enc.encode_images([np.zeros((32, 32, 3), dtype=np.uint8)])
    assert img_vecs.shape == (1, 768)
    assert img_vecs.dtype == np.float32
    assert np.all(np.isfinite(img_vecs))
    assert np.isclose(np.linalg.norm(img_vecs[0]), 1.0, atol=1e-4)

    txt_vecs = enc.encode_texts(["a person riding a bicycle"])
    assert txt_vecs.shape == (1, 768)
    assert txt_vecs.dtype == np.float32
    assert np.all(np.isfinite(txt_vecs))
    assert np.isclose(np.linalg.norm(txt_vecs[0]), 1.0, atol=1e-4)

    assert moved_calls == [{"device": "cuda", "dtype": torch.float16}]
    assert ("pixel_values", "cuda", torch.float16) in input_move_log
    assert ("input_ids", "cuda", None) in input_move_log


def test_unified_pipeline_prunes_vision_model_before_cuda_move(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model_dir = tmp_path / "siglip2"
    model_dir.mkdir()

    events: list[tuple[str, object]] = []

    class _FakeFullModel:
        def __init__(self):
            self.vision_model = object()

        def __delattr__(self, name: str):
            if name == "vision_model":
                events.append(("del_vision_model", None))
            super().__delattr__(name)

        def to(self, device=None, dtype=None):
            events.append(("to", (str(device), dtype, hasattr(self, "vision_model"))))
            return self

        def eval(self):
            events.append(("eval", None))
            return self

    import transformers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *_a, **_kw: object()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(lambda *_a, **_kw: _FakeFullModel()))

    p = UnifiedMultimodalSearchPipeline(model_dir=model_dir, device="auto", lazy_load=True, enable_videoprism=False, use_disk_cache=False)
    assert p.device == "cuda"
    p._ensure_text_encoder()

    assert events[0] == ("del_vision_model", None)
    assert events[1] == ("to", ("cuda", torch.float16, False))
    assert events[2] == ("eval", None)


def test_cli_doctor_non_strict_and_strict_missing(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    missing_db = tmp_path / "missing.db"
    rc_ok = main(["doctor", "--db-path", str(missing_db)])
    assert rc_ok == 0
    payload_ok = json.loads(capsys.readouterr().out)
    assert payload_ok["ready_for_serving"] is False
    assert "device" in payload_ok
    assert any("metadata_db" in item for item in payload_ok["missing_required"])

    rc_strict = main(["doctor", "--strict", "--db-path", str(missing_db)])
    assert rc_strict == 2
    payload_strict = json.loads(capsys.readouterr().out)
    assert payload_strict["ready_for_serving"] is False


def test_health_reports_device_and_cuda_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device_info("cpu")["cuda_available"] is False

    fake_pipeline = SimpleNamespace(
        device="cpu",
        augmented_embeddings=np.zeros((4, 864), dtype=np.float32),
        keyframe_map={},
        video_durations={"v1": 5.0},
        enable_videoprism=False,
        videoprism_retriever=None,
        ltr_ranker=MagicMock(is_loaded=True),
    )
    client = TestClient(create_app(pipeline=fake_pipeline, snapshot_id="health_test"))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "unified"
    assert body["device"] == "cpu"
    assert body["cuda_available"] is False
    assert body["ltr_loaded"] is True
