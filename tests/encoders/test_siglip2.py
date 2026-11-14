"""Unit and integration tests for SigLIP 2 encoder."""
import os
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from videntra.encoders.siglip2 import SigLIP2Config, SigLIP2Encoder, MockSigLIP2Encoder


def test_mock_siglip2_encoder():
    enc = MockSigLIP2Encoder(SigLIP2Config(embedding_dim=768))

    # Text encoding
    texts = ["a brown dog sitting on grass", "người đi xe máy trên phố"]
    text_vecs = enc.encode_texts(texts)
    assert text_vecs.shape == (2, 768)
    assert np.all(np.isfinite(text_vecs))
    norms = np.linalg.norm(text_vecs, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Image encoding
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_vecs = enc.encode_images([img])
    assert img_vecs.shape == (1, 768)
    assert np.all(np.isfinite(img_vecs))
    assert np.isclose(np.linalg.norm(img_vecs[0]), 1.0, atol=1e-4)


_siglip_path = Path(os.environ.get("VIDENTRA_MODEL_DIR", "models/siglip2"))
_local_siglip = _siglip_path / "model.safetensors"

@pytest.mark.skipif(
    not _local_siglip.exists(),
    reason="SigLIP 2 model weights not found in local models or VIDENTRA_MODEL_DIR",
)
def test_real_siglip2_encoder_inference():
    cfg = SigLIP2Config(
        model_path=str(_siglip_path),
        device="cpu",
        batch_size=2,
    )
    enc = SigLIP2Encoder(cfg)

    # Encode texts (both EN and VI)
    texts = ["A person walking in the city", "Người đi dạo trong công viên"]
    text_embs = enc.encode_texts(texts)
    assert text_embs.shape == (2, 768)
    assert np.all(np.isfinite(text_embs))
    assert np.allclose(np.linalg.norm(text_embs, axis=-1), 1.0, atol=1e-4)

    # Encode image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[60:160, 60:160] = [200, 50, 50]
    img_embs = enc.encode_images([img])
    assert img_embs.shape == (1, 768)
    assert np.all(np.isfinite(img_embs))
    assert np.isclose(np.linalg.norm(img_embs[0]), 1.0, atol=1e-4)
