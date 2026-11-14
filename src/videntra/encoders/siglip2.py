"""SigLIP 2 vision-language encoder adapter with batching, L2 normalization, and mock fallback."""
from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SigLIP2Config:
    model_path: str | Path | None = None
    model_revision: str = "siglip2_base_patch16_224"
    device: str = "cpu"
    dtype: str = "float32"
    embedding_dim: int = 768
    max_text_length: int = 64
    batch_size: int = 32

    def resolve_model_path(self) -> Path:
        if self.model_path is not None:
            p = Path(self.model_path)
            if p.exists():
                return p
        # Check environment variables or standard relative location
        env_dir = os.environ.get("VIDENTRA_MODEL_DIR")
        env_root = os.environ.get("VIDENTRA_MODEL_ROOT")
        candidates: list[Path] = []
        if env_dir:
            candidates.append(Path(env_dir))
        if env_root:
            candidates.append(Path(env_root) / "siglip2")
        candidates.append(Path("models/siglip2"))
        for c in candidates:
            if c.exists() and (c / "model.safetensors").exists():
                return c
        raise FileNotFoundError("Cannot locate SigLIP 2 model weights directory")


class SigLIP2Encoder:
    """SigLIP 2 Base-224 encoder producing 768-dimensional L2-normalized embeddings."""

    def __init__(
        self,
        config: SigLIP2Config | None = None,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ):
        if config is not None:
            self.config = config
        else:
            self.config = SigLIP2Config(model_path=model_path, device=device)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModel

        model_dir = self.config.resolve_model_path()
        self._processor = AutoProcessor.from_pretrained(str(model_dir))
        model = AutoModel.from_pretrained(str(model_dir))

        device = torch.device(self.config.device)
        model = model.to(device)
        if self.config.dtype == "float16" and self.config.device != "cpu":
            model = model.half()
        model.eval()
        self._model = model

    def encode_images(self, images: Sequence[np.ndarray | Image.Image]) -> np.ndarray:
        """Encode a sequence of RGB images into normalized 768-dim float32 embeddings."""
        if not images:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        self._ensure_loaded()
        import torch

        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        batch_size = self.config.batch_size
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i : i + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                out = self._model.get_image_features(**inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
                # L2 normalization
                norm_emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                all_embeddings.append(norm_emb.cpu().to(torch.float32).numpy())

        res = np.concatenate(all_embeddings, axis=0)
        if not np.all(np.isfinite(res)):
            raise ValueError("Computed image embeddings contain non-finite values (NaN or Inf)")
        return res

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of text queries into normalized 768-dim float32 embeddings."""
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        self._ensure_loaded()
        import torch

        batch_size = self.config.batch_size
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                inputs = self._processor(
                    text=batch,
                    padding="max_length",
                    max_length=self.config.max_text_length,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                out = self._model.get_text_features(**inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
                # L2 normalization
                norm_emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                all_embeddings.append(norm_emb.cpu().to(torch.float32).numpy())

        res = np.concatenate(all_embeddings, axis=0)
        if not np.all(np.isfinite(res)):
            raise ValueError("Computed text embeddings contain non-finite values (NaN or Inf)")
        return res


class MockSigLIP2Encoder:
    """Mock encoder generating deterministic unit-normalized 768-dim vectors for tests."""

    def __init__(self, config: SigLIP2Config | None = None):
        self.config = config or SigLIP2Config()
        self.embedding_dim = self.config.embedding_dim

    def encode_images(self, images: Sequence[np.ndarray | Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = []
        for img in images:
            if isinstance(img, np.ndarray):
                seed_val = int(np.sum(img)) % (2**31 - 1)
            else:
                seed_val = hash(img.size) % (2**31 - 1)
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.embedding_dim).astype(np.float32)
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm)
        return np.vstack(vectors)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = []
        for txt in texts:
            seed_bytes = hashlib.sha256(txt.encode("utf-8")).digest()[:4]
            seed_val = int.from_bytes(seed_bytes, "big") % (2**31 - 1)
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.embedding_dim).astype(np.float32)
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm)
        return np.vstack(vectors)
