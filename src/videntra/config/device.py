"""Runtime device auto-resolution, CUDA dtype helpers, and environment preflight checks."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


_CORE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("av", "av"),
    ("pillow", "PIL"),
    ("jsonschema", "jsonschema"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("faiss", "faiss"),
    ("lightgbm", "lightgbm"),
    ("scipy", "scipy"),
    ("opencv", "cv2"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("psutil", "psutil"),
)

_OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("rapidocr-onnxruntime", "rapidocr_onnxruntime"),
    ("faster-whisper", "faster_whisper"),
    ("ultralytics", "ultralytics"),
    ("llama-cpp-python", "llama_cpp"),
    ("python-dotenv", "dotenv"),
    ("pytest", "pytest"),
)


def _probe_cuda() -> tuple[bool, dict[str, Any]]:
    """Safely inspect PyTorch and CUDA availability without failing on CPU builds."""
    meta: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
    }
    try:
        import torch

        meta["torch_version"] = str(getattr(torch, "__version__", "")) or None
        meta["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_ok = bool(torch.cuda.is_available())
        meta["cuda_available"] = cuda_ok
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            meta["gpu_name"] = str(getattr(props, "name", "") or "") or None
            total_bytes = getattr(props, "total_memory", None)
            if isinstance(total_bytes, (int, float)) and total_bytes > 0:
                meta["vram_total_mb"] = round(float(total_bytes) / (1024.0 * 1024.0), 1)
            if hasattr(torch.cuda, "mem_get_info"):
                free_bytes, _ = torch.cuda.mem_get_info(0)
                meta["vram_free_mb"] = round(float(free_bytes) / (1024.0 * 1024.0), 1)
    except Exception:
        pass
    return bool(meta["cuda_available"]), meta


def resolve_device(requested: str | None = None, *, default: str = "auto") -> str:
    """Resolve runtime compute device ('cuda' or 'cpu'), supporting VIDENTRA_DEVICE='auto'."""
    raw = requested if (requested is not None and str(requested).strip()) else os.environ.get("VIDENTRA_DEVICE")
    if raw is None or not str(raw).strip():
        raw = default
    candidate = str(raw).strip()
    if candidate.lower() == "auto":
        cuda_ok, _ = _probe_cuda()
        return "cuda" if cuda_ok else "cpu"
    return candidate


def preferred_cuda_dtype(device: str | None) -> Any:
    """Return optimal PyTorch tensor dtype (FP16 on CUDA by default for 6GB VRAM, FP32 on CPU)."""
    import torch

    env_dtype = os.environ.get("VIDENTRA_DTYPE", "").strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if env_dtype in dtype_map:
        return dtype_map[env_dtype]
    return torch.float16 if str(device or "").lower().startswith("cuda") else torch.float32


def get_device_info(device: str | None = None) -> dict[str, Any]:
    """Collect GPU, VRAM, CPU, and system RAM telemetry for diagnostics and reporting."""
    resolved = resolve_device(device, default="auto")
    _, cuda_meta = _probe_cuda()
    ram_total_gb: float | None = None
    try:
        import psutil

        ram_total_gb = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)
    except Exception:
        ram_total_gb = None

    return {
        "resolved_device": resolved,
        "cuda_available": bool(cuda_meta["cuda_available"]),
        "torch_version": cuda_meta["torch_version"],
        "cuda_version": cuda_meta["cuda_version"],
        "gpu_name": cuda_meta["gpu_name"],
        "vram_total_mb": cuda_meta["vram_total_mb"],
        "vram_free_mb": cuda_meta["vram_free_mb"],
        "cpu_count": os.cpu_count(),
        "ram_total_gb": ram_total_gb,
        "platform": platform.platform(),
    }


def _check_packages(specs: tuple[tuple[str, str], ...]) -> dict[str, bool]:
    return {label: (importlib.util.find_spec(mod_name) is not None) for label, mod_name in specs}


def _path_status(path: Path, *, required: bool) -> dict[str, Any]:
    exists = path.exists()
    size_mb: float | None = None
    if exists and path.is_file():
        size_mb = round(path.stat().st_size / (1024.0 * 1024.0), 2)
    return {
        "path": str(path),
        "exists": exists,
        "required": required,
        "size_mb": size_mb,
    }


def collect_environment_report(
    db_path: Path | str | None = None,
    faiss_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    ltr_model_path: Path | str | None = None,
    asset_roots_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run a comprehensive non-destructive preflight check for serving and evaluation."""
    resolved_db = Path(db_path if db_path is not None else os.environ.get("VIDENTRA_DB_PATH", "data/processed/metadata.db"))
    resolved_faiss = Path(faiss_path if faiss_path is not None else os.environ.get("VIDENTRA_FAISS_PATH", "data/processed/faiss_index.bin"))
    resolved_model_dir = Path(model_dir if model_dir is not None else os.environ.get("VIDENTRA_MODEL_DIR", "models/siglip2"))
    resolved_ltr = Path(ltr_model_path if ltr_model_path is not None else "models/ranking/videntra_ltr_v1.json")
    resolved_roots = Path(asset_roots_path if asset_roots_path is not None else "configs/local/asset_roots.json")
    processed_dir = resolved_db.parent
    data_root = processed_dir.parent if processed_dir.name == "processed" else Path("data")
    datasets_reg_path = data_root / "datasets.json"
    registered_datasets: list[str] = []
    if datasets_reg_path.is_file():
        try:
            reg_doc = json.loads(datasets_reg_path.read_text(encoding="utf-8"))
            registered_datasets = sorted((reg_doc.get("datasets") or {}).keys())
        except (OSError, json.JSONDecodeError):
            registered_datasets = []

    device_info = get_device_info()
    core_pkgs = _check_packages(_CORE_PACKAGES)
    optional_pkgs = _check_packages(_OPTIONAL_PACKAGES)

    artifacts = {
        "metadata_db": _path_status(resolved_db, required=True),
        "faiss_index": _path_status(resolved_faiss, required=True),
        "siglip2_weights": _path_status(resolved_model_dir / "model.safetensors", required=True),
        "ltr_model": _path_status(resolved_ltr, required=True),
        "augmented_embeddings": _path_status(processed_dir / "augmented_embeddings.npy", required=False),
        "e5_text_index": _path_status(processed_dir / "faiss_text_e5_384.bin", required=False),
        "videoprism_clip_index": _path_status(processed_dir / "faiss_videoprism_clip_768.bin", required=False),
        "videoprism_subclips": _path_status(processed_dir / "videoprism_subclips_768.npy", required=False),
        "datasets_registry": _path_status(datasets_reg_path, required=False),
        "asset_roots_config": _path_status(resolved_roots, required=False),
    }

    missing_required: list[str] = []
    if sys.version_info < (3, 12):
        missing_required.append(f"python>=3.12 (found {platform.python_version()})")
    for pkg_name, ok in core_pkgs.items():
        if not ok:
            missing_required.append(f"package:{pkg_name}")
    for art_name, info in artifacts.items():
        if info["required"] and not info["exists"]:
            missing_required.append(f"artifact:{art_name} ({info['path']})")

    return {
        "python_version": platform.python_version(),
        "python_supported": sys.version_info >= (3, 12),
        "device": device_info,
        "core_packages": core_pkgs,
        "optional_packages": optional_pkgs,
        "registered_datasets": registered_datasets,
        "artifacts": artifacts,
        "missing_required": missing_required,
        "ready_for_serving": len(missing_required) == 0,
    }
