"""VIDENTRA Multi-Modal Offline GPU Worker for Kaggle (Chunked Streaming Engine).

Executes full dataset offline processing on Tesla T4 GPU with bounded memory:
1. Adaptive Keyframe Sampling (4x4 Spatial Grid Histogram, Laplacian Sharpness, Blank Frame Guard, PTS timestamps).
2. SigLIP 2 ViT Base FP16 Image Embeddings (Batch Size = 64).
3. YOLOv8 Object Detection (Batch Inference = 16 images).
4. Pinned Faster-Whisper small ASR with VAD filter.
5. Pinned RapidOCR on keyframes (zero synthetic placeholders).
6. Memory-bounded chunked execution and serialization of final artifacts.
"""
from __future__ import annotations

from collections import deque
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from importlib import metadata as importlib_metadata
import sys
import time
from typing import Any

import numpy as np

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_REVISION = os.environ.get("VIDENTRA_E5_REVISION", "c007d7ef6fd86656326059b28395a7a03a7c5846")
E5_EMBEDDING_DIM = 384
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE2_REVISION = os.environ.get("VIDENTRA_FLORENCE2_REVISION", "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac")
ASR_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
A5_MODEL_FILES = {
    "whisper-small/config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
    "whisper-small/model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "whisper-small/tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "whisper-small/vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
    "ocr/ch_PP-OCRv4_det_infer.onnx": "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
    "ocr/ch_PP-OCRv4_rec_infer.onnx": "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
    "ocr/ch_ppocr_mobile_v2.0_cls_infer.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    "yolo/yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
}
A5_PACKAGE_VERSIONS = {"faster-whisper": "1.2.1", "rapidocr-onnxruntime": "1.4.4",
                       "ultralytics": "8.4.202", "ctranslate2": "4.8.5", "onnxruntime": "1.31.1",
                       "av": "16.0.1", "transformers": "5.22.1"}
VAD_SHA256 = "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
OCR_CONFIG_SHA256 = "bf94a1da4cba828e67b1d61e27cee14d9e7da27c9f272e04048a17e41ae97332"


def verify_weights(path: Path, expected: str = SIGLIP_WEIGHTS_SHA256) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 checksum mismatch: {path.name}")
    return actual


def verify_a5_model_bundle(root: Path) -> dict[str, str]:
    """Check exact known model files, without downloading or trusting filenames alone."""
    # Faster-Whisper silently consumes this optional file; it is absent at the pinned revision.
    if (root / "whisper-small/preprocessor_config.json").exists():
        raise ValueError("Unpinned Whisper preprocessor_config.json is not permitted")
    result = {}
    for relative, expected in A5_MODEL_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned A5 model file: {relative}")
        result[relative] = verify_weights(path, expected)
    return result


def verify_a5_runtime() -> dict[str, str]:
    """Fail closed on missing or mismatched direct model-runtime packages."""
    versions = {}
    for name, expected in A5_PACKAGE_VERSIONS.items():
        try:
            actual = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing pinned runtime package: {name}") from exc
        if actual != expected:
            raise RuntimeError(f"Runtime mismatch for {name}: expected {expected}, got {actual}")
        versions[name] = actual
    return versions


def asr_segment_record(video_id: str, index: int, segment: Any, info: Any) -> dict[str, Any]:
    """Serialize an ASR observation using the model fixed by the A5 bundle."""
    return {
        "segment_id": f"{video_id}__asr_{index}",
        "video_id": video_id,
        "model_id": "faster_whisper_small",
        "start_ms": int(segment.start * 1000),
        "end_ms": int(segment.end * 1000),
        "raw_text": segment.text.strip(),
        "language": getattr(info, "language", "en"),
        "confidence": round(float(segment.avg_logprob), 4),
    }


def frame_time_seconds(frame: Any, stream: Any, origin_pts: int = 0) -> float:
    """Fail closed when PTS or its mapping is unavailable or inconsistent."""
    if frame.pts is None or stream.time_base is None:
        raise ValueError("Missing frame PTS or stream time_base")
    frame_time_base = getattr(frame, "time_base", None)
    if frame_time_base is not None and frame_time_base != stream.time_base:
        raise ValueError("Frame and stream time_base differ; explicit mapping required")
    return float((frame.pts - origin_pts) * stream.time_base)


def ensure_output_files_absent(out_dir: Path) -> None:
    names = ("frame_embeddings.npy", "spatial_patches_fp16.npy", "faiss_text_e5_384.bin",
             "e5_text_manifest.json", "videos.json", "keyframe_manifest.json", "frame_objects.json",
             "ocr_observations.json", "asr_segments.json", "summary.json")
    existing = [name for name in names if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Output artifacts already exist: {existing}")


def load_bundle_videos(manifest_path: Path) -> list[str]:
    """Read only explicitly listed assets, never scan unrelated mounted datasets."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED_NOT_UPLOADED" or manifest.get("media_included") is not True:
        raise ValueError("A prepared bundle containing verified media is required")
    entries = manifest.get("assets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Bundle has no assets")
    seen: set[str] = set()
    paths: list[str] = []
    for entry in entries:
        name, expected = entry.get("name"), entry.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name or name in seen or
                Path(name).suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov"}):
            raise ValueError("Invalid or duplicate bundle media name")
        candidates = [path for path in (manifest_path.parent / "videos" / name,
                                        manifest_path.parent / name) if path.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Bundle media missing or ambiguous: {name}")
        path = candidates[0]
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"Bundle media missing or wrong size: {name}")
        verify_weights(path, expected)
        seen.add(name)
        paths.append(str(path))
    return paths

# Force UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def extract_tensor_features(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def extract_spatial_patches_3x3(last_hidden_state: Any) -> np.ndarray:
    """Pool SigLIP 2 last_hidden_state (B, H*W, D) into 3x3 L2-normalized fp16 patch tokens.

    Returns:
        np.ndarray of shape (B, 9, D) with dtype float16.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(last_hidden_state, torch.Tensor):
        hidden = torch.as_tensor(last_hidden_state)
    else:
        hidden = last_hidden_state

    if hidden.ndim != 3:
        raise ValueError(f"Expected 3D last_hidden_state (B, N, D), got {tuple(hidden.shape)}")

    bsz, n_tokens, dim = hidden.shape
    side = int(math.isqrt(n_tokens))
    if side * side != n_tokens:
        # Handle ViT architectures that prepend a CLS token
        side_minus = int(math.isqrt(n_tokens - 1))
        if side_minus * side_minus == n_tokens - 1:
            hidden = hidden[:, 1:, :]
            side = side_minus
        else:
            raise ValueError(f"Cannot reshape {n_tokens} patch tokens into a square grid")

    grid = hidden.float().view(bsz, side, side, dim).permute(0, 3, 1, 2)  # (B, D, H, W)
    pooled = F.adaptive_avg_pool2d(grid, output_size=(3, 3))  # (B, D, 3, 3)
    patches = pooled.permute(0, 2, 3, 1).reshape(bsz, 9, dim)  # (B, 9, D)
    norms = patches.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    patches = patches / norms
    return patches.detach().cpu().numpy().astype(np.float16)


def encode_e5_passages(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode ASR/OCR passages into L2-normalized 384d float32 embeddings for E5."""
    import torch

    if not texts:
        return np.zeros((0, E5_EMBEDDING_DIM), dtype=np.float32)

    prefixed = [t if t.startswith("passage: ") else f"passage: {t}" for t in texts]
    chunks: list[np.ndarray] = []

    for st in range(0, len(prefixed), batch_size):
        batch = prefixed[st : st + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.vstack(chunks)


def build_e5_text_faiss_index(
    asr_segments: list[dict[str, Any]],
    ocr_observations: list[dict[str, Any]],
    out_dir: Path,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    model_id: str = E5_MODEL_ID,
    revision: str = E5_REVISION,
) -> dict[str, Any]:
    """Build and persist faiss_text_e5_384.bin + e5_text_manifest.json for ASR & OCR."""
    import faiss

    entries: list[dict[str, Any]] = []
    texts: list[str] = []

    for seg in asr_segments:
        raw_txt = str(seg.get("raw_text", "")).strip()
        if not raw_txt:
            continue
        entries.append({
            "faiss_id": len(entries),
            "source_type": "asr",
            "item_id": seg.get("segment_id", ""),
            "video_id": seg.get("video_id", ""),
            "start_ms": int(seg.get("start_ms", 0)),
            "end_ms": int(seg.get("end_ms", 0)),
            "raw_text": raw_txt,
        })
        texts.append(raw_txt)

    for obs in ocr_observations:
        raw_txt = str(obs.get("raw_text", "")).strip()
        if not raw_txt:
            continue
        ts_ms = int(obs.get("timestamp_ms", 0))
        entries.append({
            "faiss_id": len(entries),
            "source_type": "ocr",
            "item_id": obs.get("observation_id", ""),
            "video_id": obs.get("video_id", ""),
            "start_ms": ts_ms,
            "end_ms": ts_ms,
            "raw_text": raw_txt,
        })
        texts.append(raw_txt)

    index = faiss.IndexFlatIP(E5_EMBEDDING_DIM)
    if texts:
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModel.from_pretrained(model_id, revision=revision).to(device)
            model.eval()
        vectors = encode_e5_passages(texts, tokenizer=tokenizer, model=model, device=device)
        if vectors.shape[1] != E5_EMBEDDING_DIM:
            raise ValueError(f"Expected {E5_EMBEDDING_DIM}d E5 vectors, got {vectors.shape[1]}")
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "faiss_text_e5_384.bin"
    manifest_path = out_dir / "e5_text_manifest.json"
    faiss.write_index(index, str(faiss_path))
    manifest_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "index_path": str(faiss_path),
        "manifest_path": str(manifest_path),
        "num_entries": len(entries),
        "ntotal": int(index.ntotal),
    }


def extract_florence2_keyframe_captions(
    keyframe_records: list[dict[str, Any]],
    out_dir: Path | None = None,
    *,
    processor: Any = None,
    model: Any = None,
    device: str = "cuda",
    batch_size: int = 16,
    task_prompt: str = "<DETAILED_CAPTION>",
    max_new_tokens: int = 96,
    model_id: str = FLORENCE2_MODEL_ID,
    revision: str = FLORENCE2_REVISION,
) -> list[dict[str, Any]]:
    """PQ-VCA Mode B: Generate detailed keyframe captions using Florence-2-base on Kaggle T4 GPU.

    Each record in ``keyframe_records`` should provide ``video_id``, ``timestamp_ms``,
    optional ``faiss_id``, and either ``image`` (PIL/ndarray) or ``keyframe_path``.
    """
    if not keyframe_records:
        return []

    import torch
    from PIL import Image

    if processor is None or model is None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model.eval()

    captions: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))

    for start in range(0, len(keyframe_records), bs):
        batch = keyframe_records[start : start + bs]
        images: list[Any] = []
        valid_recs: list[tuple[int, dict[str, Any]]] = []

        for offset, rec in enumerate(batch):
            img = rec.get("image")
            if img is None and rec.get("keyframe_path"):
                img_path = Path(str(rec["keyframe_path"]))
                if img_path.is_file():
                    img = Image.open(img_path).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8)).convert("RGB")

            if img is not None:
                images.append(img)
                valid_recs.append((start + offset, rec))

        if not images:
            continue

        prompts = [task_prompt] * len(images)
        inputs = processor(text=prompts, images=images, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                num_beams=1,
                do_sample=False,
            )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for (global_idx, rec), raw_cap in zip(valid_recs, decoded):
            cap_text = str(raw_cap).replace(task_prompt, "").strip()
            captions.append({
                "faiss_id": int(rec.get("faiss_id", global_idx)),
                "video_id": str(rec["video_id"]),
                "timestamp_ms": int(rec["timestamp_ms"]),
                "caption_text": cap_text,
            })

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "keyframe_captions.json"
        out_file.write_text(
            json.dumps(captions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return captions


def compute_grid_hist_512(img_rgb: np.ndarray) -> np.ndarray:
    """Computes normalized 4x4 spatial grid histogram with 512-bin 3-bit color quantization."""
    h, w = img_rgb.shape[:2]
    # Resize to 128x128 for lightning-fast computation
    if h != 128 or w != 128:
        import cv2
        img_rgb = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)

    r = img_rgb[:, :, 0].astype(np.uint16) >> 5
    g = img_rgb[:, :, 1].astype(np.uint16) >> 5
    b = img_rgb[:, :, 2].astype(np.uint16) >> 5
    color_idx = (r << 6) | (g << 3) | b  # [0, 511]

    grid_h = 32
    grid_w = 32
    hist_list = []
    for gy in range(4):
        for gx in range(4):
            patch = color_idx[gy * grid_h : (gy + 1) * grid_h, gx * grid_w : (gx + 1) * grid_w]
            h_patch = np.bincount(patch.ravel(), minlength=512).astype(np.float32)
            total = h_patch.sum()
            if total > 0:
                h_patch /= total
            hist_list.append(h_patch)

    return np.mean(hist_list, axis=0)


def compute_grid_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(1.0 - np.minimum(h1, h2).sum())


def compute_sharpness_score(img_rgb: np.ndarray) -> float:
    """Computes Laplacian variance as an ultra-fast sharpness metric."""
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] > 180:
        gray = cv2.resize(gray, (240, 135), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def is_blank_or_fade_frame(img_rgb: np.ndarray, min_luma_std: float = 8.0) -> bool:
    """Detects black, solid color, or extreme fade-in/out frames."""
    luma = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    return float(np.std(luma)) < min_luma_std


def compute_phash_64(img_rgb: np.ndarray) -> int:
    """Computes 64-bit perceptual hash using 32x32 DCT."""
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    bit_arr = (dct_low > med).flatten()
    val = 0
    for b in bit_arr:
        val = (val << 1) | int(b)
    return val


def phash_dist(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def main():
    start_time = time.time()
    out_dir = Path("/kaggle/working")
    ensure_output_files_absent(out_dir)
    model_root_value = os.environ.get("VIDENTRA_A5_MODEL_ROOT")
    if not model_root_value:
        raise RuntimeError("VIDENTRA_A5_MODEL_ROOT must identify the offline pinned model bundle")
    model_root = Path(model_root_value)
    model_hashes = verify_a5_model_bundle(model_root)
    runtime_versions = verify_a5_runtime()
    log("=" * 75)
    log("STARTING VIDENTRA MULTIMODAL OFFLINE GPU WORKER (KAGGLE)")
    log("=" * 75)

    specified = os.environ.get("VIDENTRA_KAGGLE_BUNDLE")
    manifests = ([Path(specified) / "bundle_manifest.json"] if specified else
                 list(Path("/kaggle/input").rglob("bundle_manifest.json")))
    if len(manifests) != 1:
        raise ValueError("Exactly one Kaggle bundle is required; set VIDENTRA_KAGGLE_BUNDLE")
    video_files = load_bundle_videos(manifests[0])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Inference Device: {device}")
    if device == "cuda":
        log(f"GPU Name: {torch.cuda.get_device_name(0)}")
        log(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    log("Verified pinned direct model runtimes and offline model files.")

    import av
    from PIL import Image
    from transformers import AutoProcessor, AutoModel
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    # Initialize only verified local weights. A missing or broken branch aborts the run.
    import faster_whisper
    from faster_whisper import WhisperModel
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime import main as rapidocr_main

    verify_weights(Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx", VAD_SHA256)
    verify_weights(Path(rapidocr_main.DEFAULT_CFG_PATH), OCR_CONFIG_SHA256)
    whisper_compute = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(str(model_root / "whisper-small"), device=device,
                                 compute_type=whisper_compute, local_files_only=True)
    has_whisper = True
    ocr_engine = RapidOCR(config_path=str(rapidocr_main.DEFAULT_CFG_PATH),
                          det_model_path=str(model_root / "ocr/ch_PP-OCRv4_det_infer.onnx"),
                          rec_model_path=str(model_root / "ocr/ch_PP-OCRv4_rec_infer.onnx"),
                          cls_model_path=str(model_root / "ocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"))
    has_ocr = True

    # 3. Load SigLIP 2 on GPU
    siglip_model_id = SIGLIP_MODEL_ID
    log(f"Loading SigLIP 2 ({siglip_model_id})...")
    weights_path = Path(hf_hub_download(siglip_model_id, "model.safetensors", revision=SIGLIP_REVISION))
    weights_hash = verify_weights(weights_path)
    processor = AutoProcessor.from_pretrained(siglip_model_id, revision=SIGLIP_REVISION)
    siglip_model = AutoModel.from_pretrained(
        siglip_model_id,
        revision=SIGLIP_REVISION,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    siglip_model.eval()

    # 4. Load YOLOv8 Detector
    log("Loading YOLOv8 Detector (yolov8n.pt)...")
    yolo = YOLO(str(model_root / "yolo/yolov8n.pt"))

    # 5. The input universe is fixed by the verified bundle, not by mounted files.
    log("\n--- VERIFIED BUNDLE VIDEOS ---")
    total_all_videos = len(video_files)
    log(f"Discovered {total_all_videos} video files across all attached datasets!")

    global_videos: list[dict] = []
    global_keyframes: list[dict] = []
    global_embeddings_list: list[np.ndarray] = []
    global_spatial_patches_list: list[np.ndarray] = []
    global_objects: list[dict] = []
    global_ocr_obs: list[dict] = []
    global_asr_segs: list[dict] = []
    video_errors: list[dict[str, str]] = []
    asr_error_count = 0
    ocr_error_count = 0

    chunk_size = 50
    total_chunks = (total_all_videos + chunk_size - 1) // chunk_size if total_all_videos > 0 else 1
    log(f"Executing in {total_chunks} chunk(s) (chunk_size={chunk_size}).\n")

    global_kf_counter = 0

    for chunk_idx in range(total_chunks):
        c_st = chunk_idx * chunk_size
        c_end = min(total_all_videos, c_st + chunk_size)
        chunk_videos = video_files[c_st:c_end]
        log(f">>> [CHUNK {chunk_idx + 1}/{total_chunks}] Processing videos {c_st + 1} to {c_end} of {total_all_videos}...")

        chunk_keyframes: list[dict] = []
        chunk_images: list[Image.Image] = []
        chunk_img_nps: list[np.ndarray] = []

        # A. Decode & Adaptive Keyframe Sampling for this chunk
        for v_path in chunk_videos:
            vid_name = Path(v_path).stem
            container = None
            try:
                container = av.open(v_path)
                v_stream = container.streams.video[0] if container.streams.video else None
                a_stream = container.streams.audio[0] if container.streams.audio else None
                fps = float(v_stream.average_rate or 25.0) if v_stream else 25.0
                dur_sec = float(container.duration or 0) / 1e6 if container.duration else 0.0
                if v_stream is None or v_stream.time_base is None:
                    raise ValueError("Missing video stream or time_base")
                tb_num = v_stream.time_base.numerator
                tb_den = v_stream.time_base.denominator

                v_meta = {
                    "video_id": vid_name,
                    "file_path": v_path,
                    "duration_ms": int(dur_sec * 1000),
                    "fps": fps,
                    "time_base_num": tb_num,
                    "time_base_den": tb_den,
                    "width": v_stream.width if v_stream else 1280,
                    "height": v_stream.height if v_stream else 720,
                    "has_audio": 1 if a_stream is not None else 0,
                }

                # Duration-aware sampling thresholds
                if dur_sec <= 16.0:
                    min_gap_sec = 1.0
                    max_static_gap_sec = 4.0
                elif dur_sec <= 60.0:
                    min_gap_sec = 1.5
                    max_static_gap_sec = 4.0
                else:
                    min_gap_sec = 2.0
                    max_static_gap_sec = 4.5

                last_selected_ts = -999.0
                last_hist = None
                recent_phashes: deque[int] = deque(maxlen=5)
                pending_first_valid = True
                selected_samples: list[dict] = []

                # Small candidate buffer for local sharpness maximization
                cand_buffer: list[dict] = []

                def flush_candidate_buffer(force: bool = False):
                    nonlocal last_selected_ts, last_hist
                    if not cand_buffer:
                        return
                    # Pick frame with maximum sharpness score
                    best = max(cand_buffer, key=lambda c: c["sharpness"])
                    # Check pHash duplication
                    h_val = best["phash"]
                    if recent_phashes and min(phash_dist(h, h_val) for h in recent_phashes) <= 3:
                        if (best["ts_sec"] - last_selected_ts) < max_static_gap_sec and not force:
                            cand_buffer.clear()
                            return

                    selected_samples.append(best)
                    last_selected_ts = best["ts_sec"]
                    last_hist = best["hist"]
                    recent_phashes.append(h_val)
                    cand_buffer.clear()

                last_decoded_frame_record = None
                first_frame_pts = None
                previous_pts = None

                for f_idx, frame in enumerate(container.decode(video=0)):
                    img_rgb = frame.to_ndarray(format="rgb24")
                    frame_time_seconds(frame, v_stream)
                    if first_frame_pts is None:
                        first_frame_pts = int(frame.pts)
                        v_meta["time_origin_pts"] = first_frame_pts
                        v_meta["timestamp_basis"] = "clip_relative_first_decoded_frame"
                    if previous_pts is not None and frame.pts < previous_pts:
                        raise ValueError("Backward PTS requires an explicit discontinuity mapping")
                    previous_pts = frame.pts
                    ts_sec = frame_time_seconds(frame, v_stream, first_frame_pts)
                    ts_ms = int(round(ts_sec * 1000.0))

                    blank = is_blank_or_fade_frame(img_rgb)
                    sharp = compute_sharpness_score(img_rgb)
                    hist = compute_grid_hist_512(img_rgb)
                    ph = compute_phash_64(img_rgb)

                    rec = {
                        "frame_idx": f_idx,
                        "pts": int(frame.pts),
                        "ts_sec": ts_sec,
                        "ts_ms": ts_ms,
                        "img_rgb": img_rgb,
                        "sharpness": sharp,
                        "hist": hist,
                        "phash": ph,
                        "blank": blank,
                    }
                    last_decoded_frame_record = rec

                    # Handle first frame anchor
                    if f_idx == 0:
                        if not blank:
                            selected_samples.append(rec)
                            last_selected_ts = ts_sec
                            last_hist = hist
                            recent_phashes.append(ph)
                            pending_first_valid = False
                        continue

                    if pending_first_valid:
                        if not blank and ts_sec <= 1.0:
                            selected_samples.append(rec)
                            last_selected_ts = ts_sec
                            last_hist = hist
                            recent_phashes.append(ph)
                            pending_first_valid = False
                            continue
                        elif ts_sec > 1.0:
                            pending_first_valid = False

                    # Check shot cut or min_gap trigger
                    hist_dist = compute_grid_distance(last_hist, hist) if last_hist is not None else 1.0
                    gap = ts_sec - last_selected_ts

                    if (hist_dist > 0.32 and gap >= 0.5) or (gap >= min_gap_sec):
                        if not blank:
                            cand_buffer.append(rec)
                            if len(cand_buffer) >= 3:
                                flush_candidate_buffer()

                if first_frame_pts is None:
                    raise ValueError("Video stream produced no decoded frames")
                # Flush remaining candidates
                flush_candidate_buffer(force=True)

                # Ensure last frame anchor if video is long enough
                if last_decoded_frame_record is not None and len(selected_samples) > 0:
                    last_sel = selected_samples[-1]
                    if (last_decoded_frame_record["ts_sec"] - last_sel["ts_sec"]) >= 0.8:
                        if not last_decoded_frame_record["blank"]:
                            selected_samples.append(last_decoded_frame_record)

                container.close()
                global_videos.append(v_meta)

                # Add selected samples to chunk
                for k_idx, s in enumerate(selected_samples):
                    kf_id = f"{vid_name}__kf_{k_idx:05d}"
                    kf_record = {
                        "keyframe_id": kf_id,
                        "video_id": vid_name,
                        "keyframe_index": k_idx,
                        "source_frame_index": s["frame_idx"],
                        "pts": s["pts"],
                        "timestamp_ms": s["ts_ms"],
                        "global_index": global_kf_counter,
                    }
                    chunk_keyframes.append(kf_record)
                    chunk_images.append(Image.fromarray(s["img_rgb"]))
                    chunk_img_nps.append(s["img_rgb"])
                    global_kf_counter += 1

                # Real ASR transcription
                if v_meta["has_audio"] and has_whisper and whisper_model is not None:
                    try:
                        segs, info = whisper_model.transcribe(v_path, beam_size=1, vad_filter=True, temperature=0.0)
                        for s_idx, seg in enumerate(segs):
                            st_txt = seg.text.strip()
                            if st_txt:
                                global_asr_segs.append(asr_segment_record(vid_name, s_idx, seg, info))
                    except Exception as asr_err:
                        asr_error_count += 1
                        log(f"ASR error on {vid_name}: {asr_err}")

            except Exception as vid_err:
                if container is not None:
                    container.close()
                video_errors.append({"video_id": vid_name, "error": str(vid_err)})
                log(f"Error processing video {vid_name}: {vid_err}")

        num_chunk_kf = len(chunk_keyframes)
        global_keyframes.extend(chunk_keyframes)

        # B. Batch SigLIP 2 Embeddings (Batch Size = 64)
        if num_chunk_kf > 0:
            batch_sz = 64
            for b_st in range(0, num_chunk_kf, batch_sz):
                b_end = min(num_chunk_kf, b_st + batch_sz)
                batch_imgs = chunk_images[b_st:b_end]

                inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
                if device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

                with torch.no_grad():
                    if hasattr(siglip_model, "vision_model") and "pixel_values" in inputs:
                        raw_out = siglip_model.vision_model(pixel_values=inputs["pixel_values"])
                        if getattr(raw_out, "last_hidden_state", None) is not None:
                            global_spatial_patches_list.append(
                                extract_spatial_patches_3x3(raw_out.last_hidden_state)
                            )
                    elif hasattr(siglip_model, "get_image_features"):
                        raw_out = siglip_model.get_image_features(**inputs)
                    else:
                        raw_out = siglip_model(**inputs)

                    feats = extract_tensor_features(raw_out)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    global_embeddings_list.append(feats.cpu().float().numpy())

            # C. Batch YOLOv8 Detection (Batch Size = 16)
            yolo_batch_sz = 16
            for b_st in range(0, num_chunk_kf, yolo_batch_sz):
                b_end = min(num_chunk_kf, b_st + yolo_batch_sz)
                sub_imgs = chunk_img_nps[b_st:b_end]
                sub_kfs = chunk_keyframes[b_st:b_end]

                yolo_results = yolo(sub_imgs, verbose=False, device=0 if device == "cuda" else "cpu")
                for kf_rec, r in zip(sub_kfs, yolo_results):
                    orig_h, orig_w = r.orig_shape
                    for b_idx, box in enumerate(r.boxes):
                        conf = float(box.conf[0].item())
                        if conf >= 0.25:
                            xyxy = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0].item())
                            cls_name = yolo.names.get(cls_id, f"obj_{cls_id}")
                            global_objects.append({
                                "detection_id": f"{kf_rec['keyframe_id']}__det_{b_idx}",
                                "video_id": kf_rec["video_id"],
                                "keyframe_id": kf_rec["keyframe_id"],
                                "timestamp_ms": kf_rec["timestamp_ms"],
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "bbox_x1": round(xyxy[0] / orig_w, 4),
                                "bbox_y1": round(xyxy[1] / orig_h, 4),
                                "bbox_x2": round(xyxy[2] / orig_w, 4),
                                "bbox_y2": round(xyxy[3] / orig_h, 4),
                                "confidence": round(conf, 4),
                                "image_width": orig_w,
                                "image_height": orig_h,
                            })

            # D. Real OCR on sampled keyframes (even indices or keyframes with edge variance)
            if has_ocr and ocr_engine is not None:
                for kf_rec, img_np in zip(chunk_keyframes, chunk_img_nps):
                    # Sample every 2nd keyframe or initial keyframe
                    if kf_rec["keyframe_index"] % 2 != 0 and kf_rec["keyframe_index"] != 0:
                        continue
                    try:
                        h_im, w_im = img_np.shape[:2]
                        if hasattr(ocr_engine, "__call__"):
                            # RapidOCR
                            ocr_res, _ = ocr_engine(img_np)
                            if ocr_res:
                                for o_idx, item in enumerate(ocr_res):
                                    pts, ocr_txt, ocr_conf = item
                                    clean_txt = str(ocr_txt).strip()
                                    if clean_txt and float(ocr_conf) >= 0.35:
                                        xs = [p[0] for p in pts]
                                        ys = [p[1] for p in pts]
                                        global_ocr_obs.append({
                                            "observation_id": f"{kf_rec['keyframe_id']}__ocr_{o_idx}",
                                            "video_id": kf_rec["video_id"],
                                            "keyframe_id": kf_rec["keyframe_id"],
                                            "timestamp_ms": kf_rec["timestamp_ms"],
                                            "model_id": "RapidOCR",
                                            "raw_text": clean_txt,
                                            "bbox_x1": round(max(0.0, min(xs)) / w_im, 4),
                                            "bbox_y1": round(max(0.0, min(ys)) / h_im, 4),
                                            "bbox_x2": round(min(float(w_im), max(xs)) / w_im, 4),
                                            "bbox_y2": round(min(float(h_im), max(ys)) / h_im, 4),
                                            "confidence": round(float(ocr_conf), 4),
                                            "language": "en",
                                        })
                    except Exception as ocr_err:
                        ocr_error_count += 1
                        log(f"OCR error on {kf_rec['keyframe_id']}: {ocr_err}")

        # Cleanup memory after chunk
        del chunk_images
        del chunk_img_nps
        del chunk_keyframes
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        overall_pct = (c_end / total_all_videos) * 100.0 if total_all_videos > 0 else 100.0
        log(f"  --> Chunk {chunk_idx + 1} DONE: {c_end}/{total_all_videos} videos ({overall_pct:.1f}%) | Keyframes so far: {len(global_keyframes)} | Objects: {len(global_objects)} | OCR: {len(global_ocr_obs)} | ASR: {len(global_asr_segs)}")

    # 6. Save Artifacts
    log("\n--- CONCATENATING & SAVING FINAL ARTIFACTS ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    if global_embeddings_list:
        final_embeddings = np.vstack(global_embeddings_list)
    else:
        final_embeddings = np.empty((0, 768), dtype=np.float32)

    if global_spatial_patches_list:
        final_spatial_patches = np.vstack(global_spatial_patches_list)
    else:
        final_spatial_patches = np.empty((0, 9, 768), dtype=np.float16)

    np.save(str(out_dir / "frame_embeddings.npy"), final_embeddings)
    np.save(str(out_dir / "spatial_patches_fp16.npy"), final_spatial_patches)

    e5_status: dict[str, Any] = {"available": False, "num_entries": 0}
    try:
        e5_info = build_e5_text_faiss_index(
            asr_segments=global_asr_segs,
            ocr_observations=global_ocr_obs,
            out_dir=out_dir,
            device=device,
        )
        e5_status = {"available": True, **e5_info}
    except Exception as e5_err:
        log(f"E5 text index build skipped/failed: {e5_err}")
        e5_status = {"available": False, "error": str(e5_err), "num_entries": 0}
    with open(out_dir / "videos.json", "w", encoding="utf-8") as f:
        json.dump(global_videos, f, indent=2, ensure_ascii=False)
    with open(out_dir / "keyframe_manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_keyframes, f, indent=2, ensure_ascii=False)
    with open(out_dir / "frame_objects.json", "w", encoding="utf-8") as f:
        json.dump(global_objects, f, indent=2, ensure_ascii=False)
    with open(out_dir / "ocr_observations.json", "w", encoding="utf-8") as f:
        json.dump(global_ocr_obs, f, indent=2, ensure_ascii=False)
    with open(out_dir / "asr_segments.json", "w", encoding="utf-8") as f:
        json.dump(global_asr_segs, f, indent=2, ensure_ascii=False)

    summary = {
        "status": "partial" if video_errors or not has_ocr or not has_whisper or asr_error_count or ocr_error_count else "success",
        "video_errors": video_errors,
        "branch_status": {"ocr_available": has_ocr, "ocr_errors": ocr_error_count,
                          "asr_available": has_whisper, "asr_errors": asr_error_count},
        "siglip": {"model_id": SIGLIP_MODEL_ID, "revision": SIGLIP_REVISION,
                   "weights_sha256": weights_hash, "license": "apache-2.0"},
        "pinned_models": {
            "asr": {"model_id": "Systran/faster-whisper-small", "revision": ASR_REVISION,
                    "license": "MIT", "vad_sha256": VAD_SHA256},
            "ocr": {"model_id": "RapidOCR PP-OCRv4", "revision": "rapidocr-onnxruntime-1.4.4",
                    "license": "Apache-2.0", "config_sha256": OCR_CONFIG_SHA256},
            "detection": {"model_id": "YOLOv8n", "revision": "ultralytics/assets/v8.3.0",
                          "license": "AGPL-3.0"},
            "e5_text": {"model_id": E5_MODEL_ID, "revision": E5_REVISION,
                        "embedding_dim": E5_EMBEDDING_DIM, "license": "MIT"},
            "model_files_sha256": model_hashes,
        },
        "e5_text_index": e5_status,
        "runtime_versions": runtime_versions,
        "limitations": ["ASR confidence is an uncalibrated average log probability, not a probability.",
                        "Transitive dependencies, GPU drivers and runtime wheel hashes require separate verification.",
                        "PTS is not an independent decoded-ledger or origin-mapping audit."],
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_videos": len(global_videos),
        "total_keyframes": len(global_keyframes),
        "embedding_matrix_shape": list(final_embeddings.shape),
        "spatial_patches_shape": list(final_spatial_patches.shape),
        "total_objects_detected": len(global_objects),
        "total_ocr_observations": len(global_ocr_obs),
        "total_asr_segments": len(global_asr_segs),
        "completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("=" * 75)
    log(f"ALL VIDEOS IN KAGGLE PROCESSED IN {summary['elapsed_seconds']}s!")
    log(f"Summary: {summary}")
    log("=" * 75)


if __name__ == "__main__":
    main()
