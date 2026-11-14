"""Visual-only diagnostic: query text is independent of qrels and sampled frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from huggingface_hub import hf_hub_download

MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
WEIGHTS = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def features(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    return out[0] if isinstance(out, (tuple, list)) else out


def main():
    matches = list(Path("/kaggle/input").rglob("bundle_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one pilot bundle manifest")
    bundle = matches[0].parent
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    dense_configs = list(Path("/kaggle/input").rglob("visual_dense_config.json"))
    if len(dense_configs) > 1:
        raise ValueError("Ambiguous dense-run config")
    config_path = dense_configs[0] if dense_configs else bundle / "visual_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample_interval = float(config.get("sample_interval_seconds", 10.0))
    if sample_interval <= 0:
        raise ValueError("Invalid sample interval")
    if sha256(Path(__file__)) != config["worker_sha256"]:
        raise ValueError("Worker checksum mismatch")
    if sha256(bundle / "inference_queries.json") != config["queries_sha256"]:
        raise ValueError("Query checksum mismatch")
    if sha256(matches[0]) != config["manifest_sha256"]:
        raise ValueError("Bundle manifest checksum mismatch")
    if not manifest.get("media_included") or len(manifest.get("assets", [])) != 2:
        raise ValueError("Expected two verified pilot videos")
    queries = json.loads((bundle / "inference_queries.json").read_text(encoding="utf-8"))
    if not queries or any(set(q) != {"query_id", "text"} for q in queries):
        raise ValueError("Inference queries must contain only query_id and text")
    out = Path("/kaggle/working/videntra_visual_pilot")
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU is unavailable; no silent CPU fallback for this run")
    weights = Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
    if sha256(weights) != WEIGHTS:
        raise ValueError("Unexpected model weights")
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    vectors, coordinates, pending = [], [], []

    def flush():
        if not pending:
            return
        inputs = processor(images=pending, return_tensors="pt").to(device)
        with torch.inference_mode():
            x = features(model.get_image_features(**inputs)).float()
            x = torch.nn.functional.normalize(x, dim=-1)
        vectors.extend(x.cpu().numpy())
        pending.clear()

    for asset in manifest["assets"]:
        path = bundle / asset["name"]
        if not path.is_file():
            path = bundle / "videos" / asset["name"]
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise ValueError("Media hash mismatch")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            first, previous, next_sec = None, None, 0.0
            for ordinal, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Missing PTS")
                seconds = float(frame.pts * frame.time_base)
                if previous is not None and seconds < previous:
                    raise ValueError("Backward PTS")
                if first is None:
                    first = seconds
                previous = seconds
                relative = seconds - first
                if relative + 1e-9 < next_sec:
                    continue
                coordinates.append({"video_id": asset["video_id"], "ordinal": ordinal,
                                    "pts": frame.pts, "time_base_num": frame.time_base.numerator,
                                    "time_base_den": frame.time_base.denominator,
                                    "clip_seconds": relative})
                pending.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                next_sec = relative + sample_interval
                if len(pending) >= 16:
                    flush()
        flush()
    query_vectors = []
    for start in range(0, len(queries), 16):
        texts = [q["text"] for q in queries[start:start + 16]]
        prompts = texts + [f"a video frame showing {text}" for text in texts]
        inputs = tokenizer(prompts, padding="max_length",
                           truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.inference_mode():
            x = torch.nn.functional.normalize(features(model.get_text_features(**inputs)).float(), dim=-1)
            x = torch.nn.functional.normalize(0.75 * x[:len(texts)] + 0.25 * x[len(texts):], dim=-1)
        query_vectors.extend(x.cpu().numpy())
    image_matrix, query_matrix = np.asarray(vectors, dtype=np.float32), np.asarray(query_vectors, dtype=np.float32)
    if image_matrix.shape != (len(coordinates), 768) or query_matrix.shape != (len(queries), 768):
        raise ValueError("Embedding shape mismatch")
    if not np.isfinite(image_matrix).all() or not np.isfinite(query_matrix).all():
        raise ValueError("Non-finite embeddings")
    np.save(out / "frames.npy", image_matrix)
    np.save(out / "queries.npy", query_matrix)
    (out / "coordinates.json").write_text(json.dumps(coordinates), encoding="utf-8")
    (out / "query_ids.json").write_text(json.dumps([q["query_id"] for q in queries]), encoding="utf-8")
    artifact_hashes = {name: sha256(out / name) for name in ("frames.npy", "queries.npy", "coordinates.json", "query_ids.json")}
    (out / "manifest.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "weights_sha256": WEIGHTS,
        "device": device, "gpu": torch.cuda.get_device_name(0), "sample_interval_seconds": sample_interval,
        "frame_count": len(coordinates), "query_count": len(queries),
        "torch_version": torch.__version__, "av_version": av.__version__,
        "input_manifest_sha256": sha256(matches[0]), "query_input_sha256": sha256(bundle / "inference_queries.json"),
        "worker_sha256": sha256(Path(__file__)),
        "output_sha256": artifact_hashes,
        "limitations": ["Visual-only; not full A5 or relation verifier.", "Queries are model-generated; no human gold."]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
