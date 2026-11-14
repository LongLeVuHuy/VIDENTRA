"""Kaggle GPU Worker for VIDENTRA SigLIP 2 visual & query embedding inference."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    start_time = time.time()
    print("=== VIDENTRA Kaggle GPU Embedding Worker ===")

    # 1. Detect environment and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")
    if torch.cuda.is_available():
        print(f"CUDA VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 2. Locate input paths
    # Handle Kaggle input dataset paths or local test
    kaggle_input_dir = Path("/kaggle/input")
    input_root = None
    if kaggle_input_dir.exists():
        # Search for input dataset containing keyframes_manifest.jsonl
        candidates = list(kaggle_input_dir.rglob("keyframes_manifest.jsonl"))
        if candidates:
            input_root = candidates[0].parent
            print(f"Located input directory in Kaggle: {input_root}")
    
    if input_root is None:
        input_root = Path("work/kaggle/inputs")
        print(f"Fallback to working input directory: {input_root}")

    output_dir = Path("/kaggle/working/output" if Path("/kaggle/working").exists() else "work/kaggle/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Model and Processor
    model_id = "google/siglip2-base-patch16-224"
    print(f"Loading model & processor: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # 4. Process Keyframes
    kf_manifest = input_root / "keyframes_manifest.jsonl"
    images_dir = input_root / "images"

    keyframe_items = []
    with open(kf_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                keyframe_items.append(json.loads(line.strip()))

    print(f"Loaded {len(keyframe_items)} keyframe records. Encoding images...")

    batch_size = 32 if torch.cuda.is_available() else 8
    frame_embeddings_list = []
    frame_ids = []

    for i in range(0, len(keyframe_items), batch_size):
        batch_items = keyframe_items[i : i + batch_size]
        pil_images = []
        valid_batch_ids = []

        for item in batch_items:
            img_p = images_dir / item["image_file"]
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                    pil_images.append(img)
                    valid_batch_ids.append(item["sampled_frame_id"])
                except Exception as e:
                    print(f"Warning: Failed to open image {img_p}: {e}")

        if not pil_images:
            continue

        with torch.no_grad():
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            batch_vecs = feats.cpu().numpy().astype(np.float32)
            frame_embeddings_list.append(batch_vecs)
            frame_ids.extend(valid_batch_ids)

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(keyframe_items))}/{len(keyframe_items)} images...")

    if frame_embeddings_list:
        all_frame_embeddings = np.concatenate(frame_embeddings_list, axis=0)
    else:
        all_frame_embeddings = np.zeros((0, 768), dtype=np.float32)

    print(f"Frame embeddings generated: shape={all_frame_embeddings.shape}, dtype={all_frame_embeddings.dtype}")

    # 5. Process Benchmark Queries
    queries_file = input_root / "benchmark_queries.jsonl"
    query_items = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                query_items.append(json.loads(line.strip()))

    print(f"Encoding {len(query_items)} benchmark queries...")
    query_texts = [q["query"] for q in query_items]
    query_ids = [q["qid"] for q in query_items]

    with torch.no_grad():
        q_inputs = processor(text=query_texts, return_tensors="pt", padding=True).to(device)
        q_feats = model.get_text_features(**q_inputs)
        q_feats = q_feats / q_feats.norm(dim=-1, keepdim=True)
        all_query_embeddings = q_feats.cpu().numpy().astype(np.float32)

    print(f"Query embeddings generated: shape={all_query_embeddings.shape}, dtype={all_query_embeddings.dtype}")

    # 6. Validate Embeddings
    assert not np.isnan(all_frame_embeddings).any(), "NaN found in frame embeddings!"
    assert not np.isinf(all_frame_embeddings).any(), "Inf found in frame embeddings!"
    assert not np.isnan(all_query_embeddings).any(), "NaN found in query embeddings!"
    assert not np.isinf(all_query_embeddings).any(), "Inf found in query embeddings!"

    # Verify unit norm
    if len(all_frame_embeddings) > 0:
        f_norms = np.linalg.norm(all_frame_embeddings, axis=-1)
        assert np.allclose(f_norms, 1.0, atol=1e-3), f"Frame norms not unit length: min={f_norms.min()}, max={f_norms.max()}"

    q_norms = np.linalg.norm(all_query_embeddings, axis=-1)
    assert np.allclose(q_norms, 1.0, atol=1e-3), f"Query norms not unit length: min={q_norms.min()}, max={q_norms.max()}"

    # 7. Save Artifacts
    frame_emb_path = output_dir / "frame_embeddings.npy"
    query_emb_path = output_dir / "query_embeddings.npy"
    frame_ids_path = output_dir / "frame_ids.json"
    query_ids_path = output_dir / "query_ids.json"
    meta_path = output_dir / "run_metadata.json"

    np.save(frame_emb_path, all_frame_embeddings)
    np.save(query_emb_path, all_query_embeddings)

    with open(frame_ids_path, "w", encoding="utf-8") as f:
        json.dump(frame_ids, f, indent=2)

    with open(query_ids_path, "w", encoding="utf-8") as f:
        json.dump(query_ids, f, indent=2)

    elapsed_sec = time.time() - start_time
    metadata = {
        "model_id": model_id,
        "device": str(device),
        "gpu_name": gpu_name,
        "elapsed_seconds": round(elapsed_sec, 2),
        "num_frames": len(frame_ids),
        "frame_embedding_shape": list(all_frame_embeddings.shape),
        "frame_embeddings_sha256": sha256_file(frame_emb_path),
        "num_queries": len(query_ids),
        "query_embedding_shape": list(all_query_embeddings.shape),
        "query_embeddings_sha256": sha256_file(query_emb_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"=== Worker Finished in {elapsed_sec:.2f}s ===")
    print(f"Outputs written to {output_dir}")

if __name__ == "__main__":
    main()
