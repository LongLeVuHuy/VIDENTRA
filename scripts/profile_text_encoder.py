"""Measure a warm, uncached text encoder on synthetic queries, without media."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from videntra.evaluation.provenance import sha256_file
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


def synthetic_queries() -> list[str]:
    objects = ["bicycle", "bird", "boat", "car", "cat", "dog", "flower", "house", "tree", "train"]
    objects_vi = ["xe đạp", "chim", "thuyền", "xe hơi", "mèo", "chó", "hoa", "nhà", "cây", "tàu"]
    english = ["a photo of {}", "a short clip showing {}", "a close view of {}", "the scene contains {}", "look for {}"]
    vietnamese = ["hình ảnh {}", "đoạn phim có {}", "cảnh cận của {}", "trong cảnh có {}", "tìm hình {}"]
    return [template.format(obj) for templates, items in ((english, objects), (vietnamese, objects_vi))
            for template in templates for obj in items]


def run(model_dir: Path, output: Path, threads: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if threads < 1:
        raise ValueError("threads must be positive")
    files = [model_dir / name for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")]
    hashes = {p.name: sha256_file(p) for p in files}
    torch.set_num_threads(threads)
    pipeline = UnifiedMultimodalSearchPipeline(model_dir=model_dir, device="cpu", lazy_load=True,
                                               use_disk_cache=False, memory_cache_size=0)
    queries = synthetic_queries()
    pipeline.encode_texts(["warm up text encoder"])
    elapsed = []
    for query in queries:
        start = time.perf_counter()
        pipeline.encode_texts([query])
        elapsed.append((time.perf_counter() - start) * 1000)
    pipeline.close()
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        rss = info.rss / 1048576
        peak = getattr(info, "peak_wset", None)
        peak = peak / 1048576 if peak is not None else None
        memory_source = "psutil.Process.memory_info"
    except (ImportError, OSError):
        rss = peak = None
        memory_source = None
    result = {"profile": "model_warm_single_prompt_text_encoder_only", "device": "cpu",
              "query_set": "synthetic_50_en_50_vi", "query_sha256": hashlib.sha256(
                  json.dumps(queries, ensure_ascii=False).encode("utf-8")).hexdigest(),
              "n": len(elapsed), "repeats": 1, "concurrency": 1, "torch_threads": threads,
              "p50_ms": float(np.percentile(elapsed, 50)), "p95_ms": float(np.percentile(elapsed, 95)),
              "p99_ms": float(np.percentile(elapsed, 99)), "samples_ms": elapsed,
              "rss_mb": rss, "peak_working_set_mb": peak, "memory_source": memory_source,
              "model_files_sha256": hashes, "torch_version": torch.__version__,
              "platform": platform.platform(), "processor": platform.processor(),
              "limitations": ["Synthetic queries, one run; not a representative workload or SLA.",
                              "No retrieval, API, process cold start, GPU, or concurrent requests measured."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args.model_dir, args.output, args.threads)
    print(json.dumps({key: result[key] for key in ("profile", "n", "p50_ms", "p95_ms", "p99_ms", "rss_mb", "peak_working_set_mb")}, indent=2))
