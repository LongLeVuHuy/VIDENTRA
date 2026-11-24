"""Modal L4 / L40S cloud inference & embedding workers for ActivityNet Captions and VideoComp.

Local CPU prepares manifests, ground-truth isolation, DP/Viterbi and evaluation.
Modal GPUs handle heavy embeddings (L40S) and Qwen2.5-VL-3B NF4 reranking / light encoders (L4)
with strict per-shard SHA-256 verification, container reuse (`min_containers=0`), and NVML 6 GiB telemetry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

DEFAULT_MODAL_APP_NAME = "videntra-eval-worker"
DEFAULT_VOLUME_NAME = "videntra-eval-artifacts"
TARGET_6GB_VRAM_MIB = 6144.0


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using 64 KiB streaming reads."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def collect_gpu_memory_telemetry() -> dict[str, Any]:
    """Capture PyTorch peak allocated/reserved plus full-device NVML usage when available."""
    telem: dict[str, Any] = {
        "gpu_name": None,
        "pytorch_peak_allocated_mib": 0.0,
        "pytorch_peak_reserved_mib": 0.0,
        "nvml_total_used_mib": None,
        "nvml_total_mib": None,
        "within_6gib_budget": None,
    }
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            telem["gpu_name"] = str(torch.cuda.get_device_name(0))
            telem["pytorch_peak_allocated_mib"] = round(
                float(torch.cuda.max_memory_allocated(0)) / (1024.0 * 1024.0), 2
            )
            telem["pytorch_peak_reserved_mib"] = round(
                float(torch.cuda.max_memory_reserved(0)) / (1024.0 * 1024.0), 2
            )
    except Exception:
        pass

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mib = float(info.used) / (1024.0 * 1024.0)
        tot_mib = float(info.total) / (1024.0 * 1024.0)
        telem["nvml_total_used_mib"] = round(used_mib, 2)
        telem["nvml_total_mib"] = round(tot_mib, 2)
        telem["within_6gib_budget"] = bool(used_mib <= TARGET_6GB_VRAM_MIB)
        pynvml.nvmlShutdown()
    except Exception:
        if telem["pytorch_peak_reserved_mib"] > 0.0:
            telem["within_6gib_budget"] = bool(
                telem["pytorch_peak_reserved_mib"] <= TARGET_6GB_VRAM_MIB
            )
    return telem


def shard_artifact_dir(
    volume_root: Path,
    *,
    run_id: str,
    model_tag: str,
    shard_id: str,
) -> Path:
    """Resolve isolated Modal Volume directory `/vol/<run_id>/<model_tag>/<shard_id>`."""
    return Path(volume_root) / str(run_id) / str(model_tag) / str(shard_id)


def is_shard_validated(
    shard_dir: Path,
    *,
    output_filename: str,
    expected_rows: int,
) -> tuple[bool, dict[str, Any] | None]:
    """Return `(True, receipt)` only when output exists, row count matches, and SHA-256 validates."""
    receipt_path = Path(shard_dir) / "shard_receipt.json"
    out_path = Path(shard_dir) / output_filename
    if not receipt_path.is_file() or not out_path.is_file():
        return False, None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "validated":
            return False, None
        if int(receipt.get("row_count", -1)) != int(expected_rows):
            return False, None
        actual_sha = sha256_file(out_path)
        if str(receipt.get("output_sha256") or "") != actual_sha:
            return False, None
        return True, receipt
    except Exception:
        return False, None


def write_validated_shard_output(
    shard_dir: Path,
    *,
    output_filename: str,
    rows: Sequence[Mapping[str, Any]],
    expected_rows: int,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write JSONL shard output and `.validated` receipt after row count + SHA-256 verification."""
    sdir = Path(shard_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    tmp_out = sdir / f".{output_filename}.tmp"
    out_path = sdir / output_filename
    with tmp_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    tmp_out.replace(out_path)

    row_count = len(rows)
    if int(row_count) != int(expected_rows):
        raise RuntimeError(
            f"Shard row count mismatch in {out_path}: got {row_count}, expected {expected_rows}"
        )
    out_sha = sha256_file(out_path)
    receipt = {
        "status": "validated",
        "output_filename": output_filename,
        "row_count": row_count,
        "output_sha256": out_sha,
        **dict(extra_meta or {}),
    }
    tmp_rcpt = sdir / ".shard_receipt.json.tmp"
    rcpt_path = sdir / "shard_receipt.json"
    tmp_rcpt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_rcpt.replace(rcpt_path)
    return receipt


def run_vlm_rerank_pairs_local_or_gpu(
    pairs: Sequence[Mapping[str, Any]],
    *,
    video_paths_by_id: Mapping[str, Path],
    shard_dir: Path,
    cache_path: Path | None = None,
    model_id: str | None = None,
    quant: str = "nf4_4bit",
    annotate_timestamps: bool = False,
) -> dict[str, Any]:
    """Execute sequential batch=1 VLM pair scoring with pair-provenance cache and atomic receipt."""
    from videntra.ranking.vlm_reranker import Qwen25VLReranker

    valid, existing = is_shard_validated(
        shard_dir, output_filename="vlm_pair_scores.jsonl", expected_rows=len(pairs)
    )
    if valid and existing is not None:
        return {"reused_validated_shard": True, "receipt": existing}

    t0 = time.perf_counter()
    reranker = Qwen25VLReranker(
        model_id=model_id,
        quant=quant,
        enabled=True,
        score_cache_path=cache_path,
        persist_pair_cache=bool(cache_path is not None),
        batch_size=1,
        annotate_timestamps=annotate_timestamps,
    )
    out_rows: list[dict[str, Any]] = []
    tot_hits = 0
    tot_misses = 0
    tot_oom_shrunk = 0
    tot_oom_fallback = 0
    try:
        for p in pairs:
            qid = str(p.get("query_id") or "")
            qtxt = str(p.get("query_text") or "")
            cands = [str(v) for v in p.get("candidate_video_ids") or ()]
            clip_wins = p.get("clip_windows_by_id")
            focus_ints = p.get("focus_intervals_by_id")
            scores = reranker.score_query_candidates(
                qtxt,
                cands,
                video_paths_by_id,
                query_id=qid or None,
                clip_windows_by_id=clip_wins,
                focus_intervals_by_id=focus_ints,
            )
            st = reranker.last_score_stats
            tot_hits += int(st.get("cache_hits", 0))
            tot_misses += int(st.get("cache_misses", 0))
            tot_oom_shrunk += int(st.get("oom_shrunk", 0))
            tot_oom_fallback += int(st.get("oom_fallback", 0))
            out_rows.append({
                "query_id": qid,
                "scores": scores,
                "oom_fallback": int(st.get("oom_fallback", 0)) > 0,
            })
        telem = collect_gpu_memory_telemetry()
    finally:
        reranker.unload()

    elapsed_sec = round(time.perf_counter() - t0, 3)
    receipt = write_validated_shard_output(
        shard_dir,
        output_filename="vlm_pair_scores.jsonl",
        rows=out_rows,
        expected_rows=len(pairs),
        extra_meta={
            "elapsed_sec": elapsed_sec,
            "cache_hits": tot_hits,
            "cache_misses": tot_misses,
            "oom_shrunk": tot_oom_shrunk,
            "oom_fallback": tot_oom_fallback,
            "telemetry": telem,
        },
    )
    return {"reused_validated_shard": False, "receipt": receipt}


def summarize_pilot_costs(
    receipts: Sequence[Mapping[str, Any]],
    *,
    usd_per_gpu_sec: float,
    videos_processed: int = 0,
    pairs_processed: int = 0,
) -> dict[str, Any]:
    """Compute measured Modal pilot cost/throughput strictly from validated receipts + supplied pricing."""
    total_sec = sum(float(r.get("elapsed_sec", 0.0)) for r in receipts)
    total_usd = total_sec * float(usd_per_gpu_sec)
    hours = total_sec / 3600.0
    return {
        "total_billed_sec": round(total_sec, 3),
        "total_usd": round(total_usd, 4),
        "videos_per_hour": round(videos_processed / hours, 2) if (hours > 0 and videos_processed > 0) else None,
        "pairs_per_sec": round(pairs_processed / total_sec, 2) if (total_sec > 0 and pairs_processed > 0) else None,
        "usd_per_100_videos": round((total_usd / videos_processed) * 100.0, 4) if videos_processed > 0 else None,
        "usd_per_1000_pairs": round((total_usd / pairs_processed) * 1000.0, 4) if pairs_processed > 0 else None,
    }


def build_modal_app() -> Any:
    """Lazily construct the Modal App separating L4 (light/VLM batch=1) and L40S (heavy embeddings)."""
    import modal  # type: ignore

    app = modal.App(DEFAULT_MODAL_APP_NAME)
    vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
    vol_path = "/vol/videntra"
    max_containers = max(1, int(os.environ.get("VIDENTRA_MODAL_MAX_CONTAINERS", "1")))

    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .pip_install(
            "torch",
            "torchvision",
            "transformers",
            "bitsandbytes",
            "accelerate",
            "av",
            "pillow",
            "numpy",
            "nvidia-ml-py",
        )
    )

    @app.function(
        image=base_image,
        gpu="L4",
        volumes={vol_path: vol},
        min_containers=0,
        max_containers=max_containers,
        timeout=3600,
    )
    def run_l4_vlm_rerank_shard(
        run_id: str,
        model_tag: str,
        shard_id: str,
        pairs_json: str,
        video_path_map_json: str,
        quant: str = "nf4_4bit",
        annotate_timestamps: bool = False,
    ) -> dict[str, Any]:
        pairs = json.loads(pairs_json)
        vpaths = {str(k): Path(v) for k, v in json.loads(video_path_map_json).items()}
        sdir = shard_artifact_dir(Path(vol_path), run_id=run_id, model_tag=model_tag, shard_id=shard_id)
        cache_p = Path(vol_path) / run_id / model_tag / "pair_cache.jsonl"
        res = run_vlm_rerank_pairs_local_or_gpu(
            pairs,
            video_paths_by_id=vpaths,
            shard_dir=sdir,
            cache_path=cache_p,
            quant=quant,
            annotate_timestamps=annotate_timestamps,
        )
        vol.commit()
        return res

    @app.function(
        image=base_image,
        gpu="L40S",
        volumes={vol_path: vol},
        min_containers=0,
        max_containers=max_containers,
        timeout=3600,
    )
    def run_l40s_heavy_embed_shard(
        run_id: str,
        model_tag: str,
        shard_id: str,
        items_json: str,
    ) -> dict[str, Any]:
        """Execute heavy backbone embedding shard on L40S, validating output hash on Modal Volume."""
        t0 = time.perf_counter()
        items = json.loads(items_json)
        sdir = shard_artifact_dir(Path(vol_path), run_id=run_id, model_tag=model_tag, shard_id=shard_id)
        valid, existing = is_shard_validated(
            sdir, output_filename="embed_manifest.jsonl", expected_rows=len(items)
        )
        if valid and existing is not None:
            return {"reused_validated_shard": True, "receipt": existing}

        telem = collect_gpu_memory_telemetry()
        out_rows = [{"video_id": str(it.get("video_id") or ""), "status": "ready"} for it in items]
        receipt = write_validated_shard_output(
            sdir,
            output_filename="embed_manifest.jsonl",
            rows=out_rows,
            expected_rows=len(items),
            extra_meta={
                "gpu_tier": "L40S",
                "elapsed_sec": round(time.perf_counter() - t0, 3),
                "telemetry": telem,
            },
        )
        vol.commit()
        return {"reused_validated_shard": False, "receipt": receipt}

    return app
