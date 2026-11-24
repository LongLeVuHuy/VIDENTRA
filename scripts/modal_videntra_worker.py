"""Modal L4 / L40S cloud inference & embedding workers for ActivityNet Captions and VideoComp.

Local CPU prepares manifests, ground-truth isolation, DP/Viterbi and evaluation.
Modal GPUs handle heavy embeddings (L40S/L4) and Qwen2.5-VL-3B NF4 reranking (L4, batch=1)
with strict per-shard SHA-256 verification, container reuse (`min_containers=0`), and NVML 6 GiB telemetry.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

DEFAULT_MODAL_APP_NAME = "videntra-eval-worker"
DEFAULT_VOLUME_NAME = "videntra-eval-artifacts"
VOLUME_MOUNT_PATH = "/vol/videntra"
TARGET_6GB_VRAM_MIB = 6144.0

# Pinned multimodal backbones matching VIDENTRA Champion configuration
PINNED_BACKBONES: dict[str, dict[str, Any]] = {
    "pe_bigg": {
        "model_id": "timm/PE-Core-bigG-14-448",
        "revision": "17aa0c25addfa14198fa2ff73d845a22d433432e",
        "weights_file": "open_clip_model.safetensors",
        "weights_sha256": "d46f61f93c4c8533d2550ed929effe9ec5b096ec48f78f42a17a050f43093a6a",
        "fallback_arch": "PE-Core-bigG-14-448",
        "dim": 1280,
        "ctx_default": 72,
    },
    "pe_l": {
        "model_id": "timm/PE-Core-L-14-336",
        "revision": "8eff41b3f687e50a323662c2dda5eb3588c6dd35",
        "weights_file": "open_clip_model.safetensors",
        "weights_sha256": "b1fff7093c32a01d98a2d82098f9adf788b30f0efee639916520b3e4302f786b",
        "fallback_arch": "PE-Core-L-14-336",
        "dim": 1024,
        "ctx_default": 32,
    },
    "sig400": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        "weights_file": "model.safetensors",
        "weights_sha256": "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2",
        "dim": 1152,
    },
}

DEFAULT_EXTRACTION_BACKBONES: tuple[str, ...] = ("pe_bigg", "pe_l", "sig400")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using 64 KiB streaming reads."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_media_key(raw_name: str) -> str:
    """Normalize ActivityNet/YouCook2 stem (`v_xyz` or `xyz_clip`) to root key."""
    stem = Path(str(raw_name)).stem.strip()
    if stem.startswith("v_") and len(stem) > 2:
        stem = stem[2:]
    return stem


def _safe_reset_cuda_peak_stats() -> None:
    """Safely initialize CUDA context (if needed) and reset peak memory stats without crashing."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


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
    """Resolve isolated Modal Volume directory `/vol/videntra/<run_id>/<model_tag>/<shard_id>`."""
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
    num_frames: int = 8,
    annotate_timestamps: bool = False,
    query_template: str | None = None,
) -> dict[str, Any]:
    """Execute sequential batch=1 VLM pair scoring with pair-provenance cache and atomic receipt."""
    from videntra.ranking.vlm_reranker import Qwen25VLReranker

    valid, existing = is_shard_validated(
        shard_dir, output_filename="vlm_pair_scores.jsonl", expected_rows=len(pairs)
    )
    if valid and existing is not None:
        return {"reused_validated_shard": True, "receipt": existing}

    import gc

    gc.collect()
    _safe_reset_cuda_peak_stats()

    t0 = time.perf_counter()
    reranker = Qwen25VLReranker(
        model_id=model_id,
        quant=quant,
        enabled=True,
        score_cache_path=cache_path,
        persist_pair_cache=bool(cache_path is not None),
        batch_size=1,
        num_frames=num_frames,
        annotate_timestamps=annotate_timestamps,
        query_template=query_template,
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
            clip_wins_raw = p.get("clip_windows_by_id")
            clip_wins = (
                {str(k): (float(v[0]), float(v[1])) for k, v in clip_wins_raw.items()}
                if isinstance(clip_wins_raw, Mapping)
                else None
            )
            focus_raw = p.get("focus_intervals_by_id")
            focus_ints = (
                {
                    str(k): [(float(iv[0]), float(iv[1])) for iv in v]
                    for k, v in focus_raw.items()
                }
                if isinstance(focus_raw, Mapping)
                else None
            )
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
        gc.collect()

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
            "num_frames": int(num_frames),
            "annotate_timestamps": bool(annotate_timestamps),
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


class _ConcatPartReader(io.RawIOBase):
    """Stream-read concatenated multi-part archive files sequentially without extra disk copy."""

    def __init__(self, part_paths: Sequence[Path]) -> None:
        self._parts = [Path(p) for p in part_paths]
        self._idx = 0
        self._fh = self._parts[0].open("rb") if self._parts else None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._fh is None:
            return b""
        if size is None or size < 0:
            chunks = [self._fh.read()]
            while self._advance():
                chunks.append(self._fh.read())
            return b"".join(chunks)
        out = bytearray()
        rem = int(size)
        while rem > 0 and self._fh is not None:
            data = self._fh.read(rem)
            if data:
                out.extend(data)
                rem -= len(data)
            elif not self._advance():
                break
        return bytes(out)

    def _advance(self) -> bool:
        if self._fh is not None:
            self._fh.close()
        self._idx += 1
        if self._idx < len(self._parts):
            self._fh = self._parts[self._idx].open("rb")
            return True
        self._fh = None
        return False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def _probe_video_file_pts(args: tuple[str, Path, Path]) -> dict[str, Any]:
    """Fast-probe container streams, packet PTS range, and SHA-256 without full pixel decode."""
    import av  # type: ignore

    domain, root_dir, fpath = args
    key = normalize_media_key(fpath.name)
    rel_path = str(fpath.relative_to(root_dir)).replace("\\", "/")
    file_sha = sha256_file(fpath)
    valid_pts: list[float] = []
    pkt_cnt = 0
    try:
        with av.open(str(fpath)) as container:
            v_streams = [s for s in container.streams if s.type == "video"]
            if not v_streams:
                return {"ok": False, "video_id": key, "domain": domain, "rel_path": rel_path, "reason": "no_video_stream"}
            stream = v_streams[0]
            tb = float(stream.time_base) if stream.time_base is not None else 0.0
            for pkt in container.demux(stream):
                if pkt.dts is None and pkt.pts is None:
                    continue
                pkt_cnt += 1
                if pkt.pts is not None and tb > 0.0:
                    valid_pts.append(float(pkt.pts * tb))
        if not valid_pts:
            return {"ok": False, "video_id": key, "domain": domain, "rel_path": rel_path, "reason": "no_valid_pts"}
        dur_sec = max(0.1, max(valid_pts) - min(valid_pts))
        return {
            "ok": True,
            "video_id": key,
            "domain": domain,
            "rel_path": rel_path,
            "ext": fpath.suffix.lower(),
            "size_bytes": int(fpath.stat().st_size),
            "file_sha256": file_sha,
            "frame_count": pkt_cnt,
            "valid_pts_count": len(valid_pts),
            "min_pts_sec": round(min(valid_pts), 4),
            "duration_sec": round(dur_sec, 4),
        }
    except Exception as exc:
        return {"ok": False, "video_id": key, "domain": domain, "rel_path": rel_path, "reason": str(exc)[:180]}


def _decode_presentation_order_bin_frames(
    video_path: Path,
    *,
    num_frames: int = 48,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
) -> tuple[list[Any], list[int], list[float], float]:
    """Low-RAM two-pass PyAV presentation-order decoder selecting `num_frames` PTS bin centers."""
    import av  # type: ignore
    from PIL import Image  # type: ignore
    from videntra.ranking.vlm_reranker import select_bin_center_ordinals

    records: list[tuple[int, float | None]] = []
    with av.open(str(video_path)) as container:
        v_streams = [s for s in container.streams if s.type == "video"]
        if not v_streams:
            raise ValueError(f"No video stream in {video_path}")
        stream = v_streams[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base is not None else 0.0
        for cur_ord, frame in enumerate(container.decode(stream)):
            ts = float(frame.pts * tb) if (frame.pts is not None and tb > 0.0) else None
            records.append((cur_ord, ts))

    pts_by_ord = {int(o): float(ts) for o, ts in records if ts is not None}
    if not pts_by_ord:
        raise ValueError(f"Video lacks mapped PTS timestamps: {video_path}")

    min_ts = min(pts_by_ord.values())
    max_ts = max(pts_by_ord.values()) + 1e-3
    eff_start = float(clip_start_sec) if clip_start_sec is not None else min_ts
    eff_end = float(clip_end_sec) if clip_end_sec is not None else max_ts
    if eff_end <= eff_start:
        eff_end = eff_start + 1e-3

    ords = select_bin_center_ordinals(records, eff_start, eff_end, num_frames=num_frames)
    if not ords:
        raise ValueError(f"No selectable ordinals in [{eff_start}, {eff_end}) for {video_path}")

    target_set = set(ords)
    max_target = max(target_set)
    imgs_by_ord: dict[int, Any] = {}
    with av.open(str(video_path)) as container:
        stream = [s for s in container.streams if s.type == "video"][0]
        stream.thread_type = "AUTO"
        for cur_ord, frame in enumerate(container.decode(stream)):
            if cur_ord in target_set and cur_ord not in imgs_by_ord:
                rgb = frame.to_ndarray(format="rgb24")
                pil_img = Image.fromarray(rgb, mode="RGB")
                if max(pil_img.size) > 512:
                    pil_img.thumbnail((512, 512), Image.Resampling.BICUBIC)
                imgs_by_ord[cur_ord] = pil_img
            if cur_ord >= max_target and len(imgs_by_ord) >= len(target_set):
                break

    imgs = [imgs_by_ord[o] for o in ords if o in imgs_by_ord]
    pts_list = [float(pts_by_ord[o]) for o in ords if o in imgs_by_ord]
    ord_list = [int(o) for o in ords if o in imgs_by_ord]
    dur_sec = max(0.1, float(max_ts - min_ts))
    return imgs, ord_list, pts_list, dur_sec


def _encode_openclip_frames_and_texts(
    *,
    spec: Mapping[str, Any],
    all_images: Sequence[Any],
    texts: Sequence[str],
    device: str,
    cache_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode dense video frames & texts with pinned OpenCLIP backbone on CUDA FP16."""
    import gc
    from pathlib import Path as _Path
    from huggingface_hub import snapshot_download  # type: ignore
    import numpy as np
    import open_clip  # type: ignore
    import torch  # type: ignore

    t0 = time.perf_counter()
    base_cache = _Path(cache_root) if cache_root is not None else _Path("/tmp/hf_models")
    cache_dir = base_cache / "open_clip" / spec["model_id"].replace("/", "__")
    w_file = cache_dir / spec["weights_file"]
    if not (w_file.is_file() and sha256_file(w_file) == spec["weights_sha256"]):
        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
            allow_patterns=["open_clip_config.json", spec["weights_file"], "*.json", "*.txt"],
            local_dir=str(cache_dir),
        )
    if sha256_file(w_file) != spec["weights_sha256"]:
        raise RuntimeError(f"Weight SHA-256 mismatch for {spec['model_id']}")

    try:
        mdl, _, preproc = open_clip.create_model_and_transforms(f"local-dir:{cache_dir}")
        tok = open_clip.get_tokenizer(f"local-dir:{cache_dir}")
    except Exception:
        mdl, _, preproc = open_clip.create_model_and_transforms(
            spec["fallback_arch"], pretrained=str(cache_dir / spec["weights_file"])
        )
        tok = open_clip.get_tokenizer(spec["fallback_arch"])

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    mdl = mdl.eval().to(device=device, dtype=dtype)
    load_sec = time.perf_counter() - t0
    dim = int(spec["dim"])
    ctx_len = int(getattr(mdl, "context_length", spec.get("ctx_default", 72)))

    frame_vecs = np.zeros((len(all_images), dim), dtype=np.float16)
    img_bs = 32 if spec["model_id"].endswith("448") else 64
    with torch.inference_mode():
        for st in range(0, len(all_images), img_bs):
            sub = all_images[st : st + img_bs]
            tens = torch.stack([preproc(im) for im in sub]).to(device=device, dtype=dtype)
            feats = mdl.encode_image(tens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            frame_vecs[st : st + len(sub)] = feats.cpu().numpy().astype(np.float16)

        text_vecs = np.zeros((len(texts), dim), dtype=np.float32)
        txt_bs = 128
        for st in range(0, len(texts), txt_bs):
            sub_t = list(texts[st : st + txt_bs])
            try:
                tk_ids = tok(sub_t, context_length=ctx_len).to(device)
            except TypeError:
                tk_ids = tok(sub_t).to(device)
            t_feats = mdl.encode_text(tk_ids)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_vecs[st : st + len(sub_t)] = t_feats.float().cpu().numpy().astype(np.float32)

    del mdl, tok, preproc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame_vecs, text_vecs, load_sec


def _encode_siglip2_frames_and_texts(
    *,
    spec: Mapping[str, Any],
    all_images: Sequence[Any],
    texts: Sequence[str],
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode dense video frames & lowercased max_length=64 texts with SigLIP2-SO400M-384."""
    import gc
    from pathlib import Path as _Path
    from huggingface_hub import hf_hub_download  # type: ignore
    import numpy as np
    import torch  # type: ignore
    from transformers import AutoModel, AutoProcessor  # type: ignore

    t0 = time.perf_counter()
    w_path = _Path(
        hf_hub_download(
            repo_id=spec["model_id"],
            filename=spec["weights_file"],
            revision=spec["revision"],
        )
    )
    if sha256_file(w_path) != spec["weights_sha256"]:
        raise RuntimeError(f"Weight SHA-256 mismatch for {spec['model_id']}")

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    proc = AutoProcessor.from_pretrained(spec["model_id"], revision=spec["revision"])
    mdl = AutoModel.from_pretrained(
        spec["model_id"], revision=spec["revision"], torch_dtype=dtype
    ).eval().to(device)
    load_sec = time.perf_counter() - t0
    dim = int(spec["dim"])

    def _pool_feat(out_obj: Any) -> Any:
        return (
            out_obj.pooler_output
            if hasattr(out_obj, "pooler_output") and out_obj.pooler_output is not None
            else out_obj
        )

    frame_vecs = np.zeros((len(all_images), dim), dtype=np.float16)
    img_bs = 32
    with torch.inference_mode():
        for st in range(0, len(all_images), img_bs):
            sub = list(all_images[st : st + img_bs])
            inputs = proc(images=sub, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=(dtype if v.is_floating_point() else None)) for k, v in inputs.items()}
            raw = mdl.get_image_features(**inputs)
            feats = _pool_feat(raw)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            frame_vecs[st : st + len(sub)] = feats.cpu().numpy().astype(np.float16)

        text_vecs = np.zeros((len(texts), dim), dtype=np.float32)
        txt_bs = 128
        for st in range(0, len(texts), txt_bs):
            sub_t = [str(t).lower() for t in texts[st : st + txt_bs]]
            t_in = proc(
                text=sub_t,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            t_in = {k: v.to(device) for k, v in t_in.items()}
            raw_t = mdl.get_text_features(**t_in)
            tf = _pool_feat(raw_t)
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_vecs[st : st + len(sub_t)] = tf.float().cpu().numpy().astype(np.float32)

    del mdl, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame_vecs, text_vecs, load_sec


def _extract_dense_embeddings_on_gpu(
    *,
    vol_root: Path,
    run_id: str,
    gpu_tag: str,
    shard_id: str,
    video_items: Sequence[Mapping[str, Any]],
    text_items: Sequence[Mapping[str, Any]],
    num_frames: int = 48,
    backbones: Sequence[str] = DEFAULT_EXTRACTION_BACKBONES,
) -> dict[str, Any]:
    """Shared L4/L40S dense PTS-ledger extraction worker saving compact validated `.npz` shards."""
    import numpy as np
    import torch  # type: ignore

    sdir = shard_artifact_dir(vol_root, run_id=run_id, model_tag=f"features_{gpu_tag}", shard_id=shard_id)
    out_npz = sdir / "shard_features.npz"
    receipt_path = sdir / "shard_receipt.json"
    if receipt_path.is_file() and out_npz.is_file():
        try:
            rc = json.loads(receipt_path.read_text(encoding="utf-8"))
            has_frames = int(rc.get("num_videos", 0)) == 0 or int(rc.get("total_decoded_frames", 0)) > 0
            if (
                rc.get("status") == "validated"
                and has_frames
                and len(rc.get("decode_errors", [])) < max(1, int(rc.get("num_videos", 0)))
                and rc.get("output_sha256") == sha256_file(out_npz)
            ):
                return {"reused_validated_shard": True, "receipt": rc}
        except Exception:
            pass

    import gc

    gc.collect()
    _safe_reset_cuda_peak_stats()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    # Decode presentation-order bin-center frames via multi-threaded PyAV preserving exact order
    all_images: list[Any] = []
    vid_ids: list[str] = []
    vid_counts: list[int] = []
    vid_durs: list[float] = []
    vid_ords_padded = np.full((len(video_items), num_frames), -1, dtype=np.int32)
    vid_pts_padded = np.full((len(video_items), num_frames), np.nan, dtype=np.float32)

    decode_errors: list[dict[str, str]] = []
    raw_media_root = vol_root / "raw_media"

    def _decode_one(idx_it: tuple[int, Mapping[str, Any]]) -> tuple[int, str, list[Any], list[int], list[float], float, str | None]:
        idx_loc, it = idx_it
        key_id = str(it["item_id"])
        vpath = raw_media_root / str(it["rel_path"])
        try:
            imgs, ords, pts_s, dur = _decode_presentation_order_bin_frames(
                vpath,
                num_frames=num_frames,
                clip_start_sec=it.get("clip_start_sec"),
                clip_end_sec=it.get("clip_end_sec"),
            )
            return idx_loc, key_id, imgs, ords, pts_s, dur, None
        except Exception as exc:
            return idx_loc, key_id, [], [], [], 0.0, str(exc)[:180]

    with ThreadPoolExecutor(max_workers=6) as dec_pool:
        decoded_results = list(dec_pool.map(_decode_one, enumerate(video_items)))
    decoded_results.sort(key=lambda r: r[0])

    for idx, key_id, imgs, ords, pts_s, dur, err_s in decoded_results:
        if err_s is None:
            n_got = len(imgs)
            all_images.extend(imgs)
            vid_ids.append(key_id)
            vid_counts.append(n_got)
            vid_durs.append(dur)
            vid_ords_padded[idx, :n_got] = np.asarray(ords, dtype=np.int32)
            vid_pts_padded[idx, :n_got] = np.asarray(pts_s, dtype=np.float32)
        else:
            vid_ids.append(key_id)
            vid_counts.append(0)
            vid_durs.append(0.0)
            decode_errors.append({"item_id": key_id, "error": err_s})

    decode_sec = round(time.perf_counter() - t0, 3)

    text_ids = [str(t["text_id"]) for t in text_items]
    texts = [str(t["text"]) for t in text_items]

    npz_payload: dict[str, Any] = {
        "video_item_ids": np.asarray(vid_ids),
        "video_frame_counts": np.asarray(vid_counts, dtype=np.int32),
        "video_durations_sec": np.asarray(vid_durs, dtype=np.float32),
        "video_frame_ordinals": vid_ords_padded,
        "video_frame_pts_sec": vid_pts_padded,
        "text_ids": np.asarray(text_ids),
    }

    hf_cache_root = vol_root / "hf_cache"
    load_times: dict[str, float] = {}
    peak_nvml_mib = 0.0
    for b_name in backbones:
        spec = PINNED_BACKBONES[b_name]
        if b_name in ("pe_bigg", "pe_l"):
            f_mat, t_mat, l_sec = _encode_openclip_frames_and_texts(
                spec=spec, all_images=all_images, texts=texts, device=dev, cache_root=hf_cache_root
            )
        elif b_name == "sig400":
            f_mat, t_mat, l_sec = _encode_siglip2_frames_and_texts(
                spec=spec, all_images=all_images, texts=texts, device=dev
            )
        else:
            raise ValueError(f"Unsupported backbone key: {b_name}")
        cur_t = collect_gpu_memory_telemetry()
        if cur_t.get("nvml_total_used_mib"):
            peak_nvml_mib = max(peak_nvml_mib, float(cur_t["nvml_total_used_mib"]))
        load_times[b_name] = round(l_sec, 3)
        npz_payload[f"{b_name}_frames"] = f_mat
        npz_payload[f"{b_name}_texts"] = t_mat

    all_images.clear()

    sdir.mkdir(parents=True, exist_ok=True)
    tmp_npz = sdir / ".shard_features.npz.tmp"
    with tmp_npz.open("wb") as fh:
        np.savez_compressed(fh, **npz_payload)
    tmp_npz.replace(out_npz)

    telem = collect_gpu_memory_telemetry()
    if peak_nvml_mib > 0.0:
        telem["nvml_total_used_mib"] = max(float(telem.get("nvml_total_used_mib") or 0.0), round(peak_nvml_mib, 2))
        telem["within_6gib_budget"] = bool(telem["nvml_total_used_mib"] <= TARGET_6GB_VRAM_MIB)
    elapsed_sec = round(time.perf_counter() - t0, 3)
    receipt = {
        "status": "validated",
        "output_filename": "shard_features.npz",
        "shard_id": shard_id,
        "gpu_tag": gpu_tag,
        "num_videos": len(video_items),
        "total_decoded_frames": int(sum(vid_counts)),
        "num_texts": len(text_items),
        "decode_errors": decode_errors,
        "decode_sec": decode_sec,
        "model_load_sec": load_times,
        "elapsed_sec": elapsed_sec,
        "output_sha256": sha256_file(out_npz),
        "output_mb": round(out_npz.stat().st_size / 1048576.0, 3),
        "telemetry": telem,
    }
    tmp_rcpt = sdir / ".shard_receipt.json.tmp"
    tmp_rcpt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_rcpt.replace(receipt_path)
    return {"reused_validated_shard": False, "receipt": receipt}


def build_modal_app() -> Any:
    """Lazily return global-scope Modal App module defining CPU sync, L4, and L40S functions."""
    from scripts.modal_app_functions import app

    return app
