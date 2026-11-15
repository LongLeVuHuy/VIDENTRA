"""Offline Kaggle VideoPrism-LvT-B worker (pinned weights, 288x288 pad, shard verify-on-read)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.prepare_clip_vp_kaggle_bundle import (  # noqa: E402
    NUM_CLIP_FRAMES,
    assert_no_gt_leakage,
    compute_video_recipe_from_ledger,
    compute_video_recipe_from_media,
    sha256_json,
    sha256_text,
)
from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.timeline.ledger import exact_seek_frame  # noqa: E402

# Section 4.1 Pinned Upstream Metadata & Checksums
VIDEOPRISM_REPO = "google-deepmind/videoprism"
VIDEOPRISM_COMMIT = "4f831e332b80d6e77bdd43882206f23aba2a67a7"
VIDEOPRISM_CONFIG = "videoprism_lvt_public_v1_base"
VIDEOPRISM_HF_REPO = "google/videoprism-lvt-base-f16r288"
VIDEOPRISM_HF_REVISION = "0119cca7d1fe61aa1d5718e1f71605703e1562e7"
VIDEOPRISM_CHECKPOINT_FILENAME = "flax_lvt_base_f16r288_repeated.npz"
VIDEOPRISM_CHECKPOINT_BYTES = 990_527_124
VIDEOPRISM_CHECKPOINT_SHA256 = "dcc9bd31ba063221ca952e5a22370236018460faa09aa1a5649c6ccfaf071180"
VIDEOPRISM_TOKENIZER_FILENAME = "sentencepiece.model"
VIDEOPRISM_TOKENIZER_SHA256 = "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"
VIDEOPRISM_LICENSE = "apache-2.0"

VIDEOPRISM_IMAGE_SIZE = 288
VIDEOPRISM_EMBEDDING_DIM = 768

SHARD_FILENAMES = (
    "video_vectors.npy",
    "video_rows.jsonl",
    "query_vectors.npy",
    "query_rows.jsonl",
)


def pinned_model_metadata() -> dict[str, Any]:
    """Return pinned VideoPrism-LvT-B metadata dictionary for receipts."""
    return {
        "repo": VIDEOPRISM_REPO,
        "commit": VIDEOPRISM_COMMIT,
        "config": VIDEOPRISM_CONFIG,
        "hf_repo": VIDEOPRISM_HF_REPO,
        "hf_revision": VIDEOPRISM_HF_REVISION,
        "checkpoint_filename": VIDEOPRISM_CHECKPOINT_FILENAME,
        "checkpoint_bytes": VIDEOPRISM_CHECKPOINT_BYTES,
        "checkpoint_sha256": VIDEOPRISM_CHECKPOINT_SHA256,
        "tokenizer_filename": VIDEOPRISM_TOKENIZER_FILENAME,
        "tokenizer_sha256": VIDEOPRISM_TOKENIZER_SHA256,
        "license": VIDEOPRISM_LICENSE,
    }


def verify_videoprism_pins(
    assets_dir: Path,
    *,
    checkpoint_sha256: str = VIDEOPRISM_CHECKPOINT_SHA256,
    tokenizer_sha256: str = VIDEOPRISM_TOKENIZER_SHA256,
    check_size: bool = False,
) -> dict[str, Any]:
    """Fail closed if pinned VideoPrism checkpoint or tokenizer is missing or corrupted."""
    assets = Path(assets_dir)
    ckpt_path = assets / VIDEOPRISM_CHECKPOINT_FILENAME
    tok_path = assets / VIDEOPRISM_TOKENIZER_FILENAME

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing pinned VideoPrism checkpoint: {ckpt_path}")
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing pinned VideoPrism tokenizer: {tok_path}")

    if check_size and ckpt_path.stat().st_size != VIDEOPRISM_CHECKPOINT_BYTES:
        raise ValueError(
            f"Checkpoint size mismatch for {ckpt_path.name}: "
            f"expected {VIDEOPRISM_CHECKPOINT_BYTES}, got {ckpt_path.stat().st_size}"
        )

    actual_ckpt_sha = sha256_file(ckpt_path)
    if actual_ckpt_sha != checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {ckpt_path.name}: "
            f"expected {checkpoint_sha256}, got {actual_ckpt_sha}"
        )

    actual_tok_sha = sha256_file(tok_path)
    if actual_tok_sha != tokenizer_sha256:
        raise ValueError(
            f"Tokenizer SHA-256 mismatch for {tok_path.name}: "
            f"expected {tokenizer_sha256}, got {actual_tok_sha}"
        )

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": actual_ckpt_sha,
        "tokenizer_path": str(tok_path),
        "tokenizer_sha256": actual_tok_sha,
    }


def disable_tf_gpu_before_jax() -> bool:
    """Hide GPUs from TensorFlow before importing JAX so JAX owns GPU memory."""
    try:
        import tensorflow as tf  # type: ignore

        tf.config.set_visible_devices([], "GPU")
        return True
    except ImportError:
        return False


def resize_and_pad_288(
    frame_rgb: np.ndarray,
    target_size: int = VIDEOPRISM_IMAGE_SIZE,
) -> np.ndarray:
    """Aspect-preserving bilinear resize (0.5/0.5 center) + black pad to 288x288 float32 in [0, 1]."""
    arr = np.asarray(frame_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB frame [H, W, 3], got {arr.shape}")

    h, w, _ = arr.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame dimensions: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    else:
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, min(target_size, int(round(w * scale))))
    new_h = max(1, min(target_size, int(round(h * scale))))

    pil_img = Image.fromarray(arr_u8, mode="RGB")
    resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_f32 = np.asarray(resized, dtype=np.float32) / 255.0

    canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w, :] = resized_f32
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def prepare_clip_batch(
    clips_frames: Sequence[Sequence[np.ndarray]],
    *,
    num_frames: int = NUM_CLIP_FRAMES,
    target_size: int = VIDEOPRISM_IMAGE_SIZE,
) -> np.ndarray:
    """Convert a batch of 16-frame clips into [B, 16, 288, 288, 3] float32 in [0, 1]."""
    if not clips_frames:
        raise ValueError("clips_frames cannot be empty")
    batch_list: list[np.ndarray] = []
    for idx, frames in enumerate(clips_frames):
        if len(frames) != num_frames:
            raise ValueError(f"Clip {idx} expected {num_frames} frames, got {len(frames)}")
        processed = [resize_and_pad_288(f, target_size=target_size) for f in frames]
        batch_list.append(np.stack(processed, axis=0))

    batch = np.stack(batch_list, axis=0).astype(np.float32, copy=False)
    expected_shape = (len(clips_frames), num_frames, target_size, target_size, 3)
    if batch.shape != expected_shape:
        raise ValueError(f"Unexpected clip batch shape {batch.shape}, expected {expected_shape}")
    if not np.isfinite(batch).all():
        raise ValueError("Clip batch contains non-finite values")
    if batch.min() < 0.0 or batch.max() > 1.0:
        raise ValueError("Clip batch values must be in [0.0, 1.0]")
    return batch


def normalize_rows(embeddings: Any, min_norm: float = 1e-8) -> np.ndarray:
    """L2-normalize 2D embedding rows with strict finiteness and norm guards."""
    rows = np.asarray(embeddings, dtype=np.float32)
    if rows.ndim != 2 or rows.size == 0:
        raise ValueError(f"Expected [batch, embedding_dim], got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("VideoPrism returned non-finite embedding values")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= min_norm):
        raise ValueError("VideoPrism returned zero or near-zero norm embedding rows")
    normalized = (rows / norms).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embeddings contain non-finite values")
    return normalized


class SyntheticVideoPrismEncoder:
    """Deterministic CPU-only synthetic encoder for --dry-run and unit verification."""

    def __init__(self, embedding_dim: int = VIDEOPRISM_EMBEDDING_DIM) -> None:
        self.embedding_dim = int(embedding_dim)
        self.device_summary = ["cpu:synthetic"]

    def encode_videos(self, clip_batch: np.ndarray) -> np.ndarray:
        if clip_batch.ndim != 5 or clip_batch.shape[1:] != (
            NUM_CLIP_FRAMES,
            VIDEOPRISM_IMAGE_SIZE,
            VIDEOPRISM_IMAGE_SIZE,
            3,
        ):
            raise ValueError(f"Invalid clip_batch shape: {clip_batch.shape}")
        rows = []
        for clip in clip_batch:
            digest = hashlib.sha256(clip.tobytes()).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(self.embedding_dim, dtype=np.float32))
        return normalize_rows(np.stack(rows, axis=0))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts cannot be empty")
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(self.embedding_dim, dtype=np.float32))
        return normalize_rows(np.stack(rows, axis=0))


class JaxVideoPrismEncoder:
    """Real VideoPrism-LvT-B JAX/Flax encoder using pinned checkpoint and tokenizer."""

    def __init__(self, assets_dir: Path, *, allow_cpu: bool = False) -> None:
        pins = verify_videoprism_pins(assets_dir, check_size=False)
        disable_tf_gpu_before_jax()

        import jax  # type: ignore
        from videoprism import models as vp  # type: ignore
        from videoprism import tokenizers as vp_tokenizers  # type: ignore

        devices = jax.devices()
        if not allow_cpu and not any(d.platform == "gpu" for d in devices):
            raise RuntimeError(f"No GPU device found for JAX: {devices}")

        self._jax = jax
        self._vp = vp
        self.device_summary = [f"{d.platform}:{d.device_kind}" for d in devices]
        self._model = vp.get_model(VIDEOPRISM_CONFIG)
        self._state = vp.load_pretrained_weights(
            VIDEOPRISM_CONFIG,
            checkpoint_path=pins["checkpoint_path"],
        )
        self._tokenizer = vp_tokenizers.SentencePieceTokenizer(pins["tokenizer_path"])

        @jax.jit
        def _video_forward(frames: Any) -> Any:
            return self._model.apply(self._state, frames, None, None, train=False)[0]

        @jax.jit
        def _text_forward(token_ids: Any, token_paddings: Any) -> Any:
            return self._model.apply(self._state, None, token_ids, token_paddings, train=False)[1]

        self._video_forward = _video_forward
        self._text_forward = _text_forward

    def encode_videos(self, clip_batch: np.ndarray) -> np.ndarray:
        raw = self._jax.device_get(self._video_forward(clip_batch))
        return normalize_rows(raw)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        ids, pads = self._vp.tokenize_texts(self._tokenizer, list(texts))
        raw = self._jax.device_get(self._text_forward(ids, pads))
        return normalize_rows(raw)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git_commit_or_unknown() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _load_video_16_frames(
    video_entry: dict[str, Any],
    bundle_dir: Path,
    *,
    media_root: Path | None = None,
    ingest_dir: Path | None = None,
    precomputed_recipe: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load 16 presentation-order frames for a video using the bin-center recipe."""
    vid = str(video_entry["video_id"])
    clip_start = float(video_entry.get("clip_start_sec", 0.0))
    clip_end_raw = video_entry.get("clip_end_sec")
    clip_end = float(clip_end_raw) if clip_end_raw is not None else None
    media_sha = video_entry.get("media_sha256")

    # 1. Check pre-exported PNG frames in bundle_dir/frames/ if precomputed_recipe exists
    frames_dir = bundle_dir / "frames"
    if precomputed_recipe and precomputed_recipe.get("selected_ordinals") and frames_dir.is_dir():
        ordinals = [int(o) for o in precomputed_recipe["selected_ordinals"]]
        unique_cache: dict[int, np.ndarray] = {}
        for o in set(ordinals):
            img_path = frames_dir / f"{vid}__{o:06d}.png"
            if img_path.is_file():
                with Image.open(img_path) as im:
                    unique_cache[o] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if len(unique_cache) == len(set(ordinals)):
            return [unique_cache[o] for o in ordinals], precomputed_recipe

    # 2. Resolve media file
    rel_path = str(video_entry.get("relative_path") or "")
    candidates: list[Path] = []
    if rel_path:
        candidates.append(bundle_dir / rel_path)
        if media_root is not None:
            candidates.append(media_root / rel_path)
            candidates.append(media_root / Path(rel_path).name)
    media_path = next((p for p in candidates if p.is_file()), None)
    if media_path is None:
        raise FileNotFoundError(f"Media file not found for video_id={vid} (relative_path={rel_path})")

    if media_sha:
        actual_sha = sha256_file(media_path)
        if actual_sha != str(media_sha):
            raise ValueError(f"Media SHA-256 mismatch for {vid}: expected {media_sha}, got {actual_sha}")

    # 3. Check timeline ledger if ingest_dir is provided
    ledger_path: Path | None = None
    if ingest_dir is not None and ingest_dir.is_dir():
        for pat in (
            f"timeline/*{media_sha}*/*/frames.parquet" if media_sha else "",
            f"timeline/*{vid}*/*/frames.parquet",
        ):
            if pat:
                matches = sorted(ingest_dir.glob(pat))
                if matches:
                    ledger_path = matches[0]
                    break

    if ledger_path is not None:
        recipe = compute_video_recipe_from_ledger(
            ledger_path, clip_start_sec=clip_start, clip_end_sec=clip_end
        )
        if recipe["status"] != "ok":
            raise ValueError(f"Recipe failed for {vid}: {recipe['status']}")
        unique_cache = {}
        for o in set(recipe["selected_ordinals"]):
            dec = exact_seek_frame(media_path, ledger_path, target_ordinal=int(o))
            if dec.pixels is None:
                raise RuntimeError(f"Decoded frame pixels None for {vid} ordinal={o}")
            unique_cache[int(o)] = dec.pixels
        return [unique_cache[int(o)] for o in recipe["selected_ordinals"]], recipe

    # 4. Sequential presentation-order decode via videntra.ingestion.decode
    recipe, decoded_pixels = compute_video_recipe_from_media(
        media_path, clip_start_sec=clip_start, clip_end_sec=clip_end
    )
    if recipe["status"] != "ok":
        raise ValueError(f"Recipe failed for {vid}: {recipe['status']}")
    frames = [decoded_pixels[int(o)] for o in recipe["selected_ordinals"]]
    return frames, recipe


def verify_shard_outputs(output_dir: Path, expected_dim: int | None = None) -> dict[str, Any]:
    """Verify-on-read for video & query vector shards and JSONL row manifests."""
    out = Path(output_dir)
    for fname in SHARD_FILENAMES:
        if not (out / fname).is_file():
            raise FileNotFoundError(f"Missing shard output file: {out / fname}")

    v_rows = _read_jsonl(out / "video_rows.jsonl")
    q_rows = _read_jsonl(out / "query_rows.jsonl")
    v_vecs = np.load(str(out / "video_vectors.npy"), mmap_mode="r")
    q_vecs = np.load(str(out / "query_vectors.npy"), mmap_mode="r")

    if v_vecs.dtype != np.float32 or q_vecs.dtype != np.float32:
        raise ValueError(f"Expected float32 shards, got {v_vecs.dtype}, {q_vecs.dtype}")
    if v_vecs.ndim != 2 or q_vecs.ndim != 2:
        raise ValueError(f"Expected 2D vector arrays, got {v_vecs.shape}, {q_vecs.shape}")
    if v_vecs.shape[0] != len(v_rows):
        raise ValueError(f"Video row/vector count mismatch: {len(v_rows)} vs {v_vecs.shape[0]}")
    if q_vecs.shape[0] != len(q_rows):
        raise ValueError(f"Query row/vector count mismatch: {len(q_rows)} vs {q_vecs.shape[0]}")

    v_dim = int(v_vecs.shape[1])
    q_dim = int(q_vecs.shape[1])
    if v_dim != q_dim:
        raise ValueError(f"Video/query dim mismatch: {v_dim} vs {q_dim}")
    if expected_dim is not None and v_dim != expected_dim:
        raise ValueError(f"Expected embedding dim {expected_dim}, got {v_dim}")

    if v_vecs.shape[0] > 0:
        v_arr = np.asarray(v_vecs, dtype=np.float32)
        if not np.isfinite(v_arr).all():
            raise ValueError("video_vectors.npy contains non-finite values")
        v_norms = np.linalg.norm(v_arr, axis=1)
        if not np.allclose(v_norms, 1.0, atol=1e-4):
            raise ValueError("video_vectors.npy rows are not L2-normalized")

    if q_vecs.shape[0] > 0:
        q_arr = np.asarray(q_vecs, dtype=np.float32)
        if not np.isfinite(q_arr).all():
            raise ValueError("query_vectors.npy contains non-finite values")
        q_norms = np.linalg.norm(q_arr, axis=1)
        if not np.allclose(q_norms, 1.0, atol=1e-4):
            raise ValueError("query_vectors.npy rows are not L2-normalized")

    seen_vids: set[str] = set()
    for idx, row in enumerate(v_rows):
        if int(row.get("row_id", -1)) != idx:
            raise ValueError(f"Non-sequential video row_id at index {idx}")
        vid = str(row.get("video_id") or "")
        if not vid or vid in seen_vids:
            raise ValueError(f"Invalid or duplicate video_id in video_rows.jsonl: {vid}")
        seen_vids.add(vid)

    seen_qids: set[str] = set()
    for idx, row in enumerate(q_rows):
        if int(row.get("row_id", -1)) != idx:
            raise ValueError(f"Non-sequential query row_id at index {idx}")
        qid = str(row.get("query_id") or "")
        if not qid or qid in seen_qids:
            raise ValueError(f"Invalid or duplicate query_id in query_rows.jsonl: {qid}")
        seen_qids.add(qid)

    files_info = {
        fname: {
            "sha256": sha256_file(out / fname),
            "bytes": (out / fname).stat().st_size,
        }
        for fname in SHARD_FILENAMES
    }
    return {
        "video_count": len(v_rows),
        "query_count": len(q_rows),
        "embedding_dim": v_dim,
        "files": files_info,
    }


def _try_resume_shard(
    output_dir: Path,
    *,
    input_fingerprint: str,
    dry_run: bool,
) -> dict[str, Any] | None:
    receipt_path = output_dir / "receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "SUCCESS":
        raise ValueError(f"Existing receipt is not SUCCESS ({receipt.get('status')}); use a new output dir")
    if receipt.get("input_fingerprint") != input_fingerprint:
        raise ValueError("Existing shard input_fingerprint mismatch; use a new output dir")
    if bool(receipt.get("dry_run")) != bool(dry_run):
        raise ValueError("Existing shard dry_run mode mismatch; use a new output dir")

    verified = verify_shard_outputs(output_dir, expected_dim=int(receipt.get("embedding_dim", 0)) or None)
    recorded_files = receipt.get("files", {})
    for fname, info in verified["files"].items():
        if recorded_files.get(fname, {}).get("sha256") != info["sha256"]:
            raise ValueError(f"Shard hash mismatch on resume for {fname}")

    resumed_receipt = dict(receipt)
    resumed_receipt["resumed"] = True
    return resumed_receipt


def run_videoprism_worker(
    *,
    bundle_dir: Path,
    output_dir: Path,
    assets_dir: Path | None = None,
    media_root: Path | None = None,
    ingest_dir: Path | None = None,
    video_batch_size: int = 1,
    query_batch_size: int = 32,
    dry_run: bool = False,
    allow_cpu: bool = False,
    resume: bool = False,
    embedding_dim: int = VIDEOPRISM_EMBEDDING_DIM,
) -> dict[str, Any]:
    """Encode bundle videos and queries with VideoPrism-LvT-B (or synthetic dry-run)."""
    bundle = Path(bundle_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    queries_path = bundle / "queries.jsonl"
    media_manifest_path = bundle / "media_manifest.jsonl"
    if not queries_path.is_file():
        raise FileNotFoundError(f"Missing queries.jsonl in {bundle}")
    if not media_manifest_path.is_file():
        raise FileNotFoundError(f"Missing media_manifest.jsonl in {bundle}")

    queries = _read_jsonl(queries_path)
    media_entries = _read_jsonl(media_manifest_path)
    assert_no_gt_leakage(queries, context="worker_queries")
    assert_no_gt_leakage(media_entries, context="worker_media_manifest")

    precomputed_recipes: dict[str, dict[str, Any]] = {}
    recipe_path = bundle / "frame_recipe.jsonl"
    if recipe_path.is_file():
        for rec in _read_jsonl(recipe_path):
            precomputed_recipes[str(rec["video_id"])] = rec

    input_fingerprint = sha256_json({
        "queries_sha256": sha256_file(queries_path),
        "media_manifest_sha256": sha256_file(media_manifest_path),
        "model_commit": VIDEOPRISM_COMMIT,
        "model_config": VIDEOPRISM_CONFIG,
    })

    if resume:
        existing = _try_resume_shard(out, input_fingerprint=input_fingerprint, dry_run=dry_run)
        if existing is not None:
            return existing
    else:
        existing_files = [name for name in (*SHARD_FILENAMES, "receipt.json") if (out / name).exists()]
        if existing_files:
            raise FileExistsError(f"Output artifacts already exist in {out}: {existing_files}")

    if dry_run:
        encoder: Any = SyntheticVideoPrismEncoder(embedding_dim=embedding_dim)
        pin_info = {"mode": "dry_run_synthetic"}
    else:
        if assets_dir is None:
            raise ValueError("--assets-dir is required when not running with --dry-run")
        pin_info = verify_videoprism_pins(assets_dir, check_size=False)
        encoder = JaxVideoPrismEncoder(assets_dir, allow_cpu=allow_cpu)

    errors: list[dict[str, str]] = []

    # 1. Encode videos in batches
    video_rows: list[dict[str, Any]] = []
    video_vec_chunks: list[np.ndarray] = []
    pending_clips: list[list[np.ndarray]] = []
    pending_meta: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def _flush_videos() -> None:
        if not pending_clips:
            return
        clip_batch = prepare_clip_batch(pending_clips)
        vecs = encoder.encode_videos(clip_batch)
        video_vec_chunks.append(vecs)
        for entry, recipe in pending_meta:
            video_rows.append({
                "row_id": len(video_rows),
                "video_id": str(entry["video_id"]),
                "media_sha256": entry.get("media_sha256"),
                "timeline_revision": entry.get("timeline_revision"),
                "selected_ordinals": recipe.get("selected_ordinals", []),
                "selected_timestamps_sec": recipe.get("selected_timestamps_sec", []),
                "unique_frames": int(recipe.get("unique_frames", 0)),
                "recipe_hash": recipe.get("recipe_hash"),
            })
        pending_clips.clear()
        pending_meta.clear()

    for entry in media_entries:
        vid = str(entry.get("video_id") or "")
        try:
            frames, recipe = _load_video_16_frames(
                entry,
                bundle,
                media_root=media_root,
                ingest_dir=ingest_dir,
                precomputed_recipe=precomputed_recipes.get(vid),
            )
            pending_clips.append(frames)
            pending_meta.append((entry, recipe))
            if len(pending_clips) >= max(1, video_batch_size):
                _flush_videos()
        except Exception as exc:
            errors.append({"id": vid, "kind": "video", "error": str(exc)})

    _flush_videos()

    # 2. Encode queries in batches (preserving exact query_text_sha256)
    query_rows: list[dict[str, Any]] = []
    query_vec_chunks: list[np.ndarray] = []
    valid_queries: list[dict[str, str]] = []
    for q in queries:
        qid = str(q.get("query_id") or "")
        qtext = str(q.get("query_text") or "")
        expected_qhash = sha256_text(qtext)
        if q.get("query_text_sha256") and str(q["query_text_sha256"]) != expected_qhash:
            errors.append({"id": qid, "kind": "query", "error": "query_text_sha256 mismatch"})
            continue
        valid_queries.append({
            "query_id": qid,
            "query_text": qtext,
            "query_text_sha256": expected_qhash,
        })

    for start in range(0, len(valid_queries), max(1, query_batch_size)):
        batch_q = valid_queries[start : start + max(1, query_batch_size)]
        vecs = encoder.encode_texts([item["query_text"] for item in batch_q])
        query_vec_chunks.append(vecs)
        for item in batch_q:
            query_rows.append({
                "row_id": len(query_rows),
                "query_id": item["query_id"],
                "query_text_sha256": item["query_text_sha256"],
            })

    actual_dim = (
        int(video_vec_chunks[0].shape[1])
        if video_vec_chunks
        else (int(query_vec_chunks[0].shape[1]) if query_vec_chunks else int(embedding_dim))
    )
    all_video_vecs = (
        np.concatenate(video_vec_chunks, axis=0).astype(np.float32, copy=False)
        if video_vec_chunks
        else np.zeros((0, actual_dim), dtype=np.float32)
    )
    all_query_vecs = (
        np.concatenate(query_vec_chunks, axis=0).astype(np.float32, copy=False)
        if query_vec_chunks
        else np.zeros((0, actual_dim), dtype=np.float32)
    )

    np.save(str(out / "video_vectors.npy"), all_video_vecs)
    _write_jsonl(out / "video_rows.jsonl", video_rows)
    np.save(str(out / "query_vectors.npy"), all_query_vecs)
    _write_jsonl(out / "query_rows.jsonl", query_rows)

    verified = verify_shard_outputs(out, expected_dim=actual_dim)

    receipt = {
        "status": "SUCCESS",
        "dry_run": bool(dry_run),
        "resumed": False,
        "input_fingerprint": input_fingerprint,
        "model": pinned_model_metadata(),
        "pins": pin_info,
        "embedding_dim": verified["embedding_dim"],
        "dtype": "float32",
        "video_count": verified["video_count"],
        "query_count": verified["query_count"],
        "files": verified["files"],
        "errors": errors,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "devices": encoder.device_summary,
            "git_commit": _git_commit_or_unknown(),
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VideoPrism-LvT-B offline encoding on a Kaggle inference bundle."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=None, help="Directory with pinned checkpoint & tokenizer")
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--ingest-dir", type=Path, default=None)
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic encoder on CPU")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow JAX CPU execution when GPU is absent")
    parser.add_argument("--resume", action="store_true", help="Verify and resume from completed shard receipt")
    parser.add_argument("--embedding-dim", type=int, default=VIDEOPRISM_EMBEDDING_DIM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = run_videoprism_worker(
        bundle_dir=args.bundle_dir,
        output_dir=args.out_dir,
        assets_dir=args.assets_dir,
        media_root=args.media_root,
        ingest_dir=args.ingest_dir,
        video_batch_size=args.video_batch_size,
        query_batch_size=args.query_batch_size,
        dry_run=args.dry_run,
        allow_cpu=args.allow_cpu,
        resume=args.resume,
        embedding_dim=args.embedding_dim,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
