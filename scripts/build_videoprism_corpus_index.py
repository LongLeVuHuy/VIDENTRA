"""Prepare and ingest full-corpus VideoPrism-LvT-B (768d) Kaggle GPU extraction artifacts."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence
import unicodedata
import zipfile

import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.kaggle_videoprism_worker import (  # noqa: E402
    VIDEOPRISM_COMMIT,
    VIDEOPRISM_CONFIG,
    VIDEOPRISM_EMBEDDING_DIM,
    VIDEOPRISM_HF_REPO,
    VIDEOPRISM_HF_REVISION,
    pinned_model_metadata,
)
from scripts.prepare_clip_vp_kaggle_bundle import (  # noqa: E402
    assert_no_gt_leakage,
    sha256_text,
)
from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.query.rules import decompose_query_clauses  # noqa: E402


DEFAULT_KAGGLE_SLUG = "lvhuylong/videntra-videoprism-full-1324"
DEFAULT_DATASETS = (
    "lvhuylong/msrvtt-1ka-original-videos-v1",
    "lvhuylong/phase8-avc-pilot300-bundle-20261102",
    "lvhuylong/qvhighlights-practice-videos-v1",
)
SUBCLIP_WINDOW_SEC = 16.0
SUBCLIP_STRIDE_SEC = 8.0


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def collect_corpus_manifests(
    *,
    db_path: Path = Path("data/processed/metadata.db"),
    val_dir: Path = Path("validation"),
    test_dir: Path = Path("test_queries"),
    llm_plan_cache_path: Path | None = Path("data/cache/query_decomposition_cache.jsonl"),
    min_subclip_video_ms: int = 20_000,
    subclip_window_sec: float = SUBCLIP_WINDOW_SEC,
    subclip_stride_sec: float = SUBCLIP_STRIDE_SEC,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Build leakage-free media, subclip, and unique text-only query manifests."""
    if not db_path.is_file():
        raise FileNotFoundError(f"metadata.db not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        video_rows = conn.execute(
            "SELECT video_id, duration_ms FROM Videos ORDER BY video_id;"
        ).fetchall()

    indexed_videos = {str(v) for v, _ in video_rows}
    media_entries: list[dict[str, Any]] = []
    subclip_entries: list[dict[str, Any]] = []

    for vid_raw, dur_ms_raw in video_rows:
        vid = str(vid_raw)
        media_entries.append({
            "video_id": vid,
            "relative_path": f"{vid}.mp4",
            "clip_start_sec": 0.0,
            "clip_end_sec": None,
            "duration_ms": int(dur_ms_raw) if dur_ms_raw is not None else None,
        })
        if dur_ms_raw is not None and int(dur_ms_raw) > int(min_subclip_video_ms):
            dur_s = float(dur_ms_raw) / 1000.0
            starts: list[float] = []
            cur = 0.0
            while cur < dur_s - 2.0:
                end_t = min(dur_s, cur + subclip_window_sec)
                if end_t - cur >= 4.0:
                    starts.append(cur)
                if end_t >= dur_s - 0.5:
                    break
                cur += subclip_stride_sec
            for w_idx, st in enumerate(starts):
                et = min(dur_s, st + subclip_window_sec)
                subclip_entries.append({
                    "subclip_id": f"{vid}__w{w_idx:03d}",
                    "video_id": vid,
                    "relative_path": f"{vid}.mp4",
                    "clip_start_sec": round(st, 3),
                    "clip_end_sec": round(et, 3),
                })

    raw_texts: list[str] = []

    def _collect_from_split_dir(split_root: Path) -> None:
        if not split_root.exists():
            return
        qvh_path = split_root / "qvhighlights" / "highlight_val_release.jsonl"
        if qvh_path.is_file():
            for item in _read_jsonl(qvh_path):
                if item.get("vid") in indexed_videos and item.get("query"):
                    raw_texts.append(str(item["query"]))

        avc_path = split_root / "avc" / "candidate_plan_inference.json"
        if avc_path.is_file():
            avc_doc = json.loads(avc_path.read_text(encoding="utf-8"))
            for item in avc_doc.get("queries", []):
                if item.get("query_text"):
                    raw_texts.append(str(item["query_text"]))

        msr_path = split_root / "msrvtt" / "test_videodatainfo.json"
        if msr_path.is_file():
            msr_doc = json.loads(msr_path.read_text(encoding="utf-8"))
            seen_msr_vids: set[str] = set()
            for sent in msr_doc.get("sentences", []):
                v = str(sent.get("video_id") or "")
                cap = str(sent.get("caption") or "").strip()
                if v in indexed_videos and v not in seen_msr_vids and cap:
                    seen_msr_vids.add(v)
                    raw_texts.append(cap)

    _collect_from_split_dir(val_dir)
    _collect_from_split_dir(test_dir)

    # Expand sub-clauses deterministically so multi-clause VideoPrism queries hit cache cleanly
    clause_texts: list[str] = []
    for qt in raw_texts:
        for cl in decompose_query_clauses(qt):
            cl_s = cl.strip()
            if cl_s:
                clause_texts.append(cl_s)

    if llm_plan_cache_path is not None and llm_plan_cache_path.is_file():
        for rec in _read_jsonl(llm_plan_cache_path):
            q_orig = str(rec.get("original_query") or rec.get("query") or "").strip()
            if q_orig:
                clause_texts.append(q_orig)
            plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else rec
            for cl in (plan.get("visual_clauses") or []):
                if isinstance(cl, str) and cl.strip():
                    clause_texts.append(cl.strip())

    # Deduplicate while keeping both raw and NFC normalized variants cleanly
    seen_sha: set[str] = set()
    query_entries: list[dict[str, str]] = []
    for txt in [*raw_texts, *clause_texts]:
        for variant in (txt, unicodedata.normalize("NFC", txt.strip())):
            if not variant.strip():
                continue
            q_sha = sha256_text(variant)
            if q_sha not in seen_sha:
                seen_sha.add(q_sha)
                query_entries.append({
                    "query_id": f"q_{len(query_entries):05d}_{q_sha[:10]}",
                    "query_text": variant,
                    "query_text_sha256": q_sha,
                })

    assert_no_gt_leakage(media_entries, context="vp_media_manifest")
    assert_no_gt_leakage(subclip_entries, context="vp_subclip_manifest")
    assert_no_gt_leakage(query_entries, context="vp_query_manifest")
    return media_entries, subclip_entries, query_entries


_KAGGLE_RUNNER_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_vp")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda13-plugin", "jax-cuda13-pjrt"],
    check=False,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "jax[cuda12]==0.6.2",
        "flax==0.10.7",
        "tensorflow-cpu==2.20.0",
        "av==16.0.1",
        "faiss-cpu==1.13.2",
        "git+https://github.com/google-deepmind/videoprism.git@4f831e332b80d6e77bdd43882206f23aba2a67a7",
    ],
    check=True,
)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

from huggingface_hub import hf_hub_download
from scripts.kaggle_videoprism_worker import (
    JaxVideoPrismEncoder,
    NUM_CLIP_FRAMES,
    VIDEOPRISM_EMBEDDING_DIM,
    VIDEOPRISM_IMAGE_SIZE,
    normalize_rows,
    pinned_model_metadata,
    resize_and_pad_288,
    verify_videoprism_pins,
)
from scripts.prepare_clip_vp_kaggle_bundle import (
    assert_no_gt_leakage,
    select_bin_center_frames,
    sha256_json,
    sha256_text,
)
from videntra.evaluation.provenance import sha256_file
from videntra.ingestion.decode import decode_frames


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")


assets = TEMP / "assets"
assets.mkdir(exist_ok=True)
hf_hub_download(
    "google/videoprism-lvt-base-f16r288",
    "flax_lvt_base_f16r288_repeated.npz",
    revision="0119cca7d1fe61aa1d5718e1f71605703e1562e7",
    local_dir=assets,
)
urllib.request.urlretrieve(
    "https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
    assets / "sentencepiece.model",
)
pin_info = verify_videoprism_pins(assets, check_size=True)
print("Pins verified:", pin_info["checkpoint_sha256"], flush=True)

media_dir = TEMP / "media"
media_dir.mkdir(exist_ok=True)
for path in sorted(Path("/kaggle/input").rglob("*.mp4")):
    link = media_dir / path.name
    if not link.exists():
        link.symlink_to(path)

media_entries = _read_jsonl(TEMP / "media_manifest.jsonl")
subclip_entries = _read_jsonl(TEMP / "subclip_manifest.jsonl") if (TEMP / "subclip_manifest.jsonl").is_file() else []
query_entries = _read_jsonl(TEMP / "queries.jsonl")
assert_no_gt_leakage(media_entries, context="kaggle_media")
assert_no_gt_leakage(subclip_entries, context="kaggle_subclips")
assert_no_gt_leakage(query_entries, context="kaggle_queries")

print(
    f"Mounted mp4 count={len(list(media_dir.iterdir()))} | videos={len(media_entries)} | subclips={len(subclip_entries)} | queries={len(query_entries)}",
    flush=True,
)
missing_media = [e["video_id"] for e in media_entries if not (media_dir / e["relative_path"]).is_file()]
assert not missing_media, f"Missing mounted media files: {missing_media[:10]}"

# Group clips + subclips per video so each source .mp4 is scanned and decoded ONLY ONCE!
jobs_by_video = {}
for idx, e in enumerate(media_entries):
    vid = str(e["video_id"])
    jobs_by_video.setdefault(vid, []).append(("video", idx, e))
for idx, e in enumerate(subclip_entries):
    vid = str(e["video_id"])
    jobs_by_video.setdefault(vid, []).append(("subclip", idx, e))


def _decode_jobs_for_video(vid_and_jobs):
    vid, job_list = vid_and_jobs
    rel_path = str(job_list[0][2]["relative_path"])
    mpath = media_dir / rel_path
    # Pass 1: lightweight PTS scan in presentation order (no YUV->RGB conversion)
    records = []
    valid_ts = []
    for f in decode_frames(mpath, include_pixels=False):
        ts = f.timestamp_seconds
        records.append((f.ordinal, ts))
        if ts is not None:
            valid_ts.append(ts)
    if not valid_ts:
        raise ValueError(f"No valid PTS timestamps for {vid}")
    min_ts = min(valid_ts)
    max_ts_end = max(valid_ts) + 1e-3

    recipes = []
    needed_ordinals = set()
    for kind, idx, e in job_list:
        st = float(e.get("clip_start_sec", 0.0)) if e.get("clip_start_sec") is not None else min_ts
        et = float(e["clip_end_sec"]) if e.get("clip_end_sec") is not None else max_ts_end
        st = max(min_ts, st)
        if et <= st:
            et = st + 1e-3
        rec = select_bin_center_frames(records, st, et, num_frames=NUM_CLIP_FRAMES)
        if rec["status"] != "ok":
            raise ValueError(f"Recipe failed for {vid} ({kind}): {rec}")
        recipes.append((kind, idx, e, rec))
        needed_ordinals.update(int(o) for o in rec["selected_ordinals"])

    # Pass 2: selectively convert and resize ONLY needed ordinals to 288x288
    import av
    from videntra.ingestion.probe import probe_media
    pr = probe_media(mpath)
    decoded_288 = {}
    container = av.open(str(mpath))
    try:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        cur_ord = 0
        max_ord = max(needed_ordinals)
        try:
            for frame in container.decode(stream):
                if cur_ord in needed_ordinals:
                    rgb = frame.to_ndarray(format="rgb24")
                    decoded_288[cur_ord] = resize_and_pad_288(rgb, VIDEOPRISM_IMAGE_SIZE)
                if cur_ord >= max_ord:
                    break
                cur_ord += 1
        except Exception:
            if not needed_ordinals.issubset(decoded_288.keys()):
                raise
    finally:
        container.close()

    out_items = []
    for kind, idx, e, rec in recipes:
        clip_arr = np.stack([decoded_288[int(o)] for o in rec["selected_ordinals"]], axis=0)
        out_items.append((kind, idx, e, rec, clip_arr))
    return out_items


t_start = time.time()
encoder = JaxVideoPrismEncoder(assets, allow_cpu=False)
print("JaxVideoPrismEncoder ready on devices:", encoder.device_summary, flush=True)

gpu_devs = [d for d in encoder._jax.devices() if d.platform == "gpu"]
pmap_encode_fn = None
if len(gpu_devs) >= 2:
    try:
        rep_state = encoder._jax.device_put_replicated(encoder._state, gpu_devs)

        @encoder._jax.pmap
        def _pmap_video_forward(st, frames):
            return encoder._model.apply(st, frames, None, None, train=False)[0]

        def _pmap_encode(clip_arr):
            raw = np.asarray(encoder._jax.device_get(_pmap_video_forward(rep_state, clip_arr[:, None, ...])))
            raw = raw.reshape(clip_arr.shape[0], -1)
            return normalize_rows(raw)

        # Warm-check pmap with 1 dummy step so any multi-GPU quirk falls back safely before loop
        _ = _pmap_encode(np.zeros((len(gpu_devs), NUM_CLIP_FRAMES, VIDEOPRISM_IMAGE_SIZE, VIDEOPRISM_IMAGE_SIZE, 3), dtype=np.float32))
        pmap_encode_fn = _pmap_encode
        print(f"Using jax.pmap across {len(gpu_devs)} GPUs (per-GPU batch=1)", flush=True)
    except Exception as exc:
        print(f"pmap fallback to single-GPU batch=1: {exc}", flush=True)
        pmap_encode_fn = None

VIDEO_BS = len(gpu_devs) if pmap_encode_fn is not None else 1
QUERY_BS = 32

video_vecs_out = np.zeros((len(media_entries), VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)
video_rows_out = [None] * len(media_entries)
subclip_vecs_out = np.zeros((len(subclip_entries), VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)
subclip_rows_out = [None] * len(subclip_entries)

pending = []


def _flush_batch():
    if not pending:
        return
    n_real = len(pending)
    if n_real < VIDEO_BS:
        pad = [pending[-1][4]] * (VIDEO_BS - n_real)
        batch_arr = np.stack([p[4] for p in pending] + pad, axis=0)
    else:
        batch_arr = np.stack([p[4] for p in pending], axis=0)
    if pmap_encode_fn is not None:
        try:
            vecs = pmap_encode_fn(batch_arr)[:n_real]
        except Exception:
            vecs = np.vstack([encoder.encode_videos(p[4][None, ...]) for p in pending])
    else:
        vecs = encoder.encode_videos(batch_arr)[:n_real]
    for k in range(n_real):
        kind, idx, e, rec, _ = pending[k]
        v = vecs[k]
        if kind == "video":
            video_vecs_out[idx] = v
            video_rows_out[idx] = {
                "row_id": idx,
                "video_id": str(e["video_id"]),
                "selected_ordinals": rec["selected_ordinals"],
                "selected_timestamps_sec": rec["selected_timestamps_sec"],
                "recipe_hash": rec["recipe_hash"],
            }
        else:
            subclip_vecs_out[idx] = v
            subclip_rows_out[idx] = {
                "row_id": idx,
                "subclip_id": str(e["subclip_id"]),
                "video_id": str(e["video_id"]),
                "clip_start_sec": float(e["clip_start_sec"]),
                "clip_end_sec": float(e["clip_end_sec"]),
                "selected_ordinals": rec["selected_ordinals"],
                "selected_timestamps_sec": rec["selected_timestamps_sec"],
                "recipe_hash": rec["recipe_hash"],
            }
    pending.clear()


vid_items = list(jobs_by_video.items())
done_vids = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    for chunk_start in range(0, len(vid_items), 16):
        chunk_items = vid_items[chunk_start : chunk_start + 16]
        for vid_results in pool.map(_decode_jobs_for_video, chunk_items):
            for item in vid_results:
                pending.append(item)
                if len(pending) >= VIDEO_BS:
                    _flush_batch()
            done_vids += 1
            if done_vids % 200 == 0 or done_vids == len(vid_items):
                print(f"Encoded {done_vids}/{len(vid_items)} source videos in {time.time() - t_start:.1f}s", flush=True)
_flush_batch()

# Encode queries in fixed QUERY_BS=64 batches
query_vecs_out = np.zeros((len(query_entries), VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)
query_rows_out = []
for start in range(0, len(query_entries), QUERY_BS):
    chunk = query_entries[start : start + QUERY_BS]
    texts = [q["query_text"] for q in chunk]
    n_real = len(texts)
    if n_real < QUERY_BS:
        texts_padded = texts + [texts[-1]] * (QUERY_BS - n_real)
    else:
        texts_padded = texts
    q_vecs = encoder.encode_texts(texts_padded)[:n_real]
    query_vecs_out[start : start + n_real] = q_vecs
    for j, q in enumerate(chunk):
        query_rows_out.append({
            "row_id": start + j,
            "query_id": q["query_id"],
            "query_text_sha256": q["query_text_sha256"],
        })

assert all(r is not None for r in video_rows_out), "Missing video rows"
assert all(r is not None for r in subclip_rows_out), "Missing subclip rows"
assert np.isfinite(video_vecs_out).all() and np.allclose(np.linalg.norm(video_vecs_out, axis=1), 1.0, atol=1e-4)
if len(subclip_entries) > 0:
    assert np.isfinite(subclip_vecs_out).all() and np.allclose(np.linalg.norm(subclip_vecs_out, axis=1), 1.0, atol=1e-4)
assert np.isfinite(query_vecs_out).all() and np.allclose(np.linalg.norm(query_vecs_out, axis=1), 1.0, atol=1e-4)

out_dir = WORK / "full_1324"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "video_vectors.npy", video_vecs_out)
_write_jsonl(out_dir / "video_rows.jsonl", video_rows_out)
np.save(out_dir / "subclip_vectors.npy", subclip_vecs_out)
_write_jsonl(out_dir / "subclip_rows.jsonl", subclip_rows_out)
np.save(out_dir / "query_vectors.npy", query_vecs_out)
_write_jsonl(out_dir / "query_rows.jsonl", query_rows_out)

artifact_names = (
    "video_vectors.npy",
    "video_rows.jsonl",
    "subclip_vectors.npy",
    "subclip_rows.jsonl",
    "query_vectors.npy",
    "query_rows.jsonl",
)
files_info = {
    fname: {
        "sha256": sha256_file(out_dir / fname),
        "bytes": (out_dir / fname).stat().st_size,
    }
    for fname in artifact_names
}
receipt = {
    "status": "SUCCESS",
    "dry_run": False,
    "model": pinned_model_metadata(),
    "pins": pin_info,
    "embedding_dim": VIDEOPRISM_EMBEDDING_DIM,
    "dtype": "float32",
    "video_count": len(video_rows_out),
    "subclip_count": len(subclip_rows_out),
    "query_count": len(query_rows_out),
    "files": files_info,
    "errors": [],
    "elapsed_seconds": round(time.time() - t_start, 2),
    "runtime": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": encoder.device_summary,
    },
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE! Elapsed:", receipt["elapsed_seconds"], "s", flush=True)
'''


def prepare_kaggle_bundle(
    *,
    out_dir: Path = Path("work/kaggle/videoprism_full_1324"),
    db_path: Path = Path("data/processed/metadata.db"),
    val_dir: Path = Path("validation"),
    test_dir: Path = Path("test_queries"),
    kernel_id: str = DEFAULT_KAGGLE_SLUG,
) -> dict[str, Any]:
    """Build self-contained Kaggle GPU launcher script + kernel-metadata.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    media_entries, subclip_entries, query_entries = collect_corpus_manifests(
        db_path=db_path,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_dir in ("src/videntra",):
            base = _ROOT / rel_dir
            for f in sorted(base.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    zf.write(f, arcname=f.relative_to(_ROOT).as_posix())
        for script_rel in (
            "scripts/__init__.py",
            "scripts/kaggle_videoprism_worker.py",
            "scripts/prepare_clip_vp_kaggle_bundle.py",
        ):
            sp = _ROOT / script_rel
            if sp.is_file():
                zf.write(sp, arcname=script_rel)

        zf.writestr(
            "media_manifest.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in media_entries) + "\n",
        )
        zf.writestr(
            "subclip_manifest.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in subclip_entries) + "\n",
        )
        zf.writestr(
            "queries.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in query_entries) + "\n",
        )

    b64_source = base64.b64encode(buf.getvalue()).decode("ascii")
    launcher_code = f"SOURCE = {b64_source!r}\n" + _KAGGLE_RUNNER_TEMPLATE.lstrip("\n")
    (out / "launcher.py").write_text(launcher_code, encoding="utf-8")

    meta = {
        "id": kernel_id,
        "title": "VIDENTRA VideoPrism Full 1324",
        "code_file": "launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(DEFAULT_DATASETS),
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out),
        "kernel_id": kernel_id,
        "video_count": len(media_entries),
        "subclip_count": len(subclip_entries),
        "query_count": len(query_entries),
        "launcher_bytes": (out / "launcher.py").stat().st_size,
    }


def _atomic_write_bytes(target: Path, writer_fn) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        writer_fn(tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def ingest_kaggle_output(
    *,
    kaggle_out_dir: Path,
    processed_dir: Path = Path("data/processed"),
    cache_path: Path = Path("data/cache/query_videoprism_768d.npz"),
    expected_video_count: int = 1324,
) -> dict[str, Any]:
    """Verify SHA-256 Kaggle VideoPrism receipt and build local FAISS index & query cache."""
    src = Path(kaggle_out_dir)
    if (src / "full_1324").is_dir():
        src = src / "full_1324"

    receipt_path = src / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing receipt.json in {src}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "SUCCESS" or receipt.get("dry_run") is not False:
        raise ValueError(f"Invalid VideoPrism receipt status/dry_run: {receipt.get('status')}, dry_run={receipt.get('dry_run')}")
    if receipt.get("errors"):
        raise ValueError(f"VideoPrism extraction had errors: {receipt['errors']}")

    files_info = receipt.get("files") or {}
    for fname, info in files_info.items():
        fpath = src / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing VideoPrism artifact: {fpath}")
        actual_sha = sha256_file(fpath)
        if actual_sha != info.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {fname}: expected {info.get('sha256')}, got {actual_sha}")

    v_vecs = np.asarray(np.load(str(src / "video_vectors.npy")), dtype=np.float32)
    v_rows = _read_jsonl(src / "video_rows.jsonl")
    if len(v_rows) != expected_video_count or v_vecs.shape != (expected_video_count, VIDEOPRISM_EMBEDDING_DIM):
        raise ValueError(
            f"Expected {expected_video_count} x {VIDEOPRISM_EMBEDDING_DIM} video vectors, "
            f"got rows={len(v_rows)}, vecs={v_vecs.shape}"
        )
    if not np.isfinite(v_vecs).all() or not np.allclose(np.linalg.norm(v_vecs, axis=1), 1.0, atol=1e-4):
        raise ValueError("Video vectors failed finiteness / unit-norm check")

    s_vecs = np.asarray(np.load(str(src / "subclip_vectors.npy")), dtype=np.float32)
    s_rows = _read_jsonl(src / "subclip_rows.jsonl")
    if len(s_rows) != s_vecs.shape[0] or (len(s_rows) > 0 and s_vecs.shape[1] != VIDEOPRISM_EMBEDDING_DIM):
        raise ValueError(f"Subclip shape mismatch: rows={len(s_rows)}, vecs={s_vecs.shape}")
    if len(s_rows) > 0 and (not np.isfinite(s_vecs).all() or not np.allclose(np.linalg.norm(s_vecs, axis=1), 1.0, atol=1e-4)):
        raise ValueError("Subclip vectors failed finiteness / unit-norm check")

    q_vecs = np.asarray(np.load(str(src / "query_vectors.npy")), dtype=np.float32)
    q_rows = _read_jsonl(src / "query_rows.jsonl")
    if len(q_rows) != q_vecs.shape[0] or (len(q_rows) > 0 and q_vecs.shape[1] != VIDEOPRISM_EMBEDDING_DIM):
        raise ValueError(f"Query shape mismatch: rows={len(q_rows)}, vecs={q_vecs.shape}")
    if len(q_rows) > 0 and (not np.isfinite(q_vecs).all() or not np.allclose(np.linalg.norm(q_vecs, axis=1), 1.0, atol=1e-4)):
        raise ValueError("Query vectors failed finiteness / unit-norm check")

    proc = Path(processed_dir)
    proc.mkdir(parents=True, exist_ok=True)

    # 1. Build VideoPrism Whole-Clip FAISS Index (768d)
    index = faiss.IndexFlatIP(VIDEOPRISM_EMBEDDING_DIM)
    index.add(np.ascontiguousarray(v_vecs, dtype=np.float32))
    faiss_out = proc / "faiss_videoprism_clip_768.bin"
    _atomic_write_bytes(faiss_out, lambda p: faiss.write_index(index, str(p)))

    clip_manifest = {
        "schema_version": 1,
        "model": receipt.get("model", pinned_model_metadata()),
        "embedding_dim": VIDEOPRISM_EMBEDDING_DIM,
        "video_count": len(v_rows),
        "video_ids": [str(r["video_id"]) for r in v_rows],
        "rows": v_rows,
        "source_receipt_sha256": sha256_file(receipt_path),
    }
    manifest_out = proc / "videoprism_clip_manifest.json"
    _atomic_write_bytes(manifest_out, lambda p: p.write_text(json.dumps(clip_manifest, indent=2), encoding="utf-8"))

    # 2. Save Sub-clip Vectors + Map
    subclip_npy_out = proc / "videoprism_subclips_768.npy"
    subclip_map_out = proc / "videoprism_subclips_map.jsonl"
    def _save_npy(p: Path) -> None:
        with p.open("wb") as fh:
            np.save(fh, np.ascontiguousarray(s_vecs, dtype=np.float32))

    _atomic_write_bytes(subclip_npy_out, _save_npy)
    _atomic_write_bytes(
        subclip_map_out,
        lambda p: p.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in s_rows) + ("\n" if s_rows else ""),
            encoding="utf-8",
        ),
    )

    # 3. Save Query Cache keyed strictly by query_text_sha256 + model_revision
    keys_arr = np.array([str(r["query_text_sha256"]) for r in q_rows])

    def _save_npz(p: Path) -> None:
        with p.open("wb") as fh:
            np.savez(
                fh,
                keys=keys_arr,
                vectors=np.ascontiguousarray(q_vecs, dtype=np.float32),
                model_revision=np.array([VIDEOPRISM_HF_REVISION]),
                model_commit=np.array([VIDEOPRISM_COMMIT]),
            )

    _atomic_write_bytes(Path(cache_path), _save_npz)

    # Preserve provenance receipt copy in work/experiments
    exp_receipt = Path("work/experiments/videoprism_full_1324_receipt.json")
    exp_receipt.parent.mkdir(parents=True, exist_ok=True)
    exp_receipt.write_text(receipt_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "faiss_index": str(faiss_out),
        "clip_manifest": str(manifest_out),
        "subclips_npy": str(subclip_npy_out),
        "subclips_map": str(subclip_map_out),
        "query_cache": str(cache_path),
        "video_count": len(v_rows),
        "subclip_count": len(s_rows),
        "query_count": len(q_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build & ingest VideoPrism-LvT-B index via Kaggle GPU.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-kaggle", action="store_true")
    group.add_argument("--ingest-kaggle-output", action="store_true")

    parser.add_argument("--out-dir", type=Path, default=Path("work/kaggle/videoprism_full_1324"))
    parser.add_argument("--kaggle-out-dir", type=Path, default=Path("work/kaggle/videoprism_full_1324_out"))
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_queries"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-path", type=Path, default=Path("data/cache/query_videoprism_768d.npz"))
    parser.add_argument("--kernel-id", type=str, default=DEFAULT_KAGGLE_SLUG)
    args = parser.parse_args(argv)

    if args.prepare_kaggle:
        info = prepare_kaggle_bundle(
            out_dir=args.out_dir,
            db_path=args.db_path,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            kernel_id=args.kernel_id,
        )
    else:
        info = ingest_kaggle_output(
            kaggle_out_dir=args.kaggle_out_dir,
            processed_dir=args.processed_dir,
            cache_path=args.cache_path,
        )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
