"""Orchestrate Kaggle GPU extraction jobs for the MSR-VTT 1K-A benchmark matrix.

Supports:
  pilot-vp   Run 16-clip + 8-query VideoPrism pilot & inventory all 9,585 MSR-VTT videos on Kaggle.
  vp-shard   Run VideoPrism-LvT-B whole + multi-window + query encoding for a split shard.
  mm-shard   Run unified PyAV decode -> hybrid_v2_adaptive keyframes, SigLIP2, YOLOv8n + SORT/ByteTrack, RapidOCR.
  asr-shard  Run Faster-Whisper small + Silero VAD ASR extraction (in-memory PyAV audio, dual-GPU).
  cap-shard  Run Florence-2-base multi-frame offline captioning + multilingual-e5-small encoding (dual-GPU).
  vl2-shard  Run pinned PE-Core-L-14-336 + SigLIP2-SO400M-384 12-frame & query extraction (dual-GPU).
  status     Check Kaggle kernel status.
  pull       Download and verify SHA-256 receipts from Kaggle kernel output.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.prepare_clip_vp_kaggle_bundle import assert_no_gt_leakage, sha256_text  # noqa: E402
from videntra.evaluation.provenance import sha256_file  # noqa: E402

PROTOCOL_ROOT = Path("work/msrvtt_protocol_20261122")
KAGGLE_WORK_DIR = PROTOCOL_ROOT / "kaggle_kernels"
KAGGLE_OUT_DIR = PROTOCOL_ROOT / "kaggle_out"

MSRVTT_KAGGLE_DATASETS = (
    "lvhuylong/msrvtt-1ka-original-videos-v1",
    "khoahunhtngng/msrvtt",
    "vishnutheepb/msrvtt",
)

def _collect_minimal_src_files() -> tuple[str, ...]:
    """Collect all src/videntra/**/*.py except heavy server/UI/pipeline-SSOT modules."""
    excluded = {
        "src/videntra/retrieval/unified_pipeline.py",
        "src/videntra/retrieval/msrvtt_matrix.py",
        "src/videntra/ranking/vedje_reranker.py",
        "src/videntra/cli.py",
    }
    files: list[str] = []
    for f in sorted((_ROOT / "src" / "videntra").rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        rel = f.relative_to(_ROOT).as_posix()
        if rel.startswith("src/videntra/api/") or rel in excluded:
            continue
        files.append(rel)
    return tuple(files)


_MINIMAL_SRC_FILES: tuple[str, ...] = _collect_minimal_src_files()

_VP_EXTRA_SCRIPTS: tuple[str, ...] = (
    "scripts/kaggle_videoprism_worker.py",
    "scripts/prepare_clip_vp_kaggle_bundle.py",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _pack_source_b64(
    src_rel_files: Sequence[str],
    payload_files: dict[str, bytes],
) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        added: set[str] = set()
        zf.writestr("scripts/__init__.py", b"")
        added.add("scripts/__init__.py")
        for rel in src_rel_files:
            if rel in added:
                continue
            fp = _ROOT / rel
            if fp.is_file():
                zf.write(fp, arcname=rel)
                added.add(rel)
        for name, data in payload_files.items():
            zf.writestr(name, data)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_and_push_kernel(
    *,
    kernel_dir: Path,
    kernel_id: str,
    title: str,
    launcher_code: str,
    datasets: Sequence[str],
    push: bool,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    acc = str(accelerator).lower()
    if acc not in ("gpu", "tpu", "cpu"):
        raise ValueError(f"Unsupported accelerator: {accelerator}")
    size_bytes = len(launcher_code.encode("utf-8"))
    if size_bytes >= 980_000:
        raise ValueError(f"launcher.py ({size_bytes} B) approaches Kaggle 1 MB limit")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "launcher.py").write_text(launcher_code, encoding="utf-8")
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": "launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": acc == "gpu",
        "enable_tpu": acc == "tpu",
        "enable_internet": True,
        "keywords": ["gpu"] if acc == "gpu" else (["tpu"] if acc == "tpu" else []),
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [],
    }
    if acc == "tpu":
        meta["machine_shape"] = "TpuV5E8"
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    push_out = None
    if push:
        push_out = subprocess.check_output(
            [str(_ROOT / ".venv/Scripts/kaggle.exe"), "kernels", "push", "-p", str(kernel_dir)],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    return {
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "accelerator": acc,
        "launcher_bytes": size_bytes,
        "pushed": bool(push),
        "push_output": push_out,
    }


def _select_stratified_train_sub(train_rows: list[dict[str, Any]], max_videos: int = 1600) -> list[dict[str, Any]]:
    """Deterministically pick a category-stratified subset of train_9k_clean."""
    if len(train_rows) <= max_videos:
        return train_rows
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_cat[int(row["category"])].append(row)
    selected: list[dict[str, Any]] = []
    idx = 0
    cats = sorted(by_cat.keys())
    while len(selected) < max_videos:
        added_any = False
        for c in cats:
            if idx < len(by_cat[c]):
                selected.append(by_cat[c][idx])
                added_any = True
                if len(selected) >= max_videos:
                    break
        if not added_any:
            break
        idx += 1
    selected.sort(key=lambda r: r["video_id"])
    return selected


def _collect_target_payload(
    protocol_root: Path,
    target: str,
    *,
    allow_sealed_test: bool = False,
    include_1ka_gallery_videos_only: bool = False,
) -> tuple[list[str], list[str], bool]:
    """Return (unique_video_ids, unique_base_queries, expand_prompt_variants).

    Note: test_1ka queries are strictly blocked unless allow_sealed_test=True.
    When include_1ka_gallery_videos_only=True, test_1ka video IDs are encoded
    purely as gallery items with zero test query access.
    """
    m_dir = protocol_root / "manifests"
    vids: list[str] = []
    base_queries: list[str] = []
    expand_variants = True

    if target == "val_tune":
        for g in ("tune_g1", "tune_g2"):
            vids.extend(r["video_id"] for r in _read_jsonl(m_dir / g / "gallery_manifest.jsonl"))
            base_queries.extend(r["query_text"] for r in _read_jsonl(m_dir / g / "queries.jsonl"))
    elif target == "val_confirm":
        for g in ("confirm_g1", "confirm_g2"):
            vids.extend(r["video_id"] for r in _read_jsonl(m_dir / g / "gallery_manifest.jsonl"))
            base_queries.extend(r["query_text"] for r in _read_jsonl(m_dir / g / "queries.jsonl"))
    elif target == "val_4000":
        for g in ("tune_g1", "tune_g2", "confirm_g1", "confirm_g2"):
            vids.extend(r["video_id"] for r in _read_jsonl(m_dir / g / "gallery_manifest.jsonl"))
            base_queries.extend(r["query_text"] for r in _read_jsonl(m_dir / g / "queries.jsonl"))
    elif target == "train_sub":
        all_train = _read_jsonl(m_dir / "train_9k_clean" / "train_videos.jsonl")
        sub = _select_stratified_train_sub(all_train, max_videos=1600)
        vids.extend(r["video_id"] for r in sub)
        for r in sub:
            for cap in r["captions"][:2]:
                base_queries.append(str(cap["caption"]))
        # Also include 1 caption per remaining train video as Train-9K Query Bank for QB-CSLS/DSN
        sub_vids = {r["video_id"] for r in sub}
        for r in all_train:
            if r["video_id"] not in sub_vids and r["captions"]:
                base_queries.append(str(r["captions"][0]["caption"]))
        expand_variants = False
    elif target == "train_qbank_plus_1ka_videos":
        all_train = _read_jsonl(m_dir / "train_9k_clean" / "train_videos.jsonl")
        vids.extend(r["video_id"] for r in _read_jsonl(m_dir / "test_1ka" / "gallery_manifest.jsonl"))
        for r in all_train:
            if r["captions"]:
                base_queries.append(str(r["captions"][0]["caption"]))
        expand_variants = False
    elif target == "test_1ka_locked":
        if not allow_sealed_test:
            raise PermissionError(
                "Refusing to bundle sealed test_1ka queries without --allow-sealed-test (unlock only after champion lock)"
            )
        vids.extend(r["video_id"] for r in _read_jsonl(m_dir / "test_1ka" / "gallery_manifest.jsonl"))
        vault = protocol_root / "eval_only" / "test_1ka_queries_vault.jsonl"
        base_queries.extend(r["query_text"] for r in _read_jsonl(vault))
    elif target == "test_1ka_videos_only":
        vids.extend(r["video_id"] for r in _read_jsonl(m_dir / "test_1ka" / "gallery_manifest.jsonl"))
        expand_variants = False
    elif target == "test_1ka_queries_only":
        if not allow_sealed_test:
            raise PermissionError(
                "Refusing to bundle sealed test_1ka queries without --allow-sealed-test (unlock only after champion lock)"
            )
        vault = protocol_root / "eval_only" / "test_1ka_queries_vault.jsonl"
        base_queries.extend(r["query_text"] for r in _read_jsonl(vault))
    else:
        raise ValueError(f"Unknown target '{target}'")

    if include_1ka_gallery_videos_only:
        vids.extend(r["video_id"] for r in _read_jsonl(m_dir / "test_1ka" / "gallery_manifest.jsonl"))

    vids_unique = list(dict.fromkeys(str(v) for v in vids))
    queries_unique = list(dict.fromkeys(str(q).strip() for q in base_queries if str(q).strip()))
    return vids_unique, queries_unique, expand_variants


_SHARED_QUERY_EXPANDER_CODE = r'''
def expand_and_dedup_queries(base_queries, expand_variants=True):
    from videntra.query.rules import decompose_query_clauses
    seen = set()
    rows = []
    for qt in base_queries:
        cands = [qt]
        if expand_variants:
            cands.extend([
                f"a video clip showing {qt}",
                f"a video frame showing {qt}",
                f"a video of {qt}",
            ])
            for cl in decompose_query_clauses(qt):
                if cl.strip():
                    cands.append(cl.strip())
        for txt in cands:
            s = txt.strip()
            if not s:
                continue
            q_sha = hashlib.sha256(s.encode("utf-8")).hexdigest()
            if q_sha not in seen:
                seen.add(q_sha)
                rows.append({
                    "row_id": len(rows),
                    "query_text": s,
                    "query_text_sha256": q_sha,
                })
    return rows
'''


_PILOT_VP_TEMPLATE = r'''
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_vp_pilot")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

import shutil
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
_has_nvidia = shutil.which("nvidia-smi") is not None
if not _has_nvidia:
    os.environ["JAX_PLATFORMS"] = "cpu"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda13-plugin", "jax-cuda13-pjrt"], check=False)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "jax[cuda12]==0.6.2" if _has_nvidia else "jax==0.6.2",
        "flax==0.10.7", "tensorflow-cpu==2.20.0",
        "av==16.0.1", "faiss-cpu==1.13.2",
        "git+https://github.com/google-deepmind/videoprism.git@4f831e332b80d6e77bdd43882206f23aba2a67a7",
    ],
    check=True,
)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

from huggingface_hub import hf_hub_download
from scripts.kaggle_videoprism_worker import (
    JaxVideoPrismEncoder, NUM_CLIP_FRAMES, VIDEOPRISM_EMBEDDING_DIM,
    VIDEOPRISM_IMAGE_SIZE, pinned_model_metadata,
    resize_and_pad_288, verify_videoprism_pins,
)
from scripts.prepare_clip_vp_kaggle_bundle import select_bin_center_frames
from videntra.evaluation.provenance import sha256_file
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import probe_media

t_start = time.time()
with gzip.open(TEMP / "inventory_spec.json.gz", "rt", encoding="utf-8") as fh:
    spec = json.load(fh)

def _prio(p: Path) -> int:
    s = p.as_posix().lower()
    if "msrvtt-1ka" in s:
        return 0
    if "khoahunhtngng" in s or "testvideo" in s:
        return 1
    return 2

mp4_candidates = {}
for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
    mp4_candidates.setdefault(p.stem, []).append(p)
mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}

inv_summary = {}
for sname, vlist in spec["expected_by_split"].items():
    found = [v for v in vlist if v in mp4_by_vid]
    missing = [v for v in vlist if v not in mp4_by_vid]
    inv_summary[sname] = {
        "expected": len(vlist),
        "found": len(found),
        "missing_count": len(missing),
        "missing_sample": missing[:10],
    }

cross_sha_checks = []
for v in sorted(spec["expected_by_split"]["test_1ka"])[:5]:
    paths = sorted(mp4_candidates.get(v, []), key=_prio)
    if len(paths) >= 2:
        shas = [{"path": str(x), "bytes": x.stat().st_size, "sha256": sha256_file(x)} for x in paths]
        cross_sha_checks.append({"video_id": v, "variants": shas})

print("INVENTORY:", json.dumps(inv_summary), flush=True)

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
encoder = JaxVideoPrismEncoder(assets, allow_cpu=False)

pilot_vids = spec["pilot_videos"]
pilot_queries = spec["pilot_queries"]

def _decode_whole_and_windows(vid):
    import av
    mpath = mp4_by_vid[vid]
    records, valid_ts = [], []
    for f in decode_frames(mpath, include_pixels=False):
        ts = f.timestamp_seconds
        records.append((f.ordinal, ts))
        if ts is not None:
            valid_ts.append(ts)
    min_ts, max_ts = min(valid_ts), max(valid_ts) + 1e-3
    dur = max(0.5, max_ts - min_ts)
    if dur < 18.0:
        bounds = [("whole", min_ts, max_ts), ("w0", min_ts, min_ts + 0.65 * dur), ("w1", min_ts + 0.35 * dur, max_ts)]
    else:
        bounds = [("whole", min_ts, max_ts), ("w0", min_ts, min_ts + 0.55 * dur),
                  ("w1", min_ts + 0.225 * dur, min_ts + 0.775 * dur), ("w2", min_ts + 0.45 * dur, max_ts)]
    recipes, needed = [], set()
    for tag, st, et in bounds:
        rec = select_bin_center_frames(records, st, max(st + 1e-3, et), num_frames=NUM_CLIP_FRAMES)
        assert rec["status"] == "ok", f"Recipe failed {vid} {tag}: {rec}"
        recipes.append((tag, rec))
        needed.update(int(o) for o in rec["selected_ordinals"])
    pr = probe_media(mpath)
    dec_288 = {}
    with av.open(str(mpath)) as container:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        cur_ord, max_ord = 0, max(needed)
        for frame in container.decode(stream):
            if cur_ord in needed:
                dec_288[cur_ord] = resize_and_pad_288(frame.to_ndarray(format="rgb24"), VIDEOPRISM_IMAGE_SIZE)
            if cur_ord >= max_ord:
                break
            cur_ord += 1
    return [(vid, tag, np.stack([dec_288[int(o)] for o in rec["selected_ordinals"]], axis=0)) for tag, rec in recipes]

clip_items = []
for v in pilot_vids:
    clip_items.extend(_decode_whole_and_windows(v))

t_enc0 = time.time()
gpu_devs = [d for d in encoder._jax.devices() if d.platform == "gpu"]
pmap_encode_fn = None
if len(gpu_devs) >= 2:
    try:
        rep_state = encoder._jax.device_put_replicated(encoder._state, gpu_devs)

        @encoder._jax.pmap
        def _pmap_video_forward(st, frames):
            return encoder._model.apply(st, frames, None, None, train=False)[0]

        def _pmap_encode(clip_arr):
            n_g = len(gpu_devs)
            bs = clip_arr.shape[0]
            assert bs % n_g == 0
            sharded = clip_arr.reshape(n_g, bs // n_g, NUM_CLIP_FRAMES, VIDEOPRISM_IMAGE_SIZE, VIDEOPRISM_IMAGE_SIZE, 3)
            raw = np.asarray(encoder._jax.device_get(_pmap_video_forward(rep_state, sharded)))
            norms = np.linalg.norm(raw.reshape(bs, -1), axis=-1, keepdims=True)
            return (raw.reshape(bs, -1) / np.maximum(norms, 1e-8)).astype(np.float32)

        _ = _pmap_encode(np.zeros((len(gpu_devs), NUM_CLIP_FRAMES, VIDEOPRISM_IMAGE_SIZE, VIDEOPRISM_IMAGE_SIZE, 3), dtype=np.float32))
        pmap_encode_fn = _pmap_encode
    except Exception:
        pmap_encode_fn = None

vecs_list = []
v_step = len(gpu_devs) if pmap_encode_fn is not None else 1
for i in range(0, len(clip_items), v_step):
    chunk = clip_items[i : i + v_step]
    batch = np.stack([c[2] for c in chunk], axis=0)
    if pmap_encode_fn is not None and batch.shape[0] == len(gpu_devs):
        v_out = pmap_encode_fn(batch)
    else:
        v_out = np.vstack([encoder.encode_videos(c[2][None, ...]) for c in chunk])
    vecs_list.append(v_out)
clip_vecs = np.vstack(vecs_list).astype(np.float32)
q_vecs = encoder.encode_texts([q["query_text"] for q in pilot_queries]).astype(np.float32)
enc_sec = time.time() - t_enc0

assert np.isfinite(clip_vecs).all() and np.allclose(np.linalg.norm(clip_vecs, axis=1), 1.0, atol=1e-4)
assert np.isfinite(q_vecs).all() and np.allclose(np.linalg.norm(q_vecs, axis=1), 1.0, atol=1e-4)

out_dir = WORK / "vp_pilot_out"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "pilot_clip_vectors.npy", clip_vecs)
np.save(out_dir / "pilot_query_vectors.npy", q_vecs)
(out_dir / "inventory.json").write_text(json.dumps({
    "total_discovered_mp4": len(mp4_by_vid),
    "splits": inv_summary,
    "cross_dataset_1ka_samples": cross_sha_checks,
}, indent=2), encoding="utf-8")

receipt = {
    "status": "SUCCESS",
    "dry_run": False,
    "model": pinned_model_metadata(),
    "pins": pin_info,
    "pilot_video_count": len(pilot_vids),
    "pilot_clip_windows_count": len(clip_items),
    "pilot_query_count": len(pilot_queries),
    "encode_seconds": round(enc_sec, 2),
    "elapsed_seconds": round(time.time() - t_start, 2),
    "devices": encoder.device_summary,
    "inventory": inv_summary,
    "files": {
        "pilot_clip_vectors.npy": {"sha256": sha256_file(out_dir / "pilot_clip_vectors.npy")},
        "pilot_query_vectors.npy": {"sha256": sha256_file(out_dir / "pilot_query_vectors.npy")},
        "inventory.json": {"sha256": sha256_file(out_dir / "inventory.json")},
    },
}
(out_dir / "pilot_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("PILOT DONE:", json.dumps(receipt), flush=True)
'''


_VP_SHARD_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_vp_shard")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

import shutil
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
_has_nvidia = shutil.which("nvidia-smi") is not None
if not _has_nvidia:
    os.environ["JAX_PLATFORMS"] = "cpu"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda13-plugin", "jax-cuda13-pjrt"], check=False)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "jax[cuda12]==0.6.2" if _has_nvidia else "jax==0.6.2",
        "flax==0.10.7", "tensorflow-cpu==2.20.0",
        "av==16.0.1", "faiss-cpu==1.13.2",
        "git+https://github.com/google-deepmind/videoprism.git@4f831e332b80d6e77bdd43882206f23aba2a67a7",
    ],
    check=True,
)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

from huggingface_hub import hf_hub_download
from scripts.kaggle_videoprism_worker import (
    JaxVideoPrismEncoder, NUM_CLIP_FRAMES, VIDEOPRISM_EMBEDDING_DIM,
    VIDEOPRISM_IMAGE_SIZE, normalize_rows, pinned_model_metadata,
    resize_and_pad_288, verify_videoprism_pins,
)
from scripts.prepare_clip_vp_kaggle_bundle import select_bin_center_frames
from videntra.evaluation.provenance import sha256_file
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import probe_media
''' + _SHARED_QUERY_EXPANDER_CODE + r'''
t_start = time.time()
with gzip.open(TEMP / "job_payload.json.gz", "rt", encoding="utf-8") as fh:
    payload = json.load(fh)

shard_name = payload["shard_name"]
video_ids = [str(v) for v in payload["video_ids"]]
query_rows_in = expand_and_dedup_queries(payload["base_queries"], payload.get("expand_variants", True))

def _prio(p: Path) -> int:
    s = p.as_posix().lower()
    if "msrvtt-1ka" in s:
        return 0
    if "khoahunhtngng" in s or "testvideo" in s:
        return 1
    return 2

mp4_by_vid = {}
if video_ids:
    mp4_candidates = {}
    for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
        mp4_candidates.setdefault(p.stem, []).append(p)
    mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}
missing = [v for v in video_ids if v not in mp4_by_vid]
assert not missing, f"Missing {len(missing)} videos in {shard_name}: {missing[:10]}"

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
encoder = JaxVideoPrismEncoder(assets, allow_cpu=(len(video_ids) == 0))

gpu_devs = [d for d in encoder._jax.devices() if d.platform == "gpu"]
PER_GPU_BS = 1
pmap_encode_fn = None
if len(gpu_devs) >= 2:
    try:
        rep_state = encoder._jax.device_put_replicated(encoder._state, gpu_devs)

        @encoder._jax.pmap
        def _pmap_video_forward(st, frames):
            return encoder._model.apply(st, frames, None, None, train=False)[0]

        def _pmap_encode(clip_arr):
            n_g = len(gpu_devs)
            bs = clip_arr.shape[0]
            assert bs % n_g == 0
            sharded = clip_arr.reshape(n_g, bs // n_g, NUM_CLIP_FRAMES, VIDEOPRISM_IMAGE_SIZE, VIDEOPRISM_IMAGE_SIZE, 3)
            raw = np.asarray(encoder._jax.device_get(_pmap_video_forward(rep_state, sharded)))
            return normalize_rows(raw.reshape(bs, -1))

        _ = _pmap_encode(np.zeros((len(gpu_devs) * PER_GPU_BS, NUM_CLIP_FRAMES, VIDEOPRISM_IMAGE_SIZE, VIDEOPRISM_IMAGE_SIZE, 3), dtype=np.float32))
        pmap_encode_fn = _pmap_encode
        print(f"Using jax.pmap across {len(gpu_devs)} GPUs with per_gpu_batch={PER_GPU_BS}", flush=True)
    except Exception as exc:
        print("pmap batch-1 fallback:", exc, flush=True)
        pmap_encode_fn = None

VIDEO_BS = (len(gpu_devs) * PER_GPU_BS) if pmap_encode_fn is not None else 1

def _decode_video_clips(vid):
    import av
    mpath = mp4_by_vid[vid]
    records, valid_ts = [], []
    for f in decode_frames(mpath, include_pixels=False):
        ts = f.timestamp_seconds
        records.append((f.ordinal, ts))
        if ts is not None:
            valid_ts.append(ts)
    if not valid_ts:
        raise ValueError(f"No valid PTS for {vid}")
    min_ts, max_ts = min(valid_ts), max(valid_ts) + 1e-3
    dur = max(0.5, max_ts - min_ts)
    if dur < 18.0:
        bounds = [("whole", min_ts, max_ts), ("w0", min_ts, min_ts + 0.65 * dur), ("w1", min_ts + 0.35 * dur, max_ts)]
    else:
        bounds = [("whole", min_ts, max_ts), ("w0", min_ts, min_ts + 0.55 * dur),
                  ("w1", min_ts + 0.225 * dur, min_ts + 0.775 * dur), ("w2", min_ts + 0.45 * dur, max_ts)]
    recipes, needed = [], set()
    for tag, st, et in bounds:
        rec = select_bin_center_frames(records, st, max(st + 1e-3, et), num_frames=NUM_CLIP_FRAMES)
        if rec["status"] != "ok":
            raise ValueError(f"Recipe failed {vid} {tag}: {rec}")
        recipes.append((tag, st - min_ts, et - min_ts, rec))
        needed.update(int(o) for o in rec["selected_ordinals"])

    pr = probe_media(mpath)
    dec_288 = {}
    with av.open(str(mpath)) as container:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        cur_ord, max_ord = 0, max(needed)
        try:
            for frame in container.decode(stream):
                if cur_ord in needed:
                    dec_288[cur_ord] = resize_and_pad_288(frame.to_ndarray(format="rgb24"), VIDEOPRISM_IMAGE_SIZE)
                if cur_ord >= max_ord:
                    break
                cur_ord += 1
        except Exception:
            if not needed.issubset(dec_288.keys()):
                raise

    return [
        (vid, tag, round(st_rel, 3), round(et_rel, 3), rec,
         np.stack([dec_288[int(o)] for o in rec["selected_ordinals"]], axis=0))
        for tag, st_rel, et_rel, rec in recipes
    ]

whole_vecs = np.zeros((len(video_ids), VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)
whole_rows = []
win_vecs_list = []
win_rows = []
vid_to_idx = {v: i for i, v in enumerate(video_ids)}

pending = []
def _flush():
    if not pending:
        return
    n_real = len(pending)
    if pmap_encode_fn is not None:
        pad = [pending[-1][5]] * (VIDEO_BS - n_real) if n_real < VIDEO_BS else []
        arr = np.stack([p[5] for p in pending] + pad, axis=0)
        vecs = pmap_encode_fn(arr)[:n_real]
    else:
        arr = np.stack([p[5] for p in pending], axis=0)
        vecs = encoder.encode_videos(arr)
    for k in range(n_real):
        vid, tag, st_rel, et_rel, rec, _ = pending[k]
        vec = vecs[k]
        if tag == "whole":
            v_idx = vid_to_idx[vid]
            whole_vecs[v_idx] = vec
            whole_rows.append({
                "row_id": v_idx, "video_id": vid,
                "selected_ordinals": rec["selected_ordinals"],
                "selected_timestamps_sec": rec["selected_timestamps_sec"],
                "recipe_hash": rec["recipe_hash"],
            })
        else:
            win_rows.append({
                "row_id": len(win_rows), "window_id": f"{vid}__{tag}",
                "video_id": vid, "window_tag": tag,
                "clip_start_sec": st_rel, "clip_end_sec": et_rel,
                "selected_ordinals": rec["selected_ordinals"],
                "selected_timestamps_sec": rec["selected_timestamps_sec"],
                "recipe_hash": rec["recipe_hash"],
            })
            win_vecs_list.append(vec)
    pending.clear()

done_v = 0
with ThreadPoolExecutor(max_workers=6) as pool:
    for c_st in range(0, len(video_ids), 24):
        for items in pool.map(_decode_video_clips, video_ids[c_st : c_st + 24]):
            for it in items:
                pending.append(it)
                if len(pending) >= VIDEO_BS:
                    _flush()
            done_v += 1
            if done_v % 250 == 0 or done_v == len(video_ids):
                print(f"[{shard_name}] {done_v}/{len(video_ids)} videos ({len(win_rows)} wins) in {time.time()-t_start:.1f}s", flush=True)
_flush()

whole_rows.sort(key=lambda r: r["row_id"])
win_vecs = np.vstack(win_vecs_list).astype(np.float32) if win_vecs_list else np.zeros((0, VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)

QUERY_BS = 32
query_vecs = np.zeros((len(query_rows_in), VIDEOPRISM_EMBEDDING_DIM), dtype=np.float32)
query_rows_out = []
for st in range(0, len(query_rows_in), QUERY_BS):
    chunk = query_rows_in[st : st + QUERY_BS]
    texts = [q["query_text"] for q in chunk]
    n_real = len(texts)
    padded = texts if n_real == QUERY_BS else (texts + [texts[-1]] * (QUERY_BS - n_real))
    qv = encoder.encode_texts(padded)[:n_real]
    query_vecs[st : st + n_real] = qv
    for j, q in enumerate(chunk):
        query_rows_out.append({
            "row_id": st + j, "query_text": q["query_text"], "query_text_sha256": q["query_text_sha256"],
        })

assert len(whole_rows) == len(video_ids)
assert np.isfinite(whole_vecs).all() and np.allclose(np.linalg.norm(whole_vecs, axis=1), 1.0, atol=1e-4)
if len(win_vecs) > 0:
    assert np.isfinite(win_vecs).all() and np.allclose(np.linalg.norm(win_vecs, axis=1), 1.0, atol=1e-4)
if len(query_vecs) > 0:
    assert np.isfinite(query_vecs).all() and np.allclose(np.linalg.norm(query_vecs, axis=1), 1.0, atol=1e-4)

out_dir = WORK / shard_name
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "video_vectors.npy", whole_vecs)
np.save(out_dir / "window_vectors.npy", win_vecs)
np.save(out_dir / "query_vectors.npy", query_vecs)

def _wjsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

_wjsonl(out_dir / "video_rows.jsonl", whole_rows)
_wjsonl(out_dir / "window_rows.jsonl", win_rows)
_wjsonl(out_dir / "query_rows.jsonl", query_rows_out)

f_names = ("video_vectors.npy", "video_rows.jsonl", "window_vectors.npy", "window_rows.jsonl", "query_vectors.npy", "query_rows.jsonl")
files_info = {fn: {"sha256": sha256_file(out_dir / fn), "bytes": (out_dir / fn).stat().st_size} for fn in f_names}
receipt = {
    "status": "SUCCESS", "dry_run": False, "shard_name": shard_name,
    "model": pinned_model_metadata(), "pins": pin_info,
    "video_count": len(whole_rows), "window_count": len(win_rows), "query_count": len(query_rows_out),
    "errors": [], "elapsed_seconds": round(time.time() - t_start, 2),
    "runtime": {"python": platform.python_version(), "devices": encoder.device_summary},
    "files": files_info,
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("SHARD DONE:", shard_name, receipt["elapsed_seconds"], "s", flush=True)
'''


_MM_SHARD_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from importlib import metadata as importlib_metadata
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_mm_shard")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["HF_HUB_DISABLE_XET"] = "1"
constraints = []
for pkg in ("numpy", "torch", "torchvision", "transformers", "opencv-python", "opencv-contrib-python", "scipy", "pillow"):
    try:
        constraints.append(f"{pkg}=={importlib_metadata.version(pkg)}")
    except importlib_metadata.PackageNotFoundError:
        pass
cfile = TEMP / "constraints.txt"
cfile.write_text("\n".join(constraints) + "\n", encoding="utf-8")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", str(cfile),
     "av==16.0.1", "ultralytics==8.4.202", "onnxruntime", "pyclipper>=1.2.0", "Shapely>=1.7.1", "six", "PyYAML", "tqdm"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--ignore-requires-python", "--no-deps",
     "rapidocr-onnxruntime==1.4.4"],
    check=True,
)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

import av
from huggingface_hub import hf_hub_download
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import torch
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

from videntra.config.ingestion import DecodeConfig, SamplingConfig
from videntra.evaluation.provenance import sha256_file
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import ProbeResult, StreamInfo
from videntra.relations.kalman_tracker import ShotAwareKalmanTracker
from videntra.sampling.hybrid import HybridAdaptiveSampler
''' + _SHARED_QUERY_EXPANDER_CODE + r'''
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"

t_start = time.time()
with gzip.open(TEMP / "job_payload.json.gz", "rt", encoding="utf-8") as fh:
    payload = json.load(fh)

shard_name = payload["shard_name"]
video_ids = [str(v) for v in payload["video_ids"]]
query_rows_in = expand_and_dedup_queries(payload["base_queries"], payload.get("expand_variants", True))
run_ocr = bool(payload.get("run_ocr", True))
run_yolo_track = bool(payload.get("run_yolo_track", True))

weights_local = Path(hf_hub_download(repo_id=SIGLIP_MODEL_ID, filename="model.safetensors", revision=SIGLIP_REVISION))
assert sha256_file(weights_local) == SIGLIP_WEIGHTS_SHA256
yolo_path = TEMP / "yolov8n.pt"
if not yolo_path.is_file():
    urllib.request.urlretrieve(YOLO_URL, yolo_path)
assert sha256_file(yolo_path) == YOLO_SHA256

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, revision=SIGLIP_REVISION)
siglip_models = [
    AutoModel.from_pretrained(SIGLIP_MODEL_ID, revision=SIGLIP_REVISION).eval().to(dev)
    for dev in devices
]
yolo_models = [YOLO(str(yolo_path)) for _ in devices] if (run_yolo_track and video_ids) else []
ocr_tls = threading.local()

def _get_thread_ocr():
    eng = getattr(ocr_tls, "engine", None)
    if eng is None:
        eng = RapidOCR(use_angle_cls=False)
        ocr_tls.engine = eng
    return eng

def _prio(p: Path) -> int:
    s = p.as_posix().lower()
    if "msrvtt-1ka" in s:
        return 0
    if "khoahunhtngng" in s or "testvideo" in s:
        return 1
    return 2

mp4_by_vid = {}
if video_ids:
    mp4_candidates = {}
    for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
        mp4_candidates.setdefault(p.stem, []).append(p)
    mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}
missing = [v for v in video_ids if v not in mp4_by_vid]
assert not missing, f"Missing videos in {shard_name}: {missing[:10]}"

def _sample_single_video(vid):
    mpath = mp4_by_vid[vid]
    with av.open(str(mpath)) as container:
        vs = container.streams.video[0]
        tb = vs.time_base
        tb_num, tb_den = (int(tb.numerator), int(tb.denominator)) if tb else (1, 1000)
        dur_s = float(vs.duration * tb) if (vs.duration and tb) else (float(container.duration) / float(av.time_base) if container.duration else 10.0)
        avg_rate = float(vs.average_rate) if vs.average_rate else 30.0
        probe = ProbeResult(
            video_path=mpath, sha256="", container_format="mp4",
            duration_seconds=dur_s, duration_us=int(round(dur_s * 1e6)),
            video_stream=StreamInfo(index=int(vs.index), codec_name="h264", codec_type="video",
                                    time_base_num=tb_num, time_base_den=tb_den,
                                    width=int(vs.width or 0), height=int(vs.height or 0),
                                    avg_frame_rate=f"{avg_rate:.6f}", nb_frames=int(vs.frames or 0)),
            audio_stream=None, has_video=True, has_audio=len(container.streams.audio) > 0,
            selected_video_stream_index=int(vs.index), selected_audio_stream_index=None, raw_probe={},
        )
    cfg = SamplingConfig.for_duration(dur_s)
    sampler = HybridAdaptiveSampler(config=cfg, video_id=vid, timeline_revision="rev_0")
    pixels_cache, last_ts, total_dec = {}, 0.0, 0
    for df in decode_frames(mpath, probe_result=probe, recipe=DecodeConfig(thread_count=2), include_pixels=True):
        total_dec += 1
        if df.timestamp_seconds is not None:
            last_ts = float(df.timestamp_seconds)
        if df.pixels is not None:
            pixels_cache[df.ordinal] = df.pixels
        sampler.process_frame(df)
    manifest = sampler.finalize()
    if not manifest.samples:
        raise RuntimeError(f"No samples for {vid}")

    kf_items = []
    for k_idx, s in enumerate(manifest.samples):
        ord_idx = int(s.original_decoded_frame_index)
        rgb = pixels_cache[ord_idx]
        ts_ms = int(round(s.timestamp_us / 1000.0)) if s.timestamp_us is not None else 0
        reasons = list(s.reasons) if s.reasons else ["hybrid_v2_adaptive"]
        kf_items.append({
            "video_id": vid, "keyframe_id": f"{vid}__kf_{k_idx:05d}", "keyframe_index": k_idx,
            "source_frame_index": ord_idx, "timestamp_ms": ts_ms, "pts_sec": round(float(s.timestamp_seconds), 6),
            "is_shot_cut": any("cut" in str(r).lower() for r in reasons),
            "sampling_reason": ",".join(reasons), "image": Image.fromarray(rgb), "rgb": rgb,
        })
    v_meta = {
        "video_id": vid, "duration_ms": max(int(round(max(dur_s, last_ts) * 1000.0)), 1000),
        "total_frames": total_dec, "has_audio": 1 if probe.has_audio else 0, "keyframe_count": len(kf_items),
    }
    return v_meta, kf_items

videos_meta, kf_rows_out, kf_vecs_chunks = [], [], []
yolo_objects_out, track_summaries_out, ocr_rows_out, ocr_video_status = [], [], [], {}
tracker_sort = ShotAwareKalmanTracker(mode="sort", max_gap_sec=2.5, min_hits=2) if run_yolo_track else None
tracker_bt = ShotAwareKalmanTracker(mode="bytetrack", max_gap_sec=2.5, min_hits=2) if run_yolo_track else None

SIGLIP_BS, YOLO_BS, CHUNK_VIDS = 128, 64, 48

def _encode_siglip_chunk(items, dev_idx):
    dev = devices[dev_idx]
    mdl = siglip_models[dev_idx]
    out_arr = np.zeros((len(items), 768), dtype=np.float16)
    for b_st in range(0, len(items), SIGLIP_BS):
        sub = items[b_st : b_st + SIGLIP_BS]
        inputs = processor(images=[it["image"] for it in sub], return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.inference_mode():
            if dev.startswith("cuda"):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = mdl.get_image_features(**inputs)
            else:
                out = mdl.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
            emb = emb.float()
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            out_arr[b_st : b_st + len(sub)] = emb.cpu().numpy().astype(np.float16)
    return out_arr

def _yolo_chunk(items, dev_idx):
    dev = devices[dev_idx]
    ymdl = yolo_models[dev_idx]
    y_arg = int(dev.split(":")[1]) if dev.startswith("cuda:") else "cpu"
    res_map = {}
    for b_st in range(0, len(items), YOLO_BS):
        sub = items[b_st : b_st + YOLO_BS]
        y_res = ymdl([it["image"] for it in sub], verbose=False, device=y_arg, conf=0.20)
        for kf, r in zip(sub, y_res):
            oh, ow = r.orig_shape
            kf_dets = []
            for d_idx, box in enumerate(r.boxes):
                conf = float(box.conf[0].item())
                if conf < 0.20:
                    continue
                xyxy = box.xyxy[0].tolist()
                x1 = max(0.0, min(1.0, round(xyxy[0] / ow, 4)))
                y1 = max(0.0, min(1.0, round(xyxy[1] / oh, 4)))
                x2 = max(x1 + 1e-4, min(1.0, round(xyxy[2] / ow, 4)))
                y2 = max(y1 + 1e-4, min(1.0, round(xyxy[3] / oh, 4)))
                cls_id = int(box.cls[0].item())
                cname = str(ymdl.names.get(cls_id, f"obj_{cls_id}"))
                kf_dets.append({
                    "detection_id": f"{kf['keyframe_id']}__det_{d_idx}",
                    "video_id": kf["video_id"], "keyframe_id": kf["keyframe_id"],
                    "timestamp_ms": kf["timestamp_ms"], "class_id": cls_id, "class_name": cname,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": min(1.0, x2), "bbox_y2": min(1.0, y2),
                    "confidence": round(conf, 4),
                })
            res_map[kf["keyframe_id"]] = kf_dets
    return res_map

for c_st in range(0, len(video_ids), CHUNK_VIDS):
    c_vids = video_ids[c_st : c_st + CHUNK_VIDS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sampled = list(pool.map(_sample_single_video, c_vids))

    chunk_kfs = []
    for v_meta, kfs in sampled:
        videos_meta.append(v_meta)
        for kf in kfs:
            kf["faiss_id"] = len(kf_rows_out)
            kf_rows_out.append({
                "faiss_id": kf["faiss_id"], "keyframe_id": kf["keyframe_id"], "video_id": kf["video_id"],
                "keyframe_index": kf["keyframe_index"], "source_frame_index": kf["source_frame_index"],
                "timestamp_ms": kf["timestamp_ms"], "pts_sec": kf["pts_sec"],
                "is_shot_cut": kf["is_shot_cut"], "sampling_reason": kf["sampling_reason"],
            })
            chunk_kfs.append(kf)

    if len(devices) >= 2 and len(chunk_kfs) >= 2:
        mid = len(chunk_kfs) // 2
        with ThreadPoolExecutor(max_workers=2) as gpool:
            f0 = gpool.submit(_encode_siglip_chunk, chunk_kfs[:mid], 0)
            f1 = gpool.submit(_encode_siglip_chunk, chunk_kfs[mid:], 1)
            kf_vecs_chunks.append(np.vstack([f0.result(), f1.result()]))
    else:
        kf_vecs_chunks.append(_encode_siglip_chunk(chunk_kfs, 0))

    if run_yolo_track:
        dets_by_kf = {}
        if len(devices) >= 2 and len(chunk_kfs) >= 2:
            mid = len(chunk_kfs) // 2
            with ThreadPoolExecutor(max_workers=2) as gpool:
                y0 = gpool.submit(_yolo_chunk, chunk_kfs[:mid], 0)
                y1 = gpool.submit(_yolo_chunk, chunk_kfs[mid:], 1)
                dets_by_kf.update(y0.result())
                dets_by_kf.update(y1.result())
        else:
            dets_by_kf.update(_yolo_chunk(chunk_kfs, 0))

        for _, kfs in sampled:
            vid = kfs[0]["video_id"]
            f_obs = []
            for kf in kfs:
                kd = dets_by_kf.get(kf["keyframe_id"], [])
                yolo_objects_out.extend([d for d in kd if d["confidence"] >= 0.25])
                f_obs.append({"ordinal": kf["source_frame_index"], "pts_sec": kf["pts_sec"], "is_shot_cut": kf["is_shot_cut"], "detections": kd})
            t_sort = tracker_sort.track_video_frames(f_obs)
            t_bt = tracker_bt.track_video_frames(f_obs)
            def _ser(res):
                return [{
                    "track_id": tr.track_id, "shot_id": tr.shot_id, "class_name": tr.class_name,
                    "observed_count": tr.observed_count, "predicted_only_count": tr.predicted_only_count,
                    "recovered_after_occlusion": tr.recovered_after_occlusion,
                    "duration_sec": round(tr.duration_sec, 3), "mean_speed": round(tr.mean_speed, 4),
                    "net_displacement": round(tr.net_displacement, 4),
                    "steps": [{"ordinal": st.frame_ordinal, "pts_sec": st.pts_sec, "shot_id": st.shot_id,
                               "source_kind": st.source_kind, "class_name": st.class_name, "confidence": st.confidence,
                               "observed_box": st.observed_box, "predicted_box": st.predicted_box, "velocity_xy": st.velocity_xy}
                              for st in tr.steps],
                } for tr in res["tracks"]]
            track_summaries_out.append({"video_id": vid, "num_frames": len(kfs), "sort_tracks": _ser(t_sort), "bytetrack_tracks": _ser(t_bt)})

    if run_ocr:
        max_ocr_frames = 2 if "train" in shard_name else 6
        def _ocr_single_vid(kfs):
            eng = _get_thread_ocr()
            vid = kfs[0]["video_id"]
            stride = max(1, len(kfs) // max_ocr_frames)
            v_ocr = []
            for kf in kfs[::stride][:max_ocr_frames]:
                res, _ = eng(kf["rgb"])
                if not res:
                    continue
                oh, ow = kf["rgb"].shape[:2]
                for o_idx, item in enumerate(res):
                    poly, txt, conf = item[0], str(item[1] or "").strip(), float(item[2])
                    if not txt or conf < 0.45:
                        continue
                    xs = [p[0] / max(1.0, ow) for p in poly]
                    ys = [p[1] / max(1.0, oh) for p in poly]
                    v_ocr.append({
                        "observation_id": f"{kf['keyframe_id']}__ocr_{o_idx}", "video_id": vid,
                        "keyframe_id": kf["keyframe_id"], "timestamp_ms": kf["timestamp_ms"],
                        "raw_text": txt, "confidence": round(conf, 4),
                        "bbox_x1": round(max(0.0, min(xs)), 4), "bbox_y1": round(max(0.0, min(ys)), 4),
                        "bbox_x2": round(min(1.0, max(xs)), 4), "bbox_y2": round(min(1.0, max(ys)), 4),
                    })
            return vid, v_ocr

        kfs_lists = [kfs for _, kfs in sampled if kfs]
        with ThreadPoolExecutor(max_workers=4) as ocr_pool:
            for vid, v_ocr in ocr_pool.map(_ocr_single_vid, kfs_lists):
                ocr_rows_out.extend(v_ocr)
                ocr_video_status[vid] = "has_ocr" if v_ocr else "empty_observed"

    print(f"[{shard_name}] {len(videos_meta)}/{len(video_ids)} videos | {len(kf_rows_out)} kfs | {len(ocr_rows_out)} ocr | {time.time()-t_start:.1f}s", flush=True)

q_vecs = np.zeros((len(query_rows_in), 768), dtype=np.float32)
dev0, mdl0 = devices[0], siglip_models[0]
for st in range(0, len(query_rows_in), 128):
    chunk = query_rows_in[st : st + 128]
    inputs = processor(text=[q["query_text"] for q in chunk], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    inputs = {k: v.to(dev0) for k, v in inputs.items()}
    with torch.no_grad():
        if dev0.startswith("cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = mdl0.get_text_features(**inputs)
        else:
            out = mdl0.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        emb = emb.float()
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        q_vecs[st : st + len(chunk)] = emb.cpu().numpy().astype(np.float32)

kf_vecs = np.vstack(kf_vecs_chunks).astype(np.float16) if kf_vecs_chunks else np.zeros((0, 768), dtype=np.float16)
assert np.isfinite(kf_vecs).all() and np.isfinite(q_vecs).all()

out_dir = WORK / shard_name
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "keyframe_embeddings_fp16.npy", kf_vecs)
np.save(out_dir / "siglip_query_vectors.npy", q_vecs)

def _wgz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

_wgz(out_dir / "videos_meta.jsonl.gz", videos_meta)
_wgz(out_dir / "keyframes_manifest.jsonl.gz", kf_rows_out)
_wgz(out_dir / "yolo_objects.jsonl.gz", yolo_objects_out)
_wgz(out_dir / "kalman_tracks.jsonl.gz", track_summaries_out)
_wgz(out_dir / "ocr_observations.jsonl.gz", ocr_rows_out)
(out_dir / "query_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in query_rows_in) + ("\n" if query_rows_in else ""), encoding="utf-8")
(out_dir / "ocr_video_status.json").write_text(json.dumps(ocr_video_status, indent=2), encoding="utf-8")

f_names = ("keyframe_embeddings_fp16.npy", "siglip_query_vectors.npy", "query_rows.jsonl",
           "videos_meta.jsonl.gz", "keyframes_manifest.jsonl.gz", "yolo_objects.jsonl.gz",
           "kalman_tracks.jsonl.gz", "ocr_observations.jsonl.gz", "ocr_video_status.json")
files_info = {fn: {"sha256": sha256_file(out_dir / fn), "bytes": (out_dir / fn).stat().st_size} for fn in f_names}
receipt = {
    "status": "SUCCESS", "shard_name": shard_name, "siglip_model_id": SIGLIP_MODEL_ID,
    "siglip_revision": SIGLIP_REVISION, "siglip_weights_sha256": SIGLIP_WEIGHTS_SHA256,
    "yolo_sha256": YOLO_SHA256, "video_count": len(videos_meta), "keyframe_count": len(kf_rows_out),
    "object_count": len(yolo_objects_out), "ocr_count": len(ocr_rows_out), "query_count": len(query_rows_in),
    "elapsed_seconds": round(time.time() - t_start, 2), "files": files_info,
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("MM SHARD DONE:", shard_name, receipt["elapsed_seconds"], "s", flush=True)
'''


_ASR_SHARD_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_asr_shard")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["HF_HUB_DISABLE_XET"] = "1"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "faster-whisper==1.2.1", "av==16.0.1"],
    check=True,
)

import ctypes
import site
_so_candidates = []
for sp in site.getsitepackages():
    for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib"):
        lib_dir = Path(sp) / sub
        if lib_dir.is_dir():
            _so_candidates.extend(sorted(lib_dir.glob("lib*.so*")))
for _ in range(2):
    for _so in _so_candidates:
        try:
            ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import av
from faster_whisper import WhisperModel
import torch

ASR_REVISION = "536b0662742c02347bc0e980a01041f333bce120"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

t_start = time.time()
with gzip.open(TEMP / "job_payload.json.gz", "rt", encoding="utf-8") as fh:
    payload = json.load(fh)

shard_name = payload["shard_name"]
video_ids = [str(v) for v in payload["video_ids"]]

def _prio(p: Path) -> int:
    s = p.as_posix().lower()
    if "msrvtt-1ka" in s:
        return 0
    if "khoahunhtngng" in s or "testvideo" in s:
        return 1
    return 2

mp4_candidates = {}
for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
    mp4_candidates.setdefault(p.stem, []).append(p)
mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}
missing = [v for v in video_ids if v not in mp4_by_vid]
assert not missing, f"Missing videos in {shard_name}: {missing[:10]}"

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
worker_count = min(2, n_gpus) if n_gpus > 0 else 1
whispers = [
    WhisperModel("Systran/faster-whisper-small", device="cuda" if n_gpus > 0 else "cpu",
                 device_index=idx if n_gpus > 0 else 0,
                 compute_type="float16" if n_gpus > 0 else "int8")
    for idx in range(worker_count)
]
try:
    _ = list(whispers[0].transcribe(np.zeros(16000, dtype=np.float32), vad_filter=False, beam_size=1)[0])
except Exception as cuda_exc:
    print("CUDA Whisper warmup failed, falling back to CPU int8:", cuda_exc, flush=True)
    worker_count = 2
    whispers = [
        WhisperModel("Systran/faster-whisper-small", device="cpu", compute_type="int8", cpu_threads=2)
        for _ in range(worker_count)
    ]

def _decode_audio_16k_mono(mpath: Path):
    with av.open(str(mpath)) as container:
        if not container.streams.audio:
            return None
        a_stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(a_stream):
            for r_frame in resampler.resample(frame):
                arr = r_frame.to_ndarray()
                chunks.append(arr.reshape(-1))
        for r_frame in resampler.resample(None):
            chunks.append(r_frame.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)

def _process_slice(worker_idx, vid_slice):
    model = whispers[worker_idx]
    segs_out = []
    status_out = {}
    counts = {"with_speech": 0, "empty_speech": 0, "no_audio_stream": 0, "errors": 0}
    for done_i, vid in enumerate(vid_slice, 1):
        mpath = mp4_by_vid[vid]
        try:
            audio = _decode_audio_16k_mono(mpath)
            if audio is None:
                status_out[vid] = "no_audio_stream"
                counts["no_audio_stream"] += 1
                continue
            if audio.size < 1600:
                status_out[vid] = "empty_speech"
                counts["empty_speech"] += 1
                continue
            segs, info = model.transcribe(audio, vad_filter=True, beam_size=1)
            had = False
            for s_i, seg in enumerate(segs):
                txt = str(seg.text or "").strip()
                if not txt:
                    continue
                segs_out.append({
                    "segment_id": f"{vid}__asr_{s_i:05d}", "video_id": vid,
                    "start_ms": int(round(seg.start * 1000.0)), "end_ms": int(round(seg.end * 1000.0)),
                    "raw_text": txt, "language": getattr(info, "language", "en"),
                    "confidence": round(float(seg.avg_logprob), 4),
                    "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0)), 4),
                })
                had = True
            if had:
                status_out[vid] = "with_speech"
                counts["with_speech"] += 1
            else:
                status_out[vid] = "empty_speech"
                counts["empty_speech"] += 1
        except Exception as exc:
            status_out[vid] = f"error:{exc}"
            counts["errors"] += 1
        if done_i % 250 == 0 or done_i == len(vid_slice):
            print(f"[{shard_name} gpu{worker_idx}] {done_i}/{len(vid_slice)} | segs={len(segs_out)} | {time.time()-t_start:.1f}s", flush=True)
    return segs_out, status_out, counts

slices = [video_ids[i::worker_count] for i in range(worker_count)]
asr_segments, video_audio_status = [], {}
total_counts = {"with_speech": 0, "empty_speech": 0, "no_audio_stream": 0, "errors": 0}
with ThreadPoolExecutor(max_workers=worker_count) as pool:
    results = list(pool.map(lambda p: _process_slice(p[0], p[1]), enumerate(slices)))
for segs_s, st_s, c_s in results:
    asr_segments.extend(segs_s)
    video_audio_status.update(st_s)
    for k, v in c_s.items():
        total_counts[k] += v

assert total_counts["with_speech"] > 0, f"ASR produced 0 speech videos: {total_counts}"
out_dir = WORK / shard_name
out_dir.mkdir(parents=True, exist_ok=True)
with gzip.open(out_dir / "asr_segments.jsonl.gz", "wt", encoding="utf-8") as fh:
    for r in asr_segments:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
(out_dir / "asr_video_status.json").write_text(json.dumps(video_audio_status, indent=2), encoding="utf-8")

receipt = {
    "status": "SUCCESS", "shard_name": shard_name, "asr_model": "Systran/faster-whisper-small",
    "asr_revision_pin": ASR_REVISION, "video_count": len(video_ids), "segment_count": len(asr_segments),
    "counts": total_counts, "elapsed_seconds": round(time.time() - t_start, 2),
    "files": {
        "asr_segments.jsonl.gz": {"sha256": sha256_file(out_dir / "asr_segments.jsonl.gz"), "bytes": (out_dir / "asr_segments.jsonl.gz").stat().st_size},
        "asr_video_status.json": {"sha256": sha256_file(out_dir / "asr_video_status.json"), "bytes": (out_dir / "asr_video_status.json").stat().st_size},
    },
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("ASR SHARD DONE:", shard_name, json.dumps(total_counts), flush=True)
'''


_CAP_SHARD_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_cap_shard")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["HF_HUB_DISABLE_XET"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av==16.0.1", "einops", "timm"], check=True)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

import av
from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
PretrainedConfig.forced_bos_token_id = None
if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
    PreTrainedTokenizerBase.additional_special_tokens = property(
        lambda self: list(getattr(self, "extra_special_tokens", None) or getattr(self, "_additional_special_tokens", []))
    )
''' + _SHARED_QUERY_EXPANDER_CODE + r'''
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE2_REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "c007d7ef6fd86656326059b28395a7a03a7c5846"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

t_start = time.time()
with gzip.open(TEMP / "job_payload.json.gz", "rt", encoding="utf-8") as fh:
    payload = json.load(fh)

shard_name = payload["shard_name"]
video_ids = [str(v) for v in payload["video_ids"]]
query_rows_in = expand_and_dedup_queries(payload["base_queries"], payload.get("expand_variants", True))

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
dtype = torch.float16 if n_gpus > 0 else torch.float32

captions_out = []
if video_ids:
    def _prio(p: Path) -> int:
        s = p.as_posix().lower()
        if "msrvtt-1ka" in s:
            return 0
        if "khoahunhtngng" in s or "testvideo" in s:
            return 1
        return 2

    mp4_candidates = {}
    for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
        mp4_candidates.setdefault(p.stem, []).append(p)
    mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}
    missing = [v for v in video_ids if v not in mp4_by_vid]
    assert not missing, f"Missing videos in {shard_name}: {missing[:10]}"

    fl_cfg = AutoConfig.from_pretrained(FLORENCE2_MODEL_ID, revision=FLORENCE2_REVISION, trust_remote_code=True)
    fl_cfg._attn_implementation = "eager"
    _fl_cls = get_class_from_dynamic_module(
        "modeling_florence2.Florence2ForConditionalGeneration",
        FLORENCE2_MODEL_ID,
        revision=FLORENCE2_REVISION,
    )
    _fl_cls._supports_sdpa = False
    fl_proc = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, revision=FLORENCE2_REVISION, trust_remote_code=True)
    fl_models = [
        AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_ID, revision=FLORENCE2_REVISION, config=fl_cfg,
            torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True,
        ).to(dev).eval()
        for dev in devices
    ]
    for _m in fl_models:
        _sh = _m.language_model.model.shared.weight
        _m.language_model.model.encoder.embed_tokens.weight = _sh
        _m.language_model.model.decoder.embed_tokens.weight = _sh
        _m.language_model.lm_head.weight = _sh

    TASK_PROMPT = "<DETAILED_CAPTION>"
    _use_cache_flag = False
    _dummy_im = Image.new("RGB", (320, 240)).resize((768, 768), Image.Resampling.BICUBIC)
    _d_in = fl_proc(text=[TASK_PROMPT], images=[_dummy_im], return_tensors="pt")
    _d_in = {
        k: (v.to(device=devices[0], dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v.to(devices[0]))
        for k, v in _d_in.items()
    }
    with torch.no_grad():
        _ = fl_models[0].generate(**_d_in, max_new_tokens=8, num_beams=1, do_sample=False, use_cache=False)

    def _decode_3_rep_frames(vid):
        mpath = mp4_by_vid[vid]
        frames_pts = []
        with av.open(str(mpath)) as container:
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"
            tb = float(vs.time_base) if vs.time_base else 1.0 / 1000.0
            for idx, frame in enumerate(container.decode(vs)):
                ts = float(frame.pts * tb) if frame.pts is not None else idx / 30.0
                frames_pts.append((ts, idx, frame))
            if not frames_pts:
                raise RuntimeError(f"Empty decode for {vid}")
            t0, t1 = frames_pts[0][0], frames_pts[-1][0]
            dur = max(0.1, t1 - t0)
            imgs = []
            for f in (0.15, 0.50, 0.85):
                best_i = min(range(len(frames_pts)), key=lambda i: (abs(frames_pts[i][0] - (t0 + f * dur)), frames_pts[i][1]))
                rgb = frames_pts[best_i][2].to_ndarray(format="rgb24")
                imgs.append(Image.fromarray(rgb).convert("RGB").resize((768, 768), Image.Resampling.BICUBIC))
        return vid, imgs

    def _caption_worker(gpu_idx, vid_slice):
        dev = devices[gpu_idx]
        mdl = fl_models[gpu_idx]
        out_caps = []
        CHUNK_VIDS, FL_BS = 32, 32
        for c_st in range(0, len(vid_slice), CHUNK_VIDS):
            c_vids = vid_slice[c_st : c_st + CHUNK_VIDS]
            with ThreadPoolExecutor(max_workers=4) as pool:
                decoded_batch = list(pool.map(_decode_3_rep_frames, c_vids))
            flat_imgs = [im for _, imgs in decoded_batch for im in imgs]
            flat_caps = []
            for b_st in range(0, len(flat_imgs), FL_BS):
                sub_imgs = flat_imgs[b_st : b_st + FL_BS]
                inputs = fl_proc(text=[TASK_PROMPT] * len(sub_imgs), images=sub_imgs, return_tensors="pt")
                inputs = {
                    k: (v.to(device=dev, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v.to(dev))
                    for k, v in inputs.items()
                }
                with torch.inference_mode():
                    gen = mdl.generate(**inputs, max_new_tokens=48, num_beams=1, do_sample=False, use_cache=_use_cache_flag)
                dec = fl_proc.batch_decode(gen, skip_special_tokens=True)
                flat_caps.extend(str(txt).replace(TASK_PROMPT, "").strip() for txt in dec)
            pos = 0
            for vid, _ in decoded_batch:
                frame_caps = flat_caps[pos : pos + 3]
                pos += 3
                dedup = []
                for fc in frame_caps:
                    if fc and fc not in dedup:
                        dedup.append(fc)
                out_caps.append({"video_id": vid, "frame_captions": frame_caps, "merged_caption": " ".join(dedup)})
            if c_st == 0 and out_caps:
                print(f"[{shard_name} {dev}] sample caption ({out_caps[0]['video_id']}): {out_caps[0]['merged_caption'][:160]}", flush=True)
            if (c_st // CHUNK_VIDS) % 8 == 0 or c_st + CHUNK_VIDS >= len(vid_slice):
                print(f"[{shard_name} {dev}] Florence-2 {len(out_caps)}/{len(vid_slice)} in {time.time()-t_start:.1f}s", flush=True)
        return out_caps

    slices = [video_ids[i::len(devices)] for i in range(len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for res in pool.map(lambda p: _caption_worker(p[0], p[1]), enumerate(slices)):
            captions_out.extend(res)
    vid_order = {v: i for i, v in enumerate(video_ids)}
    captions_out.sort(key=lambda r: vid_order[r["video_id"]])

    del fl_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

dev0 = devices[0]
e5_tok = AutoTokenizer.from_pretrained(E5_MODEL_ID, revision=E5_REVISION)
e5_mdl = AutoModel.from_pretrained(E5_MODEL_ID, revision=E5_REVISION).to(dev0).eval()

def _encode_e5(texts, prefix):
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    prefixed = [f"{prefix}{t}" for t in texts]
    chunks = []
    for st in range(0, len(prefixed), 128):
        b = prefixed[st : st + 128]
        enc = e5_tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(dev0) for k, v in enc.items()}
        with torch.inference_mode():
            out = e5_mdl(**enc)
            hidden = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            chunks.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(chunks)

cap_e5 = _encode_e5([r["merged_caption"] for r in captions_out], "passage: ")
q_e5 = _encode_e5([q["query_text"] for q in query_rows_in], "query: ")

out_dir = WORK / shard_name
out_dir.mkdir(parents=True, exist_ok=True)
with gzip.open(out_dir / "florence2_captions.jsonl.gz", "wt", encoding="utf-8") as fh:
    for r in captions_out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
np.save(out_dir / "caption_e5_vectors.npy", cap_e5)
np.save(out_dir / "query_e5_vectors.npy", q_e5)
(out_dir / "query_rows.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in query_rows_in) + ("\n" if query_rows_in else ""),
    encoding="utf-8",
)

f_names = ("florence2_captions.jsonl.gz", "caption_e5_vectors.npy", "query_e5_vectors.npy", "query_rows.jsonl")
files_info = {fn: {"sha256": sha256_file(out_dir / fn), "bytes": (out_dir / fn).stat().st_size} for fn in f_names}
receipt = {
    "status": "SUCCESS", "shard_name": shard_name, "florence2_model": FLORENCE2_MODEL_ID,
    "florence2_revision": FLORENCE2_REVISION, "e5_model": E5_MODEL_ID, "e5_revision": E5_REVISION,
    "video_count": len(captions_out), "query_count": len(query_rows_in),
    "elapsed_seconds": round(time.time() - t_start, 2), "files": files_info,
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("CAPTION SHARD DONE:", shard_name, receipt["elapsed_seconds"], "s", flush=True)
'''


_VL2_SHARD_TEMPLATE = r'''
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from importlib import metadata as importlib_metadata
import zipfile

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/tmp/videntra_vl2_shard")
TEMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
    archive.extractall(TEMP)

os.environ["HF_HUB_DISABLE_XET"] = "1"
constraints = []
for pkg in ("numpy", "torch", "torchvision", "torch_xla", "transformers", "scipy", "pillow"):
    try:
        constraints.append(f"{pkg}=={importlib_metadata.version(pkg)}")
    except importlib_metadata.PackageNotFoundError:
        pass
cfile = TEMP / "constraints.txt"
cfile.write_text("\n".join(constraints) + "\n", encoding="utf-8")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", str(cfile),
     "av==16.0.1", "timm>=1.0.19", "open-clip-torch>=3.1.0", "ftfy", "regex"],
    check=True,
)
sys.path.insert(0, str(TEMP))
sys.path.insert(0, str(TEMP / "src"))

import av
from huggingface_hub import hf_hub_download, snapshot_download
import open_clip
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from scripts.prepare_clip_vp_kaggle_bundle import select_bin_center_frames
from videntra.evaluation.provenance import sha256_file
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import probe_media
''' + _SHARED_QUERY_EXPANDER_CODE + r'''
PE_MODEL_ID = "timm/PE-Core-L-14-336"
PE_REVISION = "8eff41b3f687e50a323662c2dda5eb3588c6dd35"
PE_WEIGHTS_FILE = "open_clip_model.safetensors"
PE_WEIGHTS_SHA256 = "b1fff7093c32a01d98a2d82098f9adf788b30f0efee639916520b3e4302f786b"
PE_LICENSE = "apache-2.0"

SIG400_MODEL_ID = "google/siglip2-so400m-patch14-384"
SIG400_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
SIG400_WEIGHTS_FILE = "model.safetensors"
SIG400_WEIGHTS_SHA256 = "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2"
SIG400_LICENSE = "apache-2.0"

NUM_VL2_FRAMES = 12
PE_DIM = 1024
SIG400_DIM = 1152

t_start = time.time()
with gzip.open(TEMP / "job_payload.json.gz", "rt", encoding="utf-8") as fh:
    payload = json.load(fh)

shard_name = payload["shard_name"]
video_ids = [str(v) for v in payload["video_ids"]]
query_rows_in = expand_and_dedup_queries(payload["base_queries"], payload.get("expand_variants", True))

pe_dir = TEMP / "pe_core_l14_336"
snapshot_download(
    repo_id=PE_MODEL_ID,
    revision=PE_REVISION,
    allow_patterns=["open_clip_config.json", PE_WEIGHTS_FILE, "*.json", "*.txt"],
    local_dir=str(pe_dir),
)
assert sha256_file(pe_dir / PE_WEIGHTS_FILE) == PE_WEIGHTS_SHA256

sig400_weights = Path(hf_hub_download(repo_id=SIG400_MODEL_ID, filename=SIG400_WEIGHTS_FILE, revision=SIG400_REVISION))
assert sha256_file(sig400_weights) == SIG400_WEIGHTS_SHA256

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
xm_mod = None
is_xla = False
if n_gpus > 0:
    devices = [f"cuda:{i}" for i in range(n_gpus)]
    dtype = torch.float16
    dev_labels = list(devices)
else:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        _xdev = xm.xla_device()
        xm_mod = xm
        is_xla = True
        devices = [_xdev]
        dtype = torch.bfloat16
        dev_labels = [str(_xdev)]
    except Exception as xla_exc:
        print("torch_xla unavailable, using CPU:", xla_exc, flush=True)
        torch.set_num_threads(max(4, os.cpu_count() or 4))
        devices = ["cpu"]
        dtype = torch.float32
        dev_labels = ["cpu"]

def _load_pe(dev):
    try:
        mdl, _, preproc = open_clip.create_model_and_transforms(f"local-dir:{pe_dir}")
        tok = open_clip.get_tokenizer(f"local-dir:{pe_dir}")
    except Exception:
        mdl, _, preproc = open_clip.create_model_and_transforms(
            "PE-Core-L-14-336", pretrained=str(pe_dir / PE_WEIGHTS_FILE)
        )
        tok = open_clip.get_tokenizer("PE-Core-L-14-336")
    mdl = mdl.eval()
    if is_xla:
        mdl = mdl.to(dtype=dtype)
    return mdl.to(dev), preproc, tok

pe_models = []
pe_preprocs = []
pe_tok = None
for dev in devices:
    m, pp, tk = _load_pe(dev)
    pe_models.append(m)
    pe_preprocs.append(pp)
    if pe_tok is None:
        pe_tok = tk

sig400_procs = [AutoProcessor.from_pretrained(SIG400_MODEL_ID, revision=SIG400_REVISION) for _ in devices]
sig400_proc = sig400_procs[0]
sig400_models = [
    AutoModel.from_pretrained(SIG400_MODEL_ID, revision=SIG400_REVISION, torch_dtype=(dtype if (n_gpus > 0 or is_xla) else torch.float32)).eval().to(dev)
    for dev in devices
]

# Smoke check text & image forward (auto-selects faster fp32 vs bf16 on CPU, falls back to float32 CPU cleanly)
_pe_ctx = int(getattr(pe_models[0], "context_length", 32))
def _smoke_check():
    chosen_dtype = dtype
    with torch.inference_mode():
        texts_smk = ["a person walking"] * (128 if is_xla else 1)
        try:
            _t_tok = pe_tok(texts_smk, context_length=_pe_ctx).to(devices[0])
        except TypeError:
            _t_tok = pe_tok(texts_smk).to(devices[0])
        _ = pe_models[0].encode_text(_t_tok)
        _s_tok = sig400_proc(text=texts_smk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        _s_tok = {k: v.to(devices[0]) for k, v in _s_tok.items()}
        _ = sig400_models[0].get_text_features(**_s_tok)
        if is_xla and xm_mod is not None:
            if video_ids:
                _pe_im = torch.zeros((48, 3, 336, 336), device=devices[0], dtype=dtype)
                _s_im = torch.zeros((48, 3, 384, 384), device=devices[0], dtype=dtype)
                _ = pe_models[0].encode_image(_pe_im)
                _ = sig400_models[0].get_image_features(pixel_values=_s_im)
            xm_mod.mark_step()
        elif n_gpus == 0 and video_ids:
            _pe_im = torch.zeros((4, 3, 336, 336), device="cpu", dtype=torch.float32)
            _s_im = torch.zeros((4, 3, 384, 384), device="cpu", dtype=torch.float32)
            _ = pe_models[0].encode_image(_pe_im)
            _ = sig400_models[0].get_image_features(pixel_values=_s_im)
            _t0 = time.time()
            _ = pe_models[0].encode_image(_pe_im)
            _ = sig400_models[0].get_image_features(pixel_values=_s_im)
            _t_fp32 = time.time() - _t0
            try:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    _ = pe_models[0].encode_image(_pe_im)
                    _ = sig400_models[0].get_image_features(pixel_values=_s_im)
                _t1 = time.time()
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    _ = pe_models[0].encode_image(_pe_im)
                    _ = sig400_models[0].get_image_features(pixel_values=_s_im)
                _t_bf16 = time.time() - _t1
                chosen_dtype = torch.bfloat16 if (_t_bf16 < _t_fp32 * 0.90) else torch.float32
                print(f"CPU 4-frame bench: fp32={_t_fp32:.2f}s bf16={_t_bf16:.2f}s -> selected={chosen_dtype}", flush=True)
            except Exception as bf_exc:
                chosen_dtype = torch.float32
                print("CPU bf16 unavailable, using fp32:", bf_exc, flush=True)
    return chosen_dtype

try:
    dtype = _smoke_check()
except Exception as smoke_exc:
    print("Smoke warning, falling back to fp32 CPU:", smoke_exc, flush=True)
    is_xla = False
    xm_mod = None
    devices = ["cpu"]
    dtype = torch.float32
    dev_labels = ["cpu"]
    pe_models = [m.float().to("cpu") for m in pe_models[:1]]
    pe_preprocs = pe_preprocs[:1]
    sig400_models = [m.float().to("cpu") for m in sig400_models[:1]]
    sig400_procs = sig400_procs[:1]
    dtype = _smoke_check()
print("VL2 backend ready:", dev_labels, "is_xla=", is_xla, "dtype=", dtype, flush=True)

mp4_by_vid = {}
if video_ids:
    def _prio(p: Path) -> int:
        s = p.as_posix().lower()
        if "msrvtt-1ka" in s:
            return 0
        if "khoahunhtngng" in s or "testvideo" in s:
            return 1
        return 2

    mp4_candidates = {}
    for p in sorted(Path("/kaggle/input").rglob("*.mp4")):
        mp4_candidates.setdefault(p.stem, []).append(p)
    mp4_by_vid = {k: sorted(v, key=_prio)[0] for k, v in mp4_candidates.items()}
    missing = [v for v in video_ids if v not in mp4_by_vid]
    assert not missing, f"Missing videos in {shard_name}: {missing[:10]}"

def _sample_12_frames(vid):
    mpath = mp4_by_vid[vid]
    records, valid_ts = [], []
    for f in decode_frames(mpath, include_pixels=False):
        ts = f.timestamp_seconds
        records.append((f.ordinal, ts))
        if ts is not None:
            valid_ts.append(ts)
    if not valid_ts:
        raise ValueError(f"No valid PTS for {vid}")
    min_ts, max_ts = min(valid_ts), max(valid_ts) + 1e-3
    rec = select_bin_center_frames(records, min_ts, max_ts, num_frames=NUM_VL2_FRAMES)
    if rec["status"] != "ok":
        raise ValueError(f"Recipe failed {vid}: {rec}")
    needed = {int(o) for o in rec["selected_ordinals"]}

    pr = probe_media(mpath)
    dec_imgs = {}
    with av.open(str(mpath)) as container:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        cur_ord, max_ord = 0, max(needed)
        try:
            for frame in container.decode(stream):
                if cur_ord in needed:
                    dec_imgs[cur_ord] = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
                if cur_ord >= max_ord:
                    break
                cur_ord += 1
        except Exception:
            if not needed.issubset(dec_imgs.keys()):
                raise

    imgs = [dec_imgs[int(o)] for o in rec["selected_ordinals"]]
    return vid, imgs, rec

pe_video_vecs = np.zeros((len(video_ids), PE_DIM), dtype=np.float32)
pe_frame_vecs = np.zeros((len(video_ids), NUM_VL2_FRAMES, PE_DIM), dtype=np.float16)
sig400_video_vecs = np.zeros((len(video_ids), SIG400_DIM), dtype=np.float32)
sig400_frame_vecs = np.zeros((len(video_ids), NUM_VL2_FRAMES, SIG400_DIM), dtype=np.float16)
video_rows_out = []

if n_gpus > 0 or is_xla:
    IMG_BS, CHUNK_VIDS, LOG_EVERY = 48, 32, 8
else:
    IMG_BS, CHUNK_VIDS, LOG_EVERY = 12, 8, 1

def _l2_norm_t(x):
    xf = x.float()
    return xf / xf.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

def _encode_image_chunk(imgs, dev_idx):
    dev = devices[dev_idx]
    p_mdl = pe_models[dev_idx]
    p_prep = pe_preprocs[dev_idx]
    s_mdl = sig400_models[dev_idx]
    s_proc = sig400_procs[dev_idx]
    n_img = len(imgs)
    pe_out = np.zeros((n_img, PE_DIM), dtype=np.float16)
    sig_out = np.zeros((n_img, SIG400_DIM), dtype=np.float16)

    for b_st in range(0, n_img, IMG_BS):
        sub = imgs[b_st : b_st + IMG_BS]
        n_real = len(sub)
        if is_xla and n_real < IMG_BS:
            sub = sub + [sub[-1]] * (IMG_BS - n_real)
        pe_t = torch.stack([p_prep(im) for im in sub])
        sig_in = s_proc(images=sub, return_tensors="pt")
        if is_xla:
            pe_t = pe_t.to(device=dev, dtype=dtype)
        else:
            pe_t = pe_t.to(dev)
        sig_in = {
            k: (v.to(device=dev, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() and (n_gpus > 0 or is_xla) else v.to(dev))
            for k, v in sig_in.items()
        }
        with torch.inference_mode():
            if n_gpus > 0:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pe_feat = p_mdl.encode_image(pe_t)
                    s_raw = s_mdl.get_image_features(**sig_in)
            elif not is_xla and dtype == torch.bfloat16:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    pe_feat = p_mdl.encode_image(pe_t)
                    s_raw = s_mdl.get_image_features(**sig_in)
            else:
                pe_feat = p_mdl.encode_image(pe_t)
                s_raw = s_mdl.get_image_features(**sig_in)
            s_feat = s_raw.pooler_output if hasattr(s_raw, "pooler_output") and s_raw.pooler_output is not None else s_raw
            pe_norm = _l2_norm_t(pe_feat)
            s_norm = _l2_norm_t(s_feat)
            if is_xla and xm_mod is not None:
                xm_mod.mark_step()
            pe_out[b_st : b_st + n_real] = pe_norm[:n_real].cpu().numpy().astype(np.float16)
            sig_out[b_st : b_st + n_real] = s_norm[:n_real].cpu().numpy().astype(np.float16)
    return pe_out, sig_out

for c_st in range(0, len(video_ids), CHUNK_VIDS):
    c_vids = video_ids[c_st : c_st + CHUNK_VIDS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sampled = list(pool.map(_sample_12_frames, c_vids))

    flat_imgs = [im for _, imgs, _ in sampled for im in imgs]
    if len(devices) >= 2 and len(flat_imgs) >= 2:
        mid = len(flat_imgs) // 2
        with ThreadPoolExecutor(max_workers=2) as gpool:
            f0 = gpool.submit(_encode_image_chunk, flat_imgs[:mid], 0)
            f1 = gpool.submit(_encode_image_chunk, flat_imgs[mid:], 1)
            pe_0, sig_0 = f0.result()
            pe_1, sig_1 = f1.result()
            pe_chunk = np.vstack([pe_0, pe_1])
            sig_chunk = np.vstack([sig_0, sig_1])
    else:
        pe_chunk, sig_chunk = _encode_image_chunk(flat_imgs, 0)

    for loc_i, (vid, _, rec) in enumerate(sampled):
        v_idx = c_st + loc_i
        st_f = loc_i * NUM_VL2_FRAMES
        ed_f = st_f + NUM_VL2_FRAMES
        pf = pe_chunk[st_f:ed_f]
        sf = sig_chunk[st_f:ed_f]
        pe_frame_vecs[v_idx] = pf
        sig400_frame_vecs[v_idx] = sf

        pm = pf.astype(np.float32).mean(axis=0)
        pe_video_vecs[v_idx] = (pm / max(1e-12, float(np.linalg.norm(pm)))).astype(np.float32)
        sm = sf.astype(np.float32).mean(axis=0)
        sig400_video_vecs[v_idx] = (sm / max(1e-12, float(np.linalg.norm(sm)))).astype(np.float32)

        video_rows_out.append({
            "row_id": v_idx,
            "video_id": vid,
            "selected_ordinals": rec["selected_ordinals"],
            "selected_pts_sec": rec["selected_timestamps_sec"],
            "recipe_hash": rec["recipe_hash"],
        })

    done_v = c_st + len(c_vids)
    if (c_st // CHUNK_VIDS) % LOG_EVERY == 0 or done_v == len(video_ids):
        print(f"[{shard_name}] {done_v}/{len(video_ids)} videos in {time.time()-t_start:.1f}s", flush=True)

n_q = len(query_rows_in)
pe_query_vecs = np.zeros((n_q, PE_DIM), dtype=np.float32)
sig400_query_vecs = np.zeros((n_q, SIG400_DIM), dtype=np.float32)
dev0, pe0, s40 = devices[0], pe_models[0], sig400_models[0]
pe_ctx = int(getattr(pe0, "context_length", 32))

Q_BS = 128
for st in range(0, n_q, Q_BS):
    chunk = query_rows_in[st : st + Q_BS]
    n_real = len(chunk)
    texts_raw = [q["query_text"] for q in chunk]
    if is_xla and n_real < Q_BS:
        texts_raw = texts_raw + [texts_raw[-1]] * (Q_BS - n_real)
    texts_lower = [t.lower() for t in texts_raw]
    try:
        pe_in = pe_tok(texts_raw, context_length=pe_ctx).to(dev0)
    except TypeError:
        pe_in = pe_tok(texts_raw).to(dev0)
    sig_in = sig400_proc(text=texts_lower, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    sig_in = {k: v.to(dev0) for k, v in sig_in.items()}
    with torch.no_grad():
        if n_gpus > 0:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pe_qf = pe0.encode_text(pe_in)
                s_qraw = s40.get_text_features(**sig_in)
        elif not is_xla and dtype == torch.bfloat16:
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                pe_qf = pe0.encode_text(pe_in)
                s_qraw = s40.get_text_features(**sig_in)
        else:
            pe_qf = pe0.encode_text(pe_in)
            s_qraw = s40.get_text_features(**sig_in)
        s_qf = s_qraw.pooler_output if hasattr(s_qraw, "pooler_output") and s_qraw.pooler_output is not None else s_qraw
        pe_qn = _l2_norm_t(pe_qf)
        s_qn = _l2_norm_t(s_qf)
        if is_xla and xm_mod is not None:
            xm_mod.mark_step()
        pe_query_vecs[st : st + n_real] = pe_qn[:n_real].cpu().numpy().astype(np.float32)
        sig400_query_vecs[st : st + n_real] = s_qn[:n_real].cpu().numpy().astype(np.float32)

assert np.isfinite(pe_video_vecs).all() and np.isfinite(pe_frame_vecs).all() and np.isfinite(pe_query_vecs).all()
assert np.isfinite(sig400_video_vecs).all() and np.isfinite(sig400_frame_vecs).all() and np.isfinite(sig400_query_vecs).all()

out_dir = WORK / shard_name
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "pe_video_vectors.npy", pe_video_vecs)
np.save(out_dir / "pe_frame_vectors_fp16.npy", pe_frame_vecs)
np.save(out_dir / "pe_query_vectors.npy", pe_query_vecs)
np.save(out_dir / "sig400_video_vectors.npy", sig400_video_vecs)
np.save(out_dir / "sig400_frame_vectors_fp16.npy", sig400_frame_vecs)
np.save(out_dir / "sig400_query_vectors.npy", sig400_query_vecs)

def _wjsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

_wjsonl(out_dir / "video_rows.jsonl", video_rows_out)
_wjsonl(out_dir / "query_rows.jsonl", query_rows_in)

f_names = (
    "pe_video_vectors.npy", "pe_frame_vectors_fp16.npy", "pe_query_vectors.npy",
    "sig400_video_vectors.npy", "sig400_frame_vectors_fp16.npy", "sig400_query_vectors.npy",
    "video_rows.jsonl", "query_rows.jsonl",
)
files_info = {fn: {"sha256": sha256_file(out_dir / fn), "bytes": (out_dir / fn).stat().st_size} for fn in f_names}
receipt = {
    "status": "SUCCESS",
    "shard_name": shard_name,
    "models": {
        "pe_core": {"model_id": PE_MODEL_ID, "revision": PE_REVISION, "weights_sha256": PE_WEIGHTS_SHA256, "license": PE_LICENSE, "num_frames": NUM_VL2_FRAMES, "dim": PE_DIM},
        "siglip2_so400m": {"model_id": SIG400_MODEL_ID, "revision": SIG400_REVISION, "weights_sha256": SIG400_WEIGHTS_SHA256, "license": SIG400_LICENSE, "num_frames": NUM_VL2_FRAMES, "dim": SIG400_DIM},
    },
    "video_count": len(video_rows_out),
    "query_count": len(query_rows_in),
    "devices": dev_labels,
    "elapsed_seconds": round(time.time() - t_start, 2),
    "files": files_info,
}
(out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("VL2 SHARD DONE:", shard_name, receipt["elapsed_seconds"], "s", flush=True)
'''


def run_pilot_vp(protocol_root: Path, *, push: bool = False) -> dict[str, Any]:
    m_dir = protocol_root / "manifests"
    expected_by_split: dict[str, list[str]] = {}
    for g in ("tune_g1", "tune_g2", "confirm_g1", "confirm_g2", "test_1ka"):
        expected_by_split[g] = [r["video_id"] for r in _read_jsonl(m_dir / g / "gallery_manifest.jsonl")]
    expected_by_split["train_9k_clean"] = [r["video_id"] for r in _read_jsonl(m_dir / "train_9k_clean" / "train_videos.jsonl")]

    tune_qs = _read_jsonl(m_dir / "tune_g1" / "queries.jsonl")[:8]
    assert_no_gt_leakage(tune_qs, context="pilot_queries")
    spec = {
        "expected_by_split": expected_by_split,
        "pilot_videos": expected_by_split["tune_g1"][:16],
        "pilot_queries": tune_qs,
    }
    gz_bytes = gzip.compress(json.dumps(spec, separators=(",", ":")).encode("utf-8"), compresslevel=9)
    src_files = (*_MINIMAL_SRC_FILES, *_VP_EXTRA_SCRIPTS)
    b64 = _pack_source_b64(src_files, {"inventory_spec.json.gz": gz_bytes})
    launcher = f"SOURCE = {b64!r}\n" + _PILOT_VP_TEMPLATE.lstrip("\n")
    return _write_and_push_kernel(
        kernel_dir=KAGGLE_WORK_DIR / "vp_pilot",
        kernel_id="lvhuylong/videntra-msrvtt-vp-pilot-20261122",
        title="VIDENTRA MSRVTT VP Pilot 20261122",
        launcher_code=launcher,
        datasets=MSRVTT_KAGGLE_DATASETS,
        push=push,
    )


def run_shard_job(
    kind: str,
    target: str,
    *,
    protocol_root: Path,
    push: bool = False,
    allow_sealed_test: bool = False,
    include_1ka_gallery_videos_only: bool = False,
    accelerator: str = "gpu",
    num_slices: int = 1,
    slice_index: int = 0,
) -> dict[str, Any]:
    if int(num_slices) < 1:
        raise ValueError(f"num_slices must be >= 1, got {num_slices}")
    if not (0 <= int(slice_index) < int(num_slices)):
        raise ValueError(f"slice_index {slice_index} out of range for num_slices={num_slices}")
    vids, base_queries, expand_variants = _collect_target_payload(
        protocol_root,
        target,
        allow_sealed_test=allow_sealed_test,
        include_1ka_gallery_videos_only=include_1ka_gallery_videos_only,
    )
    shard_name = f"{kind}_{target}"
    if int(num_slices) > 1:
        vids = vids[int(slice_index)::int(num_slices)]
        if int(slice_index) > 0:
            base_queries = []
        shard_name = f"{shard_name}_p{int(slice_index)}"
    payload_obj = {
        "shard_name": shard_name,
        "video_ids": vids,
        "base_queries": base_queries,
        "expand_variants": expand_variants,
    }
    gz_bytes = gzip.compress(json.dumps(payload_obj, separators=(",", ":")).encode("utf-8"), compresslevel=9)

    if kind == "vp":
        src_files: Sequence[str] = (*_MINIMAL_SRC_FILES, *_VP_EXTRA_SCRIPTS)
        template = _VP_SHARD_TEMPLATE
    elif kind == "mm":
        src_files = _MINIMAL_SRC_FILES
        template = _MM_SHARD_TEMPLATE
    elif kind == "asr":
        src_files = ("src/videntra/__init__.py",)
        template = _ASR_SHARD_TEMPLATE
    elif kind == "cap":
        src_files = _MINIMAL_SRC_FILES
        template = _CAP_SHARD_TEMPLATE
    elif kind == "vl2":
        src_files = (*_MINIMAL_SRC_FILES, "scripts/prepare_clip_vp_kaggle_bundle.py")
        template = _VL2_SHARD_TEMPLATE
    else:
        raise ValueError(f"Unknown shard kind: {kind}")

    b64 = _pack_source_b64(src_files, {"job_payload.json.gz": gz_bytes})
    launcher = f"SOURCE = {b64!r}\n" + template.lstrip("\n")
    acc = str(accelerator).lower()
    acc_tag = "" if acc == "gpu" else f"-{acc}"
    acc_title = "" if acc == "gpu" else f" {acc.upper()}"
    slug_safe = (
        shard_name.replace("_", "-")
        .replace("test-1ka-queries-only", "t1ka-qonly")
        .replace("test-1ka-videos-only", "t1kav")
    )
    title_core = slug_safe.replace("-", " ").upper()
    if (kind == "cap" and target == "val_tune") or (kind == "vl2" and target == "val_confirm"):
        kernel_id = f"lvhuylong/videntra-msr-{slug_safe}-r3{acc_tag}-20261122"
        title = f"VIDENTRA MSR {title_core} R3{acc_title} 20261122"
    elif kind == "vl2" and target == "train_qbank_plus_1ka_videos":
        kernel_id = f"lvhuylong/videntra-msr-vl2-qb1ka{acc_tag}-20261122"
        title = f"VIDENTRA MSR VL2 QB1KA{acc_title} 20261122"
    elif kind == "cap":
        kernel_id = f"lvhuylong/videntra-msr-{slug_safe}-r2{acc_tag}-20261122"
        title = f"VIDENTRA MSR {title_core} R2{acc_title} 20261122"
    else:
        kernel_id = f"lvhuylong/videntra-msr-{slug_safe}{acc_tag}-20261122"
        title = f"VIDENTRA MSR {title_core}{acc_title} 20261122"
    res = _write_and_push_kernel(
        kernel_dir=KAGGLE_WORK_DIR / shard_name,
        kernel_id=kernel_id,
        title=title,
        launcher_code=launcher,
        datasets=MSRVTT_KAGGLE_DATASETS,
        push=push,
        accelerator=acc,
    )
    res.update({
        "shard_name": shard_name,
        "video_count": len(vids),
        "query_count": len(base_queries),
        "num_slices": int(num_slices),
        "slice_index": int(slice_index),
    })
    return res


def pull_and_verify(kernel_id: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(_ROOT / ".venv/Scripts/kaggle.exe"), "kernels", "output", kernel_id, "-p", str(out_dir)],
        check=True,
    )
    receipts = list(out_dir.rglob("receipt.json")) + list(out_dir.rglob("pilot_receipt.json"))
    if not receipts:
        raise FileNotFoundError(f"No receipt.json found in downloaded {out_dir}")
    rpath = receipts[0]
    rec = json.loads(rpath.read_text(encoding="utf-8"))
    if rec.get("status") != "SUCCESS":
        raise ValueError(f"Receipt status != SUCCESS: {rec.get('status')}")
    for fname, info in (rec.get("files") or {}).items():
        fpath = rpath.parent / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing artifact {fpath}")
        actual = sha256_file(fpath)
        if actual != info.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {fpath}: expected {info.get('sha256')}, got {actual}")
    return {"receipt_path": str(rpath), "status": rec["status"], "verified_files": len(rec.get("files") or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate Kaggle GPU jobs for MSR-VTT 1K-A.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pilot = sub.add_parser("pilot-vp")
    p_pilot.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    p_pilot.add_argument("--push", action="store_true")

    for kind in ("vp-shard", "mm-shard", "asr-shard", "cap-shard", "vl2-shard"):
        sp = sub.add_parser(kind)
        sp.add_argument("--target", required=True)
        sp.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
        sp.add_argument("--push", action="store_true")
        sp.add_argument("--allow-sealed-test", action="store_true")
        sp.add_argument("--include-1ka-gallery-videos-only", action="store_true")
        sp.add_argument("--accelerator", choices=("gpu", "tpu", "cpu"), default="gpu")
        sp.add_argument("--num-slices", type=int, default=1)
        sp.add_argument("--slice-index", type=int, default=0)

    p_st = sub.add_parser("status")
    p_st.add_argument("--kernel-id", required=True)

    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--kernel-id", required=True)
    p_pull.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "pilot-vp":
        res = run_pilot_vp(args.protocol_root, push=args.push)
    elif args.cmd in ("vp-shard", "mm-shard", "asr-shard", "cap-shard", "vl2-shard"):
        kind = args.cmd.split("-")[0]
        res = run_shard_job(
            kind,
            args.target,
            protocol_root=args.protocol_root,
            push=args.push,
            allow_sealed_test=args.allow_sealed_test,
            include_1ka_gallery_videos_only=args.include_1ka_gallery_videos_only,
            accelerator=args.accelerator,
            num_slices=args.num_slices,
            slice_index=args.slice_index,
        )
    elif args.cmd == "status":
        out = subprocess.check_output(
            [str(_ROOT / ".venv/Scripts/kaggle.exe"), "kernels", "status", args.kernel_id],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        res = {"kernel_id": args.kernel_id, "status": out}
    elif args.cmd == "pull":
        res = pull_and_verify(args.kernel_id, args.out_dir)
    else:
        raise ValueError(args.cmd)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
