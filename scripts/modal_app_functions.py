"""Global-scope Modal App definitions for VIDENTRA ActivityNet Captions & VideoComp evaluation."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence
import urllib.error
import urllib.request

import modal

from scripts.modal_videntra_worker import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_VOLUME_NAME,
    VOLUME_MOUNT_PATH,
    _ConcatPartReader,
    _extract_dense_embeddings_on_gpu,
    _probe_video_file_pts,
    normalize_media_key,
    run_vlm_rerank_pairs_local_or_gpu,
    shard_artifact_dir,
)

app = modal.App(DEFAULT_MODAL_APP_NAME)
vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
_MAX_CONTAINERS = max(1, int(os.environ.get("VIDENTRA_MODAL_MAX_CONTAINERS", "4")))
_REPO_SCRIPTS = Path(__file__).resolve().parent

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "tar", "unzip")
    .pip_install("huggingface_hub", "hf_transfer", "av", "numpy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root"})
    .add_local_python_source("videntra")
    .add_local_dir(str(_REPO_SCRIPTS), remote_path="/root/scripts", ignore=["__pycache__", "*.pyc"])
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.49.0",
        "open_clip_torch",
        "timm",
        "bitsandbytes",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "safetensors",
        "av",
        "pillow",
        "numpy",
        "nvidia-ml-py",
        "jsonschema",
        "PyYAML",
        "scipy",
        "pyarrow",
        "pydantic",
        "psutil",
        "python-dotenv",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{VOLUME_MOUNT_PATH}/hf_cache",
        "PYTHONPATH": "/root",
    })
    .add_local_python_source("videntra")
    .add_local_dir(str(_REPO_SCRIPTS), remote_path="/root/scripts", ignore=["__pycache__", "*.pyc"])
)


@app.function(
    image=cpu_image,
    volumes={VOLUME_MOUNT_PATH: vol},
    cpu=8.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    timeout=7200,
)
def sync_raw_media_to_volume(
    target_anet_ids_json: str,
    target_yc2_ids_json: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download ActivityNet + YouCook2 HF mirrors onto Modal Volume, extract targeted media & PyAV probe."""
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    import tarfile
    import zipfile
    from huggingface_hub import hf_hub_download  # type: ignore

    vol.reload()
    vroot = Path(VOLUME_MOUNT_PATH) / "raw_media"
    anet_dir = vroot / "activitynet"
    yc2_dir = vroot / "youcook2"
    catalog_path = vroot / "media_catalog.json"
    if catalog_path.is_file() and not force:
        return json.loads(catalog_path.read_text(encoding="utf-8"))

    anet_targets = {normalize_media_key(x) for x in json.loads(target_anet_ids_json)}
    yc2_targets = {normalize_media_key(x) for x in json.loads(target_yc2_ids_json)}
    anet_dir.mkdir(parents=True, exist_ok=True)
    yc2_dir.mkdir(parents=True, exist_ok=True)
    dl_tmp = vroot / "_dl_tmp"
    dl_tmp.mkdir(parents=True, exist_ok=True)

    # 1. ActivityNet Captions multi-part tarball stream extraction (~39 GB -> target IDs only)
    existing_anet = {normalize_media_key(p.name) for p in anet_dir.iterdir() if p.is_file()}
    if len(existing_anet & anet_targets) < max(1, int(0.8 * len(anet_targets))) or force:
        parts: list[Path] = []
        for p_idx in range(8):
            fn = f"ActivityNet_Videos.tar.part-{p_idx:03d}"
            lp = Path(hf_hub_download("friedrichor/ActivityNet_Captions", fn, repo_type="dataset", local_dir=str(dl_tmp)))
            parts.append(lp)
            print(f"[sync] Downloaded {fn}", flush=True)
        with _ConcatPartReader(parts) as stream_f:
            with tarfile.open(fileobj=stream_f, mode="r|*") as tf:
                extracted = 0
                for member in tf:
                    if not member.isfile():
                        continue
                    n_key = normalize_media_key(member.name)
                    if anet_targets and n_key not in anet_targets:
                        continue
                    ext = Path(member.name).suffix.lower() or ".mp4"
                    dst = anet_dir / f"{n_key}{ext}"
                    if dst.is_file() and dst.stat().st_size > 0:
                        continue
                    src_f = tf.extractfile(member)
                    if src_f is not None:
                        with dst.open("wb") as out_f:
                            shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
                        extracted += 1
                        if extracted % 1000 == 0:
                            print(f"[sync] ActivityNet extracted {extracted} files", flush=True)
        shutil.rmtree(dl_tmp, ignore_errors=True)
        vol.commit()
        dl_tmp.mkdir(parents=True, exist_ok=True)

    # 2. YouCook2 (`lmms-eval/YouCook2` zip first, fallback `Windsao/Processed_YouCook2` for missing IDs)
    existing_yc2 = {normalize_media_key(p.name) for p in yc2_dir.iterdir() if p.is_file()}
    if len(existing_yc2 & yc2_targets) < max(1, int(0.5 * len(yc2_targets))) or force:
        zp = Path(hf_hub_download("lmms-eval/YouCook2", "YouCookIIVideos.zip", repo_type="dataset", local_dir=str(dl_tmp)))
        print("[sync] Downloaded YouCookIIVideos.zip", flush=True)
        with zipfile.ZipFile(zp, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                n_key = normalize_media_key(info.filename)
                if yc2_targets and n_key not in yc2_targets:
                    continue
                ext = Path(info.filename).suffix.lower() or ".mp4"
                dst = yc2_dir / f"{n_key}{ext}"
                if dst.is_file() and dst.stat().st_size > 0:
                    continue
                with zf.open(info, "r") as src_f, dst.open("wb") as out_f:
                    shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
        shutil.rmtree(dl_tmp, ignore_errors=True)
        vol.commit()
        dl_tmp.mkdir(parents=True, exist_ok=True)

    found_yc2 = {normalize_media_key(p.name) for p in yc2_dir.iterdir() if p.is_file()}
    missing_yc2 = yc2_targets - found_yc2
    print(f"[sync] YouCook2 after zip: found={len(found_yc2)} missing={len(missing_yc2)}", flush=True)
    if len(missing_yc2) > max(20, int(0.4 * len(yc2_targets))):
        tzp = Path(hf_hub_download("Windsao/Processed_YouCook2", "Processed_YouCook2.tar.gz", repo_type="dataset", local_dir=str(dl_tmp)))
        print("[sync] Downloaded Processed_YouCook2.tar.gz", flush=True)
        with tarfile.open(tzp, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                n_key = normalize_media_key(member.name)
                if n_key not in missing_yc2:
                    continue
                ext = Path(member.name).suffix.lower() or ".mp4"
                dst = yc2_dir / f"{n_key}{ext}"
                if dst.is_file() and dst.stat().st_size > 0:
                    continue
                src_f = tf.extractfile(member)
                if src_f is not None:
                    with dst.open("wb") as out_f:
                        shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
        shutil.rmtree(dl_tmp, ignore_errors=True)
        vol.commit()

    shutil.rmtree(dl_tmp, ignore_errors=True)

    tasks = [
        ("activitynet", vroot, p) for p in sorted(anet_dir.iterdir()) if p.is_file()
    ] + [
        ("youcook2", vroot, p) for p in sorted(yc2_dir.iterdir()) if p.is_file()
    ]
    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(_probe_video_file_pts, tasks):
            items.append(res)

    ok_items = [it for it in items if it.get("ok")]
    payload = {
        "total_probed": len(items),
        "valid_count": len(ok_items),
        "activitynet_valid": sum(1 for it in ok_items if it["domain"] == "activitynet"),
        "youcook2_valid": sum(1 for it in ok_items if it["domain"] == "youcook2"),
        "items": items,
    }
    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    vol.commit()
    return payload


class _ConcatURLReader(io.RawIOBase):
    """Stream concatenated multi-part HTTP URLs sequentially with Range-resume on drops."""

    def __init__(self, urls: Sequence[str], headers: dict[str, str] | None = None, max_retries: int = 8) -> None:
        self._urls = list(urls)
        self._headers = dict(headers or {})
        self._max_retries = int(max_retries)
        self._idx = 0
        self._offset = 0
        self._expected_part_len: int | None = None
        self._resp = None
        self.total_bytes_read = 0
        self._open_current(resume_offset=0)

    def readable(self) -> bool:
        return True

    def _open_current(self, resume_offset: int) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        if self._idx >= len(self._urls):
            return
        url = self._urls[self._idx]
        for attempt in range(self._max_retries):
            hdrs = dict(self._headers)
            hdrs.setdefault("User-Agent", "VIDENTRA-ModalSync/1.0")
            if resume_offset > 0:
                hdrs["Range"] = f"bytes={resume_offset}-"
            req = urllib.request.Request(url, headers=hdrs)
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                status = getattr(resp, "status", 200)
                if resume_offset > 0 and status == 200:
                    # Server ignored Range header; discard prefix safely or reset
                    resp.close()
                    raise RuntimeError(f"Server did not honor Range request for {url}")
                cr = resp.headers.get("Content-Range")
                cl = resp.headers.get("Content-Length")
                if cr and "/" in cr:
                    tot_s = cr.rsplit("/", 1)[-1].strip()
                    if tot_s.isdigit():
                        self._expected_part_len = int(tot_s)
                elif resume_offset == 0 and cl and cl.isdigit():
                    self._expected_part_len = int(cl)
                self._resp = resp
                return
            except Exception as exc:
                if attempt + 1 >= self._max_retries:
                    raise RuntimeError(f"Failed opening {url} at offset {resume_offset}: {exc}") from exc
                time.sleep(min(30.0, 2.0 ** attempt))

    def _advance_part(self) -> bool:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        self._idx += 1
        self._offset = 0
        self._expected_part_len = None
        if self._idx < len(self._urls):
            print(f"[sync-yc2] Advancing to archive part {self._idx + 1}/{len(self._urls)}", flush=True)
            self._open_current(resume_offset=0)
            return self._resp is not None
        return False

    def read(self, size: int = -1) -> bytes:
        if self._resp is None and self._idx >= len(self._urls):
            return b""
        want = 1024 * 1024 if (size is None or size < 0) else int(size)
        if want == 0:
            return b""
        buf = bytearray()
        retries = 0
        while len(buf) < want and self._idx < len(self._urls):
            if self._resp is None:
                self._open_current(resume_offset=self._offset)
                if self._resp is None:
                    break
            try:
                chunk = self._resp.read(want - len(buf))
                if chunk:
                    buf.extend(chunk)
                    self._offset += len(chunk)
                    self.total_bytes_read += len(chunk)
                    retries = 0
                    continue
                # Hit EOF on current response: verify whether full part arrived
                if self._expected_part_len is not None and self._offset < self._expected_part_len:
                    retries += 1
                    if retries >= self._max_retries:
                        raise EOFError(
                            f"Truncated part {self._idx}: {self._offset}/{self._expected_part_len}"
                        )
                    time.sleep(min(20.0, 1.5 ** retries))
                    self._open_current(resume_offset=self._offset)
                    continue
                if not self._advance_part():
                    break
            except Exception as exc:
                retries += 1
                if retries >= self._max_retries:
                    raise RuntimeError(
                        f"Stream read error on part {self._idx} at offset {self._offset}: {exc}"
                    ) from exc
                time.sleep(min(20.0, 1.5 ** retries))
                self._open_current(resume_offset=self._offset)
        return bytes(buf)

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        super().close()


@app.function(
    image=cpu_image,
    volumes={VOLUME_MOUNT_PATH: vol},
    cpu=8.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    timeout=7200,
)
def sync_youcook2_to_volume(
    target_yc2_ids_json: str,
    force: bool = False,
) -> dict[str, Any]:
    """Stream `pengxiang/youcook2` tar parts over HTTPS and extract matching YouCook2 videos to Volume."""
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    import tarfile
    from huggingface_hub import HfApi, hf_hub_url  # type: ignore

    vol.reload()
    vroot = Path(VOLUME_MOUNT_PATH) / "raw_media"
    anet_dir = vroot / "activitynet"
    yc2_dir = vroot / "youcook2"
    catalog_path = vroot / "media_catalog.json"
    anet_dir.mkdir(parents=True, exist_ok=True)
    yc2_dir.mkdir(parents=True, exist_ok=True)

    yc2_targets = {normalize_media_key(x) for x in json.loads(target_yc2_ids_json)}
    if force:
        for p in yc2_dir.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)

    existing_yc2 = {
        normalize_media_key(p.name)
        for p in yc2_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.stat().st_size > 0
    }
    remaining = set(yc2_targets - existing_yc2)
    print(
        f"[sync-yc2] Target IDs={len(yc2_targets)}, already_present={len(existing_yc2)}, remaining={len(remaining)}",
        flush=True,
    )

    if remaining:
        api = HfApi()
        repo_files = api.list_repo_files("pengxiang/youcook2", repo_type="dataset")
        part_files = sorted(f for f in repo_files if re.match(r"^raw_videos\.part[a-z]{2}$", f))
        if not part_files:
            raise RuntimeError(f"No raw_videos.partXX files found in pengxiang/youcook2: {repo_files[:20]}")
        urls = [hf_hub_url("pengxiang/youcook2", fn, repo_type="dataset") for fn in part_files]
        print(f"[sync-yc2] Streaming {len(urls)} tar parts: {part_files[0]} .. {part_files[-1]}", flush=True)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        extracted = 0
        last_log_gib = 0
        with _ConcatURLReader(urls, headers=headers) as stream_f:
            with tarfile.open(fileobj=stream_f, mode="r|", bufsize=1024 * 1024) as tf:
                for member in tf:
                    gib_now = int(stream_f.total_bytes_read // (20 * 1024 * 1024 * 1024))
                    if gib_now > last_log_gib:
                        last_log_gib = gib_now
                        print(
                            f"[sync-yc2] Streamed ~{stream_f.total_bytes_read / (1024**3):.1f} GiB | "
                            f"extracted={extracted} | remaining={len(remaining)}",
                            flush=True,
                        )
                    if not member.isfile():
                        continue
                    n_key = normalize_media_key(member.name)
                    if yc2_targets and n_key not in yc2_targets:
                        continue
                    ext = Path(member.name).suffix.lower() or ".mp4"
                    dst = yc2_dir / f"{n_key}{ext}"
                    if not dst.is_file() or dst.stat().st_size == 0:
                        src_f = tf.extractfile(member)
                        if src_f is not None:
                            tmp_dst = yc2_dir / f".{n_key}{ext}.tmp"
                            with tmp_dst.open("wb") as out_f:
                                shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
                            tmp_dst.replace(dst)
                            extracted += 1
                            if extracted % 200 == 0:
                                print(f"[sync-yc2] Extracted {extracted} target YouCook2 videos", flush=True)
                    remaining.discard(n_key)
                    if not remaining:
                        print("[sync-yc2] All target YouCook2 videos found! Stopping stream early.", flush=True)
                        break
        vol.commit()

    # Probe YouCook2 videos and preserve already-probed ActivityNet entries when available
    prev_items: list[dict[str, Any]] = []
    if catalog_path.is_file():
        try:
            prev_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            prev_items = [it for it in prev_payload.get("items", []) if it.get("domain") == "activitynet"]
        except Exception:
            prev_items = []

    if not prev_items:
        anet_tasks = [("activitynet", vroot, p) for p in sorted(anet_dir.iterdir()) if p.is_file()]
        with ThreadPoolExecutor(max_workers=8) as pool:
            prev_items = list(pool.map(_probe_video_file_pts, anet_tasks))

    yc2_tasks = [
        ("youcook2", vroot, p)
        for p in sorted(yc2_dir.iterdir())
        if p.is_file() and not p.name.startswith(".")
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        yc2_items = list(pool.map(_probe_video_file_pts, yc2_tasks))

    items = prev_items + yc2_items
    ok_items = [it for it in items if it.get("ok")]
    payload = {
        "total_probed": len(items),
        "valid_count": len(ok_items),
        "activitynet_valid": sum(1 for it in ok_items if it["domain"] == "activitynet"),
        "youcook2_valid": sum(1 for it in ok_items if it["domain"] == "youcook2"),
        "yc2_targets_requested": len(yc2_targets),
        "yc2_targets_missing": sorted(remaining),
        "items": items,
    }
    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    vol.commit()
    return payload


@app.function(
    image=gpu_image,
    gpu=["L4", "A10G"],
    volumes={VOLUME_MOUNT_PATH: vol},
    cpu=8.0,
    memory=32768,
    min_containers=0,
    max_containers=_MAX_CONTAINERS,
    timeout=7200,
)
def run_l4_extract_shard(
    run_id: str,
    shard_id: str,
    video_items_json: str,
    text_items_json: str,
    num_frames: int = 48,
    backbones_json: str = '["pe_bigg","pe_l","sig400"]',
) -> dict[str, Any]:
    vol.reload()
    res = _extract_dense_embeddings_on_gpu(
        vol_root=Path(VOLUME_MOUNT_PATH),
        run_id=run_id,
        gpu_tag="L4",
        shard_id=shard_id,
        video_items=json.loads(video_items_json),
        text_items=json.loads(text_items_json),
        num_frames=int(num_frames),
        backbones=json.loads(backbones_json),
    )
    vol.commit()
    return res


@app.function(
    image=gpu_image,
    gpu="L40S",
    volumes={VOLUME_MOUNT_PATH: vol},
    cpu=8.0,
    memory=32768,
    min_containers=0,
    max_containers=_MAX_CONTAINERS,
    timeout=7200,
)
def run_l40s_extract_shard(
    run_id: str,
    shard_id: str,
    video_items_json: str,
    text_items_json: str,
    num_frames: int = 48,
    backbones_json: str = '["pe_bigg","pe_l","sig400"]',
) -> dict[str, Any]:
    vol.reload()
    res = _extract_dense_embeddings_on_gpu(
        vol_root=Path(VOLUME_MOUNT_PATH),
        run_id=run_id,
        gpu_tag="L40S",
        shard_id=shard_id,
        video_items=json.loads(video_items_json),
        text_items=json.loads(text_items_json),
        num_frames=int(num_frames),
        backbones=json.loads(backbones_json),
    )
    vol.commit()
    return res


@app.function(
    image=gpu_image,
    gpu=["L40S", "A10G", "L4"],
    volumes={VOLUME_MOUNT_PATH: vol},
    min_containers=0,
    max_containers=_MAX_CONTAINERS,
    timeout=7200,
)
def run_l4_vlm_rerank_shard(
    run_id: str,
    model_tag: str,
    shard_id: str,
    pairs_json: str,
    rel_path_map_json: str,
    quant: str = "nf4_4bit",
    num_frames: int = 8,
    annotate_timestamps: bool = False,
    template_kind: str = "default",
) -> dict[str, Any]:
    from videntra.ranking.vlm_reranker import (
        COMPOSITIONAL_VLM_QUERY_TEMPLATE,
        DEFAULT_VLM_QUERY_TEMPLATE,
    )

    vol.reload()
    raw_root = Path(VOLUME_MOUNT_PATH) / "raw_media"
    pairs = json.loads(pairs_json)
    vpaths = {str(k): raw_root / str(v) for k, v in json.loads(rel_path_map_json).items()}
    sdir = shard_artifact_dir(Path(VOLUME_MOUNT_PATH), run_id=run_id, model_tag=model_tag, shard_id=shard_id)
    cache_p = sdir / "pair_cache.jsonl"
    tmpl = (
        COMPOSITIONAL_VLM_QUERY_TEMPLATE
        if str(template_kind).strip().lower() == "compositional"
        else DEFAULT_VLM_QUERY_TEMPLATE
    )
    res = run_vlm_rerank_pairs_local_or_gpu(
        pairs,
        video_paths_by_id=vpaths,
        shard_dir=sdir,
        cache_path=cache_p,
        quant=quant,
        num_frames=int(num_frames),
        annotate_timestamps=annotate_timestamps,
        query_template=tmpl,
    )
    vol.commit()
    return res
