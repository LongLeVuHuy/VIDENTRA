"""Prepare and freeze the leakage-free MSR-VTT 1K-A comparison protocol (seed=20261122).

Pins exact source files and SHA-256 digests:
- val_list_jsfusion.txt (1,000 test video IDs)
- MSRVTT_JSFUSION_test.csv (1,000 official 1K-A test queries)
- train_list_jsfusion.txt (9,000 non-1K-A video IDs)
- MSRVTT_data.json (10,000 video URLs/categories + 200,000 captions)

Excludes any 9K video sharing a normalized YouTube source_group with 1K-A,
partitions 4,000 validation videos into 4 disjoint source-grouped galleries
(tune_g1, tune_g2, confirm_g1, confirm_g2; 1,000 videos & 1,000 queries each),
assigns the remaining eligible videos (4,585) to train_9k_clean, and keeps
1K-A ground-truth labels and test query payload isolated in eval_only/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
from typing import Any, Mapping, Sequence
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.evaluation.split_lock import freeze_split, verify_split_lock  # noqa: E402

PROTOCOL_SEED = 20261122

PINNED_SOURCES: dict[str, dict[str, Any]] = {
    "val_list_jsfusion.txt": {
        "url": "https://raw.githubusercontent.com/albanie/collaborative-experts/master/misc/datasets/msrvtt/val_list_jsfusion.txt",
        "sha256": "49a09fb68f7ec52b3b9bbb7e856ffbd3f00233976ed435fded658309c3cd491a",
        "bytes": 9999,
    },
    "MSRVTT_JSFUSION_test.csv": {
        "url": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/raw_data/MSRVTT_JSFUSION_test.csv",
        "sha256": "e61144b312c8a2fc2126f96504c981f8c23c05b3ba9655a56e2af50ac8a4b16a",
        "bytes": 74475,
    },
    "train_list_jsfusion.txt": {
        "url": "https://raw.githubusercontent.com/albanie/collaborative-experts/master/misc/datasets/msrvtt/train_list_jsfusion.txt",
        "sha256": "6d34fc995c696c36efc89911eb6da41322ada4dd80941aa99e28a23cffcbb3c3",
        "bytes": 88889,
    },
    "MSRVTT_data.json": {
        "url": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/raw_data/MSRVTT_data.json",
        "sha256": "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014",
        "bytes": 23113733,
    },
}

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
_FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "target_videos",
    "target_windows",
    "relevant_windows",
    "source_group",
    "source_group_status",
    "known_target_window_id",
})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_youtube_source_group(url: str | None, video_id: str) -> str:
    raw = str(url or "").strip()
    match = _YT_ID_RE.search(raw)
    if match:
        return f"yt:{match.group(1)}"
    if raw:
        return f"url:{raw}"
    return f"vid:{video_id}"


def download_and_verify_sources(
    provenance_dir: Path,
    *,
    timeout_sec: int = 45,
) -> dict[str, dict[str, Any]]:
    """Download pinned MSR-VTT sources if absent and verify SHA-256 byte-for-byte."""
    provenance_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, Any]] = {}
    for fname, spec in PINNED_SOURCES.items():
        fpath = provenance_dir / fname
        if not fpath.is_file():
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
            tmp = fpath.with_suffix(fpath.suffix + ".tmp")
            tmp.write_bytes(raw)
            tmp.replace(fpath)
        actual_sha = sha256_file(fpath)
        actual_bytes = fpath.stat().st_size
        if actual_sha != spec["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {fname}: expected {spec['sha256']}, got {actual_sha}"
            )
        if actual_bytes != spec["bytes"]:
            raise ValueError(
                f"Byte-size mismatch for {fname}: expected {spec['bytes']}, got {actual_bytes}"
            )
        receipts[fname] = {
            "path": str(fpath),
            "url": spec["url"],
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }
    return receipts


def select_deterministic_caption(
    video_id: str,
    sentences: Sequence[Mapping[str, Any]],
    *,
    seed: int = PROTOCOL_SEED,
) -> dict[str, Any]:
    """Select one original caption deterministically from seed + sorted sen_id."""
    ordered = sorted(
        (
            {
                "sen_id": int(s["sen_id"]),
                "caption": str(s["caption"]).strip(),
                "video_id": str(s["video_id"]),
            }
            for s in sentences
            if str(s.get("caption") or "").strip()
        ),
        key=lambda row: (row["sen_id"], row["caption"]),
    )
    if not ordered:
        raise ValueError(f"No valid captions found for {video_id}")
    digest = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).digest()
    pick_idx = int.from_bytes(digest[:8], "little") % len(ordered)
    return ordered[pick_idx]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_no_leakage(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    for idx, row in enumerate(rows):
        leaked = _FORBIDDEN_INFERENCE_FIELDS & set(row.keys())
        if leaked:
            raise AssertionError(f"Leakage field(s) {sorted(leaked)} in {label}[{idx}]")


def _allocate_balanced_source_groups(
    groups: list[dict[str, Any]],
    gallery_names: Sequence[str],
    target_size: int,
    *,
    seed: int = PROTOCOL_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Partition source-groups into disjoint exact-size galleries with balanced source sizes and categories."""
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    total_eligible = sum(len(g["videos"]) for g in shuffled)
    global_cat_counts = Counter()
    for g in shuffled:
        for v in g["videos"]:
            global_cat_counts[int(v["category"])] += 1
    target_cat_dist = {cat: cnt / max(1, total_eligible) for cat, cnt in global_cat_counts.items()}

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in gallery_names}
    bucket_sizes: dict[str, int] = {name: 0 for name in gallery_names}
    bucket_cats: dict[str, Counter[int]] = {name: Counter() for name in gallery_names}
    train_groups: list[dict[str, Any]] = []

    need_total = len(gallery_names) * target_size
    if total_eligible < need_total:
        raise ValueError(f"Insufficient eligible videos ({total_eligible}) for {need_total} validation slots")

    val_ratio = float(need_total) / float(total_eligible)
    for idx, grp in enumerate(shuffled):
        g_len = len(grp["videos"])
        rem_needed = need_total - sum(bucket_sizes.values())
        if rem_needed <= 0:
            train_groups.append(grp)
            continue

        eligible_buckets = [name for name in gallery_names if bucket_sizes[name] + g_len <= target_size]
        # Deterministic quota: take into validation iff validation progress lags val_ratio or tail needs filling
        rem_available = sum(len(g["videos"]) for g in shuffled[idx:])
        must_take = rem_available - g_len < rem_needed
        target_so_far = val_ratio * sum(len(g["videos"]) for g in shuffled[: idx + 1])
        should_take = bool(eligible_buckets) and (
            must_take or (sum(bucket_sizes.values()) < target_so_far)
        )
        if not should_take:
            train_groups.append(grp)
            continue

        g_cats = Counter(int(v["category"]) for v in grp["videos"])

        def _score_bucket(bname: str) -> tuple[float, int, str]:
            sz_after = bucket_sizes[bname] + g_len
            fill_ratio = bucket_sizes[bname] / float(target_size)
            cat_err = 0.0
            for cat, target_p in target_cat_dist.items():
                new_p = (bucket_cats[bname][cat] + g_cats.get(cat, 0)) / max(1, sz_after)
                cat_err += abs(new_p - target_p)
            return (round(fill_ratio * 1.5 + cat_err, 6), bucket_sizes[bname], bname)

        chosen = min(eligible_buckets, key=_score_bucket)
        buckets[chosen].append(grp)
        bucket_sizes[chosen] += g_len
        bucket_cats[chosen].update(g_cats)

    for name in gallery_names:
        if bucket_sizes[name] != target_size:
            raise RuntimeError(
                f"Gallery {name} filled {bucket_sizes[name]} != {target_size}"
            )
    return buckets, train_groups


def build_local_inventory(
    *,
    media_dir: Path = Path("data/datasets/msrvtt/media"),
    db_path: Path = Path("data/processed/metadata.db"),
    vp_manifest_path: Path = Path("data/processed/videoprism_clip_manifest.json"),
    vp_subclips_path: Path = Path("data/processed/videoprism_subclips_map.jsonl"),
    test_1ka_ids: set[str],
    all_video_ids: set[str],
) -> dict[str, Any]:
    """Inventory local MSR-VTT mp4 media, DB tables, OCR/ASR cache, and VideoPrism coverage."""
    local_mp4s = {p.stem: p for p in sorted(media_dir.glob("*.mp4"))} if media_dir.is_dir() else {}

    db_videos: set[str] = set()
    db_kfs = 0
    db_objs = 0
    db_ocr_vids: set[str] = set()
    db_asr_vids: set[str] = set()
    if db_path.is_file():
        with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            db_videos = {
                str(r[0])
                for r in conn.execute("SELECT video_id FROM Videos WHERE video_id LIKE 'video%'").fetchall()
            }
            db_kfs = int(conn.execute("SELECT COUNT(*) FROM Keyframes WHERE video_id LIKE 'video%'").fetchone()[0])
            db_objs = int(conn.execute("SELECT COUNT(*) FROM FrameObjects WHERE video_id LIKE 'video%'").fetchone()[0])
            db_ocr_vids = {
                str(r[0])
                for r in conn.execute("SELECT DISTINCT video_id FROM OCRObservations WHERE video_id LIKE 'video%'").fetchall()
            }
            db_asr_vids = {
                str(r[0])
                for r in conn.execute("SELECT DISTINCT video_id FROM ASRSegments WHERE video_id LIKE 'video%'").fetchall()
            }

    vp_vids: set[str] = set()
    if vp_manifest_path.is_file():
        vp_man = json.loads(vp_manifest_path.read_text(encoding="utf-8"))
        vp_vids = {str(v) for v in (vp_man.get("video_ids") or []) if str(v).startswith("video")}

    vp_sub_vids: set[str] = set()
    if vp_subclips_path.is_file():
        for line in vp_subclips_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v = str(row.get("video_id") or "")
            if v.startswith("video"):
                vp_sub_vids.add(v)

    non_1ka_ids = all_video_ids - test_1ka_ids
    return {
        "local_media_dir": str(media_dir),
        "local_mp4_count": len(local_mp4s),
        "local_mp4_matches_1ka": len(set(local_mp4s.keys()) & test_1ka_ids),
        "local_mp4_non_1ka": len(set(local_mp4s.keys()) & non_1ka_ids),
        "db_msrvtt_videos": len(db_videos),
        "db_msrvtt_matches_1ka": len(db_videos & test_1ka_ids),
        "db_msrvtt_keyframes": db_kfs,
        "db_msrvtt_yolo_objects": db_objs,
        "db_msrvtt_ocr_videos": len(db_ocr_vids),
        "db_msrvtt_asr_videos": len(db_asr_vids),
        "videoprism_whole_msrvtt_videos": len(vp_vids),
        "videoprism_whole_matches_1ka": len(vp_vids & test_1ka_ids),
        "videoprism_subclip_msrvtt_videos_gt20s": len(vp_sub_vids),
        "missing_local_media_for_9k": len(non_1ka_ids - set(local_mp4s.keys())),
        "kaggle_media_datasets": [
            "lvhuylong/msrvtt-1ka-original-videos-v1",
            "khoahunhtngng/msrvtt",
            "vishnutheepb/msrvtt",
        ],
    }


def prepare_protocol(
    *,
    provenance_dir: Path = Path("data/datasets/msrvtt/provenance/protocol_20261122"),
    out_root: Path = Path("work/msrvtt_protocol_20261122"),
    seed: int = PROTOCOL_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download, verify, partition, and freeze the complete MSR-VTT 1K-A benchmark protocol."""
    source_receipts = download_and_verify_sources(provenance_dir)

    val_1ka_ids = [
        line.strip()
        for line in Path(source_receipts["val_list_jsfusion.txt"]["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_9k_ids = [
        line.strip()
        for line in Path(source_receipts["train_list_jsfusion.txt"]["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(val_1ka_ids) != 1000 or len(set(val_1ka_ids)) != 1000:
        raise ValueError(f"Expected 1,000 unique 1K-A IDs, got {len(val_1ka_ids)}")
    if len(train_9k_ids) != 9000 or len(set(train_9k_ids)) != 9000:
        raise ValueError(f"Expected 9,000 unique train IDs, got {len(train_9k_ids)}")
    if set(val_1ka_ids) & set(train_9k_ids):
        raise ValueError("1K-A and 9K lists overlap")

    msr_doc = json.loads(Path(source_receipts["MSRVTT_data.json"]["path"]).read_text(encoding="utf-8"))
    videos_by_id: dict[str, dict[str, Any]] = {}
    for v in msr_doc["videos"]:
        vid = str(v["video_id"]).strip()
        sg = normalize_youtube_source_group(v.get("url"), vid)
        videos_by_id[vid] = {
            "video_id": vid,
            "category": int(v["category"]),
            "url": str(v.get("url") or "").strip(),
            "start_time": float(v.get("start time", 0.0)),
            "end_time": float(v.get("end time", 0.0)),
            "source_group": sg,
        }

    sents_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in msr_doc["sentences"]:
        sents_by_vid[str(s["video_id"]).strip()].append(s)

    # 1. Freeze Test 1K-A first
    with Path(source_receipts["MSRVTT_JSFUSION_test.csv"]["path"]).open("r", encoding="utf-8") as fh:
        js_rows = list(csv.DictReader(fh))
    if len(js_rows) != 1000:
        raise ValueError(f"Expected 1,000 JSFusion test rows, got {len(js_rows)}")

    val_1ka_set = set(val_1ka_ids)
    test_gallery_ids = sorted(val_1ka_set, key=lambda v: (int(v[5:]) if v.startswith("video") and v[5:].isdigit() else 10**9, v))
    test_source_groups = {videos_by_id[vid]["source_group"] for vid in test_gallery_ids}

    test_queries_vault: list[dict[str, Any]] = []
    test_labels: list[dict[str, Any]] = []
    test_lock_records: list[dict[str, Any]] = []
    for row in js_rows:
        qid = str(row["key"]).strip()
        vid = str(row["video_id"]).strip()
        sent = str(row["sentence"]).strip()
        if vid not in val_1ka_set:
            raise ValueError(f"JSFusion test video {vid} missing from val_list_jsfusion.txt")
        q_sha = sha256_text(sent)
        sg = videos_by_id[vid]["source_group"]
        test_queries_vault.append({
            "query_id": qid,
            "query_text": sent,
            "query_text_sha256": q_sha,
            "dataset": "msrvtt_1ka",
            "split": "test_1ka",
        })
        test_labels.append({
            "query_id": qid,
            "query_text_sha256": q_sha,
            "dataset": "msrvtt_1ka",
            "split": "test_1ka",
            "target_videos": [vid],
            "target_windows": [],
            "source_group": sg,
            "source_group_status": "msrvtt_youtube_url",
            "category": videos_by_id[vid]["category"],
        })
        test_lock_records.append({
            "query_id": qid,
            "split": "test_1ka",
            "source_group": sg,
        })

    # 2. Exclude source-group leaks from 9K
    excluded_leaks: list[dict[str, Any]] = []
    eligible_groups_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vid in sorted(train_9k_ids):
        vinfo = videos_by_id[vid]
        if vinfo["source_group"] in test_source_groups:
            excluded_leaks.append({
                "video_id": vid,
                "source_group": vinfo["source_group"],
                "reason": "shares_source_youtube_url_with_1ka_test",
            })
            continue
        eligible_groups_map[vinfo["source_group"]].append(vinfo)

    eligible_groups: list[dict[str, Any]] = []
    for sg, vlist in sorted(eligible_groups_map.items()):
        vlist_sorted = sorted(vlist, key=lambda x: x["video_id"])
        tie = hashlib.sha256(f"{seed}:{sg}".encode("utf-8")).hexdigest()
        eligible_groups.append({
            "source_group": sg,
            "videos": vlist_sorted,
            "sort_tie": tie,
        })

    gallery_names = ("tune_g1", "tune_g2", "confirm_g1", "confirm_g2")
    val_buckets, train_groups = _allocate_balanced_source_groups(
        eligible_groups, gallery_names, target_size=1000, seed=seed
    )

    manifests_dir = out_root / "manifests"
    eval_only_dir = out_root / "eval_only"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    eval_only_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, Any] = {}
    all_split_vids: dict[str, set[str]] = {}
    all_split_sgs: dict[str, set[str]] = {}

    for gname in gallery_names:
        vlist = [v for grp in val_buckets[gname] for v in grp["videos"]]
        vlist.sort(key=lambda x: (int(x["video_id"][5:]) if x["video_id"][5:].isdigit() else 10**9, x["video_id"]))

        gallery_rows: list[dict[str, Any]] = []
        query_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        lock_records: list[dict[str, Any]] = []
        cat_hist = Counter()

        for rank_idx, v in enumerate(vlist):
            vid = v["video_id"]
            sg = v["source_group"]
            cat = int(v["category"])
            cat_hist[cat] += 1
            picked = select_deterministic_caption(vid, sents_by_vid[vid], seed=seed)
            qid = f"{gname}_{vid}"
            qtext = picked["caption"]
            q_sha = sha256_text(qtext)

            gallery_rows.append({
                "gallery_order": rank_idx,
                "video_id": vid,
                "start_time": v["start_time"],
                "end_time": v["end_time"],
                "category": cat,
            })
            query_rows.append({
                "query_id": qid,
                "query_text": qtext,
                "query_text_sha256": q_sha,
            })
            label_rows.append({
                "query_id": qid,
                "query_text_sha256": q_sha,
                "dataset": "msrvtt",
                "split": gname,
                "target_videos": [vid],
                "target_windows": [],
                "source_group": sg,
                "source_group_status": "msrvtt_youtube_url",
                "selected_sen_id": picked["sen_id"],
                "category": cat,
            })
            lock_records.append({
                "query_id": qid,
                "split": gname,
                "source_group": sg,
            })

        _assert_no_leakage(gallery_rows, f"{gname}/gallery_manifest")
        _assert_no_leakage(query_rows, f"{gname}/queries")

        gdir = manifests_dir / gname
        _write_jsonl(gdir / "gallery_manifest.jsonl", gallery_rows)
        _write_jsonl(gdir / "queries.jsonl", query_rows)
        _write_jsonl(eval_only_dir / f"{gname}_labels.jsonl", label_rows)

        lock_path = eval_only_dir / f"{gname}_split_lock.json"
        if lock_path.exists():
            if not overwrite:
                raise FileExistsError(f"Split lock exists: {lock_path}")
            lock_path.unlink()
        freeze_split(
            lock_records,
            lock_path,
            metadata={
                "split": gname,
                "seed": seed,
                "video_count": len(gallery_rows),
                "query_count": len(query_rows),
                "gallery_sha256": sha256_file(gdir / "gallery_manifest.jsonl"),
                "queries_sha256": sha256_file(gdir / "queries.jsonl"),
                "labels_sha256": sha256_file(eval_only_dir / f"{gname}_labels.jsonl"),
            },
        )
        if not verify_split_lock(lock_path):
            raise RuntimeError(f"Split lock verify failed for {lock_path}")

        all_split_vids[gname] = {v["video_id"] for v in vlist}
        all_split_sgs[gname] = {v["source_group"] for v in vlist}
        split_stats[gname] = {
            "video_count": len(gallery_rows),
            "query_count": len(query_rows),
            "source_group_count": len(all_split_sgs[gname]),
            "gallery_sha256": sha256_file(gdir / "gallery_manifest.jsonl"),
            "queries_sha256": sha256_file(gdir / "queries.jsonl"),
            "labels_sha256": sha256_file(eval_only_dir / f"{gname}_labels.jsonl"),
            "split_lock_sha256": sha256_file(lock_path),
            "category_counts": dict(sorted(cat_hist.items())),
        }

    # 3. Build train_9k_clean manifest (all 20 captions per video kept for LTR / VEDJE / QB-CSLS)
    train_vlist = [v for grp in train_groups for v in grp["videos"]]
    train_vlist.sort(key=lambda x: (int(x["video_id"][5:]) if x["video_id"][5:].isdigit() else 10**9, x["video_id"]))
    train_rows: list[dict[str, Any]] = []
    train_cat_hist = Counter()
    for v in train_vlist:
        vid = v["video_id"]
        train_cat_hist[int(v["category"])] += 1
        caps = sorted(
            [
                {"sen_id": int(s["sen_id"]), "caption": str(s["caption"]).strip()}
                for s in sents_by_vid[vid]
                if str(s.get("caption") or "").strip()
            ],
            key=lambda c: (c["sen_id"], c["caption"]),
        )
        train_rows.append({
            "video_id": vid,
            "source_group": v["source_group"],
            "category": int(v["category"]),
            "captions": caps,
        })

    train_dir = manifests_dir / "train_9k_clean"
    _write_jsonl(train_dir / "train_videos.jsonl", train_rows)
    all_split_vids["train_9k_clean"] = {v["video_id"] for v in train_vlist}
    all_split_sgs["train_9k_clean"] = {v["source_group"] for v in train_vlist}
    split_stats["train_9k_clean"] = {
        "video_count": len(train_rows),
        "caption_count": sum(len(r["captions"]) for r in train_rows),
        "source_group_count": len(all_split_sgs["train_9k_clean"]),
        "train_videos_sha256": sha256_file(train_dir / "train_videos.jsonl"),
        "category_counts": dict(sorted(train_cat_hist.items())),
    }

    # 4. Write test_1ka gallery manifest (no GT) + keep test_1ka queries/labels locked in eval_only/
    test_gallery_rows = [
        {
            "gallery_order": idx,
            "video_id": vid,
            "start_time": videos_by_id[vid]["start_time"],
            "end_time": videos_by_id[vid]["end_time"],
            "category": videos_by_id[vid]["category"],
        }
        for idx, vid in enumerate(test_gallery_ids)
    ]
    _assert_no_leakage(test_gallery_rows, "test_1ka/gallery_manifest")
    _assert_no_leakage(test_queries_vault, "test_1ka_queries_vault")

    test_gdir = manifests_dir / "test_1ka"
    _write_jsonl(test_gdir / "gallery_manifest.jsonl", test_gallery_rows)
    _write_jsonl(eval_only_dir / "test_1ka_queries_vault.jsonl", test_queries_vault)
    _write_jsonl(eval_only_dir / "test_1ka_labels.jsonl", test_labels)

    test_lock_path = eval_only_dir / "test_1ka_split_lock.json"
    if test_lock_path.exists():
        if not overwrite:
            raise FileExistsError(f"Split lock exists: {test_lock_path}")
        test_lock_path.unlink()
    freeze_split(
        test_lock_records,
        test_lock_path,
        metadata={
            "split": "test_1ka",
            "seed": seed,
            "video_count": len(test_gallery_rows),
            "query_count": len(test_queries_vault),
            "gallery_sha256": sha256_file(test_gdir / "gallery_manifest.jsonl"),
            "queries_sha256": sha256_file(eval_only_dir / "test_1ka_queries_vault.jsonl"),
            "labels_sha256": sha256_file(eval_only_dir / "test_1ka_labels.jsonl"),
        },
    )
    if not verify_split_lock(test_lock_path):
        raise RuntimeError(f"Split lock verify failed for {test_lock_path}")

    all_split_vids["test_1ka"] = set(test_gallery_ids)
    all_split_sgs["test_1ka"] = set(test_source_groups)
    split_stats["test_1ka"] = {
        "video_count": len(test_gallery_rows),
        "query_count": len(test_queries_vault),
        "source_group_count": len(test_source_groups),
        "gallery_sha256": sha256_file(test_gdir / "gallery_manifest.jsonl"),
        "queries_vault_sha256": sha256_file(eval_only_dir / "test_1ka_queries_vault.jsonl"),
        "labels_sha256": sha256_file(eval_only_dir / "test_1ka_labels.jsonl"),
        "split_lock_sha256": sha256_file(test_lock_path),
        "queries_sealed_until_champion_lock": True,
    }

    # 5. Pairwise zero-overlap verification across all 6 partitions
    split_names = list(all_split_vids.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            v_ov = all_split_vids[a] & all_split_vids[b]
            sg_ov = all_split_sgs[a] & all_split_sgs[b]
            if v_ov:
                raise AssertionError(f"Video overlap between {a} and {b}: {sorted(v_ov)[:5]}")
            if sg_ov:
                raise AssertionError(f"Source-group overlap between {a} and {b}: {sorted(sg_ov)[:5]}")

    # 6. Inventory local media, DB, and VideoPrism cache
    inventory = build_local_inventory(
        test_1ka_ids=val_1ka_set,
        all_video_ids=set(videos_by_id.keys()),
    )

    receipt = {
        "schema_version": 1,
        "protocol_name": "MSRVTT_1KA_BENCHMARK_20261122",
        "seed": seed,
        "pinned_sources": source_receipts,
        "excluded_9k_source_leaks_count": len(excluded_leaks),
        "excluded_9k_source_leaks_sample": excluded_leaks[:10],
        "eligible_9k_count": len(eligible_groups),
        "eligible_9k_video_count": sum(len(g["videos"]) for g in eligible_groups),
        "zero_video_overlap_verified": True,
        "zero_source_group_overlap_verified": True,
        "splits": split_stats,
        "local_inventory": inventory,
    }
    _write_jsonl(out_root / "excluded_9k_source_leaks.jsonl", excluded_leaks)
    receipt_path = out_root / "protocol_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and verify MSR-VTT 1K-A protocol splits.")
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=Path("data/datasets/msrvtt/provenance/protocol_20261122"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("work/msrvtt_protocol_20261122"),
    )
    parser.add_argument("--seed", type=int, default=PROTOCOL_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    receipt = prepare_protocol(
        provenance_dir=args.provenance_dir,
        out_root=args.out_root,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "protocol_name": receipt["protocol_name"],
        "seed": receipt["seed"],
        "excluded_9k_source_leaks_count": receipt["excluded_9k_source_leaks_count"],
        "zero_video_overlap_verified": receipt["zero_video_overlap_verified"],
        "zero_source_group_overlap_verified": receipt["zero_source_group_overlap_verified"],
        "splits": {
            k: {sk: sv for sk, sv in v.items() if sk != "category_counts"}
            for k, v in receipt["splits"].items()
        },
        "local_inventory": receipt["local_inventory"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
