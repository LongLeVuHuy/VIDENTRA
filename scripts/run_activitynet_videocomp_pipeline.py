"""End-to-end orchestrator for ActivityNet Captions & VideoComp across local CPU + Modal L4/L40S.

Zero leakage contract:
- Local CPU retains all Ground Truth, builds source-disjoint 40/40/20 splits, runs sweeps & evaluator.
- Modal GPUs receive ONLY blinded video paths / text strings / candidate IDs (zero split/GT labels).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modal_videntra_worker import (  # noqa: E402
    DEFAULT_VOLUME_NAME,
    normalize_media_key,
    summarize_pilot_costs,
)
from scripts.prepare_activitynet_videocomp_splits import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    assign_source_disjoint_splits,
)
from videntra.evaluation.multievent_evaluator import (  # noqa: E402
    evaluate_activitynet_full_chain,
    evaluate_videocomp,
    normalize_videocomp_type,
    paired_source_group_bootstrap_ci,
    videocomp_us_to_sec,
)
from videntra.query.rules import decompose_query_event_chain  # noqa: E402
from videntra.retrieval.alignment import (  # noqa: E402
    localize_exact_segment_chain_dp,
    score_videocomp_order,
    viterbi_subevent_localization,
)

DEFAULT_WORK_ROOT = REPO_ROOT / "work" / "activitynet_videocomp_20261124"


def pull_modal_file(remote_rel_path: str, local_dst: Path) -> Path:
    """Download an artifact from Modal Volume `videntra-eval-artifacts` cleanly into local disk."""
    import modal  # type: ignore

    vol = modal.Volume.from_name(DEFAULT_VOLUME_NAME)
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_dst.with_suffix(local_dst.suffix + ".tmp")
    rel = remote_rel_path.lstrip("/")
    if rel.startswith("vol/videntra/"):
        rel = rel[len("vol/videntra/") :]
    with tmp.open("wb") as fh:
        vol.read_file_into_fileobj(rel, fh)
    tmp.replace(local_dst)
    return local_dst


def cmd_fetch_annotations(work_root: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download  # type: ignore

    ann_dir = work_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for fn in ("raw_data/train.json", "raw_data/val_1.json", "raw_data/val_2.json"):
        p = hf_hub_download(
            "friedrichor/ActivityNet_Captions",
            fn,
            repo_type="dataset",
            local_dir=str(ann_dir / "anet_captions_hf"),
        )
        downloaded.append(str(p))
    for fn in (
        "activitynet_comp_train.json",
        "activitynet_comp_val.json",
        "youcook2_comp_train.json",
        "youcook2_comp_val.json",
    ):
        dst = ann_dir / fn
        if not dst.is_file():
            urllib.request.urlretrieve(f"https://storage.googleapis.com/video_comp/{fn}", str(dst))
        downloaded.append(str(dst))
    res = {"annotations_dir": str(ann_dir), "files": downloaded}
    print(json.dumps(res, indent=2))
    return res


def _build_target_id_lists(ann_dir: Path) -> tuple[list[str], list[str]]:
    comp_at = json.loads((ann_dir / "activitynet_comp_train.json").read_text(encoding="utf-8"))
    comp_av = json.loads((ann_dir / "activitynet_comp_val.json").read_text(encoding="utf-8"))
    comp_yt = json.loads((ann_dir / "youcook2_comp_train.json").read_text(encoding="utf-8"))
    comp_yv = json.loads((ann_dir / "youcook2_comp_val.json").read_text(encoding="utf-8"))
    anet_tr = json.loads((ann_dir / "anet_captions_hf/raw_data/train.json").read_text(encoding="utf-8"))
    anet_v1 = json.loads((ann_dir / "anet_captions_hf/raw_data/val_1.json").read_text(encoding="utf-8"))

    # Prioritize all videos overlapping Captions >=2 sentences & VideoComp
    anet_ids = sorted(
        {normalize_media_key(k) for k, v in anet_tr.items() if len(v.get("sentences", [])) >= 2}
        | {normalize_media_key(k) for k, v in anet_v1.items() if len(v.get("sentences", [])) >= 2}
        | {normalize_media_key(r["video_id"]) for r in comp_at}
        | {normalize_media_key(r["video_id"]) for r in comp_av}
    )
    yc2_ids = sorted(
        {normalize_media_key(r["video_id"]) for r in comp_yt}
        | {normalize_media_key(r["video_id"]) for r in comp_yv}
    )
    return anet_ids, yc2_ids


def cmd_sync_media(
    work_root: Path,
    *,
    force: bool = False,
    youcook2_only: bool = False,
) -> dict[str, Any]:
    from scripts.modal_app_functions import (
        app as modal_app,
        sync_raw_media_to_volume,
        sync_youcook2_to_volume,
    )

    ann_dir = work_root / "annotations"
    anet_ids, yc2_ids = _build_target_id_lists(ann_dir)
    with modal_app.run():
        if youcook2_only:
            payload = sync_youcook2_to_volume.remote(json.dumps(yc2_ids), bool(force))
        else:
            payload = sync_raw_media_to_volume.remote(
                json.dumps(anet_ids), json.dumps(yc2_ids), bool(force)
            )
    cat_path = work_root / "media_catalog.json"
    cat_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "items"}
    summary["catalog_path"] = str(cat_path)
    print(json.dumps(summary, indent=2))
    return payload


def cmd_lock_splits(
    work_root: Path,
    *,
    max_anet_groups: int = 1000,
    max_yc2_groups: int = 500,
    seed: int = DEFAULT_SPLIT_SEED,
    intersect_anet_comp: bool = True,
) -> dict[str, Any]:
    """Build deterministic source-disjoint 40/40/20 manifests using verified Modal Volume catalog."""
    ann_dir = work_root / "annotations"
    cat = json.loads((work_root / "media_catalog.json").read_text(encoding="utf-8"))
    valid_media: dict[tuple[str, str], dict[str, Any]] = {}
    hashes_by_vid: dict[str, str] = {}
    for it in cat.get("items", []):
        if not it.get("ok"):
            continue
        vid = normalize_media_key(it["video_id"])
        dom = str(it["domain"])
        valid_media[(dom, vid)] = it
        hashes_by_vid[vid] = str(it["file_sha256"])

    # Load Captions train + val_1 + val_2 + VideoComp
    anet_tr = json.loads((ann_dir / "anet_captions_hf/raw_data/train.json").read_text(encoding="utf-8"))
    anet_v1 = json.loads((ann_dir / "anet_captions_hf/raw_data/val_1.json").read_text(encoding="utf-8"))
    anet_v2 = json.loads((ann_dir / "anet_captions_hf/raw_data/val_2.json").read_text(encoding="utf-8"))

    comp_at = json.loads((ann_dir / "activitynet_comp_train.json").read_text(encoding="utf-8"))
    comp_av = json.loads((ann_dir / "activitynet_comp_val.json").read_text(encoding="utf-8"))
    comp_yt = json.loads((ann_dir / "youcook2_comp_train.json").read_text(encoding="utf-8"))
    comp_yv = json.loads((ann_dir / "youcook2_comp_val.json").read_text(encoding="utf-8"))

    comp_anet_vids = {normalize_media_key(r["video_id"]) for r in comp_at} | {normalize_media_key(r["video_id"]) for r in comp_av}
    cap_anet_vids = (
        {normalize_media_key(k) for k, v in anet_tr.items() if len(v.get("sentences", [])) >= 2}
        | {normalize_media_key(k) for k, v in anet_v1.items() if len(v.get("sentences", [])) >= 2}
        | {normalize_media_key(k) for k, v in anet_v2.items() if len(v.get("sentences", [])) >= 2}
    )
    eligible_anet_vids = (comp_anet_vids & cap_anet_vids) if intersect_anet_comp else None

    anet_records: list[dict[str, Any]] = []
    for pool_name, cap_dict, ann_tag in (
        ("train", anet_tr, "train"),
        ("validation", anet_v1, "val_1"),
        ("validation", anet_v2, "val_2"),
    ):
        for raw_vid, entry in cap_dict.items():
            vid = normalize_media_key(raw_vid)
            if ("activitynet", vid) not in valid_media:
                continue
            if eligible_anet_vids is not None and vid not in eligible_anet_vids:
                continue
            sents = [str(s).strip() for s in entry.get("sentences", [])]
            tss = entry.get("timestamps", [])
            if len(sents) < 2 or len(sents) != len(tss):
                continue
            events = [
                {"clause_id": f"e{i}", "start_sec": float(t[0]), "end_sec": float(t[1]), "text": sents[i]}
                for i, t in enumerate(tss)
                if len(t) == 2 and float(t[1]) > float(t[0])
            ]
            if len(events) < 2:
                continue
            anet_records.append({
                "query_id": f"anet_cap_{ann_tag}_{vid}",
                "video_id": vid,
                "source_group_id": vid,
                "domain": "activitynet",
                "public_pool": pool_name,
                "task_kind": "captions",
                "annotation_source": ann_tag,
                "has_comp_pair": vid in comp_anet_vids,
                "query_text": " ".join(ev["text"] for ev in events),
                "events": events,
                "media_rel_path": valid_media[("activitynet", vid)]["rel_path"],
                "duration_sec": valid_media[("activitynet", vid)]["duration_sec"],
            })

    for pool_name, rows in (("train", comp_at), ("validation", comp_av)):
        for r in rows:
            vid = normalize_media_key(r["video_id"])
            if ("activitynet", vid) not in valid_media:
                continue
            if eligible_anet_vids is not None and vid not in eligible_anet_vids:
                continue
            anet_records.append({
                "query_id": str(r["key"]),
                "pair_id": str(r["key"]),
                "video_id": vid,
                "source_group_id": vid,
                "domain": "activitynet_comp",
                "root_domain": "activitynet",
                "public_pool": pool_name,
                "task_kind": "videocomp",
                "type": normalize_videocomp_type(str(r["type"])),
                "positive_text": str(r["positive_text"]),
                "negative_text": str(r["negative_text"]),
                "clip_start_sec": videocomp_us_to_sec(r.get("query_video/start_time", 0)),
                "clip_end_sec": videocomp_us_to_sec(r.get("query_video/end_time", 0)),
                "orig_start_sec": videocomp_us_to_sec(r.get("original_video/start_time", 0)),
                "orig_end_sec": videocomp_us_to_sec(r.get("original_video/end_time", 0)),
                "media_rel_path": valid_media[("activitynet", vid)]["rel_path"],
                "duration_sec": valid_media[("activitynet", vid)]["duration_sec"],
            })

    # Normalize domain to `activitynet` during split assignment so Captions + Comp share identical split!
    for rec in anet_records:
        rec["domain_orig"] = rec["domain"]
        rec["domain"] = "activitynet"

    anet_bundle = assign_source_disjoint_splits(
        anet_records,
        seed=seed,
        max_groups_per_domain=max_anet_groups,
        annotation_version="activitynet_captions_v1_comp_cvpr25",
        file_sha256_by_video_id=hashes_by_vid,
    )

    yc2_records: list[dict[str, Any]] = []
    for pool_name, rows in (("train", comp_yt), ("validation", comp_yv)):
        for r in rows:
            vid = normalize_media_key(r["video_id"])
            if ("youcook2", vid) not in valid_media:
                continue
            yc2_records.append({
                "query_id": str(r["key"]),
                "pair_id": str(r["key"]),
                "video_id": vid,
                "source_group_id": vid,
                "domain": "youcook2",
                "domain_orig": "youcook2_comp",
                "public_pool": pool_name,
                "task_kind": "videocomp",
                "type": normalize_videocomp_type(str(r["type"])),
                "positive_text": str(r["positive_text"]),
                "negative_text": str(r["negative_text"]),
                "clip_start_sec": videocomp_us_to_sec(r.get("query_video/start_time", 0)),
                "clip_end_sec": videocomp_us_to_sec(r.get("query_video/end_time", 0)),
                "orig_start_sec": videocomp_us_to_sec(r.get("original_video/start_time", 0)),
                "orig_end_sec": videocomp_us_to_sec(r.get("original_video/end_time", 0)),
                "media_rel_path": valid_media[("youcook2", vid)]["rel_path"],
                "duration_sec": valid_media[("youcook2", vid)]["duration_sec"],
            })

    yc2_bundle = assign_source_disjoint_splits(
        yc2_records,
        seed=seed,
        max_groups_per_domain=max_yc2_groups,
        annotation_version="youcook2_comp_cvpr25",
        file_sha256_by_video_id=hashes_by_vid,
    )

    splits_dir = work_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    combined_summary: dict[str, Any] = {
        "seed": int(seed),
        "activitynet_summary": anet_bundle["per_domain_summary"],
        "activitynet_quarantined": anet_bundle["quarantined_count"],
        "youcook2_summary": yc2_bundle["per_domain_summary"],
        "youcook2_quarantined": yc2_bundle["quarantined_count"],
        "splits": {},
    }

    for sname in ("tune", "valid_confirm", "test"):
        anet_sp = anet_bundle["splits"][sname]
        yc2_sp = yc2_bundle["splits"][sname]
        anet_recs = [dict(r, domain=r.get("domain_orig", "activitynet")) for r in anet_sp["records"]]
        yc2_recs = [dict(r, domain=r.get("domain_orig", "youcook2_comp")) for r in yc2_sp["records"]]
        merged_manifest = {
            "split": sname,
            "seed": int(seed),
            "activitynet": {
                "source_groups": anet_sp["source_groups"],
                "gallery_video_ids": anet_sp["gallery_video_ids"],
                "checksums": anet_sp["checksums"],
                "records": anet_recs,
            },
            "youcook2": {
                "source_groups": yc2_sp["source_groups"],
                "gallery_video_ids": yc2_sp["gallery_video_ids"],
                "checksums": yc2_sp["checksums"],
                "records": yc2_recs,
            },
        }
        out_p = splits_dir / f"{sname}_manifest.json"
        out_p.write_text(json.dumps(merged_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        cap_cnt = sum(1 for r in anet_recs if r["task_kind"] == "captions")
        comp_a_cnt = sum(1 for r in anet_recs if r["task_kind"] == "videocomp")
        comp_y_cnt = len(yc2_recs)
        combined_summary["splits"][sname] = {
            "anet_groups": len(anet_sp["source_groups"]),
            "yc2_groups": len(yc2_sp["source_groups"]),
            "captions_queries": cap_cnt,
            "anet_comp_pairs": comp_a_cnt,
            "yc2_comp_pairs": comp_y_cnt,
            "manifest_path": str(out_p),
        }

    (splits_dir / "split_lock_receipt.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(combined_summary, indent=2))
    return combined_summary


def _collect_blinded_split_items(
    manifest: Mapping[str, Any],
    *,
    limit_anet_vids: int | None = None,
    limit_yc2_vids: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct STRICTLY blinded video extract items + blinded text items (ZERO GT labels)."""
    video_items_map: dict[str, dict[str, Any]] = {}
    texts_map: dict[str, str] = {}

    anet_recs = manifest["activitynet"]["records"]
    yc2_recs = manifest["youcook2"]["records"]

    # Map video -> rel_path & duration from records cleanly
    rel_by_vid: dict[str, str] = {}
    for r in anet_recs + yc2_recs:
        rel_by_vid[str(r["video_id"])] = str(r["media_rel_path"])

    allowed_anet = set(manifest["activitynet"]["gallery_video_ids"][:limit_anet_vids] if limit_anet_vids else manifest["activitynet"]["gallery_video_ids"])
    allowed_yc2 = set(manifest["youcook2"]["gallery_video_ids"][:limit_yc2_vids] if limit_yc2_vids else manifest["youcook2"]["gallery_video_ids"])

    # Full-video items for gallery retrieval + non-sliced VideoComp
    for vid in sorted(allowed_anet | allowed_yc2):
        if vid in rel_by_vid:
            item_id = f"full::{vid}"
            video_items_map[item_id] = {
                "item_id": item_id,
                "video_id": vid,
                "rel_path": rel_by_vid[vid],
                "clip_start_sec": None,
                "clip_end_sec": None,
            }

    # Sliced clip items specifically for VideoComp segment_mismatch (or trimmed query_video windows)
    for r in anet_recs + yc2_recs:
        if r["task_kind"] != "videocomp":
            continue
        vid = str(r["video_id"])
        if vid not in allowed_anet and vid not in allowed_yc2:
            continue
        st = float(r.get("clip_start_sec", 0.0))
        et = float(r.get("clip_end_sec", 0.0))
        o_et = float(r.get("orig_end_sec", 0.0))
        is_trimmed = st > 0.05 or (o_et > 0.0 and et > st and et + 0.05 < o_et)
        if is_trimmed:
            item_id = f"clip::{r['pair_id']}"
            video_items_map[item_id] = {
                "item_id": item_id,
                "video_id": vid,
                "rel_path": r["media_rel_path"],
                "clip_start_sec": st,
                "clip_end_sec": et,
            }

        # Blinded candidate strings: hash by text string so positive/negative never expose order!
        for txt_field in ("positive_text", "negative_text"):
            raw_t = str(r[txt_field]).strip()
            texts_map.setdefault(raw_t, raw_t)
            for ec in decompose_query_event_chain(raw_t):
                texts_map.setdefault(ec.text, ec.text)

    for r in anet_recs:
        if r["task_kind"] != "captions" or str(r["video_id"]) not in allowed_anet:
            continue
        # Only use val_1 captions as primary queries (to avoid double counting same clip unless distinct); still extract all if present
        qt = str(r["query_text"]).strip()
        texts_map.setdefault(qt, qt)
        for ev in r["events"]:
            ev_t = str(ev["text"]).strip()
            texts_map.setdefault(ev_t, ev_t)
        for ec in decompose_query_event_chain(qt):
            texts_map.setdefault(ec.text, ec.text)

    # Sort deterministically without revealing GT vs negative label
    video_items = [video_items_map[k] for k in sorted(video_items_map)]
    text_items = [
        {"text_id": f"t_{idx:06d}", "text": txt}
        for idx, txt in enumerate(sorted(texts_map.keys()))
    ]
    return video_items, text_items


def _save_text_vocab(text_items: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(text_items), ensure_ascii=False), encoding="utf-8")


def cmd_run_pilot(work_root: Path, *, l4_usd_hr: float = 0.80, l40s_usd_hr: float = 1.95) -> dict[str, Any]:
    """Compare L4 vs L40S on identical 32-video / ~64-text Tune shard + test Qwen2.5-VL-3B NF4 on L4."""
    from scripts.modal_app_functions import (
        app as modal_app,
        run_l40s_extract_shard,
        run_l4_extract_shard,
        run_l4_vlm_rerank_shard,
    )

    manifest = json.loads((work_root / "splits" / "tune_manifest.json").read_text(encoding="utf-8"))
    video_items, all_text_items = _collect_blinded_split_items(manifest, limit_anet_vids=24, limit_yc2_vids=8)
    video_items = video_items[:32]
    text_items = all_text_items[:128]

    # Blinded sample pairs for L4 VLM 6 GiB check (12 VideoComp pairs -> 24 sequential inference calls)
    sample_comp = [
        r for r in (manifest["activitynet"]["records"] + manifest["youcook2"]["records"])
        if r["task_kind"] == "videocomp" and any(v["video_id"] == r["video_id"] for v in video_items)
    ][:12]
    vlm_pairs = []
    rel_map: dict[str, str] = {}
    for p_idx, r in enumerate(sample_comp):
        vid_id = f"cand::{r['pair_id']}"
        rel_map[vid_id] = str(r["media_rel_path"])
        clip_wins = {vid_id: [float(r["clip_start_sec"]), float(r["clip_end_sec"])]} if float(r["clip_end_sec"]) > float(r["clip_start_sec"]) else None
        for tag, tfld in (("pos", "positive_text"), ("neg", "negative_text")):
            vlm_pairs.append({
                "query_id": f"pilot_{p_idx}_{tag}",
                "query_text": str(r[tfld]),
                "candidate_video_ids": [vid_id],
                "clip_windows_by_id": clip_wins,
            })

    import modal  # type: ignore

    with modal.enable_output(), modal_app.run():
        # Run L40S first so shared `/vol/videntra/hf_cache` populates quickly on available L40S
        res_l40s = run_l40s_extract_shard.remote("pilot_20261124", "pilot32", json.dumps(video_items), json.dumps(text_items), 48)
        res_l4 = run_l4_extract_shard.remote("pilot_20261124", "pilot32", json.dumps(video_items), json.dumps(text_items), 48)
        res_vlm = run_l4_vlm_rerank_shard.remote(
            "pilot_20261124",
            "qwen3b_pilot_ts",
            "pilot_vlm",
            json.dumps(vlm_pairs),
            json.dumps(rel_map),
            "nf4_4bit",
            8,
            True,
            "compositional",
        )

    rc_l4 = res_l4["receipt"]
    rc_l40s = res_l40s["receipt"]
    rc_vlm = res_vlm["receipt"]

    if int(rc_l4.get("total_decoded_frames", 0)) == 0 or int(rc_l40s.get("total_decoded_frames", 0)) == 0:
        raise RuntimeError(
            f"Pilot decode failed (total_decoded_frames==0): "
            f"L4/A10G errors={rc_l4.get('decode_errors', [])[:2]}, L40S errors={rc_l40s.get('decode_errors', [])[:2]}"
        )

    # Check numerical parity between mid-tier (L4/A10G) and L40S feature shards
    p_dir = work_root / "pilot"
    p_dir.mkdir(parents=True, exist_ok=True)
    npz_l4 = pull_modal_file("pilot_20261124/features_L4/pilot32/shard_features.npz", p_dir / "l4_pilot32.npz")
    npz_l40s = pull_modal_file("pilot_20261124/features_L40S/pilot32/shard_features.npz", p_dir / "l40s_pilot32.npz")
    d4 = np.load(npz_l4, allow_pickle=False)
    d40 = np.load(npz_l40s, allow_pickle=False)
    max_diff_bigg = float(np.max(np.abs(d4["pe_bigg_frames"].astype(np.float32) - d40["pe_bigg_frames"].astype(np.float32))))
    d4.close()
    d40.close()

    def _rate_for_gpu(gpu_name: str | None) -> float:
        gn = str(gpu_name or "").upper()
        if "L40S" in gn:
            return float(l40s_usd_hr)
        if "A10" in gn:
            return 1.10
        return float(l4_usd_hr)

    def _steady_state_cost(rc: Mapping[str, Any], usd_hr: float, nv: int, np_: int) -> dict[str, Any]:
        load_tot = sum(float(v) for v in (rc.get("model_load_sec") or {}).values())
        steady_sec = max(0.01, float(rc.get("elapsed_sec", 0.0)) - load_tot)
        return summarize_pilot_costs([{"elapsed_sec": steady_sec}], usd_per_gpu_sec=usd_hr / 3600.0, videos_processed=nv, pairs_processed=np_)

    mid_gpu_name = (rc_l4.get("telemetry") or {}).get("gpu_name") or "L4/A10G"
    l40s_gpu_name = (rc_l40s.get("telemetry") or {}).get("gpu_name") or "L40S"
    vlm_gpu_name = (rc_vlm.get("telemetry") or {}).get("gpu_name") or "L4/A10G/L40S"

    mid_rate = _rate_for_gpu(mid_gpu_name)
    l40s_rate = _rate_for_gpu(l40s_gpu_name)
    vlm_rate = _rate_for_gpu(vlm_gpu_name)

    cost_l4 = summarize_pilot_costs([rc_l4], usd_per_gpu_sec=mid_rate / 3600.0, videos_processed=len(video_items), pairs_processed=len(text_items))
    cost_l40s = summarize_pilot_costs([rc_l40s], usd_per_gpu_sec=l40s_rate / 3600.0, videos_processed=len(video_items), pairs_processed=len(text_items))
    cost_vlm = summarize_pilot_costs([rc_vlm], usd_per_gpu_sec=vlm_rate / 3600.0, videos_processed=len(sample_comp), pairs_processed=len(vlm_pairs))

    steady_mid = _steady_state_cost(rc_l4, mid_rate, len(video_items), len(text_items))
    steady_l40s = _steady_state_cost(rc_l40s, l40s_rate, len(video_items), len(text_items))

    # Pick L40S if steady-state cost <= mid-tier OR if L4 was unavailable (A10G fallback used)
    l4_actually_got = "L4" in str(mid_gpu_name).upper() and "L40" not in str(mid_gpu_name).upper()
    winner = "L4" if (l4_actually_got and steady_mid["total_usd"] < steady_l40s["total_usd"]) else "L40S"
    comp = {
        "pilot_videos": len(video_items),
        "pilot_texts": len(text_items),
        "max_abs_diff_l4_vs_l40s_pe_bigg": round(max_diff_bigg, 6),
        "L4": {
            "actual_gpu_name": mid_gpu_name,
            "receipt": rc_l4,
            "cost_estimate": cost_l4,
            "steady_state_cost_estimate": steady_mid,
            "usd_per_hr": mid_rate,
        },
        "L40S": {
            "actual_gpu_name": l40s_gpu_name,
            "receipt": rc_l40s,
            "cost_estimate": cost_l40s,
            "steady_state_cost_estimate": steady_l40s,
            "usd_per_hr": l40s_rate,
        },
        "L4_Qwen25_VL_3B_NF4": {
            "actual_gpu_name": vlm_gpu_name,
            "receipt": rc_vlm,
            "cost_estimate": cost_vlm,
            "usd_per_hr": vlm_rate,
        },
        "winner_extract_gpu": winner,
    }
    out_p = p_dir / "pilot_gpu_comparison.json"
    out_p.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comp, indent=2))
    return comp


def cmd_extract_split(
    work_root: Path,
    *,
    split: str,
    gpu_tier: str = "L40S",
    shard_size: int = 150,
    num_frames: int = 48,
) -> dict[str, Any]:
    from scripts.modal_app_functions import (
        app as modal_app,
        run_l40s_extract_shard,
        run_l4_extract_shard,
    )

    manifest = json.loads((work_root / "splits" / f"{split}_manifest.json").read_text(encoding="utf-8"))
    video_items, text_items = _collect_blinded_split_items(manifest)
    feat_dir = work_root / "features" / split
    feat_dir.mkdir(parents=True, exist_ok=True)
    _save_text_vocab(text_items, feat_dir / "text_vocab.json")

    use_l40s = gpu_tier.upper() == "L40S"
    fn = run_l40s_extract_shard if use_l40s else run_l4_extract_shard
    gpu_tag = "L40S" if use_l40s else "L4"

    shards = [
        (i // shard_size, video_items[i : i + shard_size])
        for i in range(0, len(video_items), shard_size)
    ]
    import modal  # type: ignore

    receipts: list[dict[str, Any]] = []
    todo_args: list[tuple[str, str, str, str, int]] = []
    todo_sids: list[str] = []
    for s_idx, v_chunk in shards:
        sid = f"{split}_s{s_idx:03d}"
        local_npz = feat_dir / f"{sid}.npz"
        local_rc = feat_dir / f"{sid}_receipt.json"
        if local_npz.is_file() and local_rc.is_file():
            rc_saved = json.loads(local_rc.read_text(encoding="utf-8"))
            if int(rc_saved.get("total_decoded_frames", 0)) > 0:
                receipts.append(rc_saved)
                continue
        # Include text_items on shard 0 only (texts are shared across split, avoiding redundant re-encoding!)
        t_chunk = text_items if s_idx == 0 else []
        todo_args.append((f"extract_{split}", sid, json.dumps(v_chunk), json.dumps(t_chunk), int(num_frames)))
        todo_sids.append(sid)

    if todo_args:
        with modal.enable_output(), modal_app.run():
            results = list(fn.starmap(todo_args))
        for sid, res in zip(todo_sids, results):
            rc = res["receipt"]
            receipts.append(rc)
            (feat_dir / f"{sid}_receipt.json").write_text(json.dumps(rc, indent=2, ensure_ascii=False), encoding="utf-8")
            pull_modal_file(
                f"extract_{split}/features_{gpu_tag}/{sid}/shard_features.npz",
                feat_dir / f"{sid}.npz",
            )

    summary = {"split": split, "gpu_tag": gpu_tag, "num_shards": len(shards), "total_video_items": len(video_items), "total_texts": len(text_items), "receipts": receipts}
    (feat_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}, indent=2))
    return summary


def _load_split_feature_store(work_root: Path, split: str) -> dict[str, Any]:
    """Load downloaded FP16 frame & FP32 text feature shards into compact RAM store."""
    feat_dir = work_root / "features" / split
    vocab = json.loads((feat_dir / "text_vocab.json").read_text(encoding="utf-8"))
    text_id_to_idx: dict[str, int] = {}
    text_str_to_idx: dict[str, int] = {}

    frame_data: dict[str, dict[str, Any]] = {}
    text_vecs_by_backbone: dict[str, np.ndarray] = {}

    for npz_p in sorted(feat_dir.glob(f"{split}_s*.npz")):
        d = np.load(npz_p, allow_pickle=False)
        vids = d["video_item_ids"]
        cnts = d["video_frame_counts"]
        durs = d["video_durations_sec"]
        ords = d["video_frame_ordinals"]
        ptss = d["video_frame_pts_sec"]
        pb = d["pe_bigg_frames"]
        pl = d["pe_l_frames"]
        ps = d["sig400_frames"]

        off = 0
        for i, raw_id in enumerate(vids):
            item_id = str(raw_id)
            c = int(cnts[i])
            if c > 0:
                frame_data[item_id] = {
                    "duration": float(durs[i]),
                    "ordinals": ords[i, :c].astype(np.int32),
                    "pts_sec": ptss[i, :c].astype(np.float32),
                    "pe_bigg": pb[off : off + c].astype(np.float32),
                    "pe_l": pl[off : off + c].astype(np.float32),
                    "sig400": ps[off : off + c].astype(np.float32),
                }
            off += c

        if "pe_bigg_texts" in d and len(d["text_ids"]) > 0:
            for ti, tid in enumerate(d["text_ids"]):
                text_id_to_idx[str(tid)] = ti
            for b in ("pe_bigg", "pe_l", "sig400"):
                text_vecs_by_backbone[b] = d[f"{b}_texts"].astype(np.float32)
        d.close()

    for row in vocab:
        tid = str(row["text_id"])
        if tid in text_id_to_idx:
            text_str_to_idx[str(row["text"])] = text_id_to_idx[tid]

    return {
        "frame_data": frame_data,
        "text_str_to_idx": text_str_to_idx,
        "text_vecs": text_vecs_by_backbone,
    }


DEFAULT_TUNE_HPARAMS: dict[str, Any] = {
    "weights": {"pe_bigg": 1.40, "pe_l": 1.10, "sig400": 0.75},
    "qb_alpha": 0.35,
    "dense_pool_alpha": 0.45,
    "dense_topk": 6,
    "clause_ret_alpha": 0.55,
    "dp_rank_alpha": 2.0,
    "order_rank_alpha": 0.85,
    "dp_lambda_bg": 0.30,
    "dp_lambda_peak": 0.35,
    "dp_lambda_len": 0.04,
    "span_mode": "dp_exact",
    "com_beta": 0.35,
    "order_beta": 1.00,
    "min_beta": 0.45,
    "cov_beta": 0.25,
    "vlm_margin_gamma": 1.25,
    "vlm_captions_alpha": 0.45,
    "vlm_vc_alpha": 0.35,
    "vlm_vc_margin_gamma": 999.0,
    "vlm_vc_use_timestamps": True,
    "use_vlm_captions": True,
    "use_vlm_videocomp": True,
}


def _build_or_load_tune_query_bank(work_root: Path, *, max_bank_texts: int = 1200) -> dict[str, np.ndarray]:
    """Build inductive Query-Bank strictly from `Tune` split features (zero Valid-Confirm/Test leakage)."""
    bank_path = work_root / "tune" / "tune_query_bank.npz"
    if bank_path.is_file():
        d = np.load(bank_path, allow_pickle=False)
        out = {b: d[b].astype(np.float32) for b in ("pe_bigg", "pe_l", "sig400") if b in d}
        d.close()
        return out

    manifest = json.loads((work_root / "splits" / "tune_manifest.json").read_text(encoding="utf-8"))
    store = _load_split_feature_store(work_root, "tune")
    tmap = store["text_str_to_idx"]
    tvecs = store["text_vecs"]
    idxs: list[int] = []
    seen_txt: set[str] = set()
    for r in manifest["activitynet"]["records"]:
        if r["task_kind"] != "captions":
            continue
        qt = str(r.get("query_text") or "").strip()
        if qt and qt in tmap and qt not in seen_txt:
            seen_txt.add(qt)
            idxs.append(tmap[qt])
        for ev in r.get("events", []):
            et = str(ev.get("text") or "").strip()
            if et and et in tmap and et not in seen_txt:
                seen_txt.add(et)
                idxs.append(tmap[et])
        if len(idxs) >= max_bank_texts:
            break

    idxs = idxs[:max_bank_texts]
    out: dict[str, np.ndarray] = {}
    for b in ("pe_bigg", "pe_l", "sig400"):
        mat = tvecs[b][idxs].astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-12)
        out[b] = mat
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bank_path, **out)
    return out


def _refine_dp_spans_contrast(
    timestamps_sec: list[float],
    ev_curves: np.ndarray,
    dp_spans: list[dict[str, Any]],
    *,
    v_dur: float,
) -> list[dict[str, Any]]:
    """Refine DP segment bounds around DP centers while keeping chronological precedence."""
    from videntra.localization.refine import expand_boundary_contrast

    refined: list[dict[str, Any]] = []
    ts_arr = np.asarray(timestamps_sec, dtype=np.float32)
    prev_s = 0.0
    prev_e = 0.0
    for k, sp in enumerate(dp_spans):
        mid = 0.5 * (float(sp["start_sec"]) + float(sp["end_sec"]))
        anchor_idx = int(np.argmin(np.abs(ts_arr - mid)))
        cs, ce = expand_boundary_contrast(
            timestamps_sec,
            ev_curves[k].tolist(),
            v_dur=v_dur,
            anchor_idx=anchor_idx,
        )
        # Keep tight valid ordering [start_{k-1} <= start_k, end_{k-1} <= end_k] allowing soft overlap
        cs = max(prev_s, min(float(sp["start_sec"]), cs) if k == 0 else max(prev_s, cs))
        ce = max(cs + 0.5, prev_e, ce)
        ce = min(v_dur, ce)
        cs = min(cs, max(0.0, ce - 0.5))
        refined.append({"clause_id": sp["clause_id"], "start_sec": round(cs, 4), "end_sec": round(ce, 4)})
        prev_s, prev_e = cs, ce
    return refined


def _build_captions_stage1_ctx(
    manifest: Mapping[str, Any],
    store: Mapping[str, Any],
    query_bank: Mapping[str, np.ndarray] | None,
    weights: Mapping[str, float],
    dense_topk: int,
) -> dict[str, Any]:
    """Precompute Stage-1 ActivityNet Captions gallery tensors & batched SGEMM matrices once per split."""
    fdata = store["frame_data"]
    tmap = store["text_str_to_idx"]
    tvecs = store["text_vecs"]
    w = dict(weights)
    w_sum = sum(w.values())

    anet_gallery = [str(v) for v in manifest["activitynet"]["gallery_video_ids"] if f"full::{v}" in fdata]
    indexed_anet = set(anet_gallery)
    vid_to_idx: dict[str, int] = {vid: i for i, vid in enumerate(anet_gallery)}

    g_mean: dict[str, np.ndarray] = {}
    g_frames_flat: dict[str, np.ndarray] = {}
    g_vid_slices: list[tuple[int, int]] = []
    hub_mean: dict[str, np.ndarray] = {}
    for b in ("pe_bigg", "pe_l", "sig400"):
        mats = []
        f_blocks = []
        off = 0
        for vid in anet_gallery:
            f_b = fdata[f"full::{vid}"][b]
            m = np.mean(f_b, axis=0)
            m /= max(1e-12, float(np.linalg.norm(m)))
            mats.append(m)
            if b == "pe_bigg":
                g_vid_slices.append((off, off + len(f_b)))
                off += len(f_b)
            f_blocks.append(f_b)
        g_mean[b] = np.vstack(mats).astype(np.float32) if mats else np.zeros((0, 1), dtype=np.float32)
        g_frames_flat[b] = np.vstack(f_blocks).astype(np.float32) if f_blocks else np.zeros((0, 1), dtype=np.float32)
        if query_bank and b in query_bank and len(g_mean[b]) > 0 and len(query_bank[b]) > 0:
            qb_sims = g_mean[b] @ query_bank[b].T
            hub_mean[b] = np.mean(qb_sims, axis=1).astype(np.float32)
        else:
            hub_mean[b] = np.zeros(len(anet_gallery), dtype=np.float32)

    rel_by_vid: dict[str, str] = {}
    for r in manifest["activitynet"]["records"] + manifest["youcook2"]["records"]:
        rel_by_vid[str(r["video_id"])] = str(r["media_rel_path"])

    cap_queries: list[dict[str, Any]] = []
    seen_q_vids: set[str] = set()
    for r in manifest["activitynet"]["records"]:
        if r["task_kind"] != "captions":
            continue
        vid = str(r["video_id"])
        if r.get("annotation_source") == "val_2":
            continue
        if vid in seen_q_vids:
            continue
        seen_q_vids.add(vid)
        cap_queries.append(r)

    cap_text_list: list[str] = []
    cap_text_seen: dict[str, int] = {}
    for q in cap_queries:
        for txt_s in [str(q["query_text"]).strip()] + [str(ev["text"]).strip() for ev in q["events"]]:
            if txt_s in tmap and txt_s not in cap_text_seen:
                cap_text_seen[txt_s] = len(cap_text_list)
                cap_text_list.append(txt_s)

    cap_tidxs = [tmap[t] for t in cap_text_list]
    g_sim_by_b: dict[str, np.ndarray] = {}
    f_sim_fused = (
        np.zeros((len(g_frames_flat["pe_bigg"]), len(cap_tidxs)), dtype=np.float32)
        if (cap_tidxs and len(g_frames_flat["pe_bigg"]) > 0)
        else np.zeros((0, 0), dtype=np.float32)
    )
    f_sim_by_b: dict[str, np.ndarray] = {}
    if cap_tidxs and len(anet_gallery) > 0:
        for b in ("pe_bigg", "pe_l", "sig400"):
            t_mat_b = tvecs[b][cap_tidxs]
            g_sim_by_b[b] = g_mean[b] @ t_mat_b.T
            f_b_sim = g_frames_flat[b] @ t_mat_b.T
            f_sim_by_b[b] = f_b_sim
            f_sim_fused += (w[b] / w_sum) * f_b_sim

    dense_topk_z_by_b: dict[str, np.ndarray] = {}
    clause_topk_z = np.zeros((len(anet_gallery), len(cap_tidxs)), dtype=np.float32)
    if cap_tidxs and len(anet_gallery) > 0:
        kk_e = min(4, 48)
        raw_cl = np.empty((len(anet_gallery), len(cap_tidxs)), dtype=np.float32)
        for vi, (st_i, en_i) in enumerate(g_vid_slices):
            seg = f_sim_fused[st_i:en_i]
            kk = min(kk_e, len(seg))
            raw_cl[vi] = np.mean(np.partition(seg, -kk, axis=0)[-kk:], axis=0)
        mu_c = np.mean(raw_cl, axis=0, keepdims=True)
        sd_c = np.maximum(1e-6, np.std(raw_cl, axis=0, keepdims=True))
        clause_topk_z = (raw_cl - mu_c) / sd_c

        for b in ("pe_bigg", "pe_l", "sig400"):
            fb = f_sim_by_b[b]
            raw_d = np.empty((len(anet_gallery), len(cap_tidxs)), dtype=np.float32)
            for vi, (st_i, en_i) in enumerate(g_vid_slices):
                seg = fb[st_i:en_i]
                kk = min(dense_topk, len(seg))
                raw_d[vi] = np.mean(np.partition(seg, -kk, axis=0)[-kk:], axis=0)
            mu_d = np.mean(raw_d, axis=0, keepdims=True)
            sd_d = np.maximum(1e-6, np.std(raw_d, axis=0, keepdims=True))
            dense_topk_z_by_b[b] = (raw_d - mu_d) / sd_d

    return {
        "anet_gallery": anet_gallery,
        "indexed_anet": indexed_anet,
        "vid_to_idx": vid_to_idx,
        "g_frames_flat": g_frames_flat,
        "g_vid_slices": g_vid_slices,
        "hub_mean": hub_mean,
        "rel_by_vid": rel_by_vid,
        "cap_queries": cap_queries,
        "cap_text_seen": cap_text_seen,
        "g_sim_by_b": g_sim_by_b,
        "f_sim_fused": f_sim_fused,
        "dense_topk_z_by_b": dense_topk_z_by_b,
        "clause_topk_z": clause_topk_z,
    }


def _evaluate_split_cpu_methods(
    work_root: Path,
    *,
    split: str,
    hparams: Mapping[str, Any] | None = None,
    query_bank: Mapping[str, np.ndarray] | None = None,
    top_k_rerank_stage1: int = 20,
    include_upgraded_stage1: bool = True,
    run_captions: bool = True,
    run_videocomp: bool = True,
    captions_methods_subset: Sequence[str] | None = None,
    preloaded_manifest: Mapping[str, Any] | None = None,
    preloaded_store: Mapping[str, Any] | None = None,
    dp_precomputed_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    precomputed_cap_ctx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run local CPU baselines & calibrated Semi-Markov/3-Axis upgrades on one locked split."""
    hp = dict(DEFAULT_TUNE_HPARAMS)
    if hparams:
        hp.update(hparams)
    w = dict(hp["weights"])
    w_sum = sum(w.values())
    qb_alpha = float(hp["qb_alpha"])
    dense_pool_alpha = float(hp["dense_pool_alpha"])
    dense_topk = max(1, int(hp["dense_topk"]))
    clause_ret_alpha = float(hp["clause_ret_alpha"])
    dp_rank_alpha = float(hp["dp_rank_alpha"])
    order_rank_alpha = float(hp["order_rank_alpha"])
    dp_lambda_bg = float(hp.get("dp_lambda_bg", 0.30))
    dp_lambda_peak = float(hp.get("dp_lambda_peak", 0.35))
    dp_lambda_len = float(hp.get("dp_lambda_len", 0.04))
    span_mode = str(hp.get("span_mode") or "dp_exact")
    com_beta = float(hp["com_beta"])
    order_beta = float(hp["order_beta"])
    min_beta = float(hp["min_beta"])
    cov_beta = float(hp["cov_beta"])

    manifest = (
        dict(preloaded_manifest)
        if preloaded_manifest is not None
        else json.loads((work_root / "splits" / f"{split}_manifest.json").read_text(encoding="utf-8"))
    )
    store = (
        dict(preloaded_store)
        if preloaded_store is not None
        else _load_split_feature_store(work_root, split)
    )
    fdata = store["frame_data"]
    tmap = store["text_str_to_idx"]
    tvecs = store["text_vecs"]

    if precomputed_cap_ctx is not None:
        anet_gallery = list(precomputed_cap_ctx["anet_gallery"])
        indexed_anet = set(precomputed_cap_ctx["indexed_anet"])
    else:
        anet_gallery = [str(v) for v in manifest["activitynet"]["gallery_video_ids"] if f"full::{v}" in fdata]
        indexed_anet = set(anet_gallery)

    default_methods = [
        "whole_paragraph_global",
        "independent_clause_contrast",
        "viterbi_chain",
        "exact_semi_markov_dp",
    ]
    if include_upgraded_stage1:
        default_methods.append("exact_semi_markov_dp_calibrated")
    method_names = [
        m for m in default_methods if (captions_methods_subset is None or m in captions_methods_subset)
    ]

    cap_queries: list[dict[str, Any]] = []
    preds_by_method: dict[str, dict[str, Any]] = {m: {} for m in method_names}
    given_vid_by_method: dict[str, dict[str, Any]] = {m: {} for m in method_names}
    stage1_borderline_candidates: list[dict[str, Any]] = []
    cap_results: dict[str, Any] = {}

    if run_captions:
        ctx = (
            precomputed_cap_ctx
            if precomputed_cap_ctx is not None
            else _build_captions_stage1_ctx(manifest, store, query_bank, w, dense_topk)
        )
        vid_to_idx = ctx["vid_to_idx"]
        g_frames_flat = ctx["g_frames_flat"]
        g_vid_slices = ctx["g_vid_slices"]
        hub_mean = ctx["hub_mean"]
        rel_by_vid = ctx["rel_by_vid"]
        cap_queries = list(ctx["cap_queries"])
        cap_text_seen = ctx["cap_text_seen"]
        g_sim_by_b = ctx["g_sim_by_b"]
        f_sim_fused = ctx["f_sim_fused"]
        dense_topk_z_by_b = ctx["dense_topk_z_by_b"]
        clause_topk_z = ctx["clause_topk_z"]

        def _fused_frame_curve(vid: str, txt: str) -> np.ndarray:
            vi = vid_to_idx.get(vid)
            ti = cap_text_seen.get(txt)
            if vi is not None and ti is not None and f_sim_fused.shape[1] > 0:
                st_i, en_i = g_vid_slices[vi]
                return f_sim_fused[st_i:en_i, ti]
            idx = tmap[txt]
            fd = fdata[f"full::{vid}"]
            num = np.zeros(len(fd["pts_sec"]), dtype=np.float32)
            for b in ("pe_bigg", "pe_l", "sig400"):
                num += w[b] * (fd[b] @ tvecs[b][idx])
            return num / w_sum

        from videntra.localization.refine import expand_boundary_contrast

        need_base_glob = "whole_paragraph_global" in method_names
        need_base_ind = "independent_clause_contrast" in method_names
        need_base_vit = "viterbi_chain" in method_names
        need_dp_raw = "exact_semi_markov_dp" in method_names
        need_dp_cal = "exact_semi_markov_dp_calibrated" in method_names
        need_any_dp = need_dp_raw or need_dp_cal

        for q in cap_queries:
            qid = str(q["query_id"])
            qt = str(q["query_text"]).strip()
            gt_vid = str(q["video_id"])
            events = list(q["events"])
            cids = [str(ev["clause_id"]) for ev in events]
            ev_texts = [str(ev["text"]).strip() for ev in events]
            rels = ["start"] + ["before"] * (len(events) - 1)

            if qt not in tmap or qt not in cap_text_seen:
                continue
            qt_loc = cap_text_seen[qt]

            # Baseline Stage-1 global multi-backbone fusion
            vis_scores = np.zeros(len(anet_gallery), dtype=np.float32)
            # Calibrated Stage-1: Query-Bank hubness + dense frame top-K + clause frame matching
            cal_scores = np.zeros(len(anet_gallery), dtype=np.float32)
            for b in ("pe_bigg", "pe_l", "sig400"):
                s_b = g_sim_by_b[b][:, qt_loc]
                mu_b = float(np.mean(s_b))
                sd_b = max(1e-6, float(np.std(s_b)))
                z_global = (s_b - mu_b) / sd_b
                vis_scores += w[b] * z_global

                if include_upgraded_stage1 and len(g_frames_flat[b]) > 0:
                    s_adj = s_b - qb_alpha * hub_mean[b]
                    mu_adj = float(np.mean(s_adj))
                    sd_adj = max(1e-6, float(np.std(s_adj)))
                    z_adj = (s_adj - mu_adj) / sd_adj
                    z_dense = dense_topk_z_by_b[b][:, qt_loc]
                    cal_scores += w[b] * (z_adj + dense_pool_alpha * z_dense)

            if include_upgraded_stage1 and len(anet_gallery) > 0:
                ev_locs = [cap_text_seen[et] for et in ev_texts if et in cap_text_seen]
                if ev_locs:
                    cl_mean = np.mean(clause_topk_z[:, ev_locs], axis=1)
                    cal_scores += clause_ret_alpha * cl_mean * w_sum

            top_order_idx = np.argsort(-vis_scores)[:top_k_rerank_stage1]
            top_vids = [anet_gallery[int(i)] for i in top_order_idx]
            if include_upgraded_stage1:
                cal_order_idx = np.argsort(-cal_scores)[:top_k_rerank_stage1]
                cal_top_vids = [anet_gallery[int(i)] for i in cal_order_idx]
            else:
                cal_top_vids = list(top_vids)

            use_only_cal = need_dp_cal and not (need_base_glob or need_base_ind or need_base_vit or need_dp_raw)
            pool_vids = cal_top_vids if use_only_cal else list(dict.fromkeys(top_vids + cal_top_vids))
            eval_vids = list(dict.fromkeys(pool_vids + ([gt_vid] if gt_vid in indexed_anet else [])))
            cache_vid_spans: dict[str, dict[str, Any]] = {m: {} for m in method_names}

            for vid in eval_vids:
                fd = fdata[f"full::{vid}"]
                ts = fd["pts_sec"].tolist()
                ords = fd["ordinals"].tolist()
                dur = fd["duration"]
                v_pos = vid_to_idx.get(vid, -1)
                base_g = float(vis_scores[v_pos]) if v_pos >= 0 else 0.0
                cal_g = float(cal_scores[v_pos]) if (include_upgraded_stage1 and v_pos >= 0) else base_g

                base_dp_key = (vid, qid, 0.30, 0.35, 0.04)
                cal_dp_key = (vid, qid, dp_lambda_bg, dp_lambda_peak, dp_lambda_len)
                need_base_dp_compute = need_dp_raw and (dp_precomputed_cache is None or base_dp_key not in dp_precomputed_cache)
                need_cal_dp_compute = need_dp_cal and (dp_precomputed_cache is None or cal_dp_key not in dp_precomputed_cache)
                ev_curves = None
                if need_base_ind or need_base_vit or need_base_dp_compute or need_cal_dp_compute:
                    ev_curves = np.vstack([_fused_frame_curve(vid, et) for et in ev_texts])

                ws, we = 0.0, dur
                if need_base_glob or need_base_vit:
                    p_curve = _fused_frame_curve(vid, qt)
                    pk_i = int(np.argmax(p_curve))
                    ws, we = expand_boundary_contrast(ts, p_curve.tolist(), v_dur=dur, anchor_idx=pk_i)

                if need_base_glob:
                    cache_vid_spans["whole_paragraph_global"][vid] = (
                        base_g,
                        [{"clause_id": cid, "start_sec": ws, "end_sec": we} for cid in cids],
                    )

                if need_base_ind and ev_curves is not None:
                    ind_sps = []
                    for k_c, cid in enumerate(cids):
                        ck = ev_curves[k_c]
                        ci = int(np.argmax(ck))
                        cs, ce = expand_boundary_contrast(ts, ck.tolist(), v_dur=dur, anchor_idx=ci)
                        ind_sps.append({"clause_id": cid, "start_sec": cs, "end_sec": ce})
                    cache_vid_spans["independent_clause_contrast"][vid] = (
                        float(np.mean(np.max(ev_curves, axis=1))),
                        ind_sps,
                    )

                if need_base_vit and ev_curves is not None:
                    vit = viterbi_subevent_localization(ts, ev_curves, max_gap_sec=45.0)
                    vit_sc = float(vit.avg_score) if vit is not None else 0.0
                    vit_s = float(vit.start_sec) if vit is not None else ws
                    vit_e = float(max(vit_s + 0.5, vit.end_sec)) if vit is not None else we
                    cache_vid_spans["viterbi_chain"][vid] = (
                        vit_sc,
                        [{"clause_id": cid, "start_sec": vit_s, "end_sec": vit_e} for cid in cids],
                    )

                if need_any_dp:
                    def _solve_dp_for_key(cache_key: tuple[Any, ...], lb: float, lp: float, ll: float) -> dict[str, Any]:
                        if dp_precomputed_cache is not None and cache_key in dp_precomputed_cache:
                            return dp_precomputed_cache[cache_key]
                        assert ev_curves is not None
                        seg_res = localize_exact_segment_chain_dp(
                            timestamps_sec=ts,
                            event_vis_curves=ev_curves,
                            clause_ids=cids,
                            clause_texts=ev_texts,
                            relations_to_prev=rels,
                            frame_ordinals=ords,
                            v_dur_sec=dur,
                            max_gap_sec=45.0,
                            lambda_bg=lb,
                            lambda_peak=lp,
                            lambda_len=ll,
                        )
                        if seg_res is not None:
                            raw_dp_sps = [sp.to_dict() for sp in seg_res.spans]
                            entry = {
                                "ok": True,
                                "avg_score": float(seg_res.avg_score),
                                "order_margin": float(seg_res.directional_order_margin),
                                "dp_exact": raw_dp_sps,
                            }
                        else:
                            entry = {"ok": False}
                        if dp_precomputed_cache is not None:
                            dp_precomputed_cache[cache_key] = entry
                        return entry

                    if need_dp_raw:
                        b_dp = _solve_dp_for_key(base_dp_key, 0.30, 0.35, 0.04)
                        if b_dp.get("ok"):
                            cache_vid_spans["exact_semi_markov_dp"][vid] = (
                                base_g + 2.0 * b_dp["avg_score"] + 0.8 * b_dp["order_margin"],
                                b_dp["dp_exact"],
                            )
                        else:
                            cache_vid_spans["exact_semi_markov_dp"][vid] = (base_g, [])

                    if need_dp_cal:
                        c_dp = _solve_dp_for_key(cal_dp_key, dp_lambda_bg, dp_lambda_peak, dp_lambda_len)
                        if c_dp.get("ok"):
                            raw_dp_sps = c_dp["dp_exact"]
                            if span_mode == "dp_contrast_blend":
                                if "dp_contrast_blend" not in c_dp:
                                    if ev_curves is None:
                                        ev_curves = np.vstack([_fused_frame_curve(vid, et) for et in ev_texts])
                                    c_dp["dp_contrast_blend"] = _refine_dp_spans_contrast(ts, ev_curves, raw_dp_sps, v_dur=dur)
                                cal_sps = c_dp["dp_contrast_blend"]
                            else:
                                cal_sps = raw_dp_sps
                            cache_vid_spans["exact_semi_markov_dp_calibrated"][vid] = (
                                cal_g + dp_rank_alpha * c_dp["avg_score"] + order_rank_alpha * c_dp["order_margin"],
                                cal_sps,
                            )
                        else:
                            cache_vid_spans["exact_semi_markov_dp_calibrated"][vid] = (cal_g, [])

            if need_any_dp:
                primary_m = "exact_semi_markov_dp_calibrated" if need_dp_cal else "exact_semi_markov_dp"
                primary_pool = cal_top_vids if need_dp_cal else top_vids
                primary_ranked = sorted(primary_pool, key=lambda v: -cache_vid_spans[primary_m][v][0])
                if len(primary_ranked) >= 2:
                    s_top3 = [float(cache_vid_spans[primary_m][v][0]) for v in primary_ranked[:3]]
                    stage1_borderline_candidates.append({
                        "query_id": qid,
                        "query_text": qt,
                        "candidate_video_ids": [f"full::{v}" for v in primary_ranked[:3]],
                        "stage1_scores": s_top3,
                        "margin_top1_top2": float(s_top3[0] - s_top3[1]),
                        "rel_paths": {f"full::{v}": rel_by_vid.get(v, "") for v in primary_ranked[:3]},
                        "focus_intervals_by_id": {
                            f"full::{v}": [
                                [float(sp["start_sec"]), float(sp["end_sec"])]
                                for sp in cache_vid_spans[primary_m][v][1]
                            ]
                            for v in primary_ranked[:3]
                        },
                    })

            for mname in method_names:
                if mname == "exact_semi_markov_dp":
                    ranked_vids = sorted(top_vids, key=lambda v: -cache_vid_spans[mname][v][0])
                elif mname == "exact_semi_markov_dp_calibrated":
                    ranked_vids = sorted(cal_top_vids, key=lambda v: -cache_vid_spans[mname][v][0])
                else:
                    ranked_vids = list(top_vids)
                preds_by_method[mname][qid] = {
                    "ranked_video_ids": ranked_vids,
                    "top_k_results": [
                        {"video_id": v, "event_spans": cache_vid_spans[mname][v][1]}
                        for v in ranked_vids[:5]
                    ],
                    "candidate_scores": {v: float(cache_vid_spans[mname][v][0]) for v in ranked_vids},
                }
                if gt_vid in indexed_anet:
                    given_vid_by_method[mname][qid] = {
                        "video_id": gt_vid,
                        "event_spans": cache_vid_spans[mname][gt_vid][1],
                    }

        cap_results = {
            mname: evaluate_activitynet_full_chain(
                cap_queries,
                preds_by_method[mname],
                indexed_video_ids=indexed_anet,
                given_video_predictions_by_qid=given_vid_by_method[mname],
            )
            for mname in method_names
        }

    # 2. VideoComp evaluation (ActivityNet-Comp + YouCook2-Comp)
    comp_pairs: list[dict[str, Any]] = []
    comp_scores: dict[str, dict[str, dict[str, float]]] = {}
    comp_results: dict[str, Any] = {}
    if run_videocomp:
        comp_pairs = [
            r for r in (manifest["activitynet"]["records"] + manifest["youcook2"]["records"])
            if r["task_kind"] == "videocomp"
        ]
        comp_scores = {
            "global_paragraph_3backbone": {},
            "viterbi_forward_dp": {},
            "three_axis_compositional": {},
        }
        for r in comp_pairs:
            pid = str(r["pair_id"])
            clip_key = f"clip::{pid}" if f"clip::{pid}" in fdata else f"full::{r['video_id']}"
            if clip_key not in fdata:
                continue
            fd = fdata[clip_key]
            ts = fd["pts_sec"].tolist()
            pos_t = str(r["positive_text"]).strip()
            neg_t = str(r["negative_text"]).strip()
            if pos_t not in tmap or neg_t not in tmap:
                continue

            def _score_pair_candidate(txt: str) -> dict[str, float]:
                c_idx = tmap[txt]
                g_sc = 0.0
                for b in ("pe_bigg", "pe_l", "sig400"):
                    f_mean = np.mean(fd[b], axis=0)
                    f_mean /= max(1e-12, float(np.linalg.norm(f_mean)))
                    g_sc += w[b] * float(np.dot(f_mean, tvecs[b][c_idx]))
                g_sc /= w_sum

                chain = decompose_query_event_chain(txt)
                c_texts = [ec.text for ec in chain if ec.text in tmap] or [txt]
                c_rels = [("start" if i == 0 else ec.relation_to_prev) for i, ec in enumerate(chain) if ec.text in tmap] or ["start"]

                curves = []
                for ct in c_texts:
                    ci = tmap[ct]
                    c_vec = np.zeros(len(ts), dtype=np.float32)
                    for b in ("pe_bigg", "pe_l", "sig400"):
                        c_vec += w[b] * (fd[b] @ tvecs[b][ci])
                    curves.append(c_vec / w_sum)
                c_mat = np.vstack(curves)

                vit_res = viterbi_subevent_localization(ts, c_mat, max_gap_sec=45.0)
                vit_sc = float(vit_res.avg_score) if vit_res is not None else g_sc

                three_ax = score_videocomp_order(
                    ts,
                    c_mat,
                    relations_to_prev=c_rels,
                    global_score=g_sc,
                    beta_order=order_beta,
                    beta_com=com_beta,
                    beta_min=min_beta,
                    beta_cov=cov_beta,
                    use_segment_dp=True,
                )
                return {
                    "global": g_sc,
                    "viterbi": vit_sc,
                    "three_axis": float(three_ax["combined_score"]),
                }

            sp_p = _score_pair_candidate(pos_t)
            sp_n = _score_pair_candidate(neg_t)
            comp_scores["global_paragraph_3backbone"][pid] = {"positive_score": sp_p["global"], "negative_score": sp_n["global"]}
            comp_scores["viterbi_forward_dp"][pid] = {"positive_score": sp_p["viterbi"], "negative_score": sp_n["viterbi"]}
            comp_scores["three_axis_compositional"][pid] = {"positive_score": sp_p["three_axis"], "negative_score": sp_n["three_axis"]}

        comp_results = {
            mname: evaluate_videocomp(comp_pairs, sc_map)
            for mname, sc_map in comp_scores.items()
        }

    return {
        "split": split,
        "hparams": hp,
        "indexed_anet_videos": indexed_anet,
        "activitynet_captions": cap_results,
        "videocomp": comp_results,
        "captions_queries": cap_queries,
        "captions_preds": preds_by_method,
        "given_vid_preds": given_vid_by_method,
        "comp_pairs": comp_pairs,
        "comp_scores": comp_scores,
        "stage1_borderline_candidates": stage1_borderline_candidates,
    }


def _canon_sha256(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _run_or_load_vlm_shards(
    work_root: Path,
    *,
    split: str,
    model_tag: str,
    pairs: list[dict[str, Any]],
    rel_path_map: dict[str, str],
    annotate_timestamps: bool,
    template_kind: str,
    shard_size: int = 120,
    num_frames: int = 8,
    quant: str = "nf4_4bit",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Run blinded Modal L4 Qwen2.5-VL-3B-Instruct shards (or reuse cached local JSONL receipts)."""
    from scripts.modal_app_functions import app as modal_app, run_l4_vlm_rerank_shard

    out_dir = work_root / "vlm" / split / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_by_qid: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    run_id = f"vlm_{split}"

    shards = [
        (i // shard_size, pairs[i : i + shard_size])
        for i in range(0, len(pairs), shard_size)
    ]
    missing_shards: list[tuple[str, list[dict[str, Any]], Path, Path]] = []
    for s_idx, chunk in shards:
        sid = f"{model_tag}_s{s_idx:03d}"
        lp = out_dir / f"{sid}_scores.jsonl"
        rp = out_dir / f"{sid}_receipt.json"
        if lp.is_file() and rp.is_file():
            rc = json.loads(rp.read_text(encoding="utf-8"))
            if int(rc.get("row_count", -1)) == len(chunk):
                receipts.append(rc)
                for line in lp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        scores_by_qid[str(row["query_id"])] = row
                continue
        missing_shards.append((sid, chunk, lp, rp))

    if missing_shards:
        import modal  # type: ignore

        starmap_args: list[tuple[Any, ...]] = []
        for sid, chunk, lp, rp in missing_shards:
            sub_vids = {v for p in chunk for v in p["candidate_video_ids"]}
            sub_rel = {v: rel_path_map[v] for v in sorted(sub_vids) if v in rel_path_map}
            starmap_args.append((
                run_id,
                model_tag,
                sid,
                json.dumps(chunk),
                json.dumps(sub_rel),
                quant,
                int(num_frames),
                bool(annotate_timestamps),
                template_kind,
            ))

        with modal.enable_output(), modal_app.run():
            starmap_results = list(run_l4_vlm_rerank_shard.starmap(starmap_args))

        for (sid, chunk, lp, rp), res in zip(missing_shards, starmap_results):
            rc = res["receipt"]
            receipts.append(rc)
            rp.write_text(json.dumps(rc, indent=2, ensure_ascii=False), encoding="utf-8")
            pull_modal_file(f"{run_id}/{model_tag}/{sid}/vlm_pair_scores.jsonl", lp)
            for line in lp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    scores_by_qid[str(row["query_id"])] = row

    return scores_by_qid, receipts


def _score_vlm_captions_variant(
    cpu_eval: Mapping[str, Any],
    *,
    work_root: Path,
    split: str,
    tag_suffix: str,
    annotate_timestamps: bool,
    template_kind: str,
    gamma: float,
    blend_alpha: float,
    max_queries: int | None = None,
    run_remote: bool = True,
    precomputed_vlm_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply Stage-2 Modal L4 Qwen2.5-VL-3B-Instruct reranking to borderline Captions queries."""
    cands = sorted(
        (c for c in cpu_eval["stage1_borderline_candidates"] if float(c["margin_top1_top2"]) < float(gamma)),
        key=lambda c: (float(c["margin_top1_top2"]), str(c["query_id"])),
    )
    if max_queries is not None:
        cands = cands[: int(max_queries)]
    allowed_qids = {str(c["query_id"]) for c in cands}
    rel_map: dict[str, str] = {}
    pairs: list[dict[str, Any]] = []
    for c in cands:
        rel_map.update(c["rel_paths"])
        pairs.append({
            "query_id": c["query_id"],
            "query_text": c["query_text"],
            "candidate_video_ids": c["candidate_video_ids"],
            "focus_intervals_by_id": c["focus_intervals_by_id"] if annotate_timestamps else None,
        })

    model_tag = f"cap_{tag_suffix}"
    vlm_map: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    if precomputed_vlm_map is not None:
        vlm_map = {k: v for k, v in precomputed_vlm_map.items() if k in allowed_qids}
    elif pairs and run_remote:
        vlm_map, receipts = _run_or_load_vlm_shards(
            work_root,
            split=split,
            model_tag=model_tag,
            pairs=pairs,
            rel_path_map=rel_map,
            annotate_timestamps=annotate_timestamps,
            template_kind=template_kind,
        )

    base_key = "exact_semi_markov_dp_calibrated" if "exact_semi_markov_dp_calibrated" in cpu_eval["captions_preds"] else "exact_semi_markov_dp"
    base_preds = cpu_eval["captions_preds"][base_key]
    merged_preds: dict[str, Any] = {}
    for qid, orig_p in base_preds.items():
        ranked = list(orig_p["ranked_video_ids"])
        topk_res = list(orig_p["top_k_results"])
        cand_sc = dict(orig_p.get("candidate_scores") or {})
        v_row = vlm_map.get(qid) if qid in allowed_qids else None
        raw_vlm_scores = (v_row.get("scores_by_video") or v_row.get("scores")) if v_row else None
        if isinstance(raw_vlm_scores, Mapping) and ranked:
            s_vlm = {str(k).replace("full::", "", 1): float(v) for k, v in raw_vlm_scores.items()}
            top3 = ranked[:3]
            if all(v in s_vlm for v in top3):
                vals = np.asarray([s_vlm[v] for v in top3], dtype=np.float32)
                z_vlm = (vals - float(np.mean(vals))) / max(1e-6, float(np.std(vals)))
                s1_vals = np.asarray([cand_sc.get(v, 0.0) for v in top3], dtype=np.float32)
                z_s1 = (s1_vals - float(np.mean(s1_vals))) / max(1e-6, float(np.std(s1_vals)))
                fused_top3 = z_s1 + float(blend_alpha) * z_vlm
                new_top3 = [top3[int(i)] for i in np.argsort(-fused_top3)]
                ranked = new_top3 + [v for v in ranked[3:] if v not in new_top3]
                by_v_obj = {r["video_id"]: r for r in topk_res}
                topk_res = [by_v_obj[v] for v in ranked[:5] if v in by_v_obj]
        merged_preds[qid] = {"ranked_video_ids": ranked, "top_k_results": topk_res}

    eval_out = evaluate_activitynet_full_chain(
        cpu_eval["captions_queries"],
        merged_preds,
        indexed_video_ids=cpu_eval["indexed_anet_videos"],
        given_video_predictions_by_qid=cpu_eval["given_vid_preds"][base_key],
    )
    return {
        "metrics": eval_out,
        "predictions": merged_preds,
        "vlm_queries_triggered": len(cands),
        "vlm_map": vlm_map,
        "receipts": receipts,
    }


def _score_vlm_videocomp(
    cpu_eval: Mapping[str, Any],
    *,
    work_root: Path,
    split: str,
    model_tag: str = "vc_timestamped",
    vc_alpha: float = 0.35,
    vc_margin_gamma: float = 999.0,
    max_pairs: int | None = None,
    annotate_timestamps: bool = True,
    shard_size: int = 60,
    run_remote: bool = True,
    precomputed_vlm_map: Mapping[str, Any] | None = None,
    precomputed_pair_key_map: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Run blinded Modal VLM scoring on VideoComp pairs (`cand::<pair_id>`) w/ symmetric gating."""
    base_sc = cpu_eval["comp_scores"]["three_axis_compositional"]
    def _pair_margin(r: Mapping[str, Any]) -> float:
        sc = base_sc.get(str(r["pair_id"]))
        if not sc:
            return 999.0
        return abs(float(sc["positive_score"]) - float(sc["negative_score"]))

    pairs_src = sorted(
        (r for r in cpu_eval["comp_pairs"] if _pair_margin(r) < float(vc_margin_gamma)),
        key=lambda r: (_pair_margin(r), str(r["pair_id"])),
    )
    if max_pairs is not None:
        pairs_src = pairs_src[: int(max_pairs)]
    allowed_pids = {str(r["pair_id"]) for r in pairs_src}
    rel_map: dict[str, str] = {}
    vlm_tasks: list[dict[str, Any]] = []
    pair_key_map: dict[str, tuple[str, str]] = (
        dict(precomputed_pair_key_map) if precomputed_pair_key_map is not None else {}
    )

    if precomputed_vlm_map is None or precomputed_pair_key_map is None:
        for r in pairs_src:
            pid = str(r["pair_id"])
            vid_tok = f"cand::{pid}"
            rel_map[vid_tok] = str(r["media_rel_path"])
            cwin = (
                {vid_tok: [float(r["clip_start_sec"]), float(r["clip_end_sec"])]}
                if float(r.get("clip_end_sec", 0.0)) > float(r.get("clip_start_sec", 0.0))
                else None
            )
            # Blind ordering by SHA-256 of candidate text so Modal NEVER knows which text is positive!
            h_p = hashlib.sha256(f"{pid}:{r['positive_text']}".encode("utf-8")).hexdigest()[:12]
            h_n = hashlib.sha256(f"{pid}:{r['negative_text']}".encode("utf-8")).hexdigest()[:12]
            qid_p = f"vc_{pid}_{h_p}_a"
            qid_n = f"vc_{pid}_{h_n}_b"
            if qid_p > qid_n:
                qid_p, qid_n = qid_n, qid_p
            pair_key_map[pid] = (qid_p, qid_n)
            items_sorted = sorted(
                [(qid_p, str(r["positive_text"])), (qid_n, str(r["negative_text"]))],
                key=lambda x: x[0],
            )
            for q_k, q_t in items_sorted:
                vlm_tasks.append({
                    "query_id": q_k,
                    "query_text": q_t,
                    "candidate_video_ids": [vid_tok],
                    "clip_windows_by_id": cwin,
                })

    vlm_map: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    if precomputed_vlm_map is not None:
        vlm_map = dict(precomputed_vlm_map)
    elif vlm_tasks and run_remote:
        vlm_map, receipts = _run_or_load_vlm_shards(
            work_root,
            split=split,
            model_tag=model_tag,
            pairs=vlm_tasks,
            rel_path_map=rel_map,
            annotate_timestamps=annotate_timestamps,
            template_kind="compositional",
            shard_size=int(shard_size),
        )

    fused_scores: dict[str, dict[str, float]] = {}
    vlm_only_scores: dict[str, dict[str, float]] = {}
    for r in cpu_eval["comp_pairs"]:
        pid = str(r["pair_id"])
        sc0 = dict(base_sc.get(pid, {"positive_score": 0.0, "negative_score": 0.0}))
        if pid in allowed_pids and pid in pair_key_map:
            qp, qn = pair_key_map[pid]
            vid_tok = f"cand::{pid}"
            rp_row = vlm_map.get(qp, {})
            rn_row = vlm_map.get(qn, {})
            rp = rp_row.get("scores_by_video") or rp_row.get("scores") or {}
            rn = rn_row.get("scores_by_video") or rn_row.get("scores") or {}
            if vid_tok in rp and vid_tok in rn:
                vp = float(rp[vid_tok])
                vn = float(rn[vid_tok])
                vlm_only_scores[pid] = {"positive_score": vp, "negative_score": vn}
                fused_scores[pid] = {
                    "positive_score": float(sc0["positive_score"]) + float(vc_alpha) * vp,
                    "negative_score": float(sc0["negative_score"]) + float(vc_alpha) * vn,
                }
                continue
        fused_scores[pid] = sc0

    return {
        "metrics_fused": evaluate_videocomp(cpu_eval["comp_pairs"], fused_scores),
        "metrics_vlm_only": evaluate_videocomp(pairs_src, vlm_only_scores) if vlm_only_scores else None,
        "fused_scores": fused_scores,
        "vlm_map": vlm_map,
        "pair_key_map": pair_key_map,
        "receipts": receipts,
    }


def _extract_vlm_vram_max_mib(
    receipts: Sequence[Mapping[str, Any]],
    *,
    work_root: Path | None = None,
) -> float:
    mx = 0.0
    for rc in receipts:
        t = rc.get("telemetry") or {}
        val = t.get("nvml_total_used_mib") or t.get("pytorch_peak_reserved_mib") or 0.0
        mx = max(mx, float(val))
    if mx == 0.0 and work_root is not None:
        pilot_p = work_root / "pilot" / "pilot_gpu_comparison.json"
        if pilot_p.is_file():
            try:
                p = json.loads(pilot_p.read_text(encoding="utf-8"))
                t = ((p.get("L4_Qwen25_VL_3B_NF4") or {}).get("receipt") or {}).get("telemetry") or {}
                mx = max(mx, float(t.get("nvml_total_used_mib") or t.get("pytorch_peak_reserved_mib") or 0.0))
            except Exception:
                pass
    return round(mx, 2)


def cmd_tune_and_ablate(
    work_root: Path,
    *,
    with_vlm: bool = False,
    vlm_max_cap_queries: int = 80,
    vlm_max_vc_pairs: int = 120,
    resume_vlm: bool = False,
) -> dict[str, Any]:
    """Run all hyperparameter sweeps & ablations strictly on `Tune`, saving selected Champion config."""
    t_start = time.perf_counter()
    tune_dir = work_root / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    cpu_ckpt_path = tune_dir / "tune_cpu_stage_checkpoint.json"

    qb = _build_or_load_tune_query_bank(work_root)
    tune_manifest = json.loads((work_root / "splits" / "tune_manifest.json").read_text(encoding="utf-8"))
    tune_store = _load_split_feature_store(work_root, "tune")
    cap_ctx = _build_captions_stage1_ctx(
        tune_manifest,
        tune_store,
        qb,
        DEFAULT_TUNE_HPARAMS["weights"],
        int(DEFAULT_TUNE_HPARAMS["dense_topk"]),
    )
    dp_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    print(f"[tune] Loaded store & Stage-1 ctx in {time.perf_counter() - t_start:.2f}s", flush=True)

    if resume_vlm and cpu_ckpt_path.is_file():
        ckpt = json.loads(cpu_ckpt_path.read_text(encoding="utf-8"))
        best_cap_hp = dict(ckpt["best_cap_hp"])
        cap_candidates = list(ckpt["cap_candidates"])
        vc_candidates = list(ckpt["vc_candidates"])
        base_cap_summary = dict(ckpt["baseline_captions"])
        base_vc_summary = dict(ckpt["baseline_videocomp"])
        tuned_cpu = _evaluate_split_cpu_methods(
            work_root,
            split="tune",
            hparams=best_cap_hp,
            query_bank=qb,
            captions_methods_subset=("exact_semi_markov_dp_calibrated",),
            preloaded_manifest=tune_manifest,
            preloaded_store=tune_store,
            dp_precomputed_cache=dp_cache,
            precomputed_cap_ctx=cap_ctx,
        )
        print(f"[tune] Resumed from CPU checkpoint in {time.perf_counter() - t_start:.2f}s", flush=True)
    else:
        base_eval = _evaluate_split_cpu_methods(
            work_root,
            split="tune",
            hparams=DEFAULT_TUNE_HPARAMS,
            query_bank=qb,
            preloaded_manifest=tune_manifest,
            preloaded_store=tune_store,
            dp_precomputed_cache=dp_cache,
            precomputed_cap_ctx=cap_ctx,
        )
        print(f"[tune] Base eval done at {time.perf_counter() - t_start:.2f}s", flush=True)

        # 1A. Sweep Captions retrieval/Hubness parameters on Tune (with shared DP & Stage-1 ctx!)
        cap_candidates = []
        best_cap_hp = dict(DEFAULT_TUNE_HPARAMS)
        best_cap_score = -1.0
        for qb_a in (0.0, 0.35, 0.70):
            for dense_a in (0.0, 0.45, 0.90):
                for cl_a in (0.0, 0.45, 0.90):
                    for sp_m in ("dp_exact", "dp_contrast_blend"):
                        hp_try = dict(DEFAULT_TUNE_HPARAMS)
                        hp_try.update({
                            "qb_alpha": qb_a,
                            "dense_pool_alpha": dense_a,
                            "clause_ret_alpha": cl_a,
                            "span_mode": sp_m,
                        })
                        ev_try = _evaluate_split_cpu_methods(
                            work_root,
                            split="tune",
                            hparams=hp_try,
                            query_bank=qb,
                            run_captions=True,
                            run_videocomp=False,
                            captions_methods_subset=("exact_semi_markov_dp_calibrated",),
                            preloaded_manifest=tune_manifest,
                            preloaded_store=tune_store,
                            dp_precomputed_cache=dp_cache,
                            precomputed_cap_ctx=cap_ctx,
                        )
                        m_cal = ev_try["activitynet_captions"]["exact_semi_markov_dp_calibrated"]
                        obj = float(m_cal["full_chain_hit_at_1"]) + 0.1 * float(m_cal["video_recall_at_20"])
                        cap_candidates.append({
                            "qb_alpha": qb_a,
                            "dense_pool_alpha": dense_a,
                            "clause_ret_alpha": cl_a,
                            "span_mode": sp_m,
                            "dp_lambda_len": 0.04,
                            "dp_rank_alpha": float(hp_try["dp_rank_alpha"]),
                            "order_rank_alpha": float(hp_try["order_rank_alpha"]),
                            "hit_at_1": m_cal["full_chain_hit_at_1"],
                            "recall_at_20": m_cal["video_recall_at_20"],
                            "given_vid_hit_at_1": m_cal["given_video_full_chain_hit_at_1"],
                        })
                        if obj > best_cap_score:
                            best_cap_score = obj
                            best_cap_hp.update({
                                "qb_alpha": qb_a,
                                "dense_pool_alpha": dense_a,
                                "clause_ret_alpha": cl_a,
                                "span_mode": sp_m,
                            })
        print(f"[tune] Stage 1A (54 retrieval combos) done at {time.perf_counter() - t_start:.2f}s", flush=True)

        # 1B. Sweep DP Length Prior + Order Rerank weights strictly on Tune
        for dp_len in (0.04, 0.10, 0.14, 0.22):
            for dp_ra in (1.5, 2.5, 3.5):
                for ord_ra in (0.4, 0.85, 1.4):
                    hp_try = dict(best_cap_hp)
                    hp_try.update({
                        "dp_lambda_len": dp_len,
                        "dp_rank_alpha": dp_ra,
                        "order_rank_alpha": ord_ra,
                    })
                    ev_try = _evaluate_split_cpu_methods(
                        work_root,
                        split="tune",
                        hparams=hp_try,
                        query_bank=qb,
                        run_captions=True,
                        run_videocomp=False,
                        captions_methods_subset=("exact_semi_markov_dp_calibrated",),
                        preloaded_manifest=tune_manifest,
                        preloaded_store=tune_store,
                        dp_precomputed_cache=dp_cache,
                        precomputed_cap_ctx=cap_ctx,
                    )
                    m_cal = ev_try["activitynet_captions"]["exact_semi_markov_dp_calibrated"]
                    obj = float(m_cal["full_chain_hit_at_1"]) + 0.1 * float(m_cal["video_recall_at_20"])
                    cap_candidates.append({
                        "qb_alpha": best_cap_hp["qb_alpha"],
                        "dense_pool_alpha": best_cap_hp["dense_pool_alpha"],
                        "clause_ret_alpha": best_cap_hp["clause_ret_alpha"],
                        "span_mode": best_cap_hp["span_mode"],
                        "dp_lambda_len": dp_len,
                        "dp_rank_alpha": dp_ra,
                        "order_rank_alpha": ord_ra,
                        "hit_at_1": m_cal["full_chain_hit_at_1"],
                        "recall_at_20": m_cal["video_recall_at_20"],
                        "given_vid_hit_at_1": m_cal["given_video_full_chain_hit_at_1"],
                    })
                    if obj > best_cap_score:
                        best_cap_score = obj
                        best_cap_hp.update({
                            "dp_lambda_len": dp_len,
                            "dp_rank_alpha": dp_ra,
                            "order_rank_alpha": ord_ra,
                        })

        # 2. Sweep VideoComp 3-Axis compositional weights strictly on Tune via precomputed component signals
        vc_pairs = [
            r for r in (tune_manifest["activitynet"]["records"] + tune_manifest["youcook2"]["records"])
            if r["task_kind"] == "videocomp"
        ]
        w_vc = dict(best_cap_hp["weights"])
        w_vc_sum = sum(w_vc.values())
        fdata_tune = tune_store["frame_data"]
        tmap_tune = tune_store["text_str_to_idx"]
        tvecs_tune = tune_store["text_vecs"]
        vc_comp_cache: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for r in vc_pairs:
            pid = str(r["pair_id"])
            clip_key = f"clip::{pid}" if f"clip::{pid}" in fdata_tune else f"full::{r['video_id']}"
            if clip_key not in fdata_tune:
                continue
            fd = fdata_tune[clip_key]
            ts = fd["pts_sec"].tolist()
            pos_t = str(r["positive_text"]).strip()
            neg_t = str(r["negative_text"]).strip()
            if pos_t not in tmap_tune or neg_t not in tmap_tune:
                continue

            def _extract_comp_terms(txt: str) -> dict[str, float]:
                c_idx = tmap_tune[txt]
                g_sc = 0.0
                for b in ("pe_bigg", "pe_l", "sig400"):
                    f_mean = np.mean(fd[b], axis=0)
                    f_mean /= max(1e-12, float(np.linalg.norm(f_mean)))
                    g_sc += w_vc[b] * float(np.dot(f_mean, tvecs_tune[b][c_idx]))
                g_sc /= w_vc_sum
                chain = decompose_query_event_chain(txt)
                c_texts = [ec.text for ec in chain if ec.text in tmap_tune] or [txt]
                c_rels = [("start" if i == 0 else ec.relation_to_prev) for i, ec in enumerate(chain) if ec.text in tmap_tune] or ["start"]
                curves = []
                for ct in c_texts:
                    ci = tmap_tune[ct]
                    c_vec = np.zeros(len(ts), dtype=np.float32)
                    for b in ("pe_bigg", "pe_l", "sig400"):
                        c_vec += w_vc[b] * (fd[b] @ tvecs_tune[b][ci])
                    curves.append(c_vec / w_vc_sum)
                c_mat = np.vstack(curves)
                return score_videocomp_order(ts, c_mat, relations_to_prev=c_rels, global_score=g_sc, use_segment_dp=True)

            vc_comp_cache[pid] = (_extract_comp_terms(pos_t), _extract_comp_terms(neg_t))

        vc_candidates = []
        best_vc_all = -1.0
        for b_ord in (0.6, 1.0, 1.6):
            for b_com in (0.2, 0.45, 0.75):
                for b_min in (0.3, 0.6, 1.0):
                    for b_cov in (0.15, 0.35):
                        sc_map = {}
                        for pid, (dp_pos, dp_neg) in vc_comp_cache.items():
                            def _lin(d: dict[str, float]) -> float:
                                return (
                                    0.50 * float(d["global_score"])
                                    + 0.50 * float(d["event_max_mean"])
                                    + 0.50 * float(d["forward_avg"])
                                    + b_min * (0.5 * float(d["min_clause_sim"]) + 0.5 * float(d["bottom2_clause_mean"]))
                                    + b_ord * float(d["directional_order_margin"])
                                    + b_com * float(d["com_concordance"])
                                    + b_cov * float(d["active_clause_coverage"])
                                )
                            sc_map[pid] = {"positive_score": _lin(dp_pos), "negative_score": _lin(dp_neg)}
                        vc_m = evaluate_videocomp(vc_pairs, sc_map)["overall"]
                        all_sc = float(vc_m["all"])
                        vc_candidates.append({
                            "order_beta": b_ord,
                            "com_beta": b_com,
                            "min_beta": b_min,
                            "cov_beta": b_cov,
                            "all": all_sc,
                            "temporal_reordering": vc_m.get("temporal_reordering", {}).get("accuracy"),
                            "action_replacement": vc_m.get("action_replacement", {}).get("accuracy"),
                            "segment_mismatch": vc_m.get("segment_mismatch", {}).get("accuracy"),
                        })
                        if all_sc > best_vc_all:
                            best_vc_all = all_sc
                            best_cap_hp.update({
                                "order_beta": b_ord,
                                "com_beta": b_com,
                                "min_beta": b_min,
                                "cov_beta": b_cov,
                            })

        tuned_cpu = _evaluate_split_cpu_methods(
            work_root,
            split="tune",
            hparams=best_cap_hp,
            query_bank=qb,
            captions_methods_subset=("exact_semi_markov_dp_calibrated",),
            preloaded_manifest=tune_manifest,
            preloaded_store=tune_store,
            dp_precomputed_cache=dp_cache,
            precomputed_cap_ctx=cap_ctx,
        )
        for m_k, d_v in base_eval["activitynet_captions"].items():
            tuned_cpu["activitynet_captions"].setdefault(m_k, d_v)
        for m_k, d_v in base_eval["captions_preds"].items():
            tuned_cpu["captions_preds"].setdefault(m_k, d_v)
        base_cap_summary = {m: {k: v for k, v in d.items() if k != "per_query_hits"} for m, d in base_eval["activitynet_captions"].items()}
        base_vc_summary = base_eval["videocomp"]
        cpu_ckpt_path.write_text(
            json.dumps({
                "best_cap_hp": best_cap_hp,
                "cap_candidates": cap_candidates,
                "vc_candidates": vc_candidates,
                "baseline_captions": base_cap_summary,
                "baseline_videocomp": base_vc_summary,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[tune] Stage 1B + Stage 2 VideoComp + CPU ckpt saved at {time.perf_counter() - t_start:.2f}s", flush=True)

    vlm_ablation: dict[str, Any] = {"enabled": bool(with_vlm)}
    if with_vlm:
        print(f"[tune] Running Modal VLM ablations (max_cap={vlm_max_cap_queries}, max_vc={vlm_max_vc_pairs})...", flush=True)
        plain_cap = _score_vlm_captions_variant(
            tuned_cpu,
            work_root=work_root,
            split="tune",
            tag_suffix="plain_2x4",
            annotate_timestamps=False,
            template_kind="default",
            gamma=999.0,
            blend_alpha=0.5,
            max_queries=vlm_max_cap_queries,
            run_remote=True,
        )
        ts_cap = _score_vlm_captions_variant(
            tuned_cpu,
            work_root=work_root,
            split="tune",
            tag_suffix="ts_aligned",
            annotate_timestamps=True,
            template_kind="compositional",
            gamma=999.0,
            blend_alpha=0.5,
            max_queries=vlm_max_cap_queries,
            run_remote=True,
        )
        base_ts_hit = float(tuned_cpu["activitynet_captions"]["exact_semi_markov_dp_calibrated"]["full_chain_hit_at_1"])
        best_g, best_a, best_ts_hit = 1.25, 0.5, base_ts_hit
        min_vlm_gain = 0.01
        use_cap_vlm = False
        for gm in (0.5, 1.0, 1.5, 2.5, 999.0):
            for al in (0.25, 0.5, 1.0, 1.5):
                sw = _score_vlm_captions_variant(
                    tuned_cpu,
                    work_root=work_root,
                    split="tune",
                    tag_suffix="ts_aligned",
                    annotate_timestamps=True,
                    template_kind="compositional",
                    gamma=gm,
                    blend_alpha=al,
                    max_queries=vlm_max_cap_queries,
                    run_remote=False,
                    precomputed_vlm_map=ts_cap["vlm_map"],
                )
                h_now = float(sw["metrics"]["full_chain_hit_at_1"])
                if h_now > best_ts_hit and (h_now - base_ts_hit) >= min_vlm_gain:
                    best_ts_hit = h_now
                    best_g, best_a = gm, al
                    use_cap_vlm = True
        best_cap_hp["use_vlm_captions"] = bool(use_cap_vlm)
        best_cap_hp["vlm_margin_gamma"] = float(best_g)
        best_cap_hp["vlm_captions_alpha"] = float(best_a)

        vc_vlm_eval = _score_vlm_videocomp(
            tuned_cpu,
            work_root=work_root,
            split="tune",
            model_tag="vc_timestamped",
            vc_alpha=0.5,
            vc_margin_gamma=999.0,
            max_pairs=vlm_max_vc_pairs,
            annotate_timestamps=True,
            shard_size=120,
            run_remote=True,
        )
        base_vc_all = float(tuned_cpu["videocomp"]["three_axis_compositional"]["overall"]["all"])
        best_vc_g, best_vc_a, best_vc_vlm_all = 999.0, 0.0, base_vc_all
        use_vc_vlm = False
        for vg in (0.08, 0.15, 0.30, 999.0):
            for va in (0.25, 0.5, 1.0, 2.0):
                sw_vc = _score_vlm_videocomp(
                    tuned_cpu,
                    work_root=work_root,
                    split="tune",
                    model_tag="vc_timestamped",
                    vc_alpha=va,
                    vc_margin_gamma=vg,
                    max_pairs=vlm_max_vc_pairs,
                    annotate_timestamps=True,
                    shard_size=120,
                    run_remote=False,
                    precomputed_vlm_map=vc_vlm_eval["vlm_map"],
                    precomputed_pair_key_map=vc_vlm_eval["pair_key_map"],
                )
                all_now = float(sw_vc["metrics_fused"]["overall"]["all"])
                if all_now > best_vc_vlm_all and (all_now - base_vc_all) >= min_vlm_gain:
                    best_vc_vlm_all = all_now
                    best_vc_g, best_vc_a = vg, va
                    use_vc_vlm = True
        best_cap_hp["use_vlm_videocomp"] = bool(use_vc_vlm)
        best_cap_hp["vlm_vc_margin_gamma"] = float(best_vc_g)
        best_cap_hp["vlm_vc_alpha"] = float(best_vc_a)

        all_rcpts = plain_cap["receipts"] + ts_cap["receipts"] + vc_vlm_eval["receipts"]
        vlm_ablation.update({
            "plain_2x4_captions_hit_at_1": plain_cap["metrics"]["full_chain_hit_at_1"],
            "timestamped_aligned_captions_hit_at_1": ts_cap["metrics"]["full_chain_hit_at_1"],
            "tuned_captions_hit_at_1": best_ts_hit,
            "use_vlm_captions": bool(use_cap_vlm),
            "videocomp_vlm_fused_all": best_vc_vlm_all,
            "vlm_vc_margin_gamma": float(best_vc_g),
            "vlm_vc_alpha": float(best_vc_a),
            "use_vlm_videocomp": bool(use_vc_vlm),
            "min_vlm_promotion_gain": min_vlm_gain,
            "peak_vram_mib": _extract_vlm_vram_max_mib(all_rcpts, work_root=work_root),
        })
    else:
        best_cap_hp["use_vlm_captions"] = False
        best_cap_hp["use_vlm_videocomp"] = False

    report = {
        "selected_hparams": best_cap_hp,
        "selected_config_sha256": _canon_sha256(best_cap_hp),
        "baseline_captions": base_cap_summary,
        "tuned_captions": {m: {k: v for k, v in d.items() if k != "per_query_hits"} for m, d in tuned_cpu["activitynet_captions"].items()},
        "baseline_videocomp": base_vc_summary,
        "tuned_videocomp": tuned_cpu["videocomp"],
        "captions_sweep_top5": sorted(cap_candidates, key=lambda x: -(x["hit_at_1"] + 0.1 * x["recall_at_20"]))[:5],
        "videocomp_sweep_top5": sorted(vc_candidates, key=lambda x: -x["all"])[:5],
        "vlm_ablation": vlm_ablation,
    }
    (tune_dir / "tune_ablation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (tune_dir / "tune_selected_config.json").write_text(
        json.dumps({"hparams": best_cap_hp, "config_sha256": report["selected_config_sha256"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_config_sha256": report["selected_config_sha256"],
        "tuned_captions_exact_cal": report["tuned_captions"]["exact_semi_markov_dp_calibrated"],
        "tuned_videocomp_3axis": report["tuned_videocomp"]["three_axis_compositional"]["overall"],
        "vlm_ablation": vlm_ablation,
    }, indent=2, ensure_ascii=False))
    return report


def cmd_confirm_and_lock(work_root: Path, *, with_vlm: bool = False) -> dict[str, Any]:
    """Run one-shot Champion confirmation on `Valid-Confirm`, compute 95% CI, and enforce Go/No-Go gate."""
    sel = json.loads((work_root / "tune" / "tune_selected_config.json").read_text(encoding="utf-8"))
    hp = dict(sel["hparams"])
    qb = _build_or_load_tune_query_bank(work_root)

    ev = _evaluate_split_cpu_methods(work_root, split="valid_confirm", hparams=hp, query_bank=qb)
    receipts: list[dict[str, Any]] = []
    cap_champion_res = ev["activitynet_captions"]["exact_semi_markov_dp_calibrated"]
    cap_champion_name = "exact_semi_markov_dp_calibrated"
    if with_vlm and hp.get("use_vlm_captions"):
        cap_vlm = _score_vlm_captions_variant(
            ev,
            work_root=work_root,
            split="valid_confirm",
            tag_suffix="ts_aligned",
            annotate_timestamps=True,
            template_kind="compositional",
            gamma=float(hp.get("vlm_margin_gamma", 1.25)),
            blend_alpha=float(hp.get("vlm_captions_alpha", 0.5)),
            run_remote=True,
        )
        receipts.extend(cap_vlm["receipts"])
        cap_champion_res = cap_vlm["metrics"]
        cap_champion_name = "exact_semi_markov_dp_calibrated_plus_vlm"

    vc_champion_res = ev["videocomp"]["three_axis_compositional"]
    vc_champion_scores = ev["comp_scores"]["three_axis_compositional"]
    vc_champion_name = "three_axis_compositional"
    if with_vlm and hp.get("use_vlm_videocomp"):
        vc_vlm = _score_vlm_videocomp(
            ev,
            work_root=work_root,
            split="valid_confirm",
            model_tag="vc_timestamped",
            vc_alpha=float(hp.get("vlm_vc_alpha", 0.5)),
            vc_margin_gamma=float(hp.get("vlm_vc_margin_gamma", 999.0)),
            annotate_timestamps=True,
            run_remote=True,
        )
        receipts.extend(vc_vlm["receipts"])
        vc_champion_res = vc_vlm["metrics_fused"]
        vc_champion_scores = vc_vlm["fused_scores"]
        vc_champion_name = "three_axis_compositional_plus_vlm"

    # Compute paired source-group bootstrap 95% CI on Captions AND VideoComp vs global baselines
    cap_gids = [str(q["source_group_id"]) for q in ev["captions_queries"] if str(q["query_id"]) in ev["captions_preds"]["whole_paragraph_global"]]
    hits_champ = list(cap_champion_res["per_query_hits"])
    hits_base_vit = list(ev["activitynet_captions"]["viterbi_chain"]["per_query_hits"])
    hits_base_glob = list(ev["activitynet_captions"]["whole_paragraph_global"]["per_query_hits"])
    ci_cap_vs_glob = paired_source_group_bootstrap_ci(hits_champ, hits_base_glob, cap_gids)
    ci_cap_vs_vit = paired_source_group_bootstrap_ci(hits_champ, hits_base_vit, cap_gids)

    vc_pairs = ev["comp_pairs"]
    vc_gids = [str(r["source_group_id"]) for r in vc_pairs]
    vc_hits_champ = [
        1 if vc_champion_scores[str(r["pair_id"])]["positive_score"] > vc_champion_scores[str(r["pair_id"])]["negative_score"] else 0
        for r in vc_pairs
    ]
    vc_hits_glob = [
        1 if ev["comp_scores"]["global_paragraph_3backbone"][str(r["pair_id"])]["positive_score"] > ev["comp_scores"]["global_paragraph_3backbone"][str(r["pair_id"])]["negative_score"] else 0
        for r in vc_pairs
    ]
    ci_vc_vs_glob = paired_source_group_bootstrap_ci(vc_hits_champ, vc_hits_glob, vc_gids)

    peak_vram = _extract_vlm_vram_max_mib(receipts, work_root=work_root)
    vram_ok = (peak_vram == 0.0) or (peak_vram <= 6144.0)
    gate_passed = bool(
        (ci_cap_vs_glob["significant_gt_zero"] or ci_cap_vs_vit["significant_gt_zero"])
        and ci_vc_vs_glob["significant_gt_zero"]
        and vram_ok
    )

    lock_receipt = {
        "split": "valid_confirm",
        "champion_captions_method": cap_champion_name,
        "champion_videocomp_method": vc_champion_name,
        "config_sha256": sel["config_sha256"],
        "hparams": hp,
        "captions_metrics": {
            "whole_paragraph_global": {k: v for k, v in ev["activitynet_captions"]["whole_paragraph_global"].items() if k != "per_query_hits"},
            "independent_clause_contrast": {k: v for k, v in ev["activitynet_captions"]["independent_clause_contrast"].items() if k != "per_query_hits"},
            "viterbi_chain": {k: v for k, v in ev["activitynet_captions"]["viterbi_chain"].items() if k != "per_query_hits"},
            "exact_semi_markov_dp": {k: v for k, v in ev["activitynet_captions"]["exact_semi_markov_dp"].items() if k != "per_query_hits"},
            cap_champion_name: {k: v for k, v in cap_champion_res.items() if k != "per_query_hits"},
        },
        "videocomp_metrics": {
            "global_paragraph_3backbone": ev["videocomp"]["global_paragraph_3backbone"],
            "viterbi_forward_dp": ev["videocomp"]["viterbi_forward_dp"],
            "three_axis_compositional": ev["videocomp"]["three_axis_compositional"],
            **({vc_champion_name: vc_champion_res} if vc_champion_name != "three_axis_compositional" else {}),
        },
        "bootstrap_ci95": {
            "captions_vs_global": ci_cap_vs_glob,
            "captions_vs_viterbi": ci_cap_vs_vit,
            "videocomp_vs_global": ci_vc_vs_glob,
        },
        "vlm_peak_vram_mib": peak_vram,
        "vram_within_6gib": vram_ok,
        "go_no_go_gate_passed": gate_passed,
    }
    (work_root / "champion_lock_receipt.json").write_text(json.dumps(lock_receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "go_no_go_gate_passed": gate_passed,
        "vlm_peak_vram_mib": peak_vram,
        "captions_champion": lock_receipt["captions_metrics"][cap_champion_name],
        "videocomp_champion_overall": vc_champion_res["overall"],
        "bootstrap_ci95": lock_receipt["bootstrap_ci95"],
    }, indent=2, ensure_ascii=False))
    return lock_receipt


def cmd_eval_sealed_test(work_root: Path, *, with_vlm: bool = False) -> dict[str, Any]:
    """Evaluate locked Champion ONCE on sealed `Test`, save frozen predictions first, and write benchmark receipt."""
    lock_path = work_root / "champion_lock_receipt.json"
    if not lock_path.is_file():
        raise RuntimeError("Sealed Test refused: champion_lock_receipt.json does not exist. Run confirm-and-lock first.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("go_no_go_gate_passed"):
        raise RuntimeError("Sealed Test refused: Champion failed Valid-Confirm Go/No-Go gate.")

    hp = dict(lock["hparams"])
    qb = _build_or_load_tune_query_bank(work_root)
    ev = _evaluate_split_cpu_methods(work_root, split="test", hparams=hp, query_bank=qb)
    receipts: list[dict[str, Any]] = []

    cap_champ_name = str(lock["champion_captions_method"])
    cap_champ_res = ev["activitynet_captions"]["exact_semi_markov_dp_calibrated"]
    cap_champ_preds = ev["captions_preds"]["exact_semi_markov_dp_calibrated"]
    if with_vlm and hp.get("use_vlm_captions"):
        cap_vlm = _score_vlm_captions_variant(
            ev,
            work_root=work_root,
            split="test",
            tag_suffix="ts_aligned",
            annotate_timestamps=True,
            template_kind="compositional",
            gamma=float(hp.get("vlm_margin_gamma", 1.25)),
            blend_alpha=float(hp.get("vlm_captions_alpha", 0.5)),
            run_remote=True,
        )
        receipts.extend(cap_vlm["receipts"])
        cap_champ_res = cap_vlm["metrics"]
        cap_champ_preds = cap_vlm["predictions"]

    vc_champ_name = str(lock["champion_videocomp_method"])
    vc_champ_res = ev["videocomp"]["three_axis_compositional"]
    vc_champ_scores = ev["comp_scores"]["three_axis_compositional"]
    if with_vlm and hp.get("use_vlm_videocomp"):
        vc_vlm = _score_vlm_videocomp(
            ev,
            work_root=work_root,
            split="test",
            model_tag="vc_timestamped",
            vc_alpha=float(hp.get("vlm_vc_alpha", 0.5)),
            vc_margin_gamma=float(hp.get("vlm_vc_margin_gamma", 999.0)),
            annotate_timestamps=True,
            run_remote=True,
        )
        receipts.extend(vc_vlm["receipts"])
        vc_champ_res = vc_vlm["metrics_fused"]
        vc_champ_scores = vc_vlm["fused_scores"]

    # Persist frozen Test predictions BEFORE final bootstrap / benchmark write
    test_dir = work_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    frozen_preds = {
        "split": "test",
        "config_sha256": lock["config_sha256"],
        "captions_predictions_by_method": {
            "whole_paragraph_global": ev["captions_preds"]["whole_paragraph_global"],
            "independent_clause_contrast": ev["captions_preds"]["independent_clause_contrast"],
            "viterbi_chain": ev["captions_preds"]["viterbi_chain"],
            "exact_semi_markov_dp": ev["captions_preds"]["exact_semi_markov_dp"],
            cap_champ_name: cap_champ_preds,
        },
        "videocomp_scores_by_method": {
            "global_paragraph_3backbone": ev["comp_scores"]["global_paragraph_3backbone"],
            "viterbi_forward_dp": ev["comp_scores"]["viterbi_forward_dp"],
            vc_champ_name: vc_champ_scores,
        },
    }
    (test_dir / "test_predictions.json").write_text(json.dumps(frozen_preds, indent=2, ensure_ascii=False), encoding="utf-8")

    # Paired bootstrap CIs on sealed Test
    cap_gids = [str(q["source_group_id"]) for q in ev["captions_queries"] if str(q["query_id"]) in ev["captions_preds"]["whole_paragraph_global"]]
    ci_cap_vs_glob = paired_source_group_bootstrap_ci(
        cap_champ_res["per_query_hits"],
        ev["activitynet_captions"]["whole_paragraph_global"]["per_query_hits"],
        cap_gids,
    )
    ci_cap_vs_vit = paired_source_group_bootstrap_ci(
        cap_champ_res["per_query_hits"],
        ev["activitynet_captions"]["viterbi_chain"]["per_query_hits"],
        cap_gids,
    )
    vc_pairs = ev["comp_pairs"]
    vc_gids = [str(r["source_group_id"]) for r in vc_pairs]
    vc_hits_champ = [1 if vc_champ_scores[str(r["pair_id"])]["positive_score"] > vc_champ_scores[str(r["pair_id"])]["negative_score"] else 0 for r in vc_pairs]
    vc_hits_glob = [1 if ev["comp_scores"]["global_paragraph_3backbone"][str(r["pair_id"])]["positive_score"] > ev["comp_scores"]["global_paragraph_3backbone"][str(r["pair_id"])]["negative_score"] else 0 for r in vc_pairs]
    ci_vc_vs_glob = paired_source_group_bootstrap_ci(vc_hits_champ, vc_hits_glob, vc_gids)

    peak_vram = max(float(lock.get("vlm_peak_vram_mib", 0.0)), _extract_vlm_vram_max_mib(receipts, work_root=work_root))
    final_receipt = {
        "benchmark_suite": "ActivityNet Captions + VideoComp (Test Sealed 20%)",
        "config_sha256": lock["config_sha256"],
        "champion_captions_method": cap_champ_name,
        "champion_videocomp_method": vc_champ_name,
        "vlm_peak_vram_mib": peak_vram,
        "within_6gib_budget": bool(peak_vram == 0.0 or peak_vram <= 6144.0),
        "activitynet_captions": {
            **{m: {k: v for k, v in d.items() if k != "per_query_hits"} for m, d in ev["activitynet_captions"].items()},
            **({cap_champ_name: {k: v for k, v in cap_champ_res.items() if k != "per_query_hits"}} if cap_champ_name != "exact_semi_markov_dp_calibrated" else {}),
        },
        "videocomp": {
            **ev["videocomp"],
            **({vc_champ_name: vc_champ_res} if vc_champ_name not in ev["videocomp"] else {}),
        },
        "bootstrap_ci95": {
            "captions_vs_global": ci_cap_vs_glob,
            "captions_vs_viterbi": ci_cap_vs_vit,
            "videocomp_vs_global": ci_vc_vs_glob,
        },
    }

    bench_dir = REPO_ROOT / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    json_out = bench_dir / "activitynet_videocomp_benchmark_results.json"
    md_out = bench_dir / "activitynet_videocomp_benchmark_results.md"
    json_out.write_text(json.dumps(final_receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write concise human-readable Markdown report following AGENTS.md rules
    lines = [
        "# Kết quả Benchmark ActivityNet Captions & VideoComp (Sealed Test)",
        "",
        f"- Config SHA-256: `{final_receipt['config_sha256']}`",
        f"- VLM Peak VRAM (Modal Qwen2.5-VL-3B NF4 batch=1): `{peak_vram} MiB` (`<= 6,144 MiB`: `{final_receipt['within_6gib_budget']}`)",
        "",
        "## 1. ActivityNet Captions (`>= 2` Events, Full-Chain Retrieval)",
        "",
        "| Phương pháp | Full-Chain Hit@1 (tIoU>=0.5) | Full-Chain Hit@5 | Given-Video Hit@1 | Video Recall@20 | Per-Event tIoU>=0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pretty_cap = {
        "whole_paragraph_global": "Whole-Paragraph Global (PE-bigG + PE-L + SigLIP2-400M)",
        "independent_clause_contrast": "Independent Clause Contrast (No Order DP)",
        "viterbi_chain": "Viterbi Forward DP (Unified Span Baseline)",
        "exact_semi_markov_dp": "Exact 2D Semi-Markov Segment DP + Order Margin",
        "exact_semi_markov_dp_calibrated": "Exact 2D Semi-Markov DP + Tuned Clause Alignment & Length Penalty (PE-bigG + PE-L + SigLIP2-400M)",
        "exact_semi_markov_dp_calibrated_plus_vlm": "VIDENTRA Semi-Markov DP + Qwen2.5-VL-3B NF4 (Borderline Rerank)",
    }
    for m_k, d in final_receipt["activitynet_captions"].items():
        lines.append(
            f"| {pretty_cap.get(m_k, m_k)} | {d['full_chain_hit_at_1']*100:.2f}% | "
            f"{d['full_chain_hit_at_5']*100:.2f}% | {d['given_video_full_chain_hit_at_1']*100:.2f}% | "
            f"{d['video_recall_at_20']*100:.2f}% | {d['per_event_tiou_05_rate']*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 2. VideoComp (`ActivityNet-Comp` + `YouCook2-Comp`, Binary Compositional Accuracy)",
        "",
        "| Phương pháp | Miền dữ liệu | Temporal Reordering | Action Replacement | Segment Mismatch | `All` (Tích 3 loại) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    pretty_vc = {
        "global_paragraph_3backbone": "Global Mean Cosine (PE-bigG + PE-L + SigLIP2-400M)",
        "viterbi_forward_dp": "Viterbi Forward DP",
        "three_axis_compositional": "3-Axis Compositional Scorer (Semi-Markov Order + CoM + Min-Clause + Coverage)",
        "three_axis_compositional_plus_vlm": "3-Axis Compositional + Qwen2.5-VL-3B NF4 (Timestamped Compositional)",
    }
    for m_k, d in final_receipt["videocomp"].items():
        for dom_k in sorted(d.get("by_domain", {}).keys()):
            sub = d["by_domain"][dom_k]
            lines.append(
                f"| {pretty_vc.get(m_k, m_k)} | {dom_k} | "
                f"{sub.get('temporal_reordering', {}).get('accuracy', 0.0)*100:.2f}% | "
                f"{sub.get('action_replacement', {}).get('accuracy', 0.0)*100:.2f}% | "
                f"{sub.get('segment_mismatch', {}).get('accuracy', 0.0)*100:.2f}% | "
                f"{sub.get('all', 0.0)*100:.2f}% |"
            )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_receipt, indent=2, ensure_ascii=False))
    return final_receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrate ActivityNet & VideoComp evaluation.")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch-annotations")
    p_sync = sub.add_parser("sync-media")
    p_sync.add_argument("--force", action="store_true")
    p_sync.add_argument("--youcook2-only", action="store_true")

    p_lock = sub.add_parser("lock-splits")
    p_lock.add_argument("--max-anet-groups", type=int, default=1000)
    p_lock.add_argument("--max-yc2-groups", type=int, default=500)
    p_lock.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    p_lock.add_argument("--no-intersect-anet-comp", action="store_true")

    p_pilot = sub.add_parser("run-pilot")
    p_pilot.add_argument("--l4-usd-hr", type=float, default=0.80)
    p_pilot.add_argument("--l40s-usd-hr", type=float, default=1.95)

    p_ext = sub.add_parser("extract-split")
    p_ext.add_argument("--split", required=True, choices=("tune", "valid_confirm", "test"))
    p_ext.add_argument("--gpu", default="L40S")
    p_ext.add_argument("--shard-size", type=int, default=150)
    p_ext.add_argument("--num-frames", type=int, default=48)

    p_eval = sub.add_parser("eval-cpu-split")
    p_eval.add_argument("--split", required=True, choices=("tune", "valid_confirm", "test"))

    p_tune = sub.add_parser("tune-and-ablate")
    p_tune.add_argument("--with-vlm", action="store_true")
    p_tune.add_argument("--vlm-max-cap-queries", type=int, default=80)
    p_tune.add_argument("--vlm-max-vc-pairs", type=int, default=120)
    p_tune.add_argument("--resume-vlm", action="store_true")

    p_conf = sub.add_parser("confirm-and-lock")
    p_conf.add_argument("--with-vlm", action="store_true")

    p_test = sub.add_parser("eval-sealed-test")
    p_test.add_argument("--with-vlm", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "fetch-annotations":
        cmd_fetch_annotations(args.work_root)
    elif args.cmd == "sync-media":
        cmd_sync_media(args.work_root, force=args.force, youcook2_only=args.youcook2_only)
    elif args.cmd == "lock-splits":
        cmd_lock_splits(
            args.work_root,
            max_anet_groups=args.max_anet_groups,
            max_yc2_groups=args.max_yc2_groups,
            seed=args.seed,
            intersect_anet_comp=not args.no_intersect_anet_comp,
        )
    elif args.cmd == "run-pilot":
        cmd_run_pilot(args.work_root, l4_usd_hr=args.l4_usd_hr, l40s_usd_hr=args.l40s_usd_hr)
    elif args.cmd == "extract-split":
        cmd_extract_split(args.work_root, split=args.split, gpu_tier=args.gpu, shard_size=args.shard_size, num_frames=args.num_frames)
    elif args.cmd == "eval-cpu-split":
        qb = _build_or_load_tune_query_bank(args.work_root) if (args.work_root / "features" / "tune").exists() else None
        res = _evaluate_split_cpu_methods(args.work_root, split=args.split, query_bank=qb)
        out_dir = args.work_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        compact = {
            "split": args.split,
            "activitynet_captions": {
                m: {k: v for k, v in d.items() if k != "per_query_hits"}
                for m, d in res["activitynet_captions"].items()
            },
            "videocomp": res["videocomp"],
        }
        (out_dir / f"{args.split}_cpu_eval.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
        print(json.dumps(compact, indent=2))
    elif args.cmd == "tune-and-ablate":
        cmd_tune_and_ablate(
            args.work_root,
            with_vlm=args.with_vlm,
            vlm_max_cap_queries=args.vlm_max_cap_queries,
            vlm_max_vc_pairs=args.vlm_max_vc_pairs,
            resume_vlm=args.resume_vlm,
        )
    elif args.cmd == "confirm-and-lock":
        cmd_confirm_and_lock(args.work_root, with_vlm=args.with_vlm)
    elif args.cmd == "eval-sealed-test":
        cmd_eval_sealed_test(args.work_root, with_vlm=args.with_vlm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
