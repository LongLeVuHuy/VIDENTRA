"""Upgrade orchestrator: Joint CrossEventSpanNet + LightGBM + Pairwise Soft-Overlap DP & 128-frame Coarse-to-Fine.

Zero-leakage rules:
- `train_clean` (2,400 ActivityNet train videos) is strictly disjoint from `Tune`, `Valid-Confirm`, `Test-200`, and `Fresh-Test-200`.
- `extract-dense-top3` selects candidate videos strictly via blinded Stage-1 retrieval scores (never Ground Truth).
- `tune` sweeps ensemble + DP parameters; `valid_confirm` confirms once and locks Champion v2 via 95% bootstrap CI.
- `eval-tests` runs once after gate pass on both `test` (200 videos, comparison vs Phase B) and unseen sealed `fresh_test` (200 videos).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modal_videntra_worker import normalize_media_key  # noqa: E402
from scripts.run_activitynet_videocomp_pipeline import (  # noqa: E402
    DEFAULT_WORK_ROOT,
    _build_captions_stage1_ctx,
    _build_or_load_tune_query_bank,
    _load_split_feature_store,
    cmd_extract_split,
    pull_modal_file,
)
from videntra.evaluation.multievent_evaluator import (  # noqa: E402
    evaluate_activitynet_full_chain,
    paired_source_group_bootstrap_ci,
)
from videntra.localization.span_grounder import (  # noqa: E402
    CrossEventSpanNet,
    JointMultiEventGrounder,
    SpanGrounderLGBM,
    extract_multievent_frame_features,
    extract_multievent_span_features,
)
from videntra.retrieval.alignment import (  # noqa: E402
    _voronoi_frame_edges,
    localize_soft_overlap_segment_dp,
)

DEFAULT_UPGRADE_SEED = 20261125
DEFAULT_WEIGHTS: dict[str, float] = {"pe_bigg": 1.40, "pe_l": 1.10, "sig400": 0.75}


def _load_locked_stage1_hparams(work_root: Path) -> dict[str, Any]:
    lock_p = work_root / "champion_lock_receipt.json"
    if lock_p.is_file():
        return dict(json.loads(lock_p.read_text(encoding="utf-8"))["hparams"])
    sel_p = work_root / "tune" / "tune_selected_config.json"
    return dict(json.loads(sel_p.read_text(encoding="utf-8"))["hparams"])


def _compute_near_uniform_rate(
    queries: Sequence[Mapping[str, Any]], preds_by_qid: Mapping[str, Any]
) -> float:
    near_u = 0
    tot_sp = 0
    for q in queries:
        qid = str(q["query_id"])
        dur = float(q["duration_sec"])
        k = max(1, len(q["events"]))
        unif = dur / float(k)
        pr = preds_by_qid.get(qid) or {}
        tkr = pr.get("top_k_results") or []
        if not tkr:
            continue
        for sp in tkr[0].get("event_spans") or []:
            tot_sp += 1
            d_sp = float(sp["end_sec"]) - float(sp["start_sec"])
            if abs(d_sp - unif) / max(1e-6, unif) <= 0.25:
                near_u += 1
    return round(near_u / max(1, tot_sp), 6)


def _eval_uniform_given_video_baseline(queries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[int]]:
    unif_preds: dict[str, Any] = {}
    for q in queries:
        qid = str(q["query_id"])
        vid = str(q["video_id"])
        dur = float(q["duration_sec"])
        evs = list(q["events"])
        k = max(1, len(evs))
        spans = [
            {
                "clause_id": str(ev["clause_id"]),
                "start_sec": dur * float(i) / float(k),
                "end_sec": dur * float(i + 1) / float(k),
                "status": "found",
            }
            for i, ev in enumerate(evs)
        ]
        unif_preds[qid] = {"ranked_video_ids": [vid], "top_k_results": [{"video_id": vid, "event_spans": spans}]}
    res = evaluate_activitynet_full_chain(queries, unif_preds)
    return {k: v for k, v in res.items() if k != "per_query_hits"}, list(res["per_query_hits"])


def cmd_build_clean_manifests(
    work_root: Path,
    *,
    num_train_videos: int = 2400,
    num_fresh_test_videos: int = 200,
    seed: int = DEFAULT_UPGRADE_SEED,
) -> dict[str, Any]:
    """Create strictly disjoint `train_clean` (from ANet train) and `fresh_test` (from ANet val_1)."""
    ann_dir = work_root / "annotations" / "anet_captions_hf" / "raw_data"
    tr_dict = json.loads((ann_dir / "train.json").read_text(encoding="utf-8"))
    v1_dict = json.loads((ann_dir / "val_1.json").read_text(encoding="utf-8"))
    cat = json.loads((work_root / "media_catalog.json").read_text(encoding="utf-8"))

    valid_anet: dict[str, dict[str, Any]] = {}
    for e in cat.get("items", []):
        if e.get("domain") == "activitynet" and e.get("ok") and float(e.get("duration_sec", 0.0)) > 1.0:
            vid = normalize_media_key(str(e["video_id"]))
            valid_anet[vid] = {
                "rel_path": str(e["rel_path"]),
                "duration_sec": float(e["duration_sec"]),
                "file_sha256": str(e.get("file_sha256") or ""),
            }

    excl_vids: set[str] = set()
    excl_shas: set[str] = set()
    for sp_name in ("tune", "valid_confirm", "test"):
        m = json.loads((work_root / "splits" / f"{sp_name}_manifest.json").read_text(encoding="utf-8"))
        excl_vids.update(str(g) for g in m["activitynet"]["source_groups"])
        for rec in m["activitynet"]["records"]:
            v = str(rec["video_id"])
            excl_vids.add(v)
            if v in valid_anet and valid_anet[v]["file_sha256"]:
                excl_shas.add(valid_anet[v]["file_sha256"])

    def _collect_eligible(raw_dict: Mapping[str, Any], ann_tag: str, pool_name: str) -> list[dict[str, Any]]:
        cands: list[dict[str, Any]] = []
        seen_shas_local: set[str] = set()
        for raw_vid, entry in raw_dict.items():
            vid = normalize_media_key(raw_vid)
            if vid in excl_vids or vid not in valid_anet:
                continue
            sha = valid_anet[vid]["file_sha256"]
            if sha and (sha in excl_shas or sha in seen_shas_local):
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
            if sha:
                seen_shas_local.add(sha)
            h_key = hashlib.sha256(f"{seed}:{ann_tag}:{vid}".encode("utf-8")).hexdigest()
            cands.append({
                "sort_key": h_key,
                "record": {
                    "query_id": f"anet_cap_{ann_tag}_{vid}",
                    "video_id": vid,
                    "source_group_id": vid,
                    "domain": "activitynet",
                    "domain_orig": "activitynet",
                    "public_pool": pool_name,
                    "task_kind": "captions",
                    "annotation_source": ann_tag,
                    "query_text": " ".join(ev["text"] for ev in events),
                    "events": events,
                    "media_rel_path": valid_anet[vid]["rel_path"],
                    "duration_sec": valid_anet[vid]["duration_sec"],
                },
            })
        cands.sort(key=lambda x: x["sort_key"])
        return [c["record"] for c in cands]

    tr_all = _collect_eligible(tr_dict, "train", "train")
    tr_sel = tr_all[: int(num_train_videos)]
    for rec in tr_sel:
        excl_vids.add(str(rec["video_id"]))
        sha = valid_anet[str(rec["video_id"])]["file_sha256"]
        if sha:
            excl_shas.add(sha)

    v1_all = _collect_eligible(v1_dict, "val_1", "validation")
    ft_sel = v1_all[: int(num_fresh_test_videos)]

    def _save_manifest(sp_name: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
        vids = sorted(str(r["video_id"]) for r in recs)
        manifest = {
            "split": sp_name,
            "seed": int(seed),
            "activitynet": {
                "source_groups": vids,
                "gallery_video_ids": vids,
                "records": recs,
            },
            "youcook2": {
                "source_groups": [],
                "gallery_video_ids": [],
                "records": [],
            },
        }
        out_p = work_root / "splits" / f"{sp_name}_manifest.json"
        out_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"split": sp_name, "videos": len(vids), "queries": len(recs), "manifest_path": str(out_p)}

    res_tr = _save_manifest("train_clean", tr_sel)
    res_ft = _save_manifest("fresh_test", ft_sel)

    # Zero-overlap invariant check across all 5 splits
    vid_sets: dict[str, set[str]] = {}
    for sp_n in ("train_clean", "tune", "valid_confirm", "test", "fresh_test"):
        mm = json.loads((work_root / "splits" / f"{sp_n}_manifest.json").read_text(encoding="utf-8"))
        vid_sets[sp_n] = {str(x) for x in mm["activitynet"]["source_groups"]}
    names = list(vid_sets.keys())
    overlaps = {
        f"{names[i]}_x_{names[j]}": len(vid_sets[names[i]] & vid_sets[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    if any(v > 0 for v in overlaps.values()):
        raise AssertionError(f"Split leakage detected: {overlaps}")

    out = {"train_clean": res_tr, "fresh_test": res_ft, "pairwise_overlaps": overlaps}
    print(json.dumps(out, indent=2))
    return out


def _compute_stage1_calibrated_rankings(
    manifest: Mapping[str, Any],
    store: Mapping[str, Any],
    query_bank: Mapping[str, np.ndarray] | None,
    hparams: Mapping[str, Any],
    *,
    top_k_cand: int = 20,
) -> dict[str, Any]:
    """Compute strictly blinded Stage-1 calibrated scores & top-K rankings (NO GT accessed)."""
    w = dict(hparams.get("weights") or DEFAULT_WEIGHTS)
    w_sum = sum(w.values())
    dense_topk = max(1, int(hparams.get("dense_topk", 6)))
    qb_alpha = float(hparams.get("qb_alpha", 0.0))
    dense_pool_alpha = float(hparams.get("dense_pool_alpha", 0.0))
    clause_ret_alpha = float(hparams.get("clause_ret_alpha", 0.45))

    ctx = _build_captions_stage1_ctx(manifest, store, query_bank, w, dense_topk)
    anet_gallery = list(ctx["anet_gallery"])
    hub_mean = ctx["hub_mean"]
    cap_queries = list(ctx["cap_queries"])
    cap_text_seen = ctx["cap_text_seen"]
    g_sim_by_b = ctx["g_sim_by_b"]
    dense_topk_z_by_b = ctx["dense_topk_z_by_b"]
    clause_topk_z = ctx["clause_topk_z"]
    tmap = store["text_str_to_idx"]

    ranked_by_qid: dict[str, list[tuple[str, float]]] = {}
    for q in cap_queries:
        qid = str(q["query_id"])
        qt = str(q["query_text"]).strip()
        if qt not in tmap or qt not in cap_text_seen:
            continue
        qt_loc = cap_text_seen[qt]
        cal_scores = np.zeros(len(anet_gallery), dtype=np.float32)
        for b in ("pe_bigg", "pe_l", "sig400"):
            s_b = g_sim_by_b[b][:, qt_loc]
            z_glob = (s_b - float(np.mean(s_b))) / max(1e-6, float(np.std(s_b)))
            s_qb = s_b - qb_alpha * hub_mean[b]
            z_qb = (s_qb - float(np.mean(s_qb))) / max(1e-6, float(np.std(s_qb)))
            if dense_pool_alpha > 0.0:
                z_mix = (1.0 - dense_pool_alpha) * z_qb + dense_pool_alpha * dense_topk_z_by_b[b][:, qt_loc]
            else:
                z_mix = z_qb
            cal_scores += w[b] * z_mix

        ev_z_list = []
        for ev in q["events"]:
            et = str(ev["text"]).strip()
            if et in tmap and et in cap_text_seen:
                ev_z_list.append(clause_topk_z[:, cap_text_seen[et]])
        if ev_z_list and clause_ret_alpha != 0.0:
            cal_scores = cal_scores + clause_ret_alpha * np.mean(np.stack(ev_z_list, axis=0), axis=0)

        ord_idx = np.argsort(-cal_scores, kind="mergesort")[:top_k_cand]
        ranked_by_qid[qid] = [(anet_gallery[int(i)], float(cal_scores[int(i)])) for i in ord_idx]

    return {"ctx": ctx, "ranked_by_qid": ranked_by_qid}


def cmd_extract_dense_top3(
    work_root: Path,
    *,
    split: str,
    top_k: int = 3,
    num_frames: int = 128,
    shard_size: int = 60,
) -> dict[str, Any]:
    """Extract 128-frame dense features on Modal L40S ONLY for blinded Stage-1 Top-3 candidates."""
    from scripts.modal_app_functions import app as modal_app, run_l40s_extract_shard
    import modal  # type: ignore

    hp = _load_locked_stage1_hparams(work_root)
    qb = _build_or_load_tune_query_bank(work_root)
    manifest = json.loads((work_root / "splits" / f"{split}_manifest.json").read_text(encoding="utf-8"))
    store = _load_split_feature_store(work_root, split)
    s1 = _compute_stage1_calibrated_rankings(manifest, store, qb, hp, top_k_cand=max(top_k, 5))

    top_vids = sorted({vid for cands in s1["ranked_by_qid"].values() for vid, _ in cands[:top_k]})
    rel_map = s1["ctx"]["rel_by_vid"]
    video_items = [
        {"video_id": vid, "item_id": f"full::{vid}", "rel_path": rel_map[vid], "clip_window": None}
        for vid in top_vids
        if vid in rel_map
    ]

    dense_name = f"{split}_dense{num_frames}"
    feat_dir = work_root / "features" / dense_name
    feat_dir.mkdir(parents=True, exist_ok=True)
    shards = [(i // shard_size, video_items[i : i + shard_size]) for i in range(0, len(video_items), shard_size)]

    receipts: list[dict[str, Any]] = []
    todo_args: list[tuple[str, str, str, str, int]] = []
    todo_sids: list[str] = []
    for s_idx, v_chunk in shards:
        sid = f"{dense_name}_s{s_idx:03d}"
        local_npz = feat_dir / f"{sid}.npz"
        local_rc = feat_dir / f"{sid}_receipt.json"
        if local_npz.is_file() and local_rc.is_file():
            rc_saved = json.loads(local_rc.read_text(encoding="utf-8"))
            if int(rc_saved.get("total_decoded_frames", 0)) > 0:
                receipts.append(rc_saved)
                continue
        todo_args.append((f"extract_{dense_name}", sid, json.dumps(v_chunk), "[]", int(num_frames)))
        todo_sids.append(sid)

    if todo_args:
        with modal.enable_output(), modal_app.run():
            results = list(run_l40s_extract_shard.starmap(todo_args))
        for sid, res in zip(todo_sids, results):
            rc = res["receipt"]
            receipts.append(rc)
            (feat_dir / f"{sid}_receipt.json").write_text(json.dumps(rc, indent=2, ensure_ascii=False), encoding="utf-8")
            pull_modal_file(f"extract_{dense_name}/features_L40S/{sid}/shard_features.npz", feat_dir / f"{sid}.npz")

    summary = {
        "split": split,
        "dense_split": dense_name,
        "num_frames": int(num_frames),
        "unique_top_k_videos": len(video_items),
        "num_shards": len(shards),
        "total_decoded_frames": sum(int(rc.get("total_decoded_frames", 0)) for rc in receipts),
        "decode_error_count": sum(len(rc.get("decode_errors", [])) for rc in receipts),
        "receipts": receipts,
    }
    (feat_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}, indent=2))
    return summary


def _load_dense128_frame_data(work_root: Path, split: str, num_frames: int = 128) -> dict[str, dict[str, Any]]:
    feat_dir = work_root / "features" / f"{split}_dense{num_frames}"
    if not feat_dir.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for npz_p in sorted(feat_dir.glob(f"{split}_dense{num_frames}_s*.npz")):
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
            c = int(cnts[i])
            if c > 0:
                out[str(raw_id)] = {
                    "duration": float(durs[i]),
                    "ordinals": ords[i, :c].astype(np.int32),
                    "pts_sec": ptss[i, :c].astype(np.float32),
                    "pe_bigg": pb[off : off + c].astype(np.float32),
                    "pe_l": pl[off : off + c].astype(np.float32),
                    "sig400": ps[off : off + c].astype(np.float32),
                }
            off += c
        d.close()
    return out


def _build_split_training_tensors(
    manifest: Mapping[str, Any],
    store: Mapping[str, Any],
    *,
    build_lgb_spans: bool = False,
    rng_seed: int = DEFAULT_UPGRADE_SEED,
    max_pos_per_ev: int = 60,
    max_neg_per_ev: int = 120,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Build compact `(seq_x, y_iou, y_frame, ...)` tensors for GPU training and optional sampled LGBM span arrays."""
    fdata, tmap, tvecs = store["frame_data"], store["text_str_to_idx"], store["text_vecs"]
    ta48, tb48 = np.triu_indices(48)
    rng = np.random.default_rng(rng_seed)

    seq_list, iou_list, fr_list = [], [], []
    offsets = [0]
    pts_list, durs_list, gts_list, gte_list, gta_list, gtb_list = [], [], [], [], [], []
    lgb_x_chunks, lgb_y_chunks = [], []

    for r in manifest["activitynet"]["records"]:
        if r.get("task_kind") != "captions":
            continue
        if r.get("annotation_source") == "val_2":
            continue
        vid = str(r["video_id"])
        fkey = f"full::{vid}"
        if fkey not in fdata:
            continue
        fd = fdata[fkey]
        ts = fd["pts_sec"].astype(np.float64)
        if len(ts) != 48:
            continue
        dur = max(float(r["duration_sec"]), float(fd.get("duration", 0.0)))
        le, re = _voronoi_frame_edges(ts.tolist(), dur)
        le_a, re_a = np.asarray(le, dtype=np.float64), np.asarray(re, dtype=np.float64)
        evs = list(r["events"])
        k_ev = len(evs)
        if not all(str(ev["text"]).strip() in tmap for ev in evs):
            continue

        curves = {}
        embs = {}
        for b in ("pe_bigg", "pe_l", "sig400"):
            mat = np.zeros((k_ev, 48), dtype=np.float32)
            for i, ev in enumerate(evs):
                mat[i] = fd[b] @ tvecs[b][tmap[str(ev["text"]).strip()]]
            curves[b] = mat
            embs[b] = fd[b]

        fctx = extract_multievent_frame_features(ts, dur, curves, frame_embeddings_by_backbone=embs, weights=DEFAULT_WEIGHTS)
        seq_list.append(fctx["seq_x"].astype(np.float16))

        y_iou = np.zeros((k_ev, len(ta48)), dtype=np.float32)
        y_fr = np.zeros((k_ev, 48, 3), dtype=np.float32)
        for i, ev in enumerate(evs):
            gs, ge = float(ev["start_sec"]), float(ev["end_sec"])
            inter = np.maximum(0.0, np.minimum(re_a[tb48], ge) - np.maximum(le_a[ta48], gs))
            union = np.maximum(re_a[tb48], ge) - np.minimum(le_a[ta48], gs)
            tiou_vec = np.where(union > 0.0, inter / union, 0.0).astype(np.float32)
            y_iou[i] = tiou_vec

            inside = ((re_a > gs) & (le_a < ge)).astype(np.float32)
            step_t = dur / 48.0
            sig_g = max(step_t * 1.2, 2.0)
            y_fr[i, :, 0] = inside
            y_fr[i, :, 1] = np.exp(-0.5 * ((ts - gs) / sig_g) ** 2).astype(np.float32)
            y_fr[i, :, 2] = np.exp(-0.5 * ((ts - ge) / sig_g) ** 2).astype(np.float32)

            a_gt = int(np.argmin(np.abs(ts - gs)))
            b_gt = int(np.argmin(np.abs(ts - ge)))
            gts_list.append(gs)
            gte_list.append(ge)
            gta_list.append(a_gt)
            gtb_list.append(max(a_gt, b_gt))

        iou_list.append(y_iou.astype(np.float16))
        fr_list.append(y_fr.astype(np.float16))
        pts_list.append(ts.astype(np.float32))
        durs_list.append(float(dur))
        offsets.append(offsets[-1] + k_ev)

        if build_lgb_spans:
            x_sp, _, _, _, _ = extract_multievent_span_features(ts, dur, fctx)
            for i in range(k_ev):
                yy = y_iou[i]
                pos = np.where(yy >= 0.5)[0]
                hard_n = np.where((yy >= 0.20) & (yy < 0.5))[0]
                easy_n = np.where(yy < 0.20)[0]
                sel_p = pos if len(pos) <= max_pos_per_ev else rng.choice(pos, max_pos_per_ev, replace=False)
                n_h = min(len(hard_n), max_neg_per_ev // 2)
                n_e = min(len(easy_n), max_neg_per_ev - n_h)
                sel_hn = hard_n if len(hard_n) <= n_h else rng.choice(hard_n, n_h, replace=False)
                sel_en = easy_n if len(easy_n) <= n_e else rng.choice(easy_n, n_e, replace=False)
                sel = np.concatenate([sel_p, sel_hn, sel_en])
                lgb_x_chunks.append(x_sp[i, sel])
                lgb_y_chunks.append(yy[sel])

    packed = {
        "seq_x": np.concatenate(seq_list, axis=0),
        "y_iou": np.concatenate(iou_list, axis=0),
        "y_frame": np.concatenate(fr_list, axis=0),
        "query_offsets": np.asarray(offsets, dtype=np.int32),
        "pts_sec": np.stack(pts_list, axis=0),
        "durations": np.asarray(durs_list, dtype=np.float32),
        "gt_starts": np.asarray(gts_list, dtype=np.float32),
        "gt_ends": np.asarray(gte_list, dtype=np.float32),
        "gt_a_idx": np.asarray(gta_list, dtype=np.int32),
        "gt_b_idx": np.asarray(gtb_list, dtype=np.int32),
    }
    lgb_tuple = (np.vstack(lgb_x_chunks), np.concatenate(lgb_y_chunks)) if build_lgb_spans else None
    return packed, lgb_tuple


def cmd_train_grounder(work_root: Path, *, epochs: int = 35, seeds_json: str = "[42,102,202]") -> dict[str, Any]:
    """Train `CrossEventSpanNet` (3-seed on Modal GPU) + `SpanGrounderLGBM` (CPU) strictly on `train_clean`."""
    from scripts.modal_app_functions import app as modal_app, train_cross_event_span_net_on_gpu, vol
    import modal  # type: ignore

    t0 = time.perf_counter()
    gdir = work_root / "grounder_v2"
    gdir.mkdir(parents=True, exist_ok=True)

    tr_manifest = json.loads((work_root / "splits" / "train_clean_manifest.json").read_text(encoding="utf-8"))
    va_manifest = json.loads((work_root / "splits" / "tune_manifest.json").read_text(encoding="utf-8"))
    tr_store = _load_split_feature_store(work_root, "train_clean")
    va_store = _load_split_feature_store(work_root, "tune")

    print("[train-grounder] Packing train_clean (2400) & tune (400) tensors...", flush=True)
    tr_packed, tr_lgb = _build_split_training_tensors(tr_manifest, tr_store, build_lgb_spans=True)
    va_packed, va_lgb = _build_split_training_tensors(va_manifest, va_store, build_lgb_spans=True, max_pos_per_ev=60, max_neg_per_ev=120)

    tr_npz_local = gdir / "train_clean_48f_packed.npz"
    va_npz_local = gdir / "tune_48f_packed.npz"
    np.savez_compressed(tr_npz_local, **tr_packed)
    np.savez_compressed(va_npz_local, **va_packed)
    print(f"[train-grounder] Packed .npz in {time.perf_counter()-t0:.1f}s (tr={tr_lgb[0].shape}, va={va_lgb[0].shape})", flush=True)

    # Train 3-seed CrossEventSpanNet on Modal GPU
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(tr_npz_local), "grounder_v2/train_clean_48f_packed.npz")
        batch.put_file(str(va_npz_local), "grounder_v2/tune_48f_packed.npz")

    seeds = [int(s) for s in json.loads(seeds_json)]
    starmap_args = [
        (
            "grounder_v2_run",
            "grounder_v2/train_clean_48f_packed.npz",
            "grounder_v2/tune_48f_packed.npz",
            int(epochs),
            json.dumps([s]),
        )
        for s in seeds
    ]
    with modal.enable_output(), modal_app.run():
        per_seed_receipts = list(train_cross_event_span_net_on_gpu.starmap(starmap_args))

    gpu_receipt = {
        "status": "validated",
        "device": per_seed_receipts[0]["device"],
        "gpu_name": per_seed_receipts[0]["gpu_name"],
        "train_queries": per_seed_receipts[0]["train_queries"],
        "val_queries": per_seed_receipts[0]["val_queries"],
        "epochs": int(epochs),
        "elapsed_sec": round(max(float(r["elapsed_sec"]) for r in per_seed_receipts), 2),
        "seed_reports": [rep for r in per_seed_receipts for rep in r["seed_reports"]],
        "ckpt_rel_paths": [rel for r in per_seed_receipts for rel in r["ckpt_rel_paths"]],
    }

    (gdir / "train_receipt.json").write_text(json.dumps(gpu_receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    for rel in gpu_receipt["ckpt_rel_paths"]:
        fname = Path(rel).name
        pull_modal_file(rel, gdir / fname)

    # Train SpanGrounderLGBM locally on CPU (clean train_clean -> early-stopped on tune)
    print("[train-grounder] Training SpanGrounderLGBM on local CPU...", flush=True)
    lgb_model = SpanGrounderLGBM.fit(
        tr_lgb[0],
        tr_lgb[1],
        X_val=va_lgb[0],
        y_val_iou=va_lgb[1],
        num_boost_round=500,
        num_leaves=63,
        learning_rate=0.05,
        seed=DEFAULT_UPGRADE_SEED,
        num_threads=4,
    )
    lgb_model.save(gdir)
    # Remove intermediate heavy packed npz locally to keep work folder tidy
    tr_npz_local.unlink(missing_ok=True)
    va_npz_local.unlink(missing_ok=True)

    summary = {
        "gpu_receipt": gpu_receipt,
        "lgbm_saved": str(gdir),
        "total_sec": round(time.perf_counter() - t0, 2),
    }
    (gdir / "grounder_train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _precompute_candidate_emissions(
    split: str,
    work_root: Path,
    grounder: JointMultiEventGrounder,
    hparams: Mapping[str, Any],
    query_bank: Mapping[str, np.ndarray] | None,
    *,
    use_dense128: bool = False,
    top_k_localize: int = 5,
    top_k_dense_cap: int = 3,
    reuse_48f_pack: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Precompute blinded Stage-1 rankings and `(net, lgb, pair_bonus)` caches in two contention-free phases."""
    import torch  # type: ignore

    t_start = time.perf_counter()
    tag = f"{split}_{'128f' if use_dense128 else '48f'}"
    manifest = json.loads((work_root / "splits" / f"{split}_manifest.json").read_text(encoding="utf-8"))
    store = _load_split_feature_store(work_root, split)
    dense_fdata = _load_dense128_frame_data(work_root, split, 128) if use_dense128 else {}
    s1 = _compute_stage1_calibrated_rankings(manifest, store, query_bank, hparams, top_k_cand=20)

    reuse_map: dict[tuple[str, str], dict[str, Any]] = {}
    if reuse_48f_pack is not None:
        for qc_old in reuse_48f_pack["query_caches"]:
            for ce_old in qc_old["cand_entries"]:
                reuse_map[(str(qc_old["query_id"]), str(ce_old["video_id"]))] = ce_old

    fdata, tmap, tvecs = store["frame_data"], store["text_str_to_idx"], store["text_vecs"]
    ctx = s1["ctx"]
    cap_queries = list(ctx["cap_queries"])
    ranked_by_qid = s1["ranked_by_qid"]

    # Phase 1: Pure PyTorch inference (isolated from LightGBM OpenMP thread pool)
    torch.set_num_threads(4)
    query_caches: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    n_q = len(cap_queries)
    for qi, q in enumerate(cap_queries):
        qid = str(q["query_id"])
        ranked = ranked_by_qid.get(qid)
        if not ranked:
            continue
        evs = list(q["events"])
        cids = [str(ev["clause_id"]) for ev in evs]
        etxts = [str(ev["text"]).strip() for ev in evs]
        if not all(t in tmap for t in etxts):
            continue
        k_ev = len(evs)
        top20_vids = [v for v, _ in ranked]
        cand_entries: list[dict[str, Any]] = []
        for rank_idx, (vid, s1_sc) in enumerate(ranked[:top_k_localize]):
            fkey = f"full::{vid}"
            use_dense = bool(use_dense128 and rank_idx < top_k_dense_cap and fkey in dense_fdata)
            if not use_dense and (qid, str(vid)) in reuse_map:
                cand_entries.append(reuse_map[(qid, str(vid))])
                continue
            fd = dense_fdata[fkey] if use_dense else fdata.get(fkey)
            if fd is None:
                continue
            ts = fd["pts_sec"].astype(np.float64)
            m_v = len(ts)
            dur = max(float(fd.get("duration", 0.0)), float(ts[-1]) if len(ts) else 1.0)
            curves = {}
            embs = {}
            for b in ("pe_bigg", "pe_l", "sig400"):
                mat = np.zeros((k_ev, m_v), dtype=np.float32)
                for i, et in enumerate(etxts):
                    mat[i] = fd[b] @ tvecs[b][tmap[et]]
                curves[b] = mat
                embs[b] = fd[b]

            fctx = extract_multievent_frame_features(ts, dur, curves, frame_embeddings_by_backbone=embs, weights=DEFAULT_WEIGHTS)
            ta, tb = np.triu_indices(m_v)
            net_iou_vec = np.zeros((k_ev, len(ta)), dtype=np.float32)
            net_logp_vec = np.zeros((k_ev, len(ta)), dtype=np.float32)
            pb_arr = None
            if grounder.span_nets:
                ni_acc = np.zeros((k_ev, m_v, m_v), dtype=np.float64)
                np_acc = np.zeros((k_ev, m_v, m_v), dtype=np.float64)
                pb_acc = np.zeros((max(0, k_ev - 1), m_v, m_v), dtype=np.float64) if k_ev > 1 else None
                for net in grounder.span_nets:
                    gi, gp, pb = net.predict_maps(fctx["seq_x"], target_m=m_v)
                    ni_acc += np.nan_to_num(gi, neginf=0.0)
                    np_acc += np.nan_to_num(gp, neginf=0.0)
                    if pb is not None and pb_acc is not None:
                        pb_acc += pb
                n_nets = float(len(grounder.span_nets))
                net_iou_vec = (ni_acc / n_nets)[:, ta, tb].astype(np.float32)
                net_logp_vec = np.log(np.clip(np_acc / n_nets, 1e-5, 1.0 - 1e-5))[:, ta, tb].astype(np.float32)
                if pb_acc is not None:
                    pb_arr = (pb_acc / n_nets).astype(np.float32)

            ce = {
                "video_id": vid,
                "s1_score": float(s1_sc),
                "ts": ts,
                "ordinals": fd["ordinals"],
                "dur": float(dur),
                "m_v": int(m_v),
                "k_ev": int(k_ev),
                "ta": ta,
                "tb": tb,
                "net_iou": net_iou_vec,
                "net_logp": net_logp_vec,
                "lgb_iou": np.zeros((k_ev, len(ta)), dtype=np.float32),
                "lgb_logp": np.zeros((k_ev, len(ta)), dtype=np.float32),
                "pair_bonus": pb_arr,
                "z_t": fctx["z_t"],
                "_fctx": fctx,
            }
            cand_entries.append(ce)
            pending_entries.append(ce)

        query_caches.append({
            "query_id": qid,
            "source_group_id": str(q["source_group_id"]),
            "gt_vid": str(q["video_id"]),
            "cids": cids,
            "etxts": etxts,
            "top20_vids": top20_vids,
            "cand_entries": cand_entries,
        })
        if (qi + 1) % 100 == 0 or (qi + 1) == n_q:
            print(f"[precompute:{tag}] Phase-1 PyTorch {qi + 1}/{n_q} ({time.perf_counter() - t_start:.1f}s)", flush=True)

    # Phase 2: Pure Batched LightGBM inference in ~200k-row contiguous chunks (strided for m_v > 64)
    if grounder.lgbm is not None and pending_entries:
        t_lgb0 = time.perf_counter()
        buf_x: list[np.ndarray] = []
        buf_meta: list[tuple[dict[str, Any], int, int, int, np.ndarray, np.ndarray, int]] = []
        buf_rows = 0
        done_cands = 0

        def _flush_lgb_chunk() -> None:
            if not buf_x:
                return
            x_cat = np.vstack(buf_x)
            reg_all = grounder.lgbm.reg_booster.predict(x_cat, num_threads=8)
            cls_all = grounder.lgbm.cls_booster.predict(x_cat, num_threads=8)
            pos = 0
            for ce, k_ev, m_v, step, ta_s, tb_s, n_rows in buf_meta:
                n_sub = len(ta_s)
                r_arr = reg_all[pos : pos + n_rows].reshape(k_ev, n_sub)
                c_arr = cls_all[pos : pos + n_rows].reshape(k_ev, n_sub)
                pos += n_rows
                if step == 1:
                    ce["lgb_iou"] = np.clip(r_arr, 0.0, 1.0).astype(np.float32)
                    ce["lgb_logp"] = np.log(np.clip(c_arr, 1e-6, 1.0 - 1e-6)).astype(np.float32)
                else:
                    gi, gp = SpanGrounderLGBM._upsample_strided_span_maps(r_arr, c_arr, ta_s, tb_s, step, m_v, k_ev)
                    ta_f, tb_f = ce["ta"], ce["tb"]
                    ce["lgb_iou"] = gi[:, ta_f, tb_f].astype(np.float32)
                    ce["lgb_logp"] = np.log(np.clip(gp[:, ta_f, tb_f], 1e-6, 1.0 - 1e-6)).astype(np.float32)
            buf_x.clear()
            buf_meta.clear()

        for ci, ce in enumerate(pending_entries):
            m_v = int(ce["m_v"])
            k_ev = int(ce["k_ev"])
            fctx = ce.pop("_fctx")
            x_sp, _, _, _, _ = extract_multievent_span_features(ce["ts"], ce["dur"], fctx)
            if m_v <= 64:
                step = 1
                ta_s, tb_s = ce["ta"], ce["tb"]
                sub_x = x_sp
            else:
                step = max(2, int(np.ceil(float(m_v) / 64.0)))
                mask = (ce["ta"] % step == 0) & (ce["tb"] % step == 0)
                ta_s, tb_s = ce["ta"][mask], ce["tb"][mask]
                sub_x = x_sp[:, mask, :]
            flat = sub_x.reshape(-1, sub_x.shape[-1])
            n_rows = int(flat.shape[0])
            buf_x.append(flat)
            buf_meta.append((ce, k_ev, m_v, step, ta_s, tb_s, n_rows))
            buf_rows += n_rows
            if buf_rows >= 200_000 or (ci + 1) == len(pending_entries):
                _flush_lgb_chunk()
                done_cands = ci + 1
                buf_rows = 0
                if (ci + 1) == len(pending_entries) or done_cands % 250 < len(buf_meta) + 40:
                    print(
                        f"[precompute:{tag}] Phase-2 LightGBM {done_cands}/{len(pending_entries)} ({time.perf_counter() - t_lgb0:.1f}s)",
                        flush=True,
                    )
    else:
        for ce in pending_entries:
            ce.pop("_fctx", None)

    return {
        "queries": cap_queries,
        "indexed_anet": ctx["indexed_anet"],
        "query_caches": query_caches,
    }


def _decode_cache_spans(
    eval_pack: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    ablation_mode: str = "joint",
) -> list[dict[str, Any]]:
    """Run Pairwise Soft-Overlap DP once per `(query, candidate)` for a fixed emission & DP configuration."""
    w_ni = float(cfg["w_net_iou"]) if ablation_mode in ("joint", "net_only") else 0.0
    w_np = float(cfg["w_net_logp"]) if ablation_mode in ("joint", "net_only") else 0.0
    w_li = float(cfg["w_lgb_iou"]) if ablation_mode in ("joint", "lgb_only") else 0.0
    w_lp = float(cfg["w_lgb_logp"]) if ablation_mode in ("joint", "lgb_only") else 0.0
    w_pn = float(cfg["w_pair_net"]) if ablation_mode in ("joint", "net_only") else 0.0
    w_gap = float(cfg["w_gap"])
    w_ov = float(cfg["w_ov"])
    w_edge = float(cfg["w_edge"])
    max_ov_ratio = float(cfg["max_ov_ratio"])
    enforce_end_order = bool(cfg["enforce_end_order"])

    decoded: list[dict[str, Any]] = []
    for qc in eval_pack["query_caches"]:
        qid = qc["query_id"]
        gt_vid = qc["gt_vid"]
        cids = qc["cids"]
        etxts = qc["etxts"]
        cands_out: list[tuple[float, float, str, list[dict[str, Any]]]] = []
        gv_cand: list[dict[str, Any]] = []
        for ce in qc["cand_entries"]:
            m_v = ce["m_v"]
            k_ev = len(cids)
            emit = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
            ta, tb = ce["ta"], ce["tb"]
            emit[:, ta, tb] = (
                w_ni * ce["net_iou"]
                + w_np * ce["net_logp"]
                + w_li * ce["lgb_iou"]
                + w_lp * ce["lgb_logp"]
            )
            pb = (w_pn * ce["pair_bonus"]) if (ce["pair_bonus"] is not None and w_pn != 0.0) else None
            dp_res = localize_soft_overlap_segment_dp(
                ce["ts"],
                emit,
                clause_ids=cids,
                clause_texts=etxts,
                frame_ordinals=ce["ordinals"],
                v_dur_sec=ce["dur"],
                pair_transition_bonus=pb,
                w_gap=w_gap,
                w_ov=w_ov,
                w_edge=w_edge,
                max_ov_ratio=max_ov_ratio,
                enforce_end_order=enforce_end_order,
                phi_curves=ce["z_t"],
            )
            if dp_res is None:
                continue
            spans = [
                {
                    "clause_id": sp.clause_id,
                    "start_sec": sp.start_sec,
                    "end_sec": sp.end_sec,
                    "status": sp.status,
                }
                for sp in dp_res.spans
            ]
            cands_out.append((float(ce["s1_score"]), float(dp_res.avg_score), ce["video_id"], spans))
            if ce["video_id"] == gt_vid:
                gv_cand = [{"video_id": gt_vid, "event_spans": spans}]
        decoded.append({
            "qid": qid,
            "gt_vid": gt_vid,
            "top20_vids": qc["top20_vids"],
            "cands": cands_out,
            "gv_cand": gv_cand,
        })
    return decoded


def _score_decoded_cache(
    eval_pack: Mapping[str, Any],
    decoded: Sequence[Mapping[str, Any]],
    w_dp_rank: float,
) -> dict[str, Any]:
    """Score & rank decoded candidate chains for a given `w_dp_rank` without re-running DP."""
    preds: dict[str, Any] = {}
    gv_preds: dict[str, Any] = {}
    w_r = float(w_dp_rank)
    for item in decoded:
        scored_cands = sorted(
            ((s1 + w_r * dp_sc, vid, spans) for s1, dp_sc, vid, spans in item["cands"]),
            key=lambda x: (-x[0], x[1]),
        )
        top_vids = [x[1] for x in scored_cands] + [v for v in item["top20_vids"] if all(x[1] != v for x in scored_cands)]
        qid = item["qid"]
        preds[qid] = {
            "ranked_video_ids": top_vids[:20],
            "top_k_results": [{"video_id": x[1], "event_spans": x[2]} for x in scored_cands],
        }
        gv_preds[qid] = {
            "ranked_video_ids": [item["gt_vid"]],
            "top_k_results": item["gv_cand"],
        }

    ev_res = evaluate_activitynet_full_chain(
        eval_pack["queries"],
        preds,
        indexed_video_ids=eval_pack["indexed_anet"],
        given_video_predictions_by_qid=gv_preds,
    )
    hits_list = list(ev_res["per_query_hits"])
    metrics = {k: v for k, v in ev_res.items() if k != "per_query_hits"}
    metrics["near_uniform_rate"] = _compute_near_uniform_rate(eval_pack["queries"], preds)
    return {"metrics": metrics, "per_query_hits": hits_list, "preds": preds}


def _evaluate_from_cache(
    eval_pack: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    ablation_mode: str = "joint",
) -> dict[str, Any]:
    """Decode spans & rerank top candidates directly from precomputed neural/LGBM cache."""
    decoded = _decode_cache_spans(eval_pack, cfg, ablation_mode=ablation_mode)
    return _score_decoded_cache(eval_pack, decoded, float(cfg["w_dp_rank"]))


def cmd_tune_and_lock(work_root: Path) -> dict[str, Any]:
    """Sweep Joint Grounder + Soft-Overlap DP strictly on `Tune`, then lock via one-shot `Valid-Confirm`."""
    t0 = time.perf_counter()
    hp = _load_locked_stage1_hparams(work_root)
    qb = _build_or_load_tune_query_bank(work_root)
    grounder = JointMultiEventGrounder.from_dir(work_root / "grounder_v2")

    tune_cache_48 = _precompute_candidate_emissions("tune", work_root, grounder, hp, qb, use_dense128=False)
    has_128_tune = (work_root / "features" / "tune_dense128").is_dir()
    tune_cache_128 = (
        _precompute_candidate_emissions("tune", work_root, grounder, hp, qb, use_dense128=True, reuse_48f_pack=tune_cache_48)
        if has_128_tune
        else None
    )
    print(f"[tune-v2] Precomputed caches in {time.perf_counter()-t0:.1f}s (has_128={has_128_tune})", flush=True)

    weight_modes = [
        (1.0, 0.20, 0.0, 0.0),      # Net only
        (0.0, 0.0, 1.0, 0.15),      # LGB only
        (0.65, 0.12, 0.35, 0.08),   # Joint net-heavy
        (0.50, 0.10, 0.50, 0.10),   # Balanced joint
    ]

    def _sweep_cache(cache_obj: Mapping[str, Any], use_128: bool, base_cfgs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for base in base_cfgs:
            cand_base = dict(base)
            cand_base["use_dense128"] = bool(use_128)
            decoded = _decode_cache_spans(cache_obj, cand_base, ablation_mode="joint")
            for w_dp_r in (0.0, 1.5, 3.0):
                ev = _score_decoded_cache(cache_obj, decoded, w_dp_r)
                m = ev["metrics"]
                obj = float(m["full_chain_hit_at_1"]) + 0.10 * float(m["per_event_tiou_05_rate"]) + 0.05 * float(m["full_chain_hit_at_5"])
                cand_cfg = {**cand_base, "w_dp_rank": float(w_dp_r)}
                rows.append({
                    "obj": round(obj, 6),
                    "cfg": cand_cfg,
                    "hit1": m["full_chain_hit_at_1"],
                    "hit5": m["full_chain_hit_at_5"],
                    "per_ev": m["per_event_tiou_05_rate"],
                    "near_dm": m["near_uniform_rate"],
                })
        rows.sort(key=lambda x: -x["obj"])
        return rows

    base_48_cfgs = [
        {
            "w_net_iou": w_ni,
            "w_net_logp": w_np,
            "w_lgb_iou": w_li,
            "w_lgb_logp": w_lp,
            "w_pair_net": 0.08 if w_ni > 0.0 else 0.0,
            "w_gap": wg,
            "w_ov": wo,
            "w_edge": we,
            "max_ov_ratio": max_ov,
            "enforce_end_order": end_ord,
        }
        for (w_ni, w_np, w_li, w_lp) in weight_modes
        for (wg, wo, we) in [(0.4, 0.3, 0.1), (0.8, 0.6, 0.2), (1.2, 0.9, 0.25)]
        for max_ov in (0.4, 0.65)
        for end_ord in (True, False)
    ]

    sweep_top = _sweep_cache(tune_cache_48, False, base_48_cfgs)
    if tune_cache_128 is not None:
        seen_keys: set[tuple[Any, ...]] = set()
        top_dp_cfgs: list[dict[str, Any]] = []
        for row in sweep_top:
            c = row["cfg"]
            key = (c["w_net_iou"], c["w_lgb_iou"], c["w_gap"], c["w_ov"], c["w_edge"], c["max_ov_ratio"], c["enforce_end_order"])
            if key not in seen_keys:
                seen_keys.add(key)
                top_dp_cfgs.append(c)
            if len(top_dp_cfgs) >= 12:
                break
        best48_c = sweep_top[0]["cfg"]
        for (w_ni, w_np, w_li, w_lp) in weight_modes:
            key = (w_ni, w_li, best48_c["w_gap"], best48_c["w_ov"], best48_c["w_edge"], best48_c["max_ov_ratio"], best48_c["enforce_end_order"])
            if key not in seen_keys:
                seen_keys.add(key)
                top_dp_cfgs.append({
                    **best48_c,
                    "w_net_iou": w_ni,
                    "w_net_logp": w_np,
                    "w_lgb_iou": w_li,
                    "w_lgb_logp": w_lp,
                    "w_pair_net": 0.08 if w_ni > 0.0 else 0.0,
                })
        sweep_top.extend(_sweep_cache(tune_cache_128, True, top_dp_cfgs))
        sweep_top.sort(key=lambda x: -x["obj"])

    best_cfg: dict[str, Any] = dict(sweep_top[0]["cfg"])
    best_cache = tune_cache_128 if best_cfg["use_dense128"] else tune_cache_48
    tune_net_only = _evaluate_from_cache(best_cache, best_cfg, ablation_mode="net_only")["metrics"]
    tune_lgb_only = _evaluate_from_cache(best_cache, best_cfg, ablation_mode="lgb_only")["metrics"]
    tune_joint = _evaluate_from_cache(best_cache, best_cfg, ablation_mode="joint")["metrics"]
    tune_48f = _evaluate_from_cache(tune_cache_48, {**best_cfg, "use_dense128": False}, ablation_mode="joint")["metrics"]

    # One-Shot Confirmation on Valid-Confirm!
    vc_use_128 = bool(best_cfg["use_dense128"] and (work_root / "features" / "valid_confirm_dense128").is_dir())
    vc_cache = _precompute_candidate_emissions("valid_confirm", work_root, grounder, hp, qb, use_dense128=vc_use_128)
    vc_joint = _evaluate_from_cache(vc_cache, best_cfg, ablation_mode="joint")
    vc_net_only = _evaluate_from_cache(vc_cache, best_cfg, ablation_mode="net_only")["metrics"]
    vc_lgb_only = _evaluate_from_cache(vc_cache, best_cfg, ablation_mode="lgb_only")["metrics"]

    # Paired bootstrap CI vs Phase B Locked Champion & Uniform Split on Valid-Confirm
    vc_lock_b = json.loads((work_root / "champion_lock_receipt.json").read_text(encoding="utf-8"))
    from scripts.run_activitynet_videocomp_pipeline import _evaluate_split_cpu_methods

    phase_b_vc = _evaluate_split_cpu_methods(
        work_root,
        split="valid_confirm",
        hparams=hp,
        query_bank=qb,
        captions_methods_subset=("exact_semi_markov_dp_calibrated",),
        run_videocomp=False,
    )["activitynet_captions"]["exact_semi_markov_dp_calibrated"]
    unif_vc_metrics, unif_vc_hits = _eval_uniform_given_video_baseline(vc_cache["queries"])

    gids = [str(q["source_group_id"]) for q in vc_cache["queries"]]
    ci_vs_phase_b = paired_source_group_bootstrap_ci(vc_joint["per_query_hits"], phase_b_vc["per_query_hits"], gids)
    ci_vs_uniform = paired_source_group_bootstrap_ci(vc_joint["per_query_hits"], unif_vc_hits, gids)

    gate_passed = bool(ci_vs_phase_b["significant_gt_zero"] and ci_vs_uniform["significant_gt_zero"])
    receipt = {
        "selected_config": best_cfg,
        "config_sha256": hashlib.sha256(json.dumps(best_cfg, sort_keys=True).encode("utf-8")).hexdigest(),
        "tune": {
            "joint": tune_joint,
            "net_only": tune_net_only,
            "lgb_only": tune_lgb_only,
            "joint_48f": tune_48f,
            "sweep_top5": sweep_top[:5],
        },
        "valid_confirm": {
            "phase_b_champion": {k: v for k, v in phase_b_vc.items() if k != "per_query_hits"},
            "uniform_given_video": unif_vc_metrics,
            "net_only": vc_net_only,
            "lgb_only": vc_lgb_only,
            "joint_champion_v2": vc_joint["metrics"],
            "videocomp_preserved": vc_lock_b["videocomp_metrics"][vc_lock_b["champion_videocomp_method"]],
        },
        "bootstrap_ci95": {
            "vs_phase_b_champion": ci_vs_phase_b,
            "vs_uniform_given_video": ci_vs_uniform,
        },
        "go_no_go_gate_passed": gate_passed,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (work_root / "champion_v2_lock_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "go_no_go_gate_passed": gate_passed,
        "selected_config": best_cfg,
        "tune_joint": tune_joint,
        "valid_confirm_joint": vc_joint["metrics"],
        "valid_confirm_phase_b": receipt["valid_confirm"]["phase_b_champion"],
        "bootstrap_ci95": receipt["bootstrap_ci95"],
    }, indent=2))
    return receipt


def cmd_eval_tests(work_root: Path) -> dict[str, Any]:
    """Evaluate locked Champion v2 once on `Test-200` (comparison) AND truly sealed `Fresh-Test-200`."""
    lock_p = work_root / "champion_v2_lock_receipt.json"
    if not lock_p.is_file():
        raise RuntimeError("Refused: run tune-and-lock before eval-tests.")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    if not lock.get("go_no_go_gate_passed"):
        raise RuntimeError("Refused: champion_v2_lock_receipt.json failed Go/No-Go gate.")

    cfg = dict(lock["selected_config"])
    hp = _load_locked_stage1_hparams(work_root)
    qb = _build_or_load_tune_query_bank(work_root)
    grounder = JointMultiEventGrounder.from_dir(work_root / "grounder_v2")

    # Ensure Fresh-Test 48f features exist + extract dense128 Top-3 for Test & Fresh-Test if selected
    if not (work_root / "features" / "fresh_test" / "extract_summary.json").is_file():
        cmd_extract_split(work_root, split="fresh_test", gpu_tier="L40S", shard_size=150, num_frames=48)
    if cfg.get("use_dense128"):
        for sp_t in ("test", "fresh_test"):
            if not (work_root / "features" / f"{sp_t}_dense128" / "extract_summary.json").is_file():
                cmd_extract_dense_top3(work_root, split=sp_t, top_k=3, num_frames=128, shard_size=60)

    from scripts.run_activitynet_videocomp_pipeline import _evaluate_split_cpu_methods

    results_by_split: dict[str, Any] = {}
    for sp_t in ("test", "fresh_test"):
        c_pack = _precompute_candidate_emissions(
            sp_t, work_root, grounder, hp, qb, use_dense128=bool(cfg.get("use_dense128"))
        )
        ev_joint = _evaluate_from_cache(c_pack, cfg, ablation_mode="joint")
        ev_net = _evaluate_from_cache(c_pack, cfg, ablation_mode="net_only")
        ev_lgb = _evaluate_from_cache(c_pack, cfg, ablation_mode="lgb_only")
        phase_b_sp = _evaluate_split_cpu_methods(
            work_root,
            split=sp_t,
            hparams=hp,
            query_bank=qb,
            captions_methods_subset=("whole_paragraph_global", "exact_semi_markov_dp_calibrated"),
            run_videocomp=False,
        )["activitynet_captions"]
        pb_champ = phase_b_sp["exact_semi_markov_dp_calibrated"]
        pb_glob = phase_b_sp["whole_paragraph_global"]
        unif_m, unif_hits = _eval_uniform_given_video_baseline(c_pack["queries"])

        gids = [str(q["source_group_id"]) for q in c_pack["queries"]]
        ci_pb = paired_source_group_bootstrap_ci(ev_joint["per_query_hits"], pb_champ["per_query_hits"], gids)
        ci_unif = paired_source_group_bootstrap_ci(ev_joint["per_query_hits"], unif_hits, gids)

        pred_dir = work_root / sp_t
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "grounder_v2_predictions.json").write_text(
            json.dumps({"split": sp_t, "config_sha256": lock["config_sha256"], "predictions": ev_joint["preds"]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results_by_split[sp_t] = {
            "whole_paragraph_global": {k: v for k, v in pb_glob.items() if k != "per_query_hits"},
            "phase_b_champion": {k: v for k, v in pb_champ.items() if k != "per_query_hits"},
            "uniform_given_video": unif_m,
            "net_only": ev_net["metrics"],
            "lgb_only": ev_lgb["metrics"],
            "joint_champion_v2": ev_joint["metrics"],
            "bootstrap_ci95": {
                "vs_phase_b_champion": ci_pb,
                "vs_uniform_given_video": ci_unif,
            },
        }

    bench_dir = REPO_ROOT / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    final_out = {
        "benchmark_suite": "ActivityNet Captions Multi-Event Temporal Grounding Upgrade (Valid-Confirm + Test-200 + Sealed Fresh-Test-200)",
        "config_sha256": lock["config_sha256"],
        "selected_config": cfg,
        "train_clean_videos": 2400,
        "valid_confirm": lock["valid_confirm"],
        "test": results_by_split["test"],
        "fresh_test": results_by_split["fresh_test"],
    }
    (bench_dir / "activitynet_videocomp_v2_grounder_results.json").write_text(
        json.dumps(final_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Kết quả Nâng cấp Định vị Đa Sự Kiện ActivityNet Captions (`CrossEventSpanNet` + `SpanGrounderLGBM` + Soft-Overlap DP)",
        "",
        f"- Config SHA-256: `{lock['config_sha256']}` · Huấn luyện sạch trên `2.400` video `train` tách nguồn tuyệt đối",
        "",
        "| Phương pháp | Tập kiểm tra | Full-Chain Hit@1 | Hit@5 | Given-Video Hit@1 | Per-Event `tIoU>=0.5` | Tỷ lệ gần `D/M` (±25%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for sp_lbl, block in (("Valid-Confirm (400q)", lock["valid_confirm"]), ("Test-200 (Đối chiếu)", results_by_split["test"]), ("Fresh-Test-200 (Chưa từng đụng)", results_by_split["fresh_test"])):
        pb = block["phase_b_champion"]
        net = block["net_only"]
        lgb = block["lgb_only"]
        jnt = block["joint_champion_v2"]
        for name, row in [
            ("Exact 2D Semi-Markov DP (Phase B Baseline)", pb),
            ("CrossEventSpanNet + Pairwise Soft-Overlap DP", net),
            ("SpanGrounderLGBM + Pairwise Soft-Overlap DP", lgb),
            ("Joint CrossEventSpanNet + LightGBM + Soft-Overlap DP (+ 128f Top-3)", jnt),
        ]:
            lines.append(
                f"| {name} | {sp_lbl} | {row['full_chain_hit_at_1']*100:.2f}% | {row['full_chain_hit_at_5']*100:.2f}% | "
                f"{row['given_video_full_chain_hit_at_1']*100:.2f}% | {row['per_event_tiou_05_rate']*100:.2f}% | "
                f"{row.get('near_uniform_rate', 0.0)*100:.1f}% |"
            )
    (bench_dir / "activitynet_videocomp_v2_grounder_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_out, indent=2))
    return final_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ActivityNet multi-event span grounder upgrade orchestrator.")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_m = sub.add_parser("build-clean-manifests")
    p_m.add_argument("--num-train-videos", type=int, default=2400)
    p_m.add_argument("--num-fresh-test-videos", type=int, default=200)

    p_et = sub.add_parser("extract-train-clean")
    p_et.add_argument("--shard-size", type=int, default=150)

    p_d = sub.add_parser("extract-dense-top3")
    p_d.add_argument("--split", required=True)
    p_d.add_argument("--top-k", type=int, default=3)
    p_d.add_argument("--num-frames", type=int, default=128)
    p_d.add_argument("--shard-size", type=int, default=60)

    p_tr = sub.add_parser("train-grounder")
    p_tr.add_argument("--epochs", type=int, default=35)
    p_tr.add_argument("--seeds-json", default="[42,102,202]")

    sub.add_parser("tune-and-lock")
    sub.add_parser("eval-tests")

    args = parser.parse_args(argv)
    wr = Path(args.work_root)
    if args.cmd == "build-clean-manifests":
        cmd_build_clean_manifests(wr, num_train_videos=args.num_train_videos, num_fresh_test_videos=args.num_fresh_test_videos)
    elif args.cmd == "extract-train-clean":
        cmd_extract_split(wr, split="train_clean", gpu_tier="L40S", shard_size=args.shard_size, num_frames=48)
    elif args.cmd == "extract-dense-top3":
        cmd_extract_dense_top3(wr, split=args.split, top_k=args.top_k, num_frames=args.num_frames, shard_size=args.shard_size)
    elif args.cmd == "train-grounder":
        cmd_train_grounder(wr, epochs=args.epochs, seeds_json=args.seeds_json)
    elif args.cmd == "tune-and-lock":
        cmd_tune_and_lock(wr)
    elif args.cmd == "eval-tests":
        cmd_eval_tests(wr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
