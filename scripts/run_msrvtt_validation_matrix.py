"""Run the complete MSR-VTT 1K-A validation matrix (B0-B4, R0-R7 + improvements).

1. Loads Kaggle-extracted feature banks for train_sub, tune (tune_g1 + tune_g2),
   and confirm (confirm_g1 + confirm_g2) — strictly disjoint from test_1ka.
2. Fits learned components ONLY on train_sub:
   - B4: 32d LightGBM LTR (5-fold GroupKFold by source_group)
   - R3: VEDJEJointRerankerNet on retrieval-mined Top-20 hard negatives with
         False-Negative Filtering.
   - R7: Train-9K Query-Bank Hubness priors (QB-CSLS / QB-DSN).
3. Sweeps hyperparameters strictly on tune, freezes them, evaluates on confirm,
   computes paired source-group 95% bootstrap CIs vs B1, measures latency & peak RAM,
   and locks the single Champion configuration to champion_lock_receipt.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_validation_benchmark import get_process_memory_mb  # noqa: E402
from scripts.train_ltr_ranker_cv import run_group_kfold_cv  # noqa: E402
from videntra.evaluation.provenance import compute_code_sha256, sha256_file  # noqa: E402
from videntra.ranking.features import (  # noqa: E402
    LTR_FEATURE_NAMES_V2,
    compute_query_z_scores,
    extract_ltr_features_32d,
)
from videntra.ranking.learned_ranker import LearnedRankerLTR  # noqa: E402
from videntra.ranking.vedje_reranker import (  # noqa: E402
    TrainQueryBankHubness,
    compute_spaced_top2_frame_score,
    compute_xpool_score,
    detect_false_negative_mask,
    l2_normalize_np,
    score_vedje_reranker_batch,
    train_vedje_reranker,
)
from videntra.relations.kalman_tracker import classify_motion_relation_query  # noqa: E402
from videntra.relations.spatial_action_verifier import SpatialActionVerifier  # noqa: E402
from videntra.retrieval.msrvtt_matrix import (  # noqa: E402
    GalleryFeatureBank,
    apply_confidence_gated_rerank,
    apply_hubness_csls,
    build_bm25_scorer,
    build_top20_vedje_example,
    classify_modality_query_intent,
    compute_b0_siglip_scores,
    compute_b1_videoprism_scores,
    compute_b2_videoprism_window_scores,
    compute_frame_bank_scores,
    compute_multiprompt_query_vector,
    compute_vl2_gallery_score_matrices,
    evaluate_predictions,
    fuse_b3_vp_and_siglip,
    load_gallery_feature_bank,
    rank_gallery_desc,
    rerank_fixed_pool,
    score_r6_candidates,
    sha256_list,
    sha256_text,
    zscore_normalize_row,
)

PROTOCOL_ROOT = Path("work/msrvtt_protocol_20261122")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["query_id"]): r for r in _read_jsonl(path)}


def _load_train_query_banks(vp_train_dir: Path, mm_train_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load all independent training-split caption vectors for QB-CSLS/DSN hubness priors."""
    from videntra.retrieval.msrvtt_matrix import _find_shard_subdir

    vp_bank = None
    sig_bank = None
    if vp_train_dir.exists():
        sd = _find_shard_subdir(vp_train_dir)
        q_path = sd / "query_vectors.npy"
        if q_path.is_file():
            vp_bank = np.asarray(np.load(str(q_path)), dtype=np.float32)
    if mm_train_dir.exists():
        sd = _find_shard_subdir(mm_train_dir)
        q_path = sd / "siglip_query_vectors.npy"
        if q_path.is_file():
            sig_bank = np.asarray(np.load(str(q_path)), dtype=np.float32)
    return vp_bank, sig_bank


def _load_vl2_train_query_banks(vl2_train_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load independent training-split PE-Core & SigLIP2-SO400M caption vectors for QB-CSLS."""
    from videntra.retrieval.msrvtt_matrix import _find_shard_subdir

    pe_bank = None
    sig400_bank = None
    if vl2_train_dir.exists():
        sd = _find_shard_subdir(vl2_train_dir)
        p_path = sd / "pe_query_vectors.npy"
        s_path = sd / "sig400_query_vectors.npy"
        if p_path.is_file():
            pe_bank = np.asarray(np.load(str(p_path)), dtype=np.float32)
        if s_path.is_file():
            sig400_bank = np.asarray(np.load(str(s_path)), dtype=np.float32)
    return pe_bank, sig400_bank


def _extract_b4_a5_candidates(
    bank: GalleryFeatureBank,
    query_text: str,
    *,
    ocr_bm25: Callable[[str, Sequence[str]], np.ndarray],
    asr_bm25: Callable[[str, Sequence[str]], np.ndarray],
    sa_verifier: SpatialActionVerifier,
    top_n: int = 40,
) -> tuple[list[str], list[float], np.ndarray]:
    """Extract deterministic 32d A5 features & base scores for Top-N gallery candidates."""
    vp_comb, vp_win_max = compute_b2_videoprism_window_scores(bank, query_text, whole_weight=0.65, mode="blend_max")
    vp_whole = compute_b1_videoprism_scores(bank, query_text)
    sig_frame = compute_b0_siglip_scores(bank, query_text, pooling="frame_max")
    sig_mean = compute_b0_siglip_scores(bank, query_text, pooling="video_mean")
    sig_blend = 0.65 * sig_frame + 0.35 * sig_mean

    ocr_all = ocr_bm25(query_text, bank.video_ids)
    asr_all = asr_bm25(query_text, bank.video_ids)

    n = len(bank.video_ids)
    def _ranks(arr: np.ndarray) -> np.ndarray:
        ord_idx = np.argsort(-arr, kind="mergesort")
        rk = np.empty(n, dtype=np.float64)
        rk[ord_idx] = np.arange(1, n + 1, dtype=np.float64)
        return rk

    r_vp = _ranks(vp_comb)
    r_sig = _ranks(sig_blend)
    r_ocr = np.where(ocr_all > 0.0, _ranks(ocr_all), 1e6)
    r_asr = np.where(asr_all > 0.0, _ranks(asr_all), 1e6)

    rrf_vp = 1.0 / (60.0 + r_vp)
    rrf_vis = 1.0 / (60.0 + r_sig)
    rrf_ocr = np.where(ocr_all > 0.0, 1.0 / (60.0 + r_ocr), 0.0)
    rrf_asr = np.where(asr_all > 0.0, 1.0 / (60.0 + r_asr), 0.0)
    base_all = 1.45 * rrf_vp + 1.0 * rrf_vis + 0.35 * rrf_ocr + 0.25 * rrf_asr

    top_idx = np.argsort(-base_all, kind="mergesort")[:top_n]
    cand_vids = [bank.video_ids[int(i)] for i in top_idx]
    base_scores = [float(base_all[int(i)]) for i in top_idx]

    z_stats = compute_query_z_scores(sig_blend, top_k=120)
    vp_mean = float(np.mean(vp_comb))
    vp_std = max(1e-4, float(np.std(vp_comb)))
    q_ctx = sa_verifier.prepare_query_context(query_text, [query_text])

    sig_q = bank.sig_query_map.get(sha256_text(query_text))
    feat_rows: list[np.ndarray] = []
    for pos_i, g_i in enumerate(top_idx):
        vid = bank.video_ids[int(g_i)]
        vp_hub = float(bank.vp_hub_priors["mean_topk"][int(g_i)]) if bank.vp_hub_priors is not None else 0.0
        vp_csls = float(2.0 * vp_whole[int(g_i)] - vp_hub)

        # Frame-level top3/top5 mean
        f_pos = bank.sig_video_kf_pos.get(vid)
        if sig_q is not None and f_pos is not None and len(f_pos) > 0 and bank.sig_frame_vecs is not None:
            f_sims = np.sort((bank.sig_frame_vecs[f_pos].astype(np.float32) @ l2_normalize_np(sig_q)).astype(np.float64))
            top3_m = float(np.mean(f_sims[-min(3, len(f_sims)):]))
            top5_m = float(np.mean(f_sims[-min(5, len(f_sims)):]))
            spike_drop = float(f_sims[-1] - (f_sims[-2] if len(f_sims) >= 2 else f_sims[-1]))
        else:
            top3_m = float(sig_blend[int(g_i)])
            top5_m = float(sig_blend[int(g_i)])
            spike_drop = 0.0

        # Fast spatial/action verifier
        flat_objs: list[dict[str, Any]] = []
        pts_list: list[float] = []
        for f_obs in bank.kf_detections.get(vid, ()):
            pts_val = float(f_obs.get("pts_sec", 0.0))
            for det in f_obs.get("detections") or ():
                d_copy = dict(det)
                d_copy["pts_sec"] = pts_val
                flat_objs.append(d_copy)
                pts_list.append(pts_val)
        sa = sa_verifier.verify_candidate(
            query_ctx=q_ctx,
            video_id=vid,
            window_start=0.0,
            window_end=30.0,
            video_objects=flat_objs,
            video_obj_pts=pts_list,
            detector_ran=bool(vid in bank.kf_detections),
        )
        cand_dict = {
            "s_vis_blend_max": float(sig_blend[int(g_i)]),
            "s_vis_top3_mean": top3_m,
            "s_vis_top5_mean": top5_m,
            "s_vis_ctx_coherence": float(sig_mean[int(g_i)]),
            "s_late_interaction": float(sig_frame[int(g_i)]),
            "spike_drop_ratio": spike_drop,
            "rrf_vis": float(rrf_vis[int(g_i)]),
            "rrf_ocr": float(rrf_ocr[int(g_i)]),
            "rrf_asr": float(rrf_asr[int(g_i)]),
            "fts_score": float(ocr_all[int(g_i)] + asr_all[int(g_i)]),
            "vp_whole_cos": float(vp_whole[int(g_i)]),
            "vp_subclip_max": float(vp_win_max[int(g_i)]),
            "vp_z": float((vp_comb[int(g_i)] - vp_mean) / vp_std),
            "vp_csls": vp_csls,
            "rrf_vp": float(rrf_vp[int(g_i)]),
            "lex_reliability_score": float(min(1.5, 0.5 * (ocr_all[int(g_i)] + asr_all[int(g_i)]))),
            "spatial_predicate_score": float(sa.spatial_predicate_score),
            "action_order_score": float(sa.action_order_score),
            "graph_relation_score": float(sa.score),
        }
        feat_rows.append(extract_ltr_features_32d(cand_dict, z_stats=z_stats, return_dict=False))

    return cand_vids, base_scores, np.vstack(feat_rows).astype(np.float32)


def _score_b4_a5_gallery(
    bank: GalleryFeatureBank,
    query_text: str,
    *,
    ocr_bm25: Callable[[str, Sequence[str]], np.ndarray],
    asr_bm25: Callable[[str, Sequence[str]], np.ndarray],
    sa_verifier: SpatialActionVerifier,
    ltr_ranker: LearnedRankerLTR | None,
    use_consensus: bool,
    use_ltr: bool,
    ltr_blend_weight: float = 0.08,
    cached_cands: tuple[list[str], list[float], np.ndarray] | None = None,
    cached_sig_fc: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[str]:
    """Score B4 (A5 2x2: Consensus x LTR) on a 1,000-video MSR-VTT gallery."""
    cand_vids, base_scores, x_mat = (
        cached_cands
        if cached_cands is not None
        else _extract_b4_a5_candidates(
            bank, query_text, ocr_bm25=ocr_bm25, asr_bm25=asr_bm25, sa_verifier=sa_verifier, top_n=40
        )
    )
    scores = np.asarray(base_scores, dtype=np.float64)
    if use_ltr and ltr_ranker is not None and ltr_ranker.is_loaded:
        ltr_raw = ltr_ranker.score_matrix(x_mat)
        scores = scores + float(ltr_blend_weight) * np.tanh(ltr_raw)

    if use_consensus:
        # SigLIP2 dual-rank frame + clip mean consensus promotion
        if cached_sig_fc is not None:
            sig_f, sig_c = cached_sig_fc
        else:
            sig_f = compute_b0_siglip_scores(bank, query_text, pooling="frame_max")
            sig_c = compute_b0_siglip_scores(bank, query_text, pooling="video_mean")
        n = len(bank.video_ids)
        ord_f = np.argsort(-sig_f, kind="mergesort")
        ord_c = np.argsort(-sig_c, kind="mergesort")
        rf = np.empty(n, dtype=np.int32); rf[ord_f] = np.arange(1, n + 1)
        rc = np.empty(n, dtype=np.int32); rc[ord_c] = np.arange(1, n + 1)
        c_vid = bank.video_ids[int(ord_c[0])]
        c_idx = bank.vid_to_idx[c_vid]
        if int(rf[c_idx]) <= 2 and c_vid in cand_vids:
            pos_c = cand_vids.index(c_vid)
            top_score = float(np.max(scores))
            if top_score - float(scores[pos_c]) <= 0.035:
                scores[pos_c] = top_score + 0.002

    order = sorted(range(len(cand_vids)), key=lambda i: (-float(scores[i]), cand_vids[i]))
    ranked_head = [cand_vids[i] for i in order]
    seen = set(ranked_head)
    tail = [v for v in bank.video_ids if v not in seen]
    return ranked_head + tail


def train_b4_ltr_and_r3_vedje(
    protocol_root: Path,
    *,
    train_vp_bank: np.ndarray | None,
    train_sig_bank: np.ndarray | None,
    models_dir: Path,
    seed: int = 20261122,
) -> dict[str, Any]:
    """Fit both B4 32d LightGBM LTR and R3 VEDJEJointRerankerNet strictly on train_sub."""
    models_dir.mkdir(parents=True, exist_ok=True)
    kaggle_out = protocol_root / "kaggle_out"
    vp_train_dir = kaggle_out / "vp_train_sub"
    mm_train_dir = kaggle_out / "mm_train_sub"

    # Build temporary manifest & queries for train_sub videos using deterministically selected caption #0
    train_rows = _read_jsonl(protocol_root / "manifests" / "train_9k_clean" / "train_videos.jsonl")
    from scripts.run_msrvtt_kaggle_pipeline import _select_stratified_train_sub

    sub_rows = _select_stratified_train_sub(train_rows, max_videos=1600)
    tmp_gal_path = models_dir / "_train_sub_gallery.tmp.jsonl"
    tmp_q_path = models_dir / "_train_sub_queries.tmp.jsonl"
    tmp_gal_path.write_text("\n".join(json.dumps({"video_id": r["video_id"]}) for r in sub_rows) + "\n", encoding="utf-8")

    q_records: list[dict[str, Any]] = []
    sg_by_vid = {r["video_id"]: r["source_group"] for r in sub_rows}
    caps_by_vid: dict[str, list[str]] = {r["video_id"]: [c["caption"] for c in r["captions"]] for r in sub_rows}
    for r in sub_rows:
        qtxt = str(r["captions"][0]["caption"]).strip()
        q_records.append({
            "query_id": f"train_{r['video_id']}",
            "query_text": qtxt,
            "query_text_sha256": sha256_text(qtxt),
            "target_vid": r["video_id"],
            "source_group": r["source_group"],
        })
    tmp_q_path.write_text("\n".join(json.dumps({"query_id": q["query_id"], "query_text": q["query_text"], "query_text_sha256": q["query_text_sha256"]}) for q in q_records) + "\n", encoding="utf-8")

    try:
        train_bank = load_gallery_feature_bank(
            gallery_name="train_sub",
            gallery_manifest_path=tmp_gal_path,
            queries_path=tmp_q_path,
            vp_dirs=[vp_train_dir],
            mm_dirs=[mm_train_dir],
            train_vp_bank=train_vp_bank,
            train_sig_bank=train_sig_bank,
        )
    finally:
        tmp_gal_path.unlink(missing_ok=True)
        tmp_q_path.unlink(missing_ok=True)

    # 1. Fit B4 32d LightGBM LTR on train_sub (reuse saved artifact if already fit)
    ltr_path = models_dir / "b4_ltr_train9k.json"
    if ltr_path.is_file():
        ltr_artifact = json.loads(ltr_path.read_text(encoding="utf-8"))
    else:
        ocr_bm25 = build_bm25_scorer(train_bank.ocr_docs)
        asr_bm25 = build_bm25_scorer(train_bank.asr_docs)
        sa_verifier = SpatialActionVerifier()

        x_list: list[np.ndarray] = []
        y_list: list[float] = []
        qid_list: list[str] = []
        grp_list: list[str] = []
        base_list: list[float] = []

        for qr in q_records:
            cand_vids, b_scores, x_mat = _extract_b4_a5_candidates(
                train_bank, qr["query_text"], ocr_bm25=ocr_bm25, asr_bm25=asr_bm25, sa_verifier=sa_verifier, top_n=40
            )
            t_vid = qr["target_vid"]
            if t_vid not in cand_vids:
                continue
            for pos, vid in enumerate(cand_vids):
                x_list.append(x_mat[pos])
                y_list.append(1.0 if vid == t_vid else 0.0)
                qid_list.append(qr["query_id"])
                grp_list.append(qr["source_group"])
                base_list.append(b_scores[pos])

        ltr_artifact = run_group_kfold_cv(
            np.vstack(x_list).astype(np.float64),
            np.asarray(y_list, dtype=np.float64),
            qid_list,
            grp_list,
            n_splits=5,
            backend="lightgbm",
            base_scores=np.asarray(base_list, dtype=np.float64),
            blend_weight=0.08,
        )
        ltr_path.write_text(json.dumps(ltr_artifact, indent=2), encoding="utf-8")

    # 2. Fit R3 VEDJEJointRerankerNet with retrieval-mined Top-20 hard negatives + False-Negative Filtering
    # Pre-collect caption vectors for false-negative detection on train_sub
    cap_vecs_by_vid: dict[str, np.ndarray] = {}
    for vid, cap_txts in caps_by_vid.items():
        vecs = [
            train_bank.vp_query_map[sha256_text(ct)]
            for ct in cap_txts
            if sha256_text(ct) in train_bank.vp_query_map
        ]
        if vecs:
            cap_vecs_by_vid[vid] = np.vstack(vecs).astype(np.float32)

    # Use up to 2 cached captions per train_sub video (3,200 pairs) with balanced FN filtering (0.92 / 0.96)
    vedje_q_records: list[dict[str, Any]] = []
    for r in sub_rows:
        for cap_idx, cap in enumerate(r["captions"][:2]):
            qtxt = str(cap["caption"]).strip()
            qsha = sha256_text(qtxt)
            if qsha in train_bank.vp_query_map and qsha in train_bank.sig_query_map:
                vedje_q_records.append({
                    "query_id": f"train_{r['video_id']}_c{cap_idx}",
                    "query_text": qtxt,
                    "query_text_sha256": qsha,
                    "target_vid": r["video_id"],
                    "source_group": r["source_group"],
                })
    if not vedje_q_records:
        vedje_q_records = q_records

    vedje_ckpt = models_dir / "r3_vedje_train9k.pt"
    vedje_meta_path = models_dir / "r3_vedje_train9k.meta.json"
    expected_vedje_meta = {
        "train_pairs": len(vedje_q_records),
        "text_sim_thresh": 0.92,
        "vis_sim_thresh": 0.96,
        "epochs": 16,
    }
    reuse_vedje = False
    if vedje_ckpt.is_file() and vedje_meta_path.is_file():
        loaded_meta = json.loads(vedje_meta_path.read_text(encoding="utf-8"))
        if all(loaded_meta.get(k) == v for k, v in expected_vedje_meta.items()):
            reuse_vedje = True
            vedje_res_summary = loaded_meta

    if not reuse_vedje:
        vedje_examples: list[dict[str, Any]] = []
        vedje_sgs: list[str] = []
        for qr in vedje_q_records:
            q_txt = qr["query_text"]
            t_vid = qr["target_vid"]
            vp_s, _ = compute_b2_videoprism_window_scores(train_bank, q_txt, whole_weight=0.65, mode="blend_max")
            sig_s = compute_b0_siglip_scores(train_bank, q_txt, pooling="frame_plus_mean")
            s1_all = fuse_b3_vp_and_siglip(train_bank.video_ids, vp_s, sig_s, mode="zscore", vp_weight=0.72)
            ranked = rank_gallery_desc(train_bank.video_ids, s1_all)
            top20 = ranked[:20]
            if t_vid not in top20:
                top20 = top20[:19] + [t_vid]
            top20_scores = [float(s1_all[train_bank.vid_to_idx[v]]) for v in top20]
            ex = build_top20_vedje_example(train_bank, q_txt, top20, top20_scores, target_vid=t_vid)

            q_vec = ex["vp_q"]
            t_idx = train_bank.vid_to_idx[t_vid]
            pos_v = train_bank.vp_whole_vecs[t_idx] if train_bank.vp_whole_vecs is not None else None
            cand_caps = [cap_vecs_by_vid.get(v, np.zeros((0, 768), dtype=np.float32)) for v in top20]
            is_pos = [v == t_vid for v in top20]
            ex["neg_keep_mask"] = detect_false_negative_mask(
                q_vec, cand_caps, pos_v, ex["vp_whole"], is_pos, text_sim_thresh=0.92, vis_sim_thresh=0.96
            )
            vedje_examples.append(ex)
            vedje_sgs.append(qr["source_group"])

        # Disjoint 85%/15% source-group split inside train_sub for VEDJE early stopping
        uniq_sgs = sorted(set(vedje_sgs))
        rng = np.random.default_rng(seed)
        perm_sgs = rng.permutation(len(uniq_sgs))
        val_sg_set = {uniq_sgs[int(i)] for i in perm_sgs[: max(1, int(0.15 * len(uniq_sgs)))]}
        tr_ex = [ex for ex, sg in zip(vedje_examples, vedje_sgs) if sg not in val_sg_set]
        va_ex = [ex for ex, sg in zip(vedje_examples, vedje_sgs) if sg in val_sg_set]

        vedje_res = train_vedje_reranker(tr_ex, va_ex, checkpoint_out=vedje_ckpt, epochs=16, batch_size=32, lr=8e-4, seed=seed)
        vedje_res_summary = {
            **expected_vedje_meta,
            "checkpoint_sha256": vedje_res.checkpoint_sha256,
            "best_val_hit1": vedje_res.best_val_hit1,
            "false_negatives_masked": vedje_res.false_negatives_masked,
            "total_negatives_seen": vedje_res.total_negatives_seen,
        }
        vedje_meta_path.write_text(json.dumps(vedje_res_summary, indent=2), encoding="utf-8")

    return {
        "train_video_count": len(sub_rows),
        "vedje_train_pairs": len(vedje_q_records),
        "ltr_path": str(ltr_path),
        "ltr_sha256": sha256_file(ltr_path),
        "ltr_cv_metrics": ltr_artifact["metadata"]["cv_hit_at_1"],
        "vedje_ckpt": str(vedje_ckpt),
        "vedje_sha256": str(vedje_res_summary["checkpoint_sha256"]),
        "vedje_val_hit1": float(vedje_res_summary["best_val_hit1"]),
        "vedje_false_negatives_masked": int(vedje_res_summary["false_negatives_masked"]),
        "vedje_total_negatives_seen": int(vedje_res_summary["total_negatives_seen"]),
    }


def _evaluate_split_matrix(
    split_name: str,
    galleries: Sequence[GalleryFeatureBank],
    labels_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    train_vp_bank: np.ndarray | None,
    train_sig_bank: np.ndarray | None,
    ltr_ranker: LearnedRankerLTR | None,
    vedje_ckpt: Path,
    tuned_params: dict[str, Any] | None = None,
    seed: int = 20261122,
    only_configs: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Run B0-B4 and R0-R7 on a split (or locked `only_configs` subset). Returns (results_by_id, best_params, preds)."""
    is_tune = tuned_params is None
    target_set = set(only_configs) if only_configs is not None else None
    if target_set is not None and is_tune:
        raise ValueError("only_configs cannot be used during tune sweep")
    params = dict(tuned_params) if tuned_params is not None else {}

    def _want(cid: str) -> bool:
        return target_set is None or cid in target_set

    # Precompute BM25 & query intents across galleries
    bm25_caps = {b.gallery_name: build_bm25_scorer(b.florence_captions) for b in galleries}
    bm25_ocrs = {b.gallery_name: build_bm25_scorer(b.ocr_docs) for b in galleries}
    bm25_asrs = {b.gallery_name: build_bm25_scorer(b.asr_docs) for b in galleries}
    sa_verifier = SpatialActionVerifier()
    qb_vp = TrainQueryBankHubness(train_vp_bank, k_hub=20, temperature=0.05) if train_vp_bank is not None else None
    qb_sig = TrainQueryBankHubness(train_sig_bank, k_hub=20, temperature=0.05) if train_sig_bank is not None else None

    # Slice definitions built purely from query syntax before scoring
    motion_qids: set[str] = set()
    lex_qids: set[str] = set()
    for b in galleries:
        for q in b.queries:
            qid = q["query_id"]
            if classify_motion_relation_query(q["query_text"])["is_motion_or_relation"]:
                motion_qids.add(qid)
            if classify_modality_query_intent(q["query_text"])["has_any_lexical_cue"]:
                lex_qids.add(qid)

    extra_slices = {
        "motion_or_relation": lambda qid: qid in motion_qids,
        "lexical_intent": lambda qid: qid in lex_qids,
    }

    results: dict[str, Any] = {}
    all_preds: dict[str, list[dict[str, Any]]] = {}

    def _run_full_gallery(cfg_id: str, score_fn: Callable[[GalleryFeatureBank, dict[str, str]], list[str]], b1_hits=None, s1_pools=None, n_boot: int = 10000) -> dict[str, Any]:
        preds: list[dict[str, Any]] = []
        lats: list[float] = []
        for b in galleries:
            for q in b.queries:
                t0 = time.perf_counter()
                ranked = score_fn(b, q)
                lats.append((time.perf_counter() - t0) * 1000.0)
                preds.append({"query_id": q["query_id"], "gallery": b.gallery_name, "ranked_video_ids": ranked[:20]})
        all_preds[cfg_id] = preds
        res = evaluate_predictions(
            cfg_id, preds, labels_by_qid, b1_hits_by_qid=b1_hits,
            stage1_pools_by_qid=s1_pools, latencies_ms=lats, seed=seed, extra_slices=extra_slices,
            n_boot=n_boot,
        )
        results[cfg_id] = res
        return res

    # ---- B1: VideoPrism-LvT-B standalone (main baseline) ----
    b1_all_scores: dict[str, np.ndarray] = {}
    def _b1_fn(b: GalleryFeatureBank, q: dict[str, str]) -> list[str]:
        sc = compute_b1_videoprism_scores(b, q["query_text"])
        b1_all_scores[q["query_id"]] = sc
        return rank_gallery_desc(b.video_ids, sc)

    _run_full_gallery("B1_videoprism_standalone", _b1_fn)
    b1_hits = {
        p["query_id"]: (1 if p["ranked_video_ids"][0] in set(labels_by_qid[p["query_id"]]["target_videos"]) else 0)
        for p in all_preds["B1_videoprism_standalone"]
    }
    results["B1_videoprism_standalone"] = evaluate_predictions(
        "B1_videoprism_standalone",
        all_preds["B1_videoprism_standalone"],
        labels_by_qid,
        b1_hits_by_qid=b1_hits,
        latencies_ms=[results["B1_videoprism_standalone"]["latency_p50_ms"]],
        seed=seed,
        extra_slices=extra_slices,
    )

    # ---- B0: SigLIP2 standalone ----
    if is_tune:
        best_b0_cfg, best_b0_hit = ("frame_plus_mean", 0.65), -1.0
        for pool_m in ("video_mean", "frame_max", "frame_plus_mean"):
            fw = 0.65 if pool_m == "frame_plus_mean" else 0.5
            r = _run_full_gallery(f"B0_siglip2_{pool_m}", lambda b, q, pm=pool_m, f_w=fw: rank_gallery_desc(b.video_ids, compute_b0_siglip_scores(b, q["query_text"], pooling=pm, frame_weight=f_w)), b1_hits=b1_hits, n_boot=64)
            if r["hit_at_1"] > best_b0_hit:
                best_b0_hit = r["hit_at_1"]
                best_b0_cfg = (pool_m, fw)
        params["b0_pooling"] = best_b0_cfg[0]
        params["b0_frame_weight"] = best_b0_cfg[1]
    if _want("B0_siglip2_standalone"):
        _run_full_gallery(
            "B0_siglip2_standalone",
            lambda b, q: rank_gallery_desc(b.video_ids, compute_b0_siglip_scores(b, q["query_text"], pooling=params["b0_pooling"], frame_weight=params["b0_frame_weight"])),
            b1_hits=b1_hits,
        )

    # ---- B2: VideoPrism multi-window pooling ----
    if is_tune:
        best_b2_cfg, best_b2_score = ("blend_max", 0.65), (-1.0, -1.0)
        for wm in ("blend_max", "blend_top2", "max_gate"):
            for ww in (0.45, 0.60, 0.70, 0.80):
                r = _run_full_gallery(
                    f"B2_vp_win_{wm}_{ww}",
                    lambda b, q, m=wm, w=ww: rank_gallery_desc(b.video_ids, compute_b2_videoprism_window_scores(b, q["query_text"], whole_weight=w, mode=m)[0]),
                    b1_hits=b1_hits,
                    n_boot=64,
                )
                key = (r["hit_at_1"], r["stage1_recall_at_20"])
                if key > best_b2_score:
                    best_b2_score = key
                    best_b2_cfg = (wm, ww)
        params["b2_mode"] = best_b2_cfg[0]
        params["b2_whole_weight"] = best_b2_cfg[1]
    need_b3_params = is_tune or target_set is None or any(
        c in target_set for c in ("B3_vp_siglip2_fusion", "R7_imp_qb_csls_stage1", "R7_imp_qb_csls_plus_multiprompt_stage1", "R7_combined_champion_candidate", "R8_multibackbone_stage1_fusion", "R8_multibackbone_qb_csls_mp", "R8_videntra_champion_candidate")
    ) or any(c.startswith(("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7_step", "R7_add")) for c in (target_set or ()))

    # Ensure B0/B2/B3 params exist cleanly even if only_configs skips tuning/B0/B2
    if not is_tune:
        params.setdefault("b0_pooling", "frame_plus_mean")
        params.setdefault("b0_frame_weight", 0.65)
        params.setdefault("b2_mode", "blend_max")
        params.setdefault("b2_whole_weight", 0.65)
        params.setdefault("b3_mode", "zscore")
        params.setdefault("b3_vp_weight", 0.78)
    if _want("B2_videoprism_multiwindow"):
        _run_full_gallery(
            "B2_videoprism_multiwindow",
            lambda b, q: rank_gallery_desc(b.video_ids, compute_b2_videoprism_window_scores(b, q["query_text"], whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])[0]),
            b1_hits=b1_hits,
        )

    # ---- B3: Fusion VideoPrism + SigLIP2 ----
    if is_tune:
        vp_s_cache: dict[tuple[str, str], np.ndarray] = {}
        sig_s_cache: dict[tuple[str, str], np.ndarray] = {}
        for b in galleries:
            for q in b.queries:
                c_key = (b.gallery_name, q["query_id"])
                vp_s_cache[c_key] = compute_b2_videoprism_window_scores(b, q["query_text"], whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])[0]
                sig_s_cache[c_key] = compute_b0_siglip_scores(b, q["query_text"], pooling=params["b0_pooling"], frame_weight=params["b0_frame_weight"])

        best_b3_cfg, best_b3_score = ("zscore", 0.70), (-1.0, -1.0)
        for fm in ("rrf", "zscore", "minmax"):
            for vpw in (0.60, 0.70, 0.78, 0.85):
                def _b3_cand(b: GalleryFeatureBank, q: dict[str, str], m=fm, w=vpw) -> list[str]:
                    c_key = (b.gallery_name, q["query_id"])
                    return rank_gallery_desc(b.video_ids, fuse_b3_vp_and_siglip(b.video_ids, vp_s_cache[c_key], sig_s_cache[c_key], mode=m, vp_weight=w))
                r = _run_full_gallery(f"B3_fusion_{fm}_{vpw}", _b3_cand, b1_hits=b1_hits, n_boot=64)
                key = (r["stage1_recall_at_20"], r["hit_at_1"])
                if key > best_b3_score:
                    best_b3_score = key
                    best_b3_cfg = (fm, vpw)
        params["b3_mode"] = best_b3_cfg[0]
        params["b3_vp_weight"] = best_b3_cfg[1]

    def _b3_best_scores(b: GalleryFeatureBank, q: dict[str, str], use_multiprompt: bool = False, use_hubness: bool = False) -> np.ndarray:
        qt = q["query_text"]
        if use_multiprompt:
            vp_q = compute_multiprompt_query_vector(b.vp_query_map, qt)
            sig_q = compute_multiprompt_query_vector(b.sig_query_map, qt)
            vp_s = (b.vp_whole_vecs @ vp_q).astype(np.float64) if (vp_q is not None and b.vp_whole_vecs is not None) else compute_b1_videoprism_scores(b, qt)
            if vp_q is not None and b.vp_window_vecs is not None and b.vp_win_vid_indices is not None:
                w_all = (b.vp_window_vecs @ vp_q).astype(np.float64)
                w_max = vp_s.copy()
                v_idx = b.vp_win_vid_indices
                valid_w = v_idx >= 0
                np.maximum.at(w_max, v_idx[valid_w], w_all[valid_w])
                vp_s = params["b2_whole_weight"] * vp_s + (1.0 - params["b2_whole_weight"]) * w_max
            if sig_q is not None and b.sig_frame_vecs is not None and b.sig_clip_mean_vecs is not None:
                mean_s = (b.sig_clip_mean_vecs @ sig_q).astype(np.float64)
                f_vecs = b.sig_frame_vecs if b.sig_frame_vecs.dtype == np.float32 else b.sig_frame_vecs.astype(np.float32)
                all_f = (f_vecs @ sig_q).astype(np.float64)
                if b.sig_kf_vid_indices is not None and b.sig_kf_vid_indices.size == all_f.size:
                    max_s = np.full(len(b.video_ids), -1e9, dtype=np.float64)
                    np.maximum.at(max_s, b.sig_kf_vid_indices, all_f)
                    miss = max_s < -1e8
                    if np.any(miss):
                        max_s[miss] = mean_s[miss]
                else:
                    max_s = np.array([float(np.max(all_f[b.sig_video_kf_pos[v]])) if v in b.sig_video_kf_pos else mean_s[i] for i, v in enumerate(b.video_ids)])
                sig_s = params["b0_frame_weight"] * max_s + (1.0 - params["b0_frame_weight"]) * mean_s
            else:
                sig_s = compute_b0_siglip_scores(b, qt, pooling=params["b0_pooling"], frame_weight=params["b0_frame_weight"])
        else:
            vp_s = compute_b2_videoprism_window_scores(b, qt, whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])[0]
            sig_s = compute_b0_siglip_scores(b, qt, pooling=params["b0_pooling"], frame_weight=params["b0_frame_weight"])

        if use_hubness and qb_vp is not None and b.vp_hub_priors is not None:
            vp_s = qb_vp.score_query_against_gallery(np.zeros(1), vp_s, b.vp_hub_priors, csls_weight=0.45)["csls"].astype(np.float64)
        if use_hubness and qb_sig is not None and b.sig_hub_priors is not None:
            sig_s = qb_sig.score_query_against_gallery(np.zeros(1), sig_s, b.sig_hub_priors, csls_weight=0.35)["csls"].astype(np.float64)

        return fuse_b3_vp_and_siglip(b.video_ids, vp_s, sig_s, mode=params["b3_mode"], vp_weight=params["b3_vp_weight"])

    if _want("B3_vp_siglip2_fusion"):
        _run_full_gallery("B3_vp_siglip2_fusion", lambda b, q: rank_gallery_desc(b.video_ids, _b3_best_scores(b, q)), b1_hits=b1_hits)

    # ---- B4: A5 2x2 Matrix (Consensus x LTR fit from train_sub) ----
    want_b4 = [
        (uc, ul, f"B4_a5_cons{'1' if uc else '0'}_ltr{'1' if ul else '0'}")
        for uc in (False, True)
        for ul in (False, True)
        if _want(f"B4_a5_cons{'1' if uc else '0'}_ltr{'1' if ul else '0'}")
    ]
    if want_b4:
        b4_cand_cache: dict[tuple[str, str], tuple[list[str], list[float], np.ndarray]] = {}
        b4_sigfc_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        need_cons_sig = any(uc for uc, _, _ in want_b4)
        for b in galleries:
            for q in b.queries:
                ck = (b.gallery_name, q["query_id"])
                b4_cand_cache[ck] = _extract_b4_a5_candidates(
                    b, q["query_text"], ocr_bm25=bm25_ocrs[b.gallery_name], asr_bm25=bm25_asrs[b.gallery_name], sa_verifier=sa_verifier, top_n=40
                )
                if need_cons_sig:
                    b4_sigfc_cache[ck] = (
                        compute_b0_siglip_scores(b, q["query_text"], pooling="frame_max"),
                        compute_b0_siglip_scores(b, q["query_text"], pooling="video_mean"),
                    )
        for use_cons, use_ltr, cid in want_b4:
            _run_full_gallery(
                cid,
                lambda b, q, uc=use_cons, ul=use_ltr: _score_b4_a5_gallery(
                    b, q["query_text"], ocr_bm25=bm25_ocrs[b.gallery_name],
                    asr_bm25=bm25_asrs[b.gallery_name], sa_verifier=sa_verifier,
                    ltr_ranker=ltr_ranker, use_consensus=uc, use_ltr=ul,
                    cached_cands=b4_cand_cache[(b.gallery_name, q["query_id"])],
                    cached_sig_fc=b4_sigfc_cache.get((b.gallery_name, q["query_id"])),
                ),
                b1_hits=b1_hits,
            )

    need_s1_pools = target_set is None or any(
        c.startswith(("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7_step", "R7_add"))
        for c in target_set
    )
    b1_pools: dict[str, tuple[list[str], list[float], GalleryFeatureBank, dict[str, str]]] = {}
    s1_pools: dict[str, tuple[list[str], list[float], GalleryFeatureBank, dict[str, str]]] = {}
    s1_pool_ids_only: dict[str, list[str]] = {}
    if need_s1_pools:
        for b in galleries:
            for q in b.queries:
                qid = q["query_id"]
                b1_sc = b1_all_scores[qid]
                b1_top20 = rank_gallery_desc(b.video_ids, b1_sc)[:20]
                b1_pools[qid] = (b1_top20, [float(b1_sc[b.vid_to_idx[v]]) for v in b1_top20], b, q)

                s1_sc = _b3_best_scores(b, q)
                s1_top20 = rank_gallery_desc(b.video_ids, s1_sc)[:20]
                s1_pools[qid] = (s1_top20, [float(s1_sc[b.vid_to_idx[v]]) for v in s1_top20], b, q)
                s1_pool_ids_only[qid] = s1_top20

    # ---- R0: Replay alpha=0 parity + SHA-256 verification ----
    def _run_pool_reranker(
        cfg_id: str,
        pools_dict: Mapping[str, tuple[list[str], list[float], GalleryFeatureBank, dict[str, str]]],
        signal_fn: Callable[[GalleryFeatureBank, dict[str, str], Sequence[str], Sequence[float]], Sequence[float]],
        *,
        alpha: float,
        mode: str = "norm_blend",
        allow_pool_change: bool = False,
        n_boot: int = 10000,
    ) -> dict[str, Any]:
        preds: list[dict[str, Any]] = []
        lats: list[float] = []
        pool_hashes: dict[str, str] = {}
        for b in galleries:
            for q in b.queries:
                qid = q["query_id"]
                p_vids, p_scores, bank_ref, q_ref = pools_dict[qid]
                t0 = time.perf_counter()
                if alpha == 0.0 and not allow_pool_change:
                    new_vids = list(p_vids)
                else:
                    sig_vals = signal_fn(bank_ref, q_ref, p_vids, p_scores)
                    new_vids, _ = rerank_fixed_pool(p_vids, p_scores, sig_vals, mode=mode, alpha=alpha)
                lats.append((time.perf_counter() - t0) * 1000.0)
                if not allow_pool_change:
                    assert set(new_vids) == set(p_vids) and len(new_vids) == len(p_vids), f"Membership mutated in {cfg_id}"
                pool_hashes[qid] = sha256_list(new_vids)
                preds.append({"query_id": qid, "gallery": b.gallery_name, "ranked_video_ids": new_vids})

        all_preds[cfg_id] = preds
        res = evaluate_predictions(
            cfg_id, preds, labels_by_qid, b1_hits_by_qid=b1_hits,
            stage1_pools_by_qid=s1_pool_ids_only, latencies_ms=lats, seed=seed, extra_slices=extra_slices,
            n_boot=n_boot,
        )
        res["top20_list_sha256_sample"] = {k: pool_hashes[k] for k in sorted(pool_hashes)[:5]}
        results[cfg_id] = res
        return res

    if _want("R0_replay_alpha0_b1_pool") and _want("R0_replay_alpha0_stage1_pool"):
        r0_b1 = _run_pool_reranker("R0_replay_alpha0_b1_pool", b1_pools, lambda b, q, pv, ps: ps, alpha=0.0)
        r0_s1 = _run_pool_reranker("R0_replay_alpha0_stage1_pool", s1_pools, lambda b, q, pv, ps: ps, alpha=0.0)
        assert r0_b1["hit_at_1"] == results["B1_videoprism_standalone"]["hit_at_1"], "R0 B1 replay parity mismatch!"
        if "B3_vp_siglip2_fusion" in results:
            assert r0_s1["hit_at_1"] == results["B3_vp_siglip2_fusion"]["hit_at_1"], "R0 Stage1 replay parity mismatch!"

    # ---- R1: Whole/Window VideoPrism cosine + Stage-1 prior on Top-20 ----
    # Control sanity: whole-only on B1 pool (alpha=0.25) vs window signal on B1 & Stage-1 pools
    def _vp_whole_sig(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float]) -> list[float]:
        sc = compute_b1_videoprism_scores(b, q["query_text"])
        return [float(sc[b.vid_to_idx[v]]) for v in pv]

    def _vp_win_sig(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float]) -> list[float]:
        sc, _ = compute_b2_videoprism_window_scores(b, q["query_text"], whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])
        return [float(sc[b.vid_to_idx[v]]) for v in pv]

    if _want("R1_control_whole_on_b1_pool"):
        _run_pool_reranker("R1_control_whole_on_b1_pool", b1_pools, _vp_whole_sig, alpha=0.25, mode="norm_blend")
    if is_tune:
        best_r1_cfg, best_r1_hit = ("norm_blend", 0.35), -1.0
        for rm in ("rrf", "norm_blend"):
            for al in (0.15, 0.25, 0.35, 0.50):
                r = _run_pool_reranker(f"R1_win_{rm}_{al}", s1_pools, _vp_win_sig, alpha=al, mode=rm, n_boot=64)
                if (r["hit_at_1"], r["mrr_at_10"]) > (best_r1_hit, -1.0):
                    best_r1_hit = r["hit_at_1"]
                    best_r1_cfg = (rm, al)
        params["r1_mode"] = best_r1_cfg[0]
        params["r1_alpha"] = best_r1_cfg[1]
    if _want("R1_vp_window_rerank") or any(_want(c) for c in ("R7_step0_base_r13", "R7_add_caption", "R7_add_ocr_asr", "R7_add_track_bt")):
        _run_pool_reranker("R1_vp_window_rerank", s1_pools, _vp_win_sig, alpha=params["r1_alpha"], mode=params["r1_mode"])

    # ---- R2: Query-relevant Frame Rerank (max / spaced_top2_pts / X-Pool) ----
    def _make_r2_sig(variant: str, temp: float = 0.07) -> Callable[[GalleryFeatureBank, dict[str, str], Sequence[str], Sequence[float]], list[float]]:
        def _fn(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float]) -> list[float]:
            sig_q = b.sig_query_map.get(sha256_text(q["query_text"]))
            if sig_q is None or b.sig_frame_vecs is None:
                return [0.0] * len(pv)
            out: list[float] = []
            for vid in pv:
                pos = b.sig_video_kf_pos.get(vid)
                if pos is None or len(pos) == 0:
                    out.append(0.0)
                    continue
                f_mat = b.sig_frame_vecs[pos].astype(np.float32)
                if variant == "frame_max":
                    out.append(float(np.max(f_mat @ l2_normalize_np(sig_q))))
                elif variant == "spaced_top2_pts":
                    f_pts = b.sig_frame_pts[pos] if b.sig_frame_pts is not None else np.arange(len(pos), dtype=np.float64)
                    out.append(compute_spaced_top2_frame_score(f_mat, f_pts, sig_q, min_pts_gap=1.5))
                else:
                    out.append(compute_xpool_score(f_mat, sig_q, temperature=temp))
            return out
        return _fn

    if is_tune:
        best_r2_cfg, best_r2_hit = ("xpool", 0.07, 0.25), -1.0
        for var, temp in (("frame_max", 0.07), ("spaced_top2_pts", 0.07), ("xpool", 0.05), ("xpool", 0.07), ("xpool", 0.10)):
            sig_fn_vt = _make_r2_sig(var, temp)
            r2_vt_cache = {
                q["query_id"]: sig_fn_vt(b, q, s1_pools[q["query_id"]][0], s1_pools[q["query_id"]][1])
                for b in galleries for q in b.queries
            }
            for al in (0.15, 0.25, 0.35):
                r = _run_pool_reranker(
                    f"R2_{var}_t{temp}_a{al}",
                    s1_pools,
                    lambda b, q, pv, ps, _c=r2_vt_cache: _c[q["query_id"]],
                    alpha=al,
                    n_boot=64,
                )
                if r["hit_at_1"] > best_r2_hit:
                    best_r2_hit = r["hit_at_1"]
                    best_r2_cfg = (var, temp, al)
        params["r2_variant"], params["r2_temp"], params["r2_alpha"] = best_r2_cfg

    if _want("R2_frame_max"):
        _run_pool_reranker("R2_frame_max", s1_pools, _make_r2_sig("frame_max"), alpha=params["r2_alpha"])
    if _want("R2_frame_spaced_top2"):
        _run_pool_reranker("R2_frame_spaced_top2", s1_pools, _make_r2_sig("spaced_top2_pts"), alpha=params["r2_alpha"])
    if _want("R2_frame_xpool") or any(_want(c) for c in ("R7_step0_base_r13", "R7_add_caption", "R7_add_ocr_asr", "R7_add_track_bt")):
        _run_pool_reranker("R2_frame_xpool", s1_pools, _make_r2_sig("xpool", params["r2_temp"]), alpha=params["r2_alpha"])

    # ---- R3: Lightweight VEDJE Joint Reranker ----
    need_r3 = target_set is None or any(_want(c) for c in ("R3_vedje_joint_reranker", "R7_step0_base_r13", "R7_add_caption", "R7_add_ocr_asr", "R7_add_track_bt"))
    vedje_scores_by_qid: dict[str, np.ndarray] = {}
    if need_r3:
        ordered_qids: list[str] = []
        vedje_inputs: list[dict[str, Any]] = []
        for b in galleries:
            for q in b.queries:
                qid = q["query_id"]
                pv, ps, _, _ = s1_pools[qid]
                ordered_qids.append(qid)
                vedje_inputs.append(build_top20_vedje_example(b, q["query_text"], pv, ps))
        t_v0 = time.perf_counter()
        v_outs = score_vedje_reranker_batch(vedje_ckpt, vedje_inputs)
        vedje_ms_per_q = ((time.perf_counter() - t_v0) * 1000.0) / max(1, len(ordered_qids))
        for qid, sc_arr in zip(ordered_qids, v_outs):
            vedje_scores_by_qid[qid] = sc_arr

        if is_tune:
            best_r3_al, best_r3_hit = 0.50, -1.0
            for al in (0.25, 0.40, 0.55, 0.75, 1.0):
                r = _run_pool_reranker(f"R3_vedje_a{al}", s1_pools, lambda b, q, pv, ps, _al=al: vedje_scores_by_qid[q["query_id"]].tolist(), alpha=al, n_boot=64)
                if r["hit_at_1"] > best_r3_hit:
                    best_r3_hit = r["hit_at_1"]
                    best_r3_al = al
            params["r3_alpha"] = best_r3_al
        r3_res = _run_pool_reranker("R3_vedje_joint_reranker", s1_pools, lambda b, q, pv, ps: vedje_scores_by_qid[q["query_id"]].tolist(), alpha=params["r3_alpha"])
        r3_res["model_inference_ms_per_query"] = round(vedje_ms_per_q, 4)

    # ---- R4: Offline Florence-2 Caption (sparse / dense / fusion + full-gallery Stage-1 recall) ----
    def _cap_sparse_fn(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float]) -> list[float]:
        return bm25_caps[b.gallery_name](q["query_text"], pv).tolist()

    def _cap_dense_fn(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float]) -> list[float]:
        qe = b.e5_query_map.get(sha256_text(q["query_text"]))
        if qe is None or b.cap_e5_vecs is None:
            return [0.0] * len(pv)
        sims = b.cap_e5_vecs @ l2_normalize_np(qe)
        return [float(sims[b.vid_to_idx[v]]) for v in pv]

    def _cap_fusion_fn(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], ps: Sequence[float]) -> list[float]:
        sp = zscore_normalize_row(np.asarray(_cap_sparse_fn(b, q, pv, ps)))
        dn = zscore_normalize_row(np.asarray(_cap_dense_fn(b, q, pv, ps)))
        return (0.40 * sp + 0.60 * dn).tolist()

    if is_tune:
        r4_cap_cache = {
            q["query_id"]: _cap_fusion_fn(b, q, s1_pools[q["query_id"]][0], s1_pools[q["query_id"]][1])
            for b in galleries for q in b.queries
        }
        best_r4_al, best_r4_hit = 0.20, -1.0
        for al in (0.10, 0.15, 0.20, 0.30):
            r = _run_pool_reranker(f"R4_cap_fus_{al}", s1_pools, lambda b, q, pv, ps, _c=r4_cap_cache: _c[q["query_id"]], alpha=al, n_boot=64)
            if r["hit_at_1"] > best_r4_hit:
                best_r4_hit = r["hit_at_1"]
                best_r4_al = al
        params["r4_alpha"] = best_r4_al

    if _want("R4_caption_sparse_bm25"):
        _run_pool_reranker("R4_caption_sparse_bm25", s1_pools, _cap_sparse_fn, alpha=params["r4_alpha"])
    if _want("R4_caption_dense_e5"):
        _run_pool_reranker("R4_caption_dense_e5", s1_pools, _cap_dense_fn, alpha=params["r4_alpha"])
    if _want("R4_caption_sparse_dense_fusion"):
        _run_pool_reranker("R4_caption_sparse_dense_fusion", s1_pools, _cap_fusion_fn, alpha=params["r4_alpha"])

    # Full-gallery Stage-1 retrieval with offline caption branch added
    def _r4_stage1_fn(b: GalleryFeatureBank, q: dict[str, str]) -> list[str]:
        s1 = zscore_normalize_row(_b3_best_scores(b, q))
        qe = b.e5_query_map.get(sha256_text(q["query_text"]))
        c_dn = (b.cap_e5_vecs @ l2_normalize_np(qe)).astype(np.float64) if (qe is not None and b.cap_e5_vecs is not None) else np.zeros(len(b.video_ids))
        c_sp = bm25_caps[b.gallery_name](q["query_text"], b.video_ids)
        c_fus = 0.6 * zscore_normalize_row(c_dn) + 0.4 * zscore_normalize_row(c_sp)
        return rank_gallery_desc(b.video_ids, s1 + 0.18 * c_fus)
    if _want("R4_stage1_plus_caption_recall"):
        _run_full_gallery("R4_stage1_plus_caption_recall", _r4_stage1_fn, b1_hits=b1_hits)

    # ---- R5: OCR / ASR (vision-only, +OCR, +ASR, +OCR+ASR on fixed Top-20 & union K=20) ----
    ocr_cov = sum(1 for b in galleries for v in b.video_ids if b.ocr_docs.get(v, "").strip())
    asr_cov = sum(1 for b in galleries for v in b.video_ids if b.asr_docs.get(v, "").strip())
    total_gal_vids = sum(len(b.video_ids) for b in galleries)

    def _r5_signal(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], _ps: Sequence[float], *, use_ocr: bool, use_asr: bool) -> list[float]:
        o_sc = bm25_ocrs[b.gallery_name](q["query_text"], pv) if use_ocr else np.zeros(len(pv))
        a_sc = bm25_asrs[b.gallery_name](q["query_text"], pv) if use_asr else np.zeros(len(pv))
        # Scale boundedly so zeros do not distort z-score; only boost genuine matches
        return (0.65 * np.log1p(np.maximum(0.0, o_sc)) + 0.45 * np.log1p(np.maximum(0.0, a_sc))).tolist()

    if _want("R5_vision_only_pool"):
        _run_pool_reranker("R5_vision_only_pool", s1_pools, lambda b, q, pv, ps: ps, alpha=0.0)
    if _want("R5_plus_OCR_pool"):
        _run_pool_reranker("R5_plus_OCR_pool", s1_pools, lambda b, q, pv, ps: _r5_signal(b, q, pv, ps, use_ocr=True, use_asr=False), alpha=0.12)
    if _want("R5_plus_ASR_pool"):
        _run_pool_reranker("R5_plus_ASR_pool", s1_pools, lambda b, q, pv, ps: _r5_signal(b, q, pv, ps, use_ocr=False, use_asr=True), alpha=0.12)
    if _want("R5_plus_OCR_ASR_pool"):
        _run_pool_reranker("R5_plus_OCR_ASR_pool", s1_pools, lambda b, q, pv, ps: _r5_signal(b, q, pv, ps, use_ocr=True, use_asr=True), alpha=0.12)

    # Union K=20 variant adding strong lexical hits while keeping K=20
    def _r5_union_fn(b: GalleryFeatureBank, q: dict[str, str]) -> list[str]:
        s1_sc = zscore_normalize_row(_b3_best_scores(b, q))
        o_all = np.log1p(np.maximum(0.0, bm25_ocrs[b.gallery_name](q["query_text"], b.video_ids)))
        a_all = np.log1p(np.maximum(0.0, bm25_asrs[b.gallery_name](q["query_text"], b.video_ids)))
        return rank_gallery_desc(b.video_ids, s1_sc + 0.15 * o_all + 0.10 * a_all)
    if _want("R5_union_OCR_ASR_k20"):
        r5_union = _run_full_gallery("R5_union_OCR_ASR_k20", _r5_union_fn, b1_hits=b1_hits)
        r5_union["modality_coverage"] = {
            "total_gallery_videos": total_gal_vids,
            "ocr_non_empty_videos": ocr_cov,
            "asr_non_empty_videos": asr_cov,
        }

    # ---- R6: Detector crops with/without Kalman tracking ----
    if _want("R6_global_only"):
        _run_pool_reranker("R6_global_only", s1_pools, lambda b, q, pv, ps: ps, alpha=0.0)
    if _want("R6_global_plus_crop"):
        _run_pool_reranker("R6_global_plus_crop", s1_pools, lambda b, q, pv, ps: score_r6_candidates(b, q["query_text"], pv, variant="global_crop").tolist(), alpha=0.08)
    if _want("R6_global_crop_track_sort"):
        _run_pool_reranker("R6_global_crop_track_sort", s1_pools, lambda b, q, pv, ps: score_r6_candidates(b, q["query_text"], pv, variant="global_crop_track_sort").tolist(), alpha=0.08)
    if _want("R6_global_crop_track_bytetrack"):
        _run_pool_reranker("R6_global_crop_track_bytetrack", s1_pools, lambda b, q, pv, ps: score_r6_candidates(b, q["query_text"], pv, variant="global_crop_track_bt").tolist(), alpha=0.08)

    # ---- R7: Best R1-R3 scorer + individual signals & improvements + cumulative combination ----
    need_r7_step_ablations = target_set is None or any(
        c in target_set for c in ("R7_step0_base_r13", "R7_add_caption", "R7_add_ocr_asr", "R7_add_track_bt")
    )
    best_r13_id = "R3_vedje_joint_reranker"
    if need_r7_step_ablations:
        best_r13_id = max(
            ("R1_vp_window_rerank", "R2_frame_xpool", "R3_vedje_joint_reranker"),
            key=lambda cid: (results[cid]["hit_at_1"], results[cid]["mrr_at_10"]),
        )

        def _aligned_pool_blend(
            ps_arr: Sequence[float],
            raw_vals: Sequence[float],
            *,
            mode: str,
            alpha: float,
            rrf_k: float = 60.0,
        ) -> np.ndarray:
            if float(alpha) == 0.0:
                return np.asarray(ps_arr, dtype=np.float64)
            k = len(ps_arr)
            if mode == "rrf":
                r1 = np.arange(1, k + 1, dtype=np.float64)
                s2 = np.asarray(raw_vals, dtype=np.float64)
                ord2 = sorted(range(k), key=lambda i: (-s2[i], i))
                r2 = np.empty(k, dtype=np.float64)
                for rank_pos, idx in enumerate(ord2, 1):
                    r2[idx] = float(rank_pos)
                return (1.0 - float(alpha)) / (rrf_k + r1) + float(alpha) / (rrf_k + r2)
            n1 = zscore_normalize_row(np.asarray(ps_arr, dtype=np.float64))
            n2 = zscore_normalize_row(np.asarray(raw_vals, dtype=np.float64))
            return (1.0 - float(alpha)) * n1 + float(alpha) * n2

        def _base_r13_signal(b: GalleryFeatureBank, q: dict[str, str], pv: Sequence[str], ps: Sequence[float]) -> np.ndarray:
            if best_r13_id == "R3_vedje_joint_reranker":
                raw_v = vedje_scores_by_qid[q["query_id"]]
                return _aligned_pool_blend(ps, raw_v.tolist(), mode="norm_blend", alpha=params["r3_alpha"])
            if best_r13_id == "R2_frame_xpool":
                raw_v = _make_r2_sig("xpool", params["r2_temp"])(b, q, pv, ps)
                return _aligned_pool_blend(ps, raw_v, mode="norm_blend", alpha=params["r2_alpha"])
            raw_v = _vp_win_sig(b, q, pv, ps)
            return _aligned_pool_blend(ps, raw_v, mode=params["r1_mode"], alpha=params["r1_alpha"])

        if _want("R7_step0_base_r13"):
            _run_pool_reranker("R7_step0_base_r13", s1_pools, lambda b, q, pv, ps: _base_r13_signal(b, q, pv, ps).tolist(), alpha=1.0)
        if _want("R7_add_caption"):
            _run_pool_reranker("R7_add_caption", s1_pools, lambda b, q, pv, ps: (zscore_normalize_row(_base_r13_signal(b, q, pv, ps)) + 0.15 * zscore_normalize_row(np.asarray(_cap_fusion_fn(b, q, pv, ps)))).tolist(), alpha=1.0)
        if _want("R7_add_ocr_asr"):
            _run_pool_reranker("R7_add_ocr_asr", s1_pools, lambda b, q, pv, ps: (zscore_normalize_row(_base_r13_signal(b, q, pv, ps)) + 0.10 * np.asarray(_r5_signal(b, q, pv, ps, use_ocr=True, use_asr=True))).tolist(), alpha=1.0)
        if _want("R7_add_track_bt"):
            _run_pool_reranker("R7_add_track_bt", s1_pools, lambda b, q, pv, ps: (zscore_normalize_row(_base_r13_signal(b, q, pv, ps)) + 0.06 * zscore_normalize_row(score_r6_candidates(b, q["query_text"], pv, variant="global_crop_track_bt"))).tolist(), alpha=1.0)

    if _want("R7_imp_qb_csls_stage1"):
        _run_full_gallery("R7_imp_qb_csls_stage1", lambda b, q: rank_gallery_desc(b.video_ids, _b3_best_scores(b, q, use_multiprompt=False, use_hubness=True)), b1_hits=b1_hits)
    if _want("R7_imp_qb_csls_plus_multiprompt_stage1"):
        _run_full_gallery("R7_imp_qb_csls_plus_multiprompt_stage1", lambda b, q: rank_gallery_desc(b.video_ids, _b3_best_scores(b, q, use_multiprompt=True, use_hubness=True)), b1_hits=b1_hits)

    if _want("R7_combined_champion_candidate"):
        qb_mp_pools: dict[str, tuple[list[str], list[float], GalleryFeatureBank, dict[str, str]]] = {}
        for b in galleries:
            for q in b.queries:
                qid = q["query_id"]
                sc_mp = _b3_best_scores(b, q, use_multiprompt=True, use_hubness=True)
                t20_mp = rank_gallery_desc(b.video_ids, sc_mp)[:20]
                qb_mp_pools[qid] = (t20_mp, [float(sc_mp[b.vid_to_idx[v]]) for v in t20_mp], b, q)

        # Score VEDJE + Caption + OCR/ASR + Tracking on the QB-CSLS + MultiPrompt Top-20 pool while retaining Stage-1 prior
        vedje_mp_inputs = [
            build_top20_vedje_example(b, q["query_text"], qb_mp_pools[q["query_id"]][0], qb_mp_pools[q["query_id"]][1], use_multiprompt=True)
            for b in galleries for q in b.queries
        ]
        vedje_mp_outs = score_vedje_reranker_batch(vedje_ckpt, vedje_mp_inputs)
        vedje_mp_map = {q["query_id"]: arr for q, arr in zip([q for b in galleries for q in b.queries], vedje_mp_outs)}

        if is_tune:
            s1_z_map: dict[str, np.ndarray] = {}
            vedje_z_map: dict[str, np.ndarray] = {}
            cap_z_map: dict[str, np.ndarray] = {}
            lex_map: dict[str, np.ndarray] = {}
            trk_z_map: dict[str, np.ndarray] = {}
            for b in galleries:
                for q in b.queries:
                    qid = q["query_id"]
                    pv, ps, _, _ = qb_mp_pools[qid]
                    s1_z_map[qid] = zscore_normalize_row(np.asarray(ps, dtype=np.float64))
                    vedje_z_map[qid] = zscore_normalize_row(vedje_mp_map[qid])
                    cap_z_map[qid] = zscore_normalize_row(np.asarray(_cap_fusion_fn(b, q, pv, ps)))
                    lex_map[qid] = np.asarray(_r5_signal(b, q, pv, ps, use_ocr=True, use_asr=True))
                    trk_z_map[qid] = zscore_normalize_row(score_r6_candidates(b, q["query_text"], pv, variant="global_crop_track_bt"))

            best_r7_cfg, best_r7_hit = (0.45, 0.15, 0.08, 0.05, 0.0), -1.0
            for w_vedje in (0.30, 0.50, 0.80):
                for w_cap in (0.0, 0.15, 0.25):
                    for w_lex in (0.0, 0.08):
                        for w_trk in (0.0, 0.05):
                            for m_gate in (0.0, 1.50):
                                preds_try: list[dict[str, Any]] = []
                                for b in galleries:
                                    for q in b.queries:
                                        qid = q["query_id"]
                                        pv, ps, _, _ = qb_mp_pools[qid]
                                        sc = s1_z_map[qid] + w_vedje * vedje_z_map[qid]
                                        if w_cap > 0.0:
                                            sc = sc + w_cap * cap_z_map[qid]
                                        if w_lex > 0.0:
                                            sc = sc + w_lex * lex_map[qid]
                                        if w_trk > 0.0:
                                            sc = sc + w_trk * trk_z_map[qid]
                                        ranked = apply_confidence_gated_rerank(pv, ps, sc.tolist(), top1_protect_margin=m_gate)
                                        preds_try.append({"query_id": qid, "ranked_video_ids": ranked})
                                ev = evaluate_predictions("tmp", preds_try, labels_by_qid, b1_hits_by_qid=b1_hits, seed=seed, n_boot=32)
                                if (ev["hit_at_1"], ev["mrr_at_10"]) > (best_r7_hit, -1.0):
                                    best_r7_hit = ev["hit_at_1"]
                                    best_r7_cfg = (w_vedje, w_cap, w_lex, w_trk, m_gate)
            params["r7_w_vedje"], params["r7_w_cap"], params["r7_w_lex"], params["r7_w_trk"], params["r7_margin_gate"] = best_r7_cfg

        def _r7_final_query(b: GalleryFeatureBank, q: dict[str, str]) -> list[str]:
            qid = q["query_id"]
            pv, ps, _, _ = qb_mp_pools[qid]
            sc = zscore_normalize_row(np.asarray(ps, dtype=np.float64)) + float(params.get("r7_w_vedje", 0.45)) * zscore_normalize_row(vedje_mp_map[qid])
            if params["r7_w_cap"] > 0.0:
                sc = sc + params["r7_w_cap"] * zscore_normalize_row(np.asarray(_cap_fusion_fn(b, q, pv, ps)))
            if params["r7_w_lex"] > 0.0:
                sc = sc + params["r7_w_lex"] * np.asarray(_r5_signal(b, q, pv, ps, use_ocr=True, use_asr=True))
            if params["r7_w_trk"] > 0.0:
                sc = sc + params["r7_w_trk"] * zscore_normalize_row(score_r6_candidates(b, q["query_text"], pv, variant="global_crop_track_bt"))
            return apply_confidence_gated_rerank(pv, ps, sc.tolist(), top1_protect_margin=params["r7_margin_gate"])

        preds_r7: list[dict[str, Any]] = []
        lats_r7: list[float] = []
        s1_mp_pools_only = {qid: v[0] for qid, v in qb_mp_pools.items()}
        for b in galleries:
            for q in b.queries:
                t0 = time.perf_counter()
                rk = _r7_final_query(b, q)
                lats_r7.append((time.perf_counter() - t0) * 1000.0)
                preds_r7.append({"query_id": q["query_id"], "gallery": b.gallery_name, "ranked_video_ids": rk})
        all_preds["R7_combined_champion_candidate"] = preds_r7
        results["R7_combined_champion_candidate"] = evaluate_predictions(
            "R7_combined_champion_candidate", preds_r7, labels_by_qid,
            b1_hits_by_qid=b1_hits, stage1_pools_by_qid=s1_mp_pools_only,
            latencies_ms=lats_r7, seed=seed, extra_slices=extra_slices,
        )
        results["R7_combined_champion_candidate"]["best_r13_backbone"] = best_r13_id

    # ---- R8: Vectorized Gram-Matrix X-Pool + Softmax-Calibrated Multi-Backbone + Multimodal + VEDJE ----
    has_vl2 = any(b.pe_video_vecs is not None and b.sig400_video_vecs is not None for b in galleries)
    need_r8 = has_vl2 and (target_set is None or any(c.startswith("R8_") for c in target_set))
    if need_r8:
        def _z2d(m: np.ndarray) -> np.ndarray:
            mu = np.mean(m, axis=1, keepdims=True)
            sd = np.maximum(1e-6, np.std(m, axis=1, keepdims=True))
            return (m - mu) / sd

        def _exp_z2d(m: np.ndarray, temp: float) -> np.ndarray:
            return np.exp(np.clip(_z2d(m) / max(0.25, float(temp)), -6.0, 6.0))

        def _select_vl2_mat(mats: Mapping[str, np.ndarray], prefix: str, mode: str, xpool_w: float) -> np.ndarray:
            if mode == "mean":
                return mats[f"{prefix}_mean"]
            if mode == "frame_max":
                return mats[f"{prefix}_max"]
            if mode == "xpool":
                return mats[f"{prefix}_xpool"]
            if mode == "mean_xpool_top3":
                return mats[f"{prefix}_blend"]
            return (1.0 - float(xpool_w)) * mats[f"{prefix}_mean"] + float(xpool_w) * mats[f"{prefix}_xpool"]

        def _eval_fast_matrix_tune(scores_by_gal: Sequence[tuple[GalleryFeatureBank, np.ndarray]]) -> dict[str, Any]:
            preds_try: list[dict[str, Any]] = []
            for b, mat in scores_by_gal:
                top20_idx = np.argsort(-mat, axis=1, kind="mergesort")[:, :20]
                vids_b = b.video_ids
                for q_i, q in enumerate(b.queries):
                    preds_try.append({
                        "query_id": q["query_id"],
                        "ranked_video_ids": [vids_b[int(j)] for j in top20_idx[q_i]],
                    })
            return evaluate_predictions("tmp", preds_try, labels_by_qid, b1_hits_by_qid=b1_hits, seed=seed, n_boot=8)

        # Precompute exact (N_q, N_v) Gram-matrix X-Pool & base visual/lexical matrices once per gallery
        vl2_no_mp = {b.gallery_name: compute_vl2_gallery_score_matrices(b, use_mp=False, temperature=0.05) for b in galleries}
        vl2_yes_mp: dict[str, dict[str, np.ndarray]] = {}

        def _build_aux_mats(use_mp: bool) -> dict[str, dict[str, np.ndarray]]:
            aux: dict[str, dict[str, np.ndarray]] = {}
            for b in galleries:
                nq, nv = len(b.queries), len(b.video_ids)
                m_vp = np.zeros((nq, nv), dtype=np.float64)
                m_sg = np.zeros((nq, nv), dtype=np.float64)
                m_e5 = np.zeros((nq, nv), dtype=np.float64)
                m_cb = np.zeros((nq, nv), dtype=np.float64)
                m_as = np.zeros((nq, nv), dtype=np.float64)
                m_oc = np.zeros((nq, nv), dtype=np.float64)
                for qi, q in enumerate(b.queries):
                    qt = q["query_text"]
                    if use_mp:
                        vp_q = compute_multiprompt_query_vector(b.vp_query_map, qt)
                        vp_w0, _ = compute_b2_videoprism_window_scores(b, qt, whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])
                        m_vp[qi] = 0.70 * vp_w0 + 0.30 * (b.vp_whole_vecs @ vp_q).astype(np.float64) if (vp_q is not None and b.vp_whole_vecs is not None) else vp_w0
                    else:
                        m_vp[qi], _ = compute_b2_videoprism_window_scores(b, qt, whole_weight=params["b2_whole_weight"], mode=params["b2_mode"])
                    m_sg[qi] = compute_b0_siglip_scores(b, qt, pooling=params["b0_pooling"], frame_weight=params["b0_frame_weight"])
                    qe = b.e5_query_map.get(sha256_text(qt))
                    if qe is not None and b.cap_e5_vecs is not None:
                        m_e5[qi] = (b.cap_e5_vecs @ l2_normalize_np(qe)).astype(np.float64)
                    m_cb[qi] = np.log1p(np.maximum(0.0, bm25_caps[b.gallery_name](qt, b.video_ids)))
                    m_as[qi] = np.log1p(np.maximum(0.0, bm25_asrs[b.gallery_name](qt, b.video_ids)))
                    m_oc[qi] = np.log1p(np.maximum(0.0, bm25_ocrs[b.gallery_name](qt, b.video_ids)))
                aux[b.gallery_name] = {"vp": m_vp, "sg": m_sg, "e5": m_e5, "cb": m_cb, "as": m_as, "oc": m_oc}
            return aux

        aux_no_mp = _build_aux_mats(use_mp=False)
        aux_yes_mp: dict[str, dict[str, np.ndarray]] = {}

        if is_tune:
            best_pe_cfg, best_pe_sc = ("mean_xpool_top3", 0.55), (-1.0, -1.0)
            for pm, xw in (("mean", 0.0), ("frame_max", 0.0), ("xpool", 0.0), ("mean_xpool_top3", 0.0), ("mean_plus_xpool", 0.55), ("mean_plus_xpool", 0.70)):
                ev = _eval_fast_matrix_tune([(b, _select_vl2_mat(vl2_no_mp[b.gallery_name], "pe", pm, xw)) for b in galleries])
                key = (ev["hit_at_1"], ev["stage1_recall_at_20"])
                if key > best_pe_sc:
                    best_pe_sc = key
                    best_pe_cfg = (pm, xw)
            params["r8_pe_mode"], params["r8_pe_xpool_w"] = best_pe_cfg

            best_s4_cfg, best_s4_sc = ("mean_xpool_top3", 0.55), (-1.0, -1.0)
            for sm, xw in (("mean", 0.0), ("frame_max", 0.0), ("xpool", 0.0), ("mean_xpool_top3", 0.0), ("mean_plus_xpool", 0.55), ("mean_plus_xpool", 0.70)):
                ev = _eval_fast_matrix_tune([(b, _select_vl2_mat(vl2_no_mp[b.gallery_name], "s4", sm, xw)) for b in galleries])
                key = (ev["hit_at_1"], ev["stage1_recall_at_20"])
                if key > best_s4_sc:
                    best_s4_sc = key
                    best_s4_cfg = (sm, xw)
            params["r8_s4_mode"], params["r8_s4_xpool_w"] = best_s4_cfg
        else:
            params.setdefault("r8_pe_mode", "mean_xpool_top3")
            params.setdefault("r8_pe_xpool_w", 0.55)
            params.setdefault("r8_s4_mode", "mean_xpool_top3")
            params.setdefault("r8_s4_xpool_w", 0.55)
            params.setdefault("r8_s1_weights", (1.10, 1.35, 0.80, 0.18))
            params.setdefault("r8_use_mp", False)
            params.setdefault("r8_boltz_temp", 2.3)
            params.setdefault("r8_csls_weights", (0.30, 0.0, 0.0, 0.0))
            params.setdefault("r8_w_e5", 0.16)
            params.setdefault("r8_w_cap_bm", 0.04)
            params.setdefault("r8_w_asr", 0.08)
            params.setdefault("r8_w_ocr", 0.04)
            params.setdefault("r8_w_vedje", 0.15)
            params.setdefault("r8_margin_gate", 0.0)

        def _eval_full_from_mats(cfg_id: str, mats_by_gal: Sequence[tuple[GalleryFeatureBank, np.ndarray]]) -> dict[str, Any]:
            preds_out: list[dict[str, Any]] = []
            lats_out: list[float] = []
            for b, mat in mats_by_gal:
                t_g0 = time.perf_counter()
                top20_idx = np.argsort(-mat, axis=1, kind="mergesort")[:, :20]
                dt_q = ((time.perf_counter() - t_g0) * 1000.0) / max(1, len(b.queries))
                for qi, q in enumerate(b.queries):
                    preds_out.append({"query_id": q["query_id"], "gallery": b.gallery_name, "ranked_video_ids": [b.video_ids[int(j)] for j in top20_idx[qi]]})
                    lats_out.append(dt_q)
            all_preds[cfg_id] = preds_out
            res = evaluate_predictions(cfg_id, preds_out, labels_by_qid, b1_hits_by_qid=b1_hits, latencies_ms=lats_out, seed=seed, extra_slices=extra_slices)
            results[cfg_id] = res
            return res

        if _want("R8_pecore_standalone"):
            _eval_full_from_mats(
                "R8_pecore_standalone",
                [(b, _select_vl2_mat(vl2_no_mp[b.gallery_name], "pe", params["r8_pe_mode"], params["r8_pe_xpool_w"])) for b in galleries],
            )
        if _want("R8_siglip2_so400m_standalone"):
            _eval_full_from_mats(
                "R8_siglip2_so400m_standalone",
                [(b, _select_vl2_mat(vl2_no_mp[b.gallery_name], "s4", params["r8_s4_mode"], params["r8_s4_xpool_w"])) for b in galleries],
            )

        if is_tune:
            # 1. Tune linear 4-backbone Stage-1 fusion weights cleanly
            z_cache = [
                (
                    b,
                    _z2d(aux_no_mp[b.gallery_name]["vp"]),
                    _z2d(_select_vl2_mat(vl2_no_mp[b.gallery_name], "pe", params["r8_pe_mode"], params["r8_pe_xpool_w"])),
                    _z2d(_select_vl2_mat(vl2_no_mp[b.gallery_name], "s4", params["r8_s4_mode"], params["r8_s4_xpool_w"])),
                    _z2d(aux_no_mp[b.gallery_name]["sg"]),
                )
                for b in galleries
            ]
            best_s1_w, best_s1_sc = (1.10, 1.35, 0.80, 0.18), (-1.0, -1.0)
            for w_vp in (0.95, 1.10, 1.30):
                for w_pe in (1.15, 1.35, 1.55):
                    for w_s4 in (0.70, 0.85, 1.05):
                        for w_sg in (0.12, 0.20):
                            ev = _eval_fast_matrix_tune([(b, w_vp * zvp + w_pe * zpe + w_s4 * zs4 + w_sg * zsg) for b, zvp, zpe, zs4, zsg in z_cache])
                            key = (ev["hit_at_1"], ev["stage1_recall_at_20"])
                            if key > best_s1_sc:
                                best_s1_sc = key
                                best_s1_w = (w_vp, w_pe, w_s4, w_sg)
            params["r8_s1_weights"] = best_s1_w

            # 2. Tune Multi-Prompt + QB-CSLS + Boltzmann Temperature calibration on tune
            vl2_yes_mp = {b.gallery_name: compute_vl2_gallery_score_matrices(b, use_mp=True, temperature=0.05) for b in galleries}
            aux_yes_mp = _build_aux_mats(use_mp=True)
            w_vp, w_pe, w_s4, w_sg = params["r8_s1_weights"]
            best_cal, best_cal_sc = (False, 2.3, (0.30, 0.0, 0.0, 0.0)), (-1.0, -1.0)
            for use_mp_c, v_src, a_src in ((False, vl2_no_mp, aux_no_mp), (True, vl2_yes_mp, aux_yes_mp)):
                for temp_c in (1.6, 2.0, 2.3, 2.6):
                    for c_vp in (0.0, 0.30, 0.45):
                        for c_vl2 in (0.0, 0.25):
                            sc_gals = []
                            for b in galleries:
                                gn = b.gallery_name
                                vp_m = a_src[gn]["vp"]
                                pe_m = _select_vl2_mat(v_src[gn], "pe", params["r8_pe_mode"], params["r8_pe_xpool_w"])
                                s4_m = _select_vl2_mat(v_src[gn], "s4", params["r8_s4_mode"], params["r8_s4_xpool_w"])
                                sg_m = a_src[gn]["sg"]
                                if c_vp > 0.0 and b.vp_hub_priors:
                                    vp_m = vp_m - c_vp * b.vp_hub_priors["mean_topk"][None, :]
                                if c_vl2 > 0.0 and b.pe_hub_priors:
                                    pe_m = pe_m - c_vl2 * b.pe_hub_priors["mean_topk"][None, :]
                                if c_vl2 > 0.0 and b.sig400_hub_priors:
                                    s4_m = s4_m - c_vl2 * b.sig400_hub_priors["mean_topk"][None, :]
                                if c_vp > 0.0 and b.sig_hub_priors:
                                    sg_m = sg_m - 0.20 * b.sig_hub_priors["mean_topk"][None, :]
                                mat = w_vp * _exp_z2d(vp_m, temp_c) + w_pe * _exp_z2d(pe_m, temp_c) + w_s4 * _exp_z2d(s4_m, temp_c) + w_sg * _exp_z2d(sg_m, temp_c)
                                sc_gals.append((b, mat))
                            ev = _eval_fast_matrix_tune(sc_gals)
                            key = (ev["hit_at_1"], ev["mrr_at_10"])
                            if key > best_cal_sc:
                                best_cal_sc = key
                                best_cal = (use_mp_c, temp_c, (c_vp, c_vl2, c_vl2, 0.20 if c_vp > 0 else 0.0))
            params["r8_use_mp"], params["r8_boltz_temp"], params["r8_csls_weights"] = best_cal

        if _want("R8_multibackbone_stage1_fusion"):
            w_vp, w_pe, w_s4, w_sg = params["r8_s1_weights"]
            s1_mats = [
                (
                    b,
                    w_vp * _z2d(aux_no_mp[b.gallery_name]["vp"])
                    + w_pe * _z2d(_select_vl2_mat(vl2_no_mp[b.gallery_name], "pe", params["r8_pe_mode"], params["r8_pe_xpool_w"]))
                    + w_s4 * _z2d(_select_vl2_mat(vl2_no_mp[b.gallery_name], "s4", params["r8_s4_mode"], params["r8_s4_xpool_w"]))
                    + w_sg * _z2d(aux_no_mp[b.gallery_name]["sg"]),
                )
                for b in galleries
            ]
            _eval_full_from_mats("R8_multibackbone_stage1_fusion", s1_mats)

        def _build_r8_visual_boltzmann_mats() -> list[tuple[GalleryFeatureBank, np.ndarray]]:
            use_mp_r8 = bool(params["r8_use_mp"])
            if use_mp_r8 and not vl2_yes_mp:
                vl2_yes_mp.update({b.gallery_name: compute_vl2_gallery_score_matrices(b, use_mp=True, temperature=0.05) for b in galleries})
                aux_yes_mp.update(_build_aux_mats(use_mp=True))
            v_src = vl2_yes_mp if use_mp_r8 else vl2_no_mp
            a_src = aux_yes_mp if use_mp_r8 else aux_no_mp
            w_vp, w_pe, w_s4, w_sg = params["r8_s1_weights"]
            c_vp, c_pe, c_s4, c_sg = params["r8_csls_weights"]
            temp_b = float(params["r8_boltz_temp"])
            out_mats: list[tuple[GalleryFeatureBank, np.ndarray]] = []
            for b in galleries:
                gn = b.gallery_name
                vp_m = a_src[gn]["vp"]
                pe_m = _select_vl2_mat(v_src[gn], "pe", params["r8_pe_mode"], params["r8_pe_xpool_w"])
                s4_m = _select_vl2_mat(v_src[gn], "s4", params["r8_s4_mode"], params["r8_s4_xpool_w"])
                sg_m = a_src[gn]["sg"]
                if c_vp > 0.0 and b.vp_hub_priors:
                    vp_m = vp_m - c_vp * b.vp_hub_priors["mean_topk"][None, :]
                if c_pe > 0.0 and b.pe_hub_priors:
                    pe_m = pe_m - c_pe * b.pe_hub_priors["mean_topk"][None, :]
                if c_s4 > 0.0 and b.sig400_hub_priors:
                    s4_m = s4_m - c_s4 * b.sig400_hub_priors["mean_topk"][None, :]
                if c_sg > 0.0 and b.sig_hub_priors:
                    sg_m = sg_m - c_sg * b.sig_hub_priors["mean_topk"][None, :]
                mat = w_vp * _exp_z2d(vp_m, temp_b) + w_pe * _exp_z2d(pe_m, temp_b) + w_s4 * _exp_z2d(s4_m, temp_b) + w_sg * _exp_z2d(sg_m, temp_b)
                out_mats.append((b, mat))
            return out_mats

        if _want("R8_multibackbone_qb_csls_mp") or _want("R8_videntra_champion_candidate"):
            vis_boltz_mats = _build_r8_visual_boltzmann_mats()
        if _want("R8_multibackbone_qb_csls_mp"):
            _eval_full_from_mats("R8_multibackbone_qb_csls_mp", vis_boltz_mats)

        if _want("R8_videntra_champion_candidate"):
            temp_b = float(params["r8_boltz_temp"])
            if is_tune:
                best_mm, best_mm_sc = (0.16, 0.04, 0.08, 0.04), (-1.0, -1.0)
                for w_e5 in (0.0, 0.14, 0.20):
                    for w_cb in (0.0, 0.04, 0.08):
                        for w_as in (0.0, 0.08, 0.12):
                            for w_oc in (0.0, 0.04):
                                sc_gals = [
                                    (
                                        b,
                                        v_mat
                                        + w_e5 * _exp_z2d(aux_no_mp[b.gallery_name]["e5"], temp_b)
                                        + w_cb * aux_no_mp[b.gallery_name]["cb"]
                                        + w_as * aux_no_mp[b.gallery_name]["as"]
                                        + w_oc * aux_no_mp[b.gallery_name]["oc"],
                                    )
                                    for b, v_mat in vis_boltz_mats
                                ]
                                ev = _eval_fast_matrix_tune(sc_gals)
                                key = (ev["hit_at_1"], ev["stage1_recall_at_20"])
                                if key > best_mm_sc:
                                    best_mm_sc = key
                                    best_mm = (w_e5, w_cb, w_as, w_oc)
                params["r8_w_e5"], params["r8_w_cap_bm"], params["r8_w_asr"], params["r8_w_ocr"] = best_mm

            mm_mats = [
                (
                    b,
                    v_mat
                    + float(params["r8_w_e5"]) * _exp_z2d(aux_no_mp[b.gallery_name]["e5"], temp_b)
                    + float(params["r8_w_cap_bm"]) * aux_no_mp[b.gallery_name]["cb"]
                    + float(params["r8_w_asr"]) * aux_no_mp[b.gallery_name]["as"]
                    + float(params["r8_w_ocr"]) * aux_no_mp[b.gallery_name]["oc"],
                )
                for b, v_mat in vis_boltz_mats
            ]
            r8_pools: dict[str, tuple[list[str], list[float], GalleryFeatureBank, dict[str, str]]] = {}
            for b, mat in mm_mats:
                top20_idx = np.argsort(-mat, axis=1, kind="mergesort")[:, :20]
                for qi, q in enumerate(b.queries):
                    t20 = [b.video_ids[int(j)] for j in top20_idx[qi]]
                    s20 = [float(mat[qi, int(j)]) for j in top20_idx[qi]]
                    r8_pools[q["query_id"]] = (t20, s20, b, q)

            use_mp_r8 = bool(params["r8_use_mp"])
            vedje_r8_inputs = [
                build_top20_vedje_example(b, q["query_text"], r8_pools[q["query_id"]][0], r8_pools[q["query_id"]][1], use_multiprompt=use_mp_r8)
                for b in galleries for q in b.queries
            ]
            vedje_r8_outs = score_vedje_reranker_batch(vedje_ckpt, vedje_r8_inputs)
            vedje_r8_map = {q["query_id"]: arr for q, arr in zip([q for b in galleries for q in b.queries], vedje_r8_outs)}

            if is_tune:
                best_v_cfg, best_v_sc = (0.12, 0.0), (-1.0, -1.0)
                for w_vd in (0.0, 0.08, 0.15, 0.25):
                    for m_g in (0.0, 1.50):
                        preds_try: list[dict[str, Any]] = []
                        for b in galleries:
                            for q in b.queries:
                                qid = q["query_id"]
                                pv, ps, _, _ = r8_pools[qid]
                                sc = zscore_normalize_row(np.asarray(ps, dtype=np.float64))
                                if w_vd > 0.0:
                                    sc = sc + w_vd * zscore_normalize_row(vedje_r8_map[qid])
                                preds_try.append({"query_id": qid, "ranked_video_ids": apply_confidence_gated_rerank(pv, ps, sc.tolist(), top1_protect_margin=m_g)})
                        ev = evaluate_predictions("tmp", preds_try, labels_by_qid, b1_hits_by_qid=b1_hits, seed=seed, n_boot=8)
                        key = (ev["hit_at_1"], ev["mrr_at_10"])
                        if key > best_v_sc:
                            best_v_sc = key
                            best_v_cfg = (w_vd, m_g)
                params["r8_w_vedje"], params["r8_margin_gate"] = best_v_cfg

            preds_r8: list[dict[str, Any]] = []
            lats_r8: list[float] = []
            r8_pools_only = {qid: v[0] for qid, v in r8_pools.items()}
            w_vd_final = float(params.get("r8_w_vedje", 0.0))
            m_gate_final = float(params.get("r8_margin_gate", 0.0))
            for b in galleries:
                for q in b.queries:
                    t0 = time.perf_counter()
                    qid = q["query_id"]
                    pv, ps, _, _ = r8_pools[qid]
                    sc = zscore_normalize_row(np.asarray(ps, dtype=np.float64))
                    if w_vd_final > 0.0:
                        sc = sc + w_vd_final * zscore_normalize_row(vedje_r8_map[qid])
                    rk = apply_confidence_gated_rerank(pv, ps, sc.tolist(), top1_protect_margin=m_gate_final)
                    lats_r8.append((time.perf_counter() - t0) * 1000.0)
                    preds_r8.append({"query_id": qid, "gallery": b.gallery_name, "ranked_video_ids": rk})
            all_preds["R8_videntra_champion_candidate"] = preds_r8
            results["R8_videntra_champion_candidate"] = evaluate_predictions(
                "R8_videntra_champion_candidate", preds_r8, labels_by_qid,
                b1_hits_by_qid=b1_hits, stage1_pools_by_qid=r8_pools_only,
                latencies_ms=lats_r8, seed=seed, extra_slices=extra_slices,
            )

    return results, params, all_preds


def _format_matrix_markdown(split_label: str, table: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        f"## Validation Matrix ({split_label})\n\n",
        "| ID | Hits / N | Hit@1 | R@5 | R@10 | MRR@10 | Stage-1 R@20 | Rescued/Lost vs B1 | Delta (pp) | 95% CI | p50/p95 ms |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    ]
    for cid, r in table.items():
        ci = r["ci95_pp_vs_b1"]
        lines.append(
            f"| `{cid}` | {r['hits_at_1']}/{r['n_queries']} | {r['hit_at_1']*100:.2f}% | "
            f"{r['hit_at_5']*100:.2f}% | {r['hit_at_10']*100:.2f}% | {r['mrr_at_10']:.4f} | "
            f"{r['stage1_recall_at_20']*100:.2f}% | +{r['rescued_vs_b1']}/-{r['lost_vs_b1']} | "
            f"{r['delta_pp_vs_b1']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{r['latency_p50_ms']:.2f}/{r['latency_p95_ms']:.2f} |\n"
        )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MSR-VTT validation matrix (tune -> confirm) and lock Champion.")
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROTOCOL_ROOT / "matrix_run")
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    t_start = time.perf_counter()
    p_root = args.protocol_root
    k_out = p_root / "kaggle_out"
    m_dir = p_root / "manifests"
    e_dir = p_root / "eval_only"

    train_vp_bank, train_sig_bank = _load_train_query_banks(k_out / "vp_train_sub", k_out / "mm_train_sub")
    train_pe_bank, train_sig400_bank = _load_vl2_train_query_banks(k_out / "vl2_train_sub")
    train_summary = train_b4_ltr_and_r3_vedje(
        p_root, train_vp_bank=train_vp_bank, train_sig_bank=train_sig_bank,
        models_dir=p_root / "models", seed=args.seed,
    )
    ltr_ranker = LearnedRankerLTR(model_path=Path(train_summary["ltr_path"]))
    vedje_ckpt = Path(train_summary["vedje_ckpt"])

    def _load_split_galleries(names: Sequence[str], vp_sub: str, mm_sub: str, vl2_sub: str) -> tuple[list[GalleryFeatureBank], dict[str, dict[str, Any]]]:
        banks: list[GalleryFeatureBank] = []
        labels: dict[str, dict[str, Any]] = {}
        vl2_dir = k_out / vl2_sub
        for gname in names:
            bank = load_gallery_feature_bank(
                gallery_name=gname,
                gallery_manifest_path=m_dir / gname / "gallery_manifest.jsonl",
                queries_path=m_dir / gname / "queries.jsonl",
                vp_dirs=[k_out / vp_sub],
                mm_dirs=[k_out / mm_sub],
                asr_dirs=[k_out / "asr_val_4000"],
                cap_dirs=[p for p in (k_out / "cap_val_4000", k_out / "cap_val_tune", k_out / "cap_val_confirm") if p.exists()],
                vl2_dirs=[vl2_dir] if vl2_dir.exists() else (),
                train_vp_bank=train_vp_bank,
                train_sig_bank=train_sig_bank,
                train_pe_bank=train_pe_bank,
                train_sig400_bank=train_sig400_bank,
            )
            banks.append(bank)
            labels.update(_load_labels(e_dir / f"{gname}_labels.jsonl"))
        return banks, labels

    tune_banks, tune_labels = _load_split_galleries(("tune_g1", "tune_g2"), "vp_val_tune", "mm_val_tune", "vl2_val_tune")
    confirm_banks, confirm_labels = _load_split_galleries(("confirm_g1", "confirm_g2"), "vp_val_confirm", "mm_val_confirm", "vl2_val_confirm")

    tune_results, tuned_params, tune_preds = _evaluate_split_matrix(
        "tune", tune_banks, tune_labels,
        train_vp_bank=train_vp_bank, train_sig_bank=train_sig_bank,
        ltr_ranker=ltr_ranker, vedje_ckpt=vedje_ckpt, tuned_params=None, seed=args.seed,
    )
    confirm_results, _, confirm_preds = _evaluate_split_matrix(
        "confirm", confirm_banks, confirm_labels,
        train_vp_bank=train_vp_bank, train_sig_bank=train_sig_bank,
        ltr_ranker=ltr_ranker, vedje_ckpt=vedje_ckpt, tuned_params=tuned_params, seed=args.seed,
    )

    # Select single Champion strictly by highest Hit@1 on confirm, tie-break MRR@10 then lower p95
    eligible_ids = [
        cid for cid in confirm_results
        if not cid.startswith(("R0_", "B0_siglip2_", "B2_vp_win_", "B3_fusion_", "R1_win_", "R2_xpool", "R2_frame_max_t", "R2_spaced", "R3_vedje_a", "R4_cap_fus_", "R8_pe_", "R8_s4_"))
        and "control" not in cid
    ]
    # Include the canonical rows cleanly
    canonical_ids = [
        "B0_siglip2_standalone",
        "B1_videoprism_standalone",
        "B2_videoprism_multiwindow",
        "B3_vp_siglip2_fusion",
        "B4_a5_cons0_ltr0", "B4_a5_cons0_ltr1", "B4_a5_cons1_ltr0", "B4_a5_cons1_ltr1",
        "R1_vp_window_rerank",
        "R2_frame_max", "R2_frame_spaced_top2", "R2_frame_xpool",
        "R3_vedje_joint_reranker",
        "R4_caption_sparse_bm25", "R4_caption_dense_e5", "R4_caption_sparse_dense_fusion", "R4_stage1_plus_caption_recall",
        "R5_vision_only_pool", "R5_plus_OCR_pool", "R5_plus_ASR_pool", "R5_plus_OCR_ASR_pool", "R5_union_OCR_ASR_k20",
        "R6_global_only", "R6_global_plus_crop", "R6_global_crop_track_sort", "R6_global_crop_track_bytetrack",
        "R7_step0_base_r13", "R7_add_caption", "R7_add_ocr_asr", "R7_add_track_bt",
        "R7_imp_qb_csls_stage1", "R7_imp_qb_csls_plus_multiprompt_stage1", "R7_combined_champion_candidate",
        "R8_pecore_standalone", "R8_siglip2_so400m_standalone",
        "R8_multibackbone_stage1_fusion", "R8_multibackbone_qb_csls_mp", "R8_videntra_champion_candidate",
    ]
    cand_pool_ids = [cid for cid in canonical_ids if cid in confirm_results] or eligible_ids
    champion_id = max(
        cand_pool_ids,
        key=lambda cid: (
            confirm_results[cid]["hits_at_1"],
            confirm_results[cid]["mrr_at_10"],
            -confirm_results[cid]["latency_p95_ms"],
        ),
    )

    rss_mb, peak_mb = get_process_memory_mb()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
    except Exception:
        git_commit = "unknown"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cid, plist in confirm_preds.items():
        if cid in cand_pool_ids or cid == champion_id:
            p_out = args.out_dir / "confirm_predictions" / f"{cid}.jsonl"
            p_out.parent.mkdir(parents=True, exist_ok=True)
            with p_out.open("w", encoding="utf-8") as fh:
                for row in plist:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lock_payload = {
        "schema_version": 1,
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "champion_id": champion_id,
        "champion_confirm_metrics": confirm_results[champion_id],
        "b1_confirm_metrics": confirm_results["B1_videoprism_standalone"],
        "b0_confirm_metrics": confirm_results["B0_siglip2_standalone"],
        "tuned_hyperparameters": tuned_params,
        "config_sha256": hashlib.sha256(
            json.dumps({"champion_id": champion_id, "params": tuned_params}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "code_sha256": compute_code_sha256([Path(__file__)]),
        "git_commit": git_commit,
        "train_models": train_summary,
        "runtime_memory": {"rss_mb": rss_mb, "peak_ram_mb": peak_mb, "elapsed_seconds": round(time.perf_counter() - t_start, 2)},
    }
    lock_path = p_root / "champion_lock_receipt.json"
    lock_path.write_text(json.dumps(lock_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "champion_lock": lock_payload,
        "tune_results": {k: tune_results[k] for k in tune_results if k in cand_pool_ids or k.startswith("R0_")},
        "confirm_results": {k: confirm_results[k] for k in confirm_results if k in cand_pool_ids or k.startswith("R0_")},
    }
    (args.out_dir / "validation_matrix_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_report = (
        f"# MSR-VTT 1K-A Validation Matrix Report\n\n"
        f"- **Selected Champion:** `{champion_id}` (Confirm Hit@1 = **{confirm_results[champion_id]['hits_at_1']}/{confirm_results[champion_id]['n_queries']} = {confirm_results[champion_id]['hit_at_1']*100:.2f}%**)\n"
        f"- **Peak RAM:** {peak_mb} MB | **Git:** `{git_commit}`\n\n"
        + _format_matrix_markdown("Tune (2,000 queries)", summary["tune_results"])
        + "\n"
        + _format_matrix_markdown("Confirm (2,000 queries)", summary["confirm_results"])
    )
    (args.out_dir / "validation_matrix_report.md").write_text(md_report, encoding="utf-8")
    print(md_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
