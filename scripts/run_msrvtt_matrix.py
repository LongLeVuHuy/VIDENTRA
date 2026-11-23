"""VIDENTRA v2 (R9) Research & Evaluation Matrix on MSR-VTT 3-Split Protocol.

Strictly enforces the Zero Leakage Rule:
- Training & Hubness Priors: `train_9k_clean` ONLY (`train_sub` + expanded query bank).
- Hyperparameter & Threshold Tuning: `Tune` (`tune_g1` + `tune_g2`, 2,000 videos) ONLY.
- Confirmation & Champion Lock: `Confirm` (`confirm_g1` + `confirm_g2`, 2,000 videos) evaluated ONCE.
- Sealed Final Benchmark: `test_1ka` (1,000 videos) evaluated SINGLE-SHOT after lock.

Implements V2 inductive Hit@1 upgrades:
1. 6 heterogeneous VL/spatio-temporal backbones (PE-Core-bigG-14-448, DFN5B-H-14-378,
   PE-Core-L-14-336, SigLIP2-SO400M-384, VideoPrism-LvT-B, SigLIP2-Base-384).
2. Multi-prompt query ensembling + Query-Bank Inductive Hubness Calibration
   (QB-DIS Dynamic Inverted Softmax + QB-Z-Norm + QB-CSLS) strictly vs train_9k_clean.
3. MultiBackboneCrossModalAdapter trained on train_9k_clean with false-negative masking.
4. Margin-Gated Stage-2 VLM Cross-Encoder Reranker (Qwen2.5-VL-7B/3B on Kaggle Dual-T4).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p_sys in (ROOT, SRC):
    if str(p_sys) not in sys.path:
        sys.path.insert(0, str(p_sys))

from scripts.prepare_clip_vp_kaggle_bundle import assert_no_gt_leakage
from videntra.evaluation.provenance import sha256_file
from videntra.ranking.cross_modal_adapter import (
    adapt_vectors_with_checkpoint,
    train_cross_modal_adapter,
)
from videntra.ranking.vedje_reranker import (
    TrainQueryBankHubness,
    apply_qb_dis_matrix,
    apply_qb_znorm_matrix,
    l2_normalize_np,
)
from videntra.ranking.vlm_reranker import (
    blend_stage1_and_vlm_predictions,
    combine_stage1_and_vlm_pool,
    debias_vlm_scores,
)
from videntra.retrieval.msrvtt_matrix import (
    GalleryFeatureBank,
    apply_hubness_csls,
    build_bm25_scorer,
    compute_b0_siglip_scores,
    compute_b1_videoprism_scores,
    compute_multiprompt_query_vector,
    compute_vl2_gallery_score_matrices,
    compute_vl3_gallery_score_matrices,
    evaluate_predictions,
    load_gallery_feature_bank,
    rank_gallery_desc,
    sha256_text,
)

PROTOCOL_ROOT = ROOT / "work/msrvtt_protocol_20261122"
KAGGLE_OUT = PROTOCOL_ROOT / "kaggle_out"
MODELS_DIR = PROTOCOL_ROOT / "models"
V2_RUN_DIR = PROTOCOL_ROOT / "v2_run"
BENCHMARK_DIR = ROOT / "benchmark"

VAL_TUNE_DIR = ROOT / "validation/msrvtt/valid_tuning"
VAL_CONFIRM_DIR = ROOT / "validation/msrvtt/valid_confirm"
TEST_1KA_DIR = ROOT / "test_queries/msrvtt/test_1ka_benchmark"

ALL_BACKBONES = ("pe_bigg", "dfn", "pe", "s4", "vp", "sig")
SEED = 20261123


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _find_shard_dir(raw_dir: Path) -> Path:
    if (raw_dir / "receipt.json").is_file():
        return raw_dir
    sub = raw_dir / raw_dir.name
    if (sub / "receipt.json").is_file():
        return sub
    for r in sorted(raw_dir.rglob("receipt.json")):
        return r.parent
    raise FileNotFoundError(f"receipt.json not found under {raw_dir}")


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.size == 0:
        return arr
    mu = np.mean(arr, axis=1, keepdims=True)
    sd = np.maximum(1e-9, np.std(arr, axis=1, keepdims=True))
    return (arr - mu) / sd


def _softmax_rows(mat: np.ndarray, temp: float = 0.05) -> np.ndarray:
    z = _zscore_rows(mat) / max(1e-4, float(temp))
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(1e-12, np.sum(e, axis=1, keepdims=True))


def _load_query_dict(dirs: Sequence[Path], npy_name: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for d in dirs:
        if not d.exists():
            continue
        sd = _find_shard_dir(d)
        qp = sd / npy_name
        rp = sd / "query_rows.jsonl"
        if not (qp.is_file() and rp.is_file()):
            continue
        qvecs = np.load(str(qp), mmap_mode="r")
        qrows = _read_jsonl(rp)
        for row, vec in zip(qrows, qvecs):
            qsha = str(row["query_text_sha256"])
            if qsha not in out:
                out[qsha] = l2_normalize_np(np.asarray(vec, dtype=np.float32))
    return out


def _collect_existing(paths: Sequence[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def load_all_train_query_banks(k_out: Path = KAGGLE_OUT) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Load deduplicated `train_9k_clean` query banks for all 6 backbones."""
    specs: dict[str, tuple[list[Path], str]] = {
        "vp": (_collect_existing([*sorted(k_out.glob("vp_train_sub*")), *sorted(k_out.glob("vp_train_9k_qbank*"))]), "query_vectors.npy"),
        "sig": (_collect_existing([*sorted(k_out.glob("mm_train_sub*")), *sorted(k_out.glob("mm_train_9k_qbank*"))]), "siglip_query_vectors.npy"),
        "pe": (_collect_existing([*sorted(k_out.glob("vl2_train_sub*")), *sorted(k_out.glob("vl2_train_9k_qbank*"))]), "pe_query_vectors.npy"),
        "s4": (_collect_existing([*sorted(k_out.glob("vl2_train_sub*")), *sorted(k_out.glob("vl2_train_9k_qbank*"))]), "sig400_query_vectors.npy"),
        "pe_bigg": (_collect_existing([*sorted(k_out.glob("vl3_train_sub*")), *sorted(k_out.glob("vl3_train_9k_qbank*"))]), "pe_bigg_query_vectors.npy"),
        "dfn": (_collect_existing([*sorted(k_out.glob("vl3_train_sub*")), *sorted(k_out.glob("vl3_train_9k_qbank*"))]), "dfn_query_vectors.npy"),
    }
    banks: dict[str, np.ndarray] = {}
    dicts: dict[str, dict[str, np.ndarray]] = {}
    for bname, (dirs, npy_name) in specs.items():
        qmap = _load_query_dict(dirs, npy_name)
        dicts[bname] = qmap
        if qmap:
            banks[bname] = np.vstack(list(qmap.values())).astype(np.float32)
    return banks, dicts


def _load_train_video_vectors(k_out: Path = KAGGLE_OUT) -> dict[str, dict[str, np.ndarray]]:
    """Load train_sub L2-normalized video vectors for all 6 backbones."""
    out: dict[str, dict[str, np.ndarray]] = {b: {} for b in ALL_BACKBONES}

    # VideoPrism
    vp_dir = _find_shard_dir(k_out / "vp_train_sub")
    vp_vecs = np.load(str(vp_dir / "video_vectors.npy"), mmap_mode="r")
    for row, vec in zip(_read_jsonl(vp_dir / "video_rows.jsonl"), vp_vecs):
        out["vp"][str(row["video_id"])] = l2_normalize_np(np.asarray(vec, dtype=np.float32))

    # SigLIP2-Base (mean over keyframes)
    mm_dir = _find_shard_dir(k_out / "mm_train_sub")
    kf_emb = np.load(str(mm_dir / "keyframe_embeddings_fp16.npy"), mmap_mode="r")
    by_vid: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in _read_jsonl(mm_dir / "keyframes_manifest.jsonl.gz"):
        by_vid[str(row["video_id"])].append(np.asarray(kf_emb[int(row["faiss_id"])], dtype=np.float32))
    for vid, lst in by_vid.items():
        out["sig"][vid] = l2_normalize_np(np.mean(np.vstack(lst), axis=0))

    # VL2 & VL3 dual shards (single shard or multi-slice p0..pN)
    dual_specs = [
        ("vl2_train_sub*", ("pe", "s4"), ("pe_video_vectors.npy", "sig400_video_vectors.npy")),
        ("vl3_train_sub*", ("pe_bigg", "dfn"), ("pe_bigg_video_vectors.npy", "dfn_video_vectors.npy")),
    ]
    for pattern, (ba, bb), (fa, fb) in dual_specs:
        for raw_d in sorted(k_out.glob(pattern)):
            if not raw_d.is_dir():
                continue
            sd = _find_shard_dir(raw_d)
            rows = _read_jsonl(sd / "video_rows.jsonl")
            va = np.load(str(sd / fa), mmap_mode="r")
            vb = np.load(str(sd / fb), mmap_mode="r")
            for r, a, b in zip(rows, va, vb):
                vid = str(r["video_id"])
                out[ba][vid] = l2_normalize_np(np.asarray(a, dtype=np.float32))
                out[bb][vid] = l2_normalize_np(np.asarray(b, dtype=np.float32))
    return out


def ensure_cross_modal_adapter(
    protocol_root: Path = PROTOCOL_ROOT,
    k_out: Path = KAGGLE_OUT,
    *,
    retrain: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Train or reuse MultiBackboneCrossModalAdapter strictly on `train_9k_clean`."""
    ckpt_path = MODELS_DIR / "v2_cross_modal_adapter.pt"
    meta_path = MODELS_DIR / "v2_cross_modal_adapter.meta.json"
    if ckpt_path.is_file() and meta_path.is_file() and not retrain:
        return ckpt_path, json.loads(meta_path.read_text(encoding="utf-8"))

    _, q_dicts = load_all_train_query_banks(k_out)
    v_dicts = _load_train_video_vectors(k_out)
    active_b = [b for b in ALL_BACKBONES if v_dicts[b] and q_dicts[b]]
    assert len(active_b) >= 4, f"Too few active backbones for adapter training: {active_b}"

    train_rows = _read_jsonl(protocol_root / "manifests/train_9k_clean/train_videos.jsonl")
    vid_sets = [set(v_dicts[b].keys()) for b in active_b]
    common_vids = set.intersection(*vid_sets)

    selected_videos = [r for r in train_rows if str(r["video_id"]) in common_vids]
    vid_idx_map = {str(r["video_id"]): i for i, r in enumerate(selected_videos)}
    vid_sgs = [str(r["source_group"]) for r in selected_videos]

    train_v_feats = {
        b: np.vstack([v_dicts[b][str(r["video_id"])] for r in selected_videos]).astype(np.float32)
        for b in active_b
    }

    pair_v_idx: list[int] = []
    pair_sgs: list[str] = []
    pair_q_shas: list[str] = []
    for r in selected_videos:
        vid = str(r["video_id"])
        sg = str(r["source_group"])
        v_i = vid_idx_map[vid]
        for cap in r.get("captions") or ():
            qsha = sha256_text(str(cap.get("caption") or ""))
            if all(qsha in q_dicts[b] for b in active_b):
                pair_v_idx.append(v_i)
                pair_sgs.append(sg)
                pair_q_shas.append(qsha)

    train_q_feats = {
        b: np.vstack([q_dicts[b][qs] for qs in pair_q_shas]).astype(np.float32)
        for b in active_b
    }
    backbone_dims = {b: int(train_v_feats[b].shape[1]) for b in active_b}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = train_cross_modal_adapter(
        backbone_dims=backbone_dims,
        train_video_feats=train_v_feats,
        train_query_feats=train_q_feats,
        pair_video_idx=np.asarray(pair_v_idx, dtype=np.int64),
        pair_source_groups=pair_sgs,
        video_source_groups=vid_sgs,
        checkpoint_out=ckpt_path,
        joint_dim=512,
        hidden_dim=256,
        epochs=24,
        batch_size=256,
        lr=4e-4,
        weight_decay=0.04,
        fn_cos_threshold=0.88,
        seed=SEED,
        device=device,
    )
    meta = {
        "backbones": active_b,
        "backbone_dims": backbone_dims,
        "epochs_ran": res.epochs_ran,
        "best_val_hit1": res.best_val_hit1,
        "train_loss_final": res.train_loss_final,
        "train_videos": res.train_videos,
        "train_pairs": res.train_pairs,
        "val_videos": res.val_videos,
        "checkpoint_sha256": res.checkpoint_sha256,
    }
    _write_json(meta_path, meta)
    return ckpt_path, meta


def _load_split_galleries(
    split: str,
    *,
    train_banks: Mapping[str, np.ndarray],
    k_out: Path = KAGGLE_OUT,
) -> tuple[list[GalleryFeatureBank], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load Tune, Confirm, or sealed Test-1KA galleries with all 6 train query banks."""
    if split == "tune":
        sub_names = ["tune_g1", "tune_g2"]
        base_dir = VAL_TUNE_DIR
        q_file_tpl = "{g}_queries.jsonl"
        dir_cfg = {
            "vp": [k_out / "vp_val_tune"],
            "mm": [k_out / "mm_val_tune"],
            "asr": [k_out / "asr_val_4000"],
            "cap": [k_out / "cap_val_tune"],
            "vl2": [k_out / "vl2_val_tune"],
            "vl3": [k_out / "vl3_val_tune"],
        }
    elif split == "confirm":
        sub_names = ["confirm_g1", "confirm_g2"]
        base_dir = VAL_CONFIRM_DIR
        q_file_tpl = "{g}_queries.jsonl"
        dir_cfg = {
            "vp": [k_out / "vp_val_confirm"],
            "mm": [k_out / "mm_val_confirm"],
            "asr": [k_out / "asr_val_4000"],
            "cap": [k_out / "cap_val_confirm"],
            "vl2": [k_out / "vl2_val_confirm"],
            "vl3": [k_out / "vl3_val_confirm"],
        }
    elif split == "test_1ka":
        sub_names = ["test_1ka"]
        base_dir = TEST_1KA_DIR
        q_file_tpl = "{g}_queries_vault.jsonl"
        dir_cfg = {
            "vp": _collect_existing([k_out / "vp_train_sub", k_out / "vp_test_1ka_queries_only"]),
            "mm": _collect_existing([k_out / "mm_test_1ka_videos_only", k_out / "mm_test_1ka_queries_only"]),
            "asr": _collect_existing([k_out / "asr_test_1ka_videos_only"]),
            "cap": _collect_existing([k_out / "cap_val_tune", k_out / "cap_test_1ka_queries_only"]),
            "vl2": _collect_existing([*sorted(k_out.glob("vl2_test_1ka_videos*")), k_out / "vl2_test_1ka_queries_only"]),
            "vl3": _collect_existing([*sorted(k_out.glob("vl3_train_sub*")), *sorted(k_out.glob("vl3_test_1ka_videos*")), k_out / "vl3_test_1ka_queries_only"]),
        }
    else:
        raise ValueError(f"Unknown split: {split}")

    banks: list[GalleryFeatureBank] = []
    labels_by_qid: dict[str, dict[str, Any]] = {}
    qmeta_by_qid: dict[str, dict[str, Any]] = {}
    for g in sub_names:
        b = load_gallery_feature_bank(
            gallery_name=g,
            gallery_manifest_path=base_dir / f"{g}_gallery_manifest.jsonl",
            queries_path=base_dir / q_file_tpl.format(g=g),
            vp_dirs=dir_cfg["vp"],
            mm_dirs=dir_cfg["mm"],
            asr_dirs=dir_cfg["asr"],
            cap_dirs=dir_cfg["cap"],
            vl2_dirs=dir_cfg["vl2"],
            vl3_dirs=dir_cfg["vl3"],
            train_vp_bank=train_banks.get("vp"),
            train_sig_bank=train_banks.get("sig"),
            train_pe_bank=train_banks.get("pe"),
            train_sig400_bank=train_banks.get("s4"),
            train_pe_bigg_bank=train_banks.get("pe_bigg"),
            train_dfn_bank=train_banks.get("dfn"),
        )
        banks.append(b)
        for row in _read_jsonl(base_dir / f"{g}_labels.jsonl"):
            labels_by_qid[str(row["query_id"])] = row
        for q in b.queries:
            qmeta_by_qid[str(q["query_id"])] = q
    return banks, labels_by_qid, qmeta_by_qid


def _compute_vp_sig_matrices(bank: GalleryFeatureBank, *, use_mp: bool = True) -> dict[str, np.ndarray]:
    """Compute raw `(N_q, N_v)` similarity matrices for VideoPrism and SigLIP2-Base."""
    nq, nv = len(bank.queries), len(bank.video_ids)
    vp_raw = np.zeros((nq, nv), dtype=np.float64)
    sig_raw = np.zeros((nq, nv), dtype=np.float64)
    if nv == 0 or nq == 0:
        return {"vp": vp_raw, "sig": sig_raw}

    # VideoPrism whole + window blend
    vp_qmat = np.zeros((nq, 768), dtype=np.float32)
    for i, q in enumerate(bank.queries):
        qt = q["query_text"]
        qv = compute_multiprompt_query_vector(bank.vp_query_map, qt) if use_mp else bank.vp_query_map.get(sha256_text(qt))
        if qv is not None:
            vp_qmat[i] = l2_normalize_np(qv)
    if bank.vp_whole_vecs is not None:
        whole = (vp_qmat @ bank.vp_whole_vecs.astype(np.float32, copy=False).T).astype(np.float64)
        if bank.vp_window_vecs is not None and bank.vp_win_vid_indices is not None:
            w_all = (vp_qmat @ bank.vp_window_vecs.astype(np.float32, copy=False).T).astype(np.float64)
            tmp_max = np.full((nv, nq), -1e9, dtype=np.float64)
            np.maximum.at(tmp_max, bank.vp_win_vid_indices, w_all.T)
            w_max = tmp_max.T
            miss = w_max < -1e8
            w_max[miss] = whole[miss]
            vp_raw = 0.45 * whole + 0.55 * w_max
        else:
            vp_raw = whole

    # SigLIP2-Base frame_max + clip_mean blend
    sig_qmat = np.zeros((nq, 768), dtype=np.float32)
    for i, q in enumerate(bank.queries):
        qt = q["query_text"]
        qv = compute_multiprompt_query_vector(bank.sig_query_map, qt) if use_mp else bank.sig_query_map.get(sha256_text(qt))
        if qv is not None:
            sig_qmat[i] = l2_normalize_np(qv)
    if bank.sig_clip_mean_vecs is not None:
        s_mean = (sig_qmat @ bank.sig_clip_mean_vecs.astype(np.float32, copy=False).T).astype(np.float64)
        if bank.sig_frame_vecs is not None and bank.sig_kf_vid_indices is not None:
            f32 = bank.sig_frame_vecs.astype(np.float32, copy=False)
            all_f = (sig_qmat @ f32.T).astype(np.float64)
            tmp_max = np.full((nv, nq), -1e9, dtype=np.float64)
            np.maximum.at(tmp_max, bank.sig_kf_vid_indices, all_f.T)
            s_max = tmp_max.T
            miss = s_max < -1e8
            s_max[miss] = s_mean[miss]
            sig_raw = 0.65 * s_max + 0.35 * s_mean
        else:
            sig_raw = s_mean
    return {"vp": vp_raw, "sig": sig_raw}


def _hub_transform_backbone(
    raw_mat: np.ndarray,
    priors: Mapping[str, np.ndarray] | None,
    *,
    dis_tau: float,
    w_raw: float,
    w_dis: float,
    w_znorm: float,
    w_csls: float,
) -> np.ndarray:
    """Apply inductive train-9K query-bank hubness transforms strictly independently per query."""
    comp = float(w_raw) * _zscore_rows(raw_mat)
    if priors is None:
        return comp
    if float(w_dis) != 0.0:
        comp = comp + float(w_dis) * _zscore_rows(apply_qb_dis_matrix(raw_mat, priors, tau=dis_tau))
    if float(w_znorm) != 0.0:
        comp = comp + float(w_znorm) * _zscore_rows(apply_qb_znorm_matrix(raw_mat, priors))
    if float(w_csls) != 0.0:
        comp = comp + float(w_csls) * _zscore_rows(apply_hubness_csls(raw_mat, priors, csls_weight=0.45))
    return comp


def _precompute_gallery_signals(
    bank: GalleryFeatureBank,
    *,
    ckpt_path: Path,
    train_adapter_joint_bank: np.ndarray | None,
    use_mp: bool = True,
) -> dict[str, Any]:
    """Precompute raw matrices, train-9K priors, adapter matrices, and multimodal signals."""
    vl2_mp = compute_vl2_gallery_score_matrices(bank, use_mp=use_mp, temperature=0.05)
    vl3_mp = compute_vl3_gallery_score_matrices(bank, use_mp=use_mp, temperature=0.05)
    vp_sig = _compute_vp_sig_matrices(bank, use_mp=use_mp)

    raw_mats: dict[str, np.ndarray] = {
        "pe_bigg": vl3_mp["pe_bigg_blend"],
        "dfn": vl3_mp["dfn_blend"],
        "pe": vl2_mp["pe_blend"],
        "s4": 0.30 * vl2_mp["s4_mean"] + 0.70 * vl2_mp["s4_xpool"],
        "vp": vp_sig["vp"],
        "sig": vp_sig["sig"],
    }
    priors_map: dict[str, Mapping[str, np.ndarray] | None] = {
        "pe_bigg": bank.pe_bigg_hub_priors,
        "dfn": bank.dfn_hub_priors,
        "pe": bank.pe_hub_priors,
        "s4": bank.sig400_hub_priors,
        "vp": bank.vp_hub_priors,
        "sig": bank.sig_hub_priors,
    }

    # Build gallery video + query dicts for CrossModalAdapter
    nq, nv = len(bank.queries), len(bank.video_ids)
    sig_mean_vecs = bank.sig_clip_mean_vecs if bank.sig_clip_mean_vecs is not None else np.zeros((nv, 768), dtype=np.float32)
    vid_feats = {
        "pe_bigg": bank.pe_bigg_video_vecs,
        "dfn": bank.dfn_video_vecs,
        "pe": bank.pe_video_vecs,
        "s4": bank.sig400_video_vecs,
        "vp": bank.vp_whole_vecs,
        "sig": sig_mean_vecs,
    }
    q_maps = {
        "pe_bigg": bank.pe_bigg_query_map,
        "dfn": bank.dfn_query_map,
        "pe": bank.pe_query_map,
        "s4": bank.sig400_query_map,
        "vp": bank.vp_query_map,
        "sig": bank.sig_query_map,
    }
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    adapter_bs = tuple(ckpt["backbone_dims"].keys())

    qry_feats: dict[str, np.ndarray] = {}
    for bname in adapter_bs:
        dim = int(ckpt["backbone_dims"][bname])
        qmat = np.zeros((nq, dim), dtype=np.float32)
        qmap = q_maps[bname]
        for i, q in enumerate(bank.queries):
            qt = q["query_text"]
            qv = compute_multiprompt_query_vector(qmap, qt) if use_mp else qmap.get(sha256_text(qt))
            if qv is not None:
                qmat[i] = l2_normalize_np(qv)
        qry_feats[bname] = qmat

    v_in = {b: np.asarray(vid_feats[b], dtype=np.float32) for b in adapter_bs}
    proj = adapt_vectors_with_checkpoint(ckpt_path, video_feats=v_in, query_feats=qry_feats, device="cpu")
    vj, qj = proj["videos"]["joint"], proj["queries"]["joint"]
    ad_joint_raw = (qj @ vj.T).astype(np.float64)
    ad_res_raw = sum((proj["queries"][b] @ proj["videos"][b].T).astype(np.float64) for b in adapter_bs) / max(1, len(adapter_bs))

    ad_joint_priors = None
    if train_adapter_joint_bank is not None and len(train_adapter_joint_bank) > 0:
        ad_joint_priors = TrainQueryBankHubness(train_adapter_joint_bank, k_hub=20, temperature=0.05).compute_video_priors(vj)

    # Multimodal lexical/caption matrices
    bm25_cap = build_bm25_scorer({v: bank.florence_captions.get(v, "") for v in bank.video_ids})
    bm25_ocr = build_bm25_scorer({v: bank.ocr_docs.get(v, "") for v in bank.video_ids})
    bm25_asr = build_bm25_scorer({v: bank.asr_docs.get(v, "") for v in bank.video_ids})

    cap_e5_m = np.zeros((nq, nv), dtype=np.float64)
    cap_bm_m = np.zeros((nq, nv), dtype=np.float64)
    ocr_m = np.zeros((nq, nv), dtype=np.float64)
    asr_m = np.zeros((nq, nv), dtype=np.float64)
    for i, q in enumerate(bank.queries):
        qt = q["query_text"]
        if bank.cap_e5_vecs is not None:
            qe = bank.e5_query_map.get(sha256_text(qt))
            if qe is not None:
                cap_e5_m[i] = (bank.cap_e5_vecs @ l2_normalize_np(qe)).astype(np.float64)
        cap_bm_m[i] = bm25_cap(qt, bank.video_ids)
        ocr_m[i] = np.log1p(np.maximum(0.0, bm25_ocr(qt, bank.video_ids)))
        asr_m[i] = np.log1p(np.maximum(0.0, bm25_asr(qt, bank.video_ids)))

    return {
        "bank": bank,
        "raw_mats": raw_mats,
        "priors_map": priors_map,
        "ad_joint_raw": ad_joint_raw,
        "ad_joint_priors": ad_joint_priors,
        "ad_res_raw": ad_res_raw,
        "cap_fus_z": 0.6 * _zscore_rows(cap_e5_m) + 0.4 * _zscore_rows(cap_bm_m),
        "ocr_raw": ocr_m,
        "asr_raw": asr_m,
    }


def _build_fused_matrix(
    sig: Mapping[str, Any],
    *,
    b_weights: Mapping[str, float],
    dis_tau: float,
    w_raw: float,
    w_dis: float,
    w_znorm: float,
    w_csls: float,
    boltz_temp: float = 0.0,
    w_ad_joint: float = 0.0,
    w_ad_res: float = 0.0,
    w_cap: float = 0.0,
    w_asr: float = 0.0,
    w_ocr: float = 0.0,
) -> np.ndarray:
    """Compute Stage-1 combined `(N_q, N_v)` similarity matrix with optional Boltzmann sharpening."""
    raw_mats = sig["raw_mats"]
    priors = sig["priors_map"]
    nq, nv = raw_mats["vp"].shape
    total = np.zeros((nq, nv), dtype=np.float64)
    for bname, bw in b_weights.items():
        if float(bw) == 0.0:
            continue
        tmat = _hub_transform_backbone(
            raw_mats[bname], priors[bname],
            dis_tau=dis_tau, w_raw=w_raw, w_dis=w_dis, w_znorm=w_znorm, w_csls=w_csls,
        )
        term = _softmax_rows(tmat, temp=boltz_temp) if float(boltz_temp) > 0.0 else _zscore_rows(tmat)
        total += float(bw) * term

    total_z = _zscore_rows(total)
    if float(w_ad_joint) != 0.0:
        adj = _hub_transform_backbone(
            sig["ad_joint_raw"], sig["ad_joint_priors"],
            dis_tau=dis_tau, w_raw=w_raw, w_dis=w_dis, w_znorm=w_znorm, w_csls=0.0,
        )
        total_z = total_z + float(w_ad_joint) * _zscore_rows(adj)
    if float(w_ad_res) != 0.0:
        total_z = total_z + float(w_ad_res) * _zscore_rows(sig["ad_res_raw"])
    if float(w_cap) != 0.0:
        total_z = total_z + float(w_cap) * sig["cap_fus_z"]
    if float(w_asr) != 0.0:
        total_z = total_z + float(w_asr) * sig["asr_raw"]
    if float(w_ocr) != 0.0:
        total_z = total_z + float(w_ocr) * sig["ocr_raw"]
    return total_z


def _eval_score_dict(
    config_id: str,
    banks: Sequence[GalleryFeatureBank],
    mats_by_gal: Mapping[str, np.ndarray],
    labels_by_qid: Mapping[str, dict[str, Any]],
    *,
    b1_hits_by_qid: Mapping[str, int] | None = None,
    n_boot: int = 10000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds: list[dict[str, Any]] = []
    for b in banks:
        mat = mats_by_gal[b.gallery_name]
        for i, q in enumerate(b.queries):
            preds.append({
                "query_id": q["query_id"],
                "gallery": b.gallery_name,
                "ranked_video_ids": rank_gallery_desc(b.video_ids, mat[i]),
            })
    metrics = evaluate_predictions(
        config_id, preds, labels_by_qid,
        b1_hits_by_qid=b1_hits_by_qid,
        seed=SEED,
        n_boot=n_boot,
    )
    return metrics, preds


def _eval_r8_v1_reference(
    banks: Sequence[GalleryFeatureBank],
    labels_by_qid: Mapping[str, dict[str, Any]],
    b1_hits: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reproduce V1 Champion (`R8`) reference scores directly from frozen V1 params."""
    preds: list[dict[str, Any]] = []
    for b in banks:
        bm_asr = build_bm25_scorer({v: b.asr_docs.get(v, "") for v in b.video_ids})
        vl2 = compute_vl2_gallery_score_matrices(b, use_mp=True, temperature=0.05)
        vp_sig = _compute_vp_sig_matrices(b, use_mp=False)
        pe_s = _zscore_rows((vl2["pe_mean"] + vl2["pe_xpool"] + vl2["pe_top3"]) / 3.0)
        s4_s = _zscore_rows(0.30 * vl2["s4_mean"] + 0.70 * vl2["s4_xpool"])
        vp_s = _zscore_rows(apply_hubness_csls(vp_sig["vp"], b.vp_hub_priors, csls_weight=0.30))
        sig_s = _zscore_rows(apply_hubness_csls(vp_sig["sig"], b.sig_hub_priors, csls_weight=0.20))

        # Boltzmann 2.0 softmax ensemble matching V1 R8 weights [0.95, 1.35, 0.70, 0.12]
        comb = (
            0.95 * _softmax_rows(pe_s, temp=2.0)
            + 1.35 * _softmax_rows(s4_s, temp=2.0)
            + 0.70 * _softmax_rows(vp_s, temp=2.0)
            + 0.12 * _softmax_rows(sig_s, temp=2.0)
        )
        comb_z = _zscore_rows(comb)
        for i, q in enumerate(b.queries):
            asr_sc = np.log1p(np.maximum(0.0, bm_asr(q["query_text"], b.video_ids)))
            row = comb_z[i] + 0.12 * asr_sc
            preds.append({"query_id": q["query_id"], "gallery": b.gallery_name, "ranked_video_ids": rank_gallery_desc(b.video_ids, row)})
    metrics = evaluate_predictions("R8_v1_baseline", preds, labels_by_qid, b1_hits_by_qid=b1_hits, seed=SEED)
    return metrics, preds


def _tune_stage1_hyperparams(
    tune_sigs: Mapping[str, Mapping[str, Any]],
    tune_banks: Sequence[GalleryFeatureBank],
    tune_labels: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Tune R9a, R9b, and R9c strictly on Tune using low-DOF structured search."""
    def _obj(bw: Mapping[str, float], kw: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        mats = {g: _build_fused_matrix(tune_sigs[g], b_weights=bw, **kw) for g in tune_sigs}
        m, _ = _eval_score_dict("tmp", tune_banks, mats, tune_labels, n_boot=32)
        return m["hit_at_1"] + 0.15 * m["mrr_at_10"], m

    # 1. Tune R9a (6 backbones, raw zscore vs boltzmann, coordinate search)
    r9a_w: dict[str, float] = {"pe_bigg": 1.35, "dfn": 1.15, "pe": 0.90, "s4": 1.05, "vp": 0.55, "sig": 0.10}
    best_r9a_t, best_r9a_obj = 0.0, -1.0
    for bt in (0.0, 2.0, 1.5):
        o, _ = _obj(r9a_w, dict(dis_tau=0.05, w_raw=1.0, w_dis=0.0, w_znorm=0.0, w_csls=0.0, boltz_temp=bt))
        if o > best_r9a_obj:
            best_r9a_obj, best_r9a_t = o, bt
    for _pass in range(2):
        for bk in ("pe_bigg", "dfn", "pe", "s4", "vp", "sig"):
            cur = r9a_w[bk]
            for cand in sorted({max(0.0, round(cur + d, 2)) for d in (-0.30, -0.15, 0.0, 0.15, 0.30)}):
                trial = dict(r9a_w)
                trial[bk] = cand
                o, _ = _obj(trial, dict(dis_tau=0.05, w_raw=1.0, w_dis=0.0, w_znorm=0.0, w_csls=0.0, boltz_temp=best_r9a_t))
                if o > best_r9a_obj + 1e-6:
                    best_r9a_obj = o
                    r9a_w = trial
    r9a_cfg = {"b_weights": r9a_w, "dis_tau": 0.05, "w_raw": 1.0, "w_dis": 0.0, "w_znorm": 0.0, "w_csls": 0.0, "boltz_temp": best_r9a_t}

    # 2. Tune R9b (QB-DIS + QB-Z-Norm + QB-CSLS + 1 backbone weight pass)
    best_hub = {"dis_tau": 0.05, "w_raw": 0.30, "w_dis": 0.45, "w_znorm": 0.15, "w_csls": 0.10}
    best_r9b_obj = -1.0
    hub_grid = [
        (tau, wr, wd, wz, wc)
        for tau in (0.03, 0.05, 0.07)
        for (wr, wd, wz, wc) in (
            (0.40, 0.40, 0.10, 0.10),
            (0.30, 0.50, 0.15, 0.05),
            (0.25, 0.55, 0.15, 0.05),
            (0.20, 0.50, 0.25, 0.05),
            (0.35, 0.35, 0.20, 0.10),
            (0.50, 0.00, 0.30, 0.20),
            (0.60, 0.00, 0.00, 0.40),
        )
    ]
    r9b_w = dict(r9a_w)
    for tau, wr, wd, wz, wc in hub_grid:
        o, _ = _obj(r9b_w, dict(dis_tau=tau, w_raw=wr, w_dis=wd, w_znorm=wz, w_csls=wc, boltz_temp=best_r9a_t))
        if o > best_r9b_obj:
            best_r9b_obj = o
            best_hub = {"dis_tau": tau, "w_raw": wr, "w_dis": wd, "w_znorm": wz, "w_csls": wc}
    # Light 1-pass weight polish with hubness active
    for bk in ("pe_bigg", "dfn", "pe", "s4", "vp", "sig"):
        cur = r9b_w[bk]
        for cand in sorted({max(0.0, round(cur + d, 2)) for d in (-0.20, 0.0, 0.20)}):
            trial = dict(r9b_w)
            trial[bk] = cand
            o, _ = _obj(trial, dict(**best_hub, boltz_temp=best_r9a_t))
            if o > best_r9b_obj + 1e-6:
                best_r9b_obj = o
                r9b_w = trial
    r9b_cfg = {"b_weights": r9b_w, **best_hub, "boltz_temp": best_r9a_t}

    # 3. Tune R9c (Adapter Joint/Res + Multimodal signals strictly on top of R9b)
    r9c_mm = {"w_ad_joint": 0.15, "w_ad_res": 0.0, "w_cap": 0.05, "w_asr": 0.10, "w_ocr": 0.0}
    best_r9c_obj, _ = _obj(r9b_w, dict(**best_hub, boltz_temp=best_r9a_t, **r9c_mm))
    mm_ranges = {
        "w_ad_joint": (0.0, 0.10, 0.20, 0.35, 0.50),
        "w_ad_res": (0.0, 0.10, 0.20),
        "w_cap": (0.0, 0.05, 0.10, 0.18),
        "w_asr": (0.0, 0.08, 0.12, 0.18),
        "w_ocr": (0.0, 0.06, 0.12),
    }
    for _pass in range(2):
        for mk, cands in mm_ranges.items():
            for cand in cands:
                trial = dict(r9c_mm)
                trial[mk] = cand
                o, _ = _obj(r9b_w, dict(**best_hub, boltz_temp=best_r9a_t, **trial))
                if o > best_r9c_obj + 1e-6:
                    best_r9c_obj = o
                    r9c_mm = trial
    r9c_cfg = {**r9b_cfg, **r9c_mm}
    return r9a_cfg, r9b_cfg, r9c_cfg


def _export_vlm_candidates_for_split(
    banks: Sequence[GalleryFeatureBank],
    sigs: Mapping[str, Mapping[str, Any]],
    r9c_cfg: Mapping[str, Any],
    out_path: Path,
    *,
    top_k: int = 8,
    margin_threshold: float,
) -> dict[str, Any]:
    """Export uncertain Stage-1 Top-K candidates (strictly zero GT!) for Kaggle VLM scoring."""
    rows: list[dict[str, Any]] = []
    all_margins: list[float] = []
    total_q = 0
    for b in banks:
        mat = _build_fused_matrix(sigs[b.gallery_name], **r9c_cfg)
        for i, q in enumerate(b.queries):
            total_q += 1
            sc_row = mat[i]
            top_vids = rank_gallery_desc(b.video_ids, sc_row)[:top_k]
            top_scores = [float(sc_row[b.vid_to_idx[v]]) for v in top_vids]
            m1_z = float(top_scores[0] - top_scores[1]) if len(top_scores) >= 2 else 999.0
            all_margins.append(m1_z)
            if m1_z < float(margin_threshold):
                rows.append({
                    "query_id": str(q["query_id"]),
                    "gallery": b.gallery_name,
                    "query_text": str(q["query_text"]),
                    "candidate_video_ids": top_vids,
                    "stage1_scores": [round(s, 6) for s in top_scores],
                    "stage1_margin_z": round(m1_z, 6),
                })

    assert_no_gt_leakage(rows, context=out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "path": str(out_path),
        "total_queries": total_q,
        "exported_queries": len(rows),
        "top_k": top_k,
        "margin_threshold": float(margin_threshold),
    }


def _collect_margins(
    banks: Sequence[GalleryFeatureBank],
    sigs: Mapping[str, Mapping[str, Any]],
    r9c_cfg: Mapping[str, Any],
) -> list[float]:
    margins: list[float] = []
    for b in banks:
        mat = _build_fused_matrix(sigs[b.gallery_name], **r9c_cfg)
        for i in range(len(b.queries)):
            s_sorted = np.sort(mat[i])
            margins.append(float(s_sorted[-1] - s_sorted[-2]) if len(s_sorted) >= 2 else 999.0)
    return margins


def _write_preds(path: Path, preds: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(dict(p), ensure_ascii=False) + "\n")


def _read_preds(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def cmd_stage1_and_export(args: argparse.Namespace) -> int:
    V2_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path, ad_meta = ensure_cross_modal_adapter(retrain=args.retrain_adapter)
    train_banks, train_dicts = load_all_train_query_banks()

    # Project train_sub query bank into adapter joint space for inductive joint hubness norm
    active_bs = tuple(ad_meta["backbones"])
    q_shas = [qs for qs in train_dicts[active_bs[0]].keys() if all(qs in train_dicts[b] for b in active_bs)]
    ad_q_bank = np.vstack([train_dicts[active_bs[0]][q_shas[0]]]) if q_shas else None
    if q_shas:
        q_in = {b: np.vstack([train_dicts[b][qs] for qs in q_shas]).astype(np.float32) for b in active_bs}
        ad_q_bank = adapt_vectors_with_checkpoint(ckpt_path, query_feats=q_in, device="cpu")["queries"]["joint"]

    tune_banks, tune_labels, _ = _load_split_galleries("tune", train_banks=train_banks)
    confirm_banks, confirm_labels, _ = _load_split_galleries("confirm", train_banks=train_banks)

    tune_sigs = {
        b.gallery_name: _precompute_gallery_signals(b, ckpt_path=ckpt_path, train_adapter_joint_bank=ad_q_bank, use_mp=True)
        for b in tune_banks
    }
    confirm_sigs = {
        b.gallery_name: _precompute_gallery_signals(b, ckpt_path=ckpt_path, train_adapter_joint_bank=ad_q_bank, use_mp=True)
        for b in confirm_banks
    }

    # B1 reference hits
    def _b1_hits(banks: Sequence[GalleryFeatureBank], lbls: Mapping[str, dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in banks:
            for q in b.queries:
                ranked = rank_gallery_desc(b.video_ids, compute_b1_videoprism_scores(b, q["query_text"]))
                tset = {str(v) for v in lbls[q["query_id"]]["target_videos"]}
                out[q["query_id"]] = 1 if ranked and ranked[0] in tset else 0
        return out

    tune_b1_hits = _b1_hits(tune_banks, tune_labels)
    confirm_b1_hits = _b1_hits(confirm_banks, confirm_labels)

    r9a_cfg, r9b_cfg, r9c_cfg = _tune_stage1_hyperparams(tune_sigs, tune_banks, tune_labels)

    def _eval_stage1_set(split: str, banks: Sequence[GalleryFeatureBank], sigs: Mapping[str, Mapping[str, Any]], lbls: Mapping[str, dict[str, Any]], b1_h: Mapping[str, int]):
        res: dict[str, Any] = {}
        res["R8_v1_baseline"], _ = _eval_r8_v1_reference(banks, lbls, b1_h)
        for cid, bk in (("R9_pe_bigg_standalone", "pe_bigg"), ("R9_dfn5b_h_standalone", "dfn")):
            m, _ = _eval_score_dict(cid, banks, {g: sigs[g]["raw_mats"][bk] for g in sigs}, lbls, b1_hits_by_qid=b1_h)
            res[cid] = m
        for cid, cfg in (("R9a_6backbone_raw", r9a_cfg), ("R9b_6backbone_qb_dis_znorm_mp", r9b_cfg), ("R9c_6backbone_adapter_multimodal", r9c_cfg)):
            mats = {g: _build_fused_matrix(sigs[g], **cfg) for g in sigs}
            m, preds = _eval_score_dict(cid, banks, mats, lbls, b1_hits_by_qid=b1_h)
            res[cid] = m
            if cid == "R9c_6backbone_adapter_multimodal":
                _write_preds(V2_RUN_DIR / f"r9c_{split}_preds.jsonl.gz", preds)
                # Save full row margins & top-20 score dicts for VLM blending without recomputation
                top20_scores: dict[str, dict[str, float]] = {}
                for b in banks:
                    bmat = mats[b.gallery_name]
                    for i, q in enumerate(b.queries):
                        row = bmat[i]
                        top20_v = rank_gallery_desc(b.video_ids, row)[:20]
                        top20_scores[q["query_id"]] = {v: round(float(row[b.vid_to_idx[v]]), 6) for v in top20_v}
                _write_json(V2_RUN_DIR / f"r9c_{split}_top20_scores.json", top20_scores)
        return res

    tune_res = _eval_stage1_set("tune", tune_banks, tune_sigs, tune_labels, tune_b1_hits)
    confirm_res = _eval_stage1_set("confirm", confirm_banks, confirm_sigs, confirm_labels, confirm_b1_hits)

    # Fixed inductive margin threshold chosen solely on Tune (default quantile covers uncertain 70%)
    tune_margins = _collect_margins(tune_banks, tune_sigs, r9c_cfg)
    export_margin_thresh = float(np.quantile(tune_margins, float(args.vlm_export_quantile)))

    exp_tune = _export_vlm_candidates_for_split(
        tune_banks, tune_sigs, r9c_cfg, V2_RUN_DIR / "vlm_candidates_tune.jsonl",
        top_k=args.vlm_top_k, margin_threshold=export_margin_thresh,
    )
    exp_confirm = _export_vlm_candidates_for_split(
        confirm_banks, confirm_sigs, r9c_cfg, V2_RUN_DIR / "vlm_candidates_confirm.jsonl",
        top_k=args.vlm_top_k, margin_threshold=export_margin_thresh,
    )

    state = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapter_meta": ad_meta,
        "train_query_bank_sizes": {k: int(v.shape[0]) for k, v in train_banks.items()},
        "configs": {"R9a": r9a_cfg, "R9b": r9b_cfg, "R9c": r9c_cfg},
        "vlm_export_top_k": int(args.vlm_top_k),
        "vlm_export_margin_threshold": round(export_margin_thresh, 6),
        "vlm_exports": {"tune": exp_tune, "confirm": exp_confirm},
        "tune_results": tune_res,
        "confirm_results": confirm_res,
    }
    _write_json(V2_RUN_DIR / "v2_stage1_state.json", state)
    print(json.dumps({
        "tune": {k: v["hit_at_1"] for k, v in tune_res.items()},
        "confirm": {k: v["hit_at_1"] for k, v in confirm_res.items()},
        "vlm_exports": state["vlm_exports"],
    }, indent=2))
    return 0


def _load_vlm_scores(vlm_dir: Path) -> dict[str, dict[str, float]]:
    sd = _find_shard_dir(vlm_dir)
    out: dict[str, dict[str, float]] = {}
    for r in _read_jsonl(sd / "vlm_rerank_scores.jsonl.gz"):
        qid = str(r["query_id"])
        out[qid] = {str(v): float(s) for v, s in zip(r["candidate_video_ids"], r["vlm_scores"])}
    return out


_debias_vlm_scores = debias_vlm_scores
_combine_stage1_and_vlm_pool = combine_stage1_and_vlm_pool
_blend_stage1_and_vlm_preds = blend_stage1_and_vlm_predictions


def _load_llm_chunk_bundle(chunk_dir: Path) -> dict[str, Any]:
    sd = _find_shard_dir(chunk_dir)
    plans_by_sha = {str(r["query_text_sha256"]): r for r in _read_jsonl(sd / "query_plans.jsonl")}
    chunk_idx_by_sha = {str(r["chunk_text_sha256"]): int(r["row_id"]) for r in _read_jsonl(sd / "chunk_rows.jsonl")}
    vecs = {
        "pe_bigg": np.load(str(sd / "pe_bigg_chunk_vectors.npy"), mmap_mode="r"),
        "dfn": np.load(str(sd / "dfn_chunk_vectors.npy"), mmap_mode="r"),
        "pe": np.load(str(sd / "pe_l_chunk_vectors.npy"), mmap_mode="r"),
        "s4": np.load(str(sd / "sig400_chunk_vectors.npy"), mmap_mode="r"),
    }
    receipt = json.loads((sd / "receipt.json").read_text(encoding="utf-8"))
    return {
        "plans_by_sha": plans_by_sha,
        "chunk_idx_by_sha": chunk_idx_by_sha,
        "vecs": vecs,
        "receipt": receipt,
    }


def _compute_clause_matrix_for_bank(
    bank: GalleryFeatureBank,
    *,
    bundle: Mapping[str, Any],
    clause_field: str,
    b_weights: Mapping[str, float],
) -> np.ndarray:
    """Compute `(N_q, N_v)` row-z-scored multi-backbone max-frame clause similarity matrix."""
    plans_by_sha = bundle["plans_by_sha"]
    chunk_idx_by_sha = bundle["chunk_idx_by_sha"]
    vecs = bundle["vecs"]
    nq, nv = len(bank.queries), len(bank.video_ids)
    if nq == 0 or nv == 0:
        return np.zeros((nq, nv), dtype=np.float64)

    q_clause_ids: list[list[int]] = []
    used_ids: list[int] = []
    seen_id: set[int] = set()
    for q in bank.queries:
        qt = str(q["query_text"]).strip()
        qsha = sha256_text(qt)
        plan = plans_by_sha.get(qsha) or {}
        clauses = [str(c).strip() for c in (plan.get(clause_field) or []) if str(c).strip()] or [qt]
        cids: list[int] = []
        for c in clauses:
            idx = chunk_idx_by_sha.get(sha256_text(c))
            if idx is not None:
                cids.append(idx)
                if idx not in seen_id:
                    seen_id.add(idx)
                    used_ids.append(idx)
        if not cids:
            fb_idx = chunk_idx_by_sha.get(qsha)
            if fb_idx is not None:
                cids = [fb_idx]
                if fb_idx not in seen_id:
                    seen_id.add(fb_idx)
                    used_ids.append(fb_idx)
        q_clause_ids.append(cids)

    local_pos = {cid: i for i, cid in enumerate(used_ids)}
    uid_arr = np.asarray(used_ids, dtype=np.int64)

    frame_map = {
        "pe_bigg": bank.pe_bigg_frame_vecs,
        "dfn": bank.dfn_frame_vecs,
        "pe": bank.pe_frame_vecs,
        "s4": bank.sig400_frame_vecs,
    }
    total = np.zeros((nq, nv), dtype=np.float64)
    w_sum = 0.0
    for bk, bw in b_weights.items():
        if bk not in vecs or float(bw) == 0.0 or frame_map.get(bk) is None or len(used_ids) == 0:
            continue
        f32 = np.asarray(frame_map[bk], dtype=np.float32)  # (N_v, F, D)
        nf, dim = int(f32.shape[1]), int(f32.shape[2])
        q_c = l2_normalize_np(np.asarray(vecs[bk][uid_arr], dtype=np.float32))  # (U, D)
        sims = (q_c @ f32.reshape(nv * nf, dim).T).reshape(len(used_ids), nv, nf).astype(np.float64)
        c_max = np.max(sims, axis=2)  # (U, N_v)

        bmat = np.zeros((nq, nv), dtype=np.float64)
        for qi, cids in enumerate(q_clause_ids):
            if cids:
                pos = [local_pos[cid] for cid in cids]
                bmat[qi] = np.mean(c_max[pos], axis=0)
        total += float(bw) * _zscore_rows(bmat)
        w_sum += float(bw)

    if w_sum <= 0.0:
        return total
    return _zscore_rows(total / w_sum)


def cmd_ablate_llm_chunks(args: argparse.Namespace) -> int:
    """Ablate deterministic rule chunking vs CUDA Local LLM (`Qwen2.5-1.5B-Instruct`) on Tune -> Confirm."""
    state = json.loads((V2_RUN_DIR / "v2_stage1_state.json").read_text(encoding="utf-8"))
    r9c_cfg = state["configs"]["R9c"]
    b_weights = r9c_cfg["b_weights"]
    ckpt_path = MODELS_DIR / "v2_cross_modal_adapter.pt"
    train_banks, train_dicts = load_all_train_query_banks()
    active_bs = tuple(state["adapter_meta"]["backbones"])
    q_shas = [qs for qs in train_dicts[active_bs[0]].keys() if all(qs in train_dicts[b] for b in active_bs)]
    q_in = {b: np.vstack([train_dicts[b][qs] for qs in q_shas]).astype(np.float32) for b in active_bs}
    ad_q_bank = adapt_vectors_with_checkpoint(ckpt_path, query_feats=q_in, device="cpu")["queries"]["joint"]

    bundle = _load_llm_chunk_bundle(Path(args.llm_chunk_dir))
    plans_by_sha = bundle["plans_by_sha"]

    def _prep_split(split_name: str):
        banks, labels, _ = _load_split_galleries(split_name, train_banks=train_banks)
        sigs = {
            b.gallery_name: _precompute_gallery_signals(b, ckpt_path=ckpt_path, train_adapter_joint_bank=ad_q_bank, use_mp=True)
            for b in banks
        }
        base_mats = {g: _build_fused_matrix(sigs[g], **r9c_cfg) for g in sigs}
        rule_mats = {
            b.gallery_name: _compute_clause_matrix_for_bank(b, bundle=bundle, clause_field="rule_visual_clauses", b_weights=b_weights)
            for b in banks
        }
        llm_mats = {
            b.gallery_name: _compute_clause_matrix_for_bank(b, bundle=bundle, clause_field="visual_clauses", b_weights=b_weights)
            for b in banks
        }
        med_long_qids = {
            q["query_id"]
            for b in banks
            for q in b.queries
            if (plans_by_sha.get(sha256_text(str(q["query_text"]).strip())) or {}).get("tier") in ("medium", "long")
        }
        return banks, labels, base_mats, rule_mats, llm_mats, med_long_qids

    tune_banks, tune_labels, tune_base, tune_rule, tune_llm, tune_ml_qids = _prep_split("tune")
    conf_banks, conf_labels, conf_base, conf_rule, conf_llm, conf_ml_qids = _prep_split("confirm")

    def _subset_hit1(preds: Sequence[Mapping[str, Any]], lbls: Mapping[str, Mapping[str, Any]], qset: set[str]) -> dict[str, Any]:
        sub = [p for p in preds if p["query_id"] in qset]
        hits = sum(
            1 for p in sub
            if p["ranked_video_ids"] and p["ranked_video_ids"][0] in {str(v) for v in lbls[p["query_id"]]["target_videos"]}
        )
        return {"n_queries": len(sub), "hits_at_1": hits, "hit_at_1": round(hits / max(1, len(sub)), 6)}

    def _sweep(variant: str, t_c: Mapping[str, np.ndarray]):
        best_w, best_m, best_obj = 0.0, None, -1.0
        rows = []
        for w in (0.0, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
            mats = {g: tune_base[g] + float(w) * t_c[g] for g in tune_base}
            m, preds = _eval_score_dict(f"tune_{variant}_w{w:.2f}", tune_banks, mats, tune_labels, n_boot=500)
            obj = float(m["hit_at_1"]) + 0.10 * float(m["mrr_at_10"])
            rows.append({
                "w_chunk": w,
                "hit_at_1": m["hit_at_1"],
                "mrr_at_10": m["mrr_at_10"],
                "med_long": _subset_hit1(preds, tune_labels, tune_ml_qids),
            })
            if obj > best_obj + 1e-6:
                best_obj = obj
                best_w = w
                best_m = m
        return best_w, best_m, rows

    best_w_rule, _, tune_rule_curve = _sweep("rule", tune_rule)
    best_w_llm, _, tune_llm_curve = _sweep("llm", tune_llm)

    base_conf_m, base_conf_preds = _eval_score_dict("confirm_r9c_baseline", conf_banks, conf_base, conf_labels)
    rule_conf_m, rule_conf_preds = _eval_score_dict(
        "confirm_rule_chunks",
        conf_banks,
        {g: conf_base[g] + float(best_w_rule) * conf_rule[g] for g in conf_base},
        conf_labels,
    )
    llm_conf_m, llm_conf_preds = _eval_score_dict(
        "confirm_llm_chunks",
        conf_banks,
        {g: conf_base[g] + float(best_w_llm) * conf_llm[g] for g in conf_base},
        conf_labels,
    )
    # Also evaluate forced fixed weight (w=0.08) so Confirm behavior is transparent even if Tune preferred w=0.0
    llm_fixed_m, llm_fixed_preds = _eval_score_dict(
        "confirm_llm_chunks_w008",
        conf_banks,
        {g: conf_base[g] + 0.08 * conf_llm[g] for g in conf_base},
        conf_labels,
    )
    rule_fixed_m, rule_fixed_preds = _eval_score_dict(
        "confirm_rule_chunks_w008",
        conf_banks,
        {g: conf_base[g] + 0.08 * conf_rule[g] for g in conf_base},
        conf_labels,
    )

    out = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_receipt": bundle["receipt"],
        "tune_sweeps": {
            "rule": {"selected_w": best_w_rule, "curve": tune_rule_curve},
            "llm": {"selected_w": best_w_llm, "curve": tune_llm_curve},
        },
        "confirm_results": {
            "r9c_baseline": {**base_conf_m, "med_long": _subset_hit1(base_conf_preds, conf_labels, conf_ml_qids)},
            "rule_tuned": {"w_chunk": best_w_rule, **rule_conf_m, "med_long": _subset_hit1(rule_conf_preds, conf_labels, conf_ml_qids)},
            "llm_tuned": {"w_chunk": best_w_llm, **llm_conf_m, "med_long": _subset_hit1(llm_conf_preds, conf_labels, conf_ml_qids)},
            "rule_w008": {"w_chunk": 0.08, **rule_fixed_m, "med_long": _subset_hit1(rule_fixed_preds, conf_labels, conf_ml_qids)},
            "llm_w008": {"w_chunk": 0.08, **llm_fixed_m, "med_long": _subset_hit1(llm_fixed_preds, conf_labels, conf_ml_qids)},
        },
        "llm_improves_over_r9c_on_confirm": bool(float(llm_conf_m["hit_at_1"]) > float(base_conf_m["hit_at_1"])),
    }
    _write_json(V2_RUN_DIR / "llm_chunk_ablation.json", out)
    print(json.dumps(out, indent=2))
    return 0


def _sweep_vlm_fusion_on_tune(
    tune_preds: Sequence[Mapping[str, Any]],
    tune_top20: Mapping[str, Mapping[str, float]],
    vlm_tune: Mapping[str, Mapping[str, float]],
    tune_labels: Mapping[str, Mapping[str, Any]],
    *,
    max_mg: float,
    max_k: int,
) -> dict[str, Any]:
    """Vectorized hyperparameter selection strictly on Tune (zero Confirm/Test exposure)."""
    n_total = len(tune_preds)
    base_hit1 = np.zeros(n_total, dtype=np.float64)
    base_rr10 = np.zeros(n_total, dtype=np.float64)
    margins = np.full(n_total, 999.0, dtype=np.float64)
    pools: list[list[str]] = []
    s1_vecs: list[np.ndarray] = []
    gt_sets: list[set[str]] = []
    qids: list[str] = []

    for i, p in enumerate(tune_preds):
        qid = str(p["query_id"])
        qids.append(qid)
        ranked = [str(v) for v in p["ranked_video_ids"]]
        gt = {str(v) for v in tune_labels[qid]["target_videos"]}
        gt_sets.append(gt)
        if ranked and ranked[0] in gt:
            base_hit1[i] = 1.0
        for r_idx, vid in enumerate(ranked[:10]):
            if vid in gt:
                base_rr10[i] = 1.0 / float(r_idx + 1)
                break
        s1m = tune_top20.get(qid, {})
        if len(ranked) >= 2:
            margins[i] = float(s1m.get(ranked[0], 0.0) - s1m.get(ranked[1], 0.0))
        if qid in vlm_tune and len(ranked) >= 2:
            pool = [v for v in ranked[:max_k] if v in vlm_tune[qid]]
        else:
            pool = []
        pools.append(pool)
        s1_vecs.append(np.asarray([s1m[v] for v in pool], dtype=np.float64) if pool else np.empty(0))

    base_h1_mean = float(np.mean(base_hit1))
    base_rr_mean = float(np.mean(base_rr10))
    best_obj = base_h1_mean + 0.15 * base_rr_mean
    best_cfg: dict[str, Any] = {
        "margin_gate": max_mg,
        "w_vlm": 0.0,
        "top_k_rerank": max_k,
        "mode": "z_pool",
        "debias_alpha": 0.0,
    }

    mg_cands = [mg for mg in sorted({0.2, 0.35, 0.5, 0.8, 1.2, 1.5, 2.0, 3.5, max_mg}) if mg <= max_mg + 1e-9]
    k_cands = [k for k in (2, 3, 4, 5, 6, 8) if k <= max_k]
    w_cands = (-2.0, -1.6, -1.35, -1.2, -1.0, -0.8, -0.6, -0.45, -0.35, -0.25, -0.15, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 1.05, 1.35)
    active_idx = [i for i, pl in enumerate(pools) if len(pl) >= 2]

    for alpha in (0.0, 0.3, 0.5, 0.75):
        eff_vlm = _debias_vlm_scores(vlm_tune, alpha=alpha, shrink=2.0)
        for k_rr in k_cands:
            for mode in ("z_pool", "margin_raw", "margin_vz"):
                for wv in w_cands:
                    cand_hit1 = base_hit1.copy()
                    cand_rr10 = base_rr10.copy()
                    for idx in active_idx:
                        pl = pools[idx][:k_rr]
                        if len(pl) < 2:
                            continue
                        va = np.asarray([eff_vlm[qids[idx]][v] for v in pl], dtype=np.float64)
                        comb = _combine_stage1_and_vlm_pool(s1_vecs[idx][:k_rr], va, w_vlm=wv, mode=mode)
                        order = sorted(range(len(pl)), key=lambda j: (-comb[j], j))
                        gt = gt_sets[idx]
                        cand_hit1[idx] = 1.0 if pl[order[0]] in gt else 0.0
                        if any(v in gt for v in pl):
                            rr = 0.0
                            for r_pos, oj in enumerate(order):
                                if pl[oj] in gt:
                                    rr = 1.0 / float(r_pos + 1)
                                    break
                            cand_rr10[idx] = rr
                    for mg in mg_cands:
                        mask = margins < mg
                        h1 = float(np.mean(np.where(mask, cand_hit1, base_hit1)))
                        rr = float(np.mean(np.where(mask, cand_rr10, base_rr10)))
                        obj = h1 + 0.15 * rr
                        if obj > best_obj + 1e-6:
                            best_obj = obj
                            best_cfg = {
                                "margin_gate": mg,
                                "w_vlm": wv,
                                "top_k_rerank": k_rr,
                                "mode": mode,
                                "debias_alpha": alpha,
                            }
    return best_cfg


def cmd_fuse_vlm_and_lock(args: argparse.Namespace) -> int:
    state = json.loads((V2_RUN_DIR / "v2_stage1_state.json").read_text(encoding="utf-8"))
    stage1_configs = {k: dict(v) for k, v in state["configs"].items()}
    w_llm_chunk = float(getattr(args, "w_llm_chunk", 0.0) or 0.0)
    llm_chunk_dir = getattr(args, "llm_chunk_dir", None)

    tune_labels = {
        str(r["query_id"]): r
        for g in ("tune_g1", "tune_g2")
        for r in _read_jsonl(VAL_TUNE_DIR / f"{g}_labels.jsonl")
    }
    confirm_labels = {
        str(r["query_id"]): r
        for g in ("confirm_g1", "confirm_g2")
        for r in _read_jsonl(VAL_CONFIRM_DIR / f"{g}_labels.jsonl")
    }

    if w_llm_chunk > 0.0 and llm_chunk_dir is not None:
        r9c_cfg = stage1_configs["R9c"]
        r9c_cfg["w_llm_chunk"] = w_llm_chunk
        bundle = _load_llm_chunk_bundle(Path(llm_chunk_dir))
        ckpt_path = MODELS_DIR / "v2_cross_modal_adapter.pt"
        train_banks, train_dicts = load_all_train_query_banks()
        active_bs = tuple(state["adapter_meta"]["backbones"])
        q_shas = [qs for qs in train_dicts[active_bs[0]].keys() if all(qs in train_dicts[b] for b in active_bs)]
        q_in = {b: np.vstack([train_dicts[b][qs] for qs in q_shas]).astype(np.float32) for b in active_bs}
        ad_q_bank = adapt_vectors_with_checkpoint(ckpt_path, query_feats=q_in, device="cpu")["queries"]["joint"]

        def _recompute_split(split_name: str):
            banks, _, _ = _load_split_galleries(split_name, train_banks=train_banks)
            preds: list[dict[str, Any]] = []
            top20: dict[str, dict[str, float]] = {}
            for b in banks:
                sig = _precompute_gallery_signals(b, ckpt_path=ckpt_path, train_adapter_joint_bank=ad_q_bank, use_mp=True)
                base_m = _build_fused_matrix(sig, **{k: v for k, v in r9c_cfg.items() if k != "w_llm_chunk"})
                c_m = _compute_clause_matrix_for_bank(b, bundle=bundle, clause_field="visual_clauses", b_weights=r9c_cfg["b_weights"])
                mat = base_m + w_llm_chunk * c_m
                for i, q in enumerate(b.queries):
                    ranked = rank_gallery_desc(b.video_ids, mat[i])
                    preds.append({"query_id": q["query_id"], "gallery": b.gallery_name, "ranked_video_ids": ranked})
                    top20[q["query_id"]] = {v: round(float(mat[i][b.vid_to_idx[v]]), 6) for v in ranked[:20]}
            return preds, top20

        tune_preds, tune_top20 = _recompute_split("tune")
        confirm_preds, confirm_top20 = _recompute_split("confirm")
        _write_preds(V2_RUN_DIR / "r9c_with_llm_confirm_preds.jsonl.gz", confirm_preds)
    else:
        tune_preds = _read_preds(V2_RUN_DIR / "r9c_tune_preds.jsonl.gz")
        confirm_preds = _read_preds(V2_RUN_DIR / "r9c_confirm_preds.jsonl.gz")
        tune_top20 = json.loads((V2_RUN_DIR / "r9c_tune_top20_scores.json").read_text(encoding="utf-8"))
        confirm_top20 = json.loads((V2_RUN_DIR / "r9c_confirm_top20_scores.json").read_text(encoding="utf-8"))

    vlm_tune_dir = Path(args.vlm_tune_dir)
    vlm_confirm_dir = Path(args.vlm_confirm_dir)
    vlm_tune = _load_vlm_scores(vlm_tune_dir)
    vlm_confirm = _load_vlm_scores(vlm_confirm_dir)
    vlm_tune_rec = json.loads((_find_shard_dir(vlm_tune_dir) / "receipt.json").read_text(encoding="utf-8"))
    vlm_conf_rec = json.loads((_find_shard_dir(vlm_confirm_dir) / "receipt.json").read_text(encoding="utf-8"))
    vlm_model_meta = vlm_conf_rec.get("model") or vlm_tune_rec.get("model") or {}
    is_3b = "3b" in str(vlm_model_meta.get("model_id", "")).lower()
    champion_id = "R9d_videntra_v2_1_3b_6gb" if is_3b else "R9d_videntra_v2_champion"

    # Tune VLM fusion strictly on Tune
    max_mg = float(state["vlm_export_margin_threshold"])
    max_k = int(state["vlm_export_top_k"])
    best_cfg = _sweep_vlm_fusion_on_tune(
        tune_preds, tune_top20, vlm_tune, tune_labels, max_mg=max_mg, max_k=max_k,
    )

    tune_r9d_preds = _blend_stage1_and_vlm_preds(tune_preds, tune_top20, vlm_tune, **best_cfg)
    confirm_r9d_preds = _blend_stage1_and_vlm_preds(confirm_preds, confirm_top20, vlm_confirm, **best_cfg)
    _write_preds(V2_RUN_DIR / "r9d_confirm_preds.jsonl.gz", confirm_r9d_preds)

    tune_r9d = evaluate_predictions(champion_id, tune_r9d_preds, tune_labels, seed=SEED)
    confirm_r9d = evaluate_predictions(champion_id, confirm_r9d_preds, confirm_labels, seed=SEED)

    v1_receipt_path = PROTOCOL_ROOT / "champion_lock_receipt_v1_r8.json"
    v1_official_confirm = (
        json.loads(v1_receipt_path.read_text(encoding="utf-8")).get("champion_confirm_metrics")
        if v1_receipt_path.is_file()
        else None
    )
    r8_confirm_h1 = max(
        float(state["confirm_results"]["R8_v1_baseline"]["hit_at_1"]),
        float(v1_official_confirm["hit_at_1"]) if v1_official_confirm else 0.502,
    )
    improved = float(confirm_r9d["hit_at_1"]) > max(r8_confirm_h1, 0.502)

    lock_payload = {
        "schema_version": 2,
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "champion_id": champion_id,
        "previous_champion_confirm_hit_at_1": r8_confirm_h1,
        "improved_over_v1_r8": improved,
        "champion_confirm_metrics": confirm_r9d,
        "champion_tune_metrics": tune_r9d,
        "stage1_configs": stage1_configs,
        "vlm_fusion_config": best_cfg,
        "vlm_model": vlm_model_meta,
        "vlm_peak_vram_mb": {
            "tune": vlm_tune_rec.get("peak_vram_mb"),
            "confirm": vlm_conf_rec.get("peak_vram_mb"),
        },
        "vlm_export_top_k": int(state["vlm_export_top_k"]),
        "vlm_export_margin_threshold": float(state["vlm_export_margin_threshold"]),
        "adapter_meta": state["adapter_meta"],
        "config_sha256": sha256_text(json.dumps({"s1": stage1_configs["R9c"], "vlm": best_cfg, "model": vlm_model_meta}, sort_keys=True)),
        "code_sha256": sha256_file(Path(__file__)),
    }
    _write_json(V2_RUN_DIR / "v2_validation_summary.json", {**state, "tune_r9d": tune_r9d, "confirm_r9d": confirm_r9d, "vlm_fusion_config": best_cfg, "vlm_model": vlm_model_meta, "v1_official_confirm": v1_official_confirm})
    _write_json(PROTOCOL_ROOT / "champion_lock_receipt_v2.json", lock_payload)
    if improved:
        main_lock = PROTOCOL_ROOT / "champion_lock_receipt.json"
        if main_lock.is_file() and not v1_receipt_path.is_file():
            shutil.copy2(main_lock, v1_receipt_path)
        _write_json(main_lock, lock_payload)

    # Write Markdown summary
    rows_md = [
        "# VIDENTRA v2.1 (`R9`) Validation Report (Tune → Confirm)\n",
        f"- **Champion**: `{champion_id}` | **VLM**: `{vlm_model_meta.get('model_id')}` (`{vlm_model_meta.get('quant')}`) | **Locked Confirm Hit@1**: `{confirm_r9d['hit_at_1']*100:.2f}%` (`{confirm_r9d['hits_at_1']}/2000`) | **MRR@10**: `{confirm_r9d['mrr_at_10']:.4f}`\n",
        f"- **VLM Config (tuned strictly on Tune)**: `{json.dumps(best_cfg)}`\n",
        "| Config ID | Tune Hit@1 | Confirm Hit@1 | Confirm Hit@5 | Confirm Hit@10 | Confirm MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if v1_official_confirm is not None:
        rows_md.append(
            f"| `R8_videntra_v1_locked` | 49.25% | {v1_official_confirm['hit_at_1']*100:.2f}% | "
            f"{v1_official_confirm['hit_at_5']*100:.2f}% | {v1_official_confirm['hit_at_10']*100:.2f}% | {v1_official_confirm['mrr_at_10']:.4f} |"
        )
    all_keys = list(state["confirm_results"].keys())
    for k in all_keys:
        if k == "R8_v1_baseline" and v1_official_confirm is not None:
            continue
        tm, cm = state["tune_results"][k], state["confirm_results"][k]
        rows_md.append(f"| `{k}` | {tm['hit_at_1']*100:.2f}% | {cm['hit_at_1']*100:.2f}% | {cm['hit_at_5']*100:.2f}% | {cm['hit_at_10']*100:.2f}% | {cm['mrr_at_10']:.4f} |")
    rows_md.append(
        f"| `{champion_id}` | {tune_r9d['hit_at_1']*100:.2f}% | {confirm_r9d['hit_at_1']*100:.2f}% | "
        f"{confirm_r9d['hit_at_5']*100:.2f}% | {confirm_r9d['hit_at_10']*100:.2f}% | {confirm_r9d['mrr_at_10']:.4f} |"
    )
    (V2_RUN_DIR / "v2_validation_report.md").write_text("\n".join(rows_md) + "\n", encoding="utf-8")

    print(json.dumps({"champion_id": champion_id, "tune_r9d": tune_r9d, "confirm_r9d": confirm_r9d, "improved_over_r8": improved, "vlm_fusion_config": best_cfg, "vlm_peak_vram_mb": lock_payload["vlm_peak_vram_mb"]}, indent=2))
    return 0


def cmd_eval_sealed_test(args: argparse.Namespace) -> int:
    lock_doc = json.loads((PROTOCOL_ROOT / "champion_lock_receipt_v2.json").read_text(encoding="utf-8"))
    if not lock_doc.get("improved_over_v1_r8") and not args.force:
        raise RuntimeError("Champion V2.1 did not exceed V1 R8 on Confirm; pass --force only for diagnostic run.")

    champion_id = str(lock_doc.get("champion_id") or "R9d_videntra_v2_1_3b_6gb")
    ckpt_path = MODELS_DIR / "v2_cross_modal_adapter.pt"
    train_banks, train_dicts = load_all_train_query_banks()
    active_bs = tuple(lock_doc["adapter_meta"]["backbones"])
    q_shas = [qs for qs in train_dicts[active_bs[0]].keys() if all(qs in train_dicts[b] for b in active_bs)]
    q_in = {b: np.vstack([train_dicts[b][qs] for qs in q_shas]).astype(np.float32) for b in active_bs}
    ad_q_bank = adapt_vectors_with_checkpoint(ckpt_path, query_feats=q_in, device="cpu")["queries"]["joint"]

    test_banks, test_labels, _ = _load_split_galleries("test_1ka", train_banks=train_banks)
    bank = test_banks[0]
    sigs = {bank.gallery_name: _precompute_gallery_signals(bank, ckpt_path=ckpt_path, train_adapter_joint_bank=ad_q_bank, use_mp=True)}

    s1_cfgs = lock_doc["stage1_configs"]
    r9c_cfg = dict(s1_cfgs["R9c"])
    w_llm_chunk = float(r9c_cfg.pop("w_llm_chunk", 0.0) or 0.0)
    if w_llm_chunk > 0.0:
        if args.llm_chunk_dir is None:
            raise ValueError("Locked config uses w_llm_chunk > 0; pass --llm-chunk-dir for test_1ka.")
        sigs[bank.gallery_name]["llm_clause_mat"] = _compute_clause_matrix_for_bank(
            bank,
            bundle=_load_llm_chunk_bundle(Path(args.llm_chunk_dir)),
            clause_field="visual_clauses",
            b_weights=r9c_cfg["b_weights"],
        )

    vlm_cfg = dict(lock_doc["vlm_fusion_config"])
    use_vlm = abs(float(vlm_cfg.get("w_vlm", 0.0))) > 0.0
    if args.export_vlm_candidates:
        if not use_vlm:
            exp = {"path": None, "total_queries": len(bank.queries), "exported_queries": 0, "vlm_skipped": True}
        else:
            r9c_mat = _build_fused_matrix(sigs[bank.gallery_name], **r9c_cfg)
            if w_llm_chunk > 0.0:
                r9c_mat = r9c_mat + w_llm_chunk * sigs[bank.gallery_name]["llm_clause_mat"]
            mg_thresh = float(vlm_cfg.get("margin_gate", lock_doc["vlm_export_margin_threshold"]))
            top_k = int(vlm_cfg.get("top_k_rerank", lock_doc["vlm_export_top_k"]))
            out_cand = V2_RUN_DIR / "vlm_candidates_test_1ka.jsonl"
            exp_rows: list[dict[str, Any]] = []
            for i, q in enumerate(bank.queries):
                ranked = rank_gallery_desc(bank.video_ids, r9c_mat[i])
                mg = float(r9c_mat[i][bank.vid_to_idx[ranked[0]]] - r9c_mat[i][bank.vid_to_idx[ranked[1]]]) if len(ranked) >= 2 else 0.0
                if mg < mg_thresh:
                    exp_rows.append({"query_id": q["query_id"], "query_text": q["query_text"], "gallery": bank.gallery_name, "stage1_margin": round(mg, 6), "candidate_video_ids": ranked[:top_k]})
            assert_no_gt_leakage(exp_rows, context="vlm_export_test_1ka")
            out_cand.parent.mkdir(parents=True, exist_ok=True)
            with out_cand.open("w", encoding="utf-8") as fh:
                for r in exp_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            exp = {"path": str(out_cand), "total_queries": len(bank.queries), "exported_queries": len(exp_rows)}
        print(json.dumps({"test_1ka_vlm_export": exp}, indent=2))
        if args.vlm_dir is None:
            return 0

    # B0, B1, R8 reference + R9a..R9d single-shot evaluation
    b0_preds = [{"query_id": q["query_id"], "gallery": bank.gallery_name, "ranked_video_ids": rank_gallery_desc(bank.video_ids, compute_b0_siglip_scores(bank, q["query_text"]))} for q in bank.queries]
    b1_preds = [{"query_id": q["query_id"], "gallery": bank.gallery_name, "ranked_video_ids": rank_gallery_desc(bank.video_ids, compute_b1_videoprism_scores(bank, q["query_text"]))} for q in bank.queries]
    b1_hits = {
        p["query_id"]: (1 if p["ranked_video_ids"] and p["ranked_video_ids"][0] in {str(v) for v in test_labels[p["query_id"]]["target_videos"]} else 0)
        for p in b1_preds
    }
    b0_m = evaluate_predictions("B0_siglip2_standalone", b0_preds, test_labels, b1_hits_by_qid=b1_hits, seed=SEED)
    b1_m = evaluate_predictions("B1_videoprism_standalone", b1_preds, test_labels, b1_hits_by_qid=b1_hits, seed=SEED)
    v1_test_path = V2_RUN_DIR / "v1_r8_test_1ka_results.json"
    if v1_test_path.is_file():
        v1_test_res = json.loads(v1_test_path.read_text(encoding="utf-8")).get("results", {})
        r8_m = v1_test_res.get("R8_videntra_champion_candidate") or _eval_r8_v1_reference(test_banks, test_labels, b1_hits)[0]
    else:
        r8_m, _ = _eval_r8_v1_reference(test_banks, test_labels, b1_hits)

    test_metrics: dict[str, Any] = {
        "B0_siglip2_standalone": b0_m,
        "B1_videoprism_standalone": b1_m,
        "R8_videntra_v1_champion": r8_m,
    }
    for cid, bk in (("R9_pe_bigg_standalone", "pe_bigg"), ("R9_dfn5b_h_standalone", "dfn")):
        m, _ = _eval_score_dict(cid, test_banks, {bank.gallery_name: sigs[bank.gallery_name]["raw_mats"][bk]}, test_labels, b1_hits_by_qid=b1_hits)
        test_metrics[cid] = m

    r9c_preds: list[dict[str, Any]] = []
    r9c_top20: dict[str, dict[str, float]] = {}
    for cid, kname in (("R9a_6backbone_raw", "R9a"), ("R9b_6backbone_qb_dis_znorm_mp", "R9b"), ("R9c_6backbone_adapter_multimodal", "R9c")):
        cfg_k = dict(s1_cfgs[kname])
        w_lk = float(cfg_k.pop("w_llm_chunk", 0.0) or 0.0)
        mat = _build_fused_matrix(sigs[bank.gallery_name], **cfg_k)
        if kname == "R9c" and w_lk > 0.0:
            mat = mat + w_lk * sigs[bank.gallery_name]["llm_clause_mat"]
        m, preds = _eval_score_dict(cid, test_banks, {bank.gallery_name: mat}, test_labels, b1_hits_by_qid=b1_hits)
        test_metrics[cid] = m
        if kname == "R9c":
            r9c_preds = preds
            for i, q in enumerate(bank.queries):
                row = mat[i]
                top20_v = rank_gallery_desc(bank.video_ids, row)[:20]
                r9c_top20[q["query_id"]] = {v: round(float(row[bank.vid_to_idx[v]]), 6) for v in top20_v}

    if use_vlm and args.vlm_dir is None:
        raise ValueError("Pass --vlm-dir to evaluate final V2.1 Champion R9d on test_1ka")
    vlm_test = _load_vlm_scores(Path(args.vlm_dir)) if (use_vlm and args.vlm_dir is not None) else {}
    vlm_test_rec = json.loads((_find_shard_dir(Path(args.vlm_dir)) / "receipt.json").read_text(encoding="utf-8")) if (use_vlm and args.vlm_dir is not None) else {}
    r9d_preds = _blend_stage1_and_vlm_preds(r9c_preds, r9c_top20, vlm_test, **vlm_cfg)
    r9d_m = evaluate_predictions(champion_id, r9d_preds, test_labels, b1_hits_by_qid=b1_hits, seed=SEED)
    test_metrics[champion_id] = r9d_m

    v2_7b_ref_path = V2_RUN_DIR / "v2_7b_test_1ka_results.json"
    if v2_7b_ref_path.is_file() and champion_id != "R9d_videntra_v2_champion":
        ref_doc = json.loads(v2_7b_ref_path.read_text(encoding="utf-8"))
        ref_7b = (ref_doc.get("test_1ka_results") or {}).get("R9d_videntra_v2_champion")
        if ref_7b:
            test_metrics["R9d_videntra_v2_7b_cloud_ref"] = {**ref_7b, "config_id": "R9d_videntra_v2_7b_cloud_ref"}

    out_json = {
        "schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "MSR-VTT 1K-A (JSFusion, 1,000 videos / 1,000 sealed queries)",
        "champion_id": champion_id,
        "champion_lock_v2": lock_doc["config_sha256"],
        "confirm_champion_hit_at_1": lock_doc["champion_confirm_metrics"]["hit_at_1"],
        "vlm_model": lock_doc.get("vlm_model"),
        "vlm_peak_vram_mb": {
            **(lock_doc.get("vlm_peak_vram_mb") or {}),
            "test_1ka": vlm_test_rec.get("peak_vram_mb"),
        },
        "test_1ka_results": test_metrics,
    }
    _write_json(BENCHMARK_DIR / "msrvtt_1ka_benchmark_results.json", out_json)

    vlm_desc = f"{(lock_doc.get('vlm_model') or {}).get('model_id', 'Qwen2.5-VL')} ({(lock_doc.get('vlm_model') or {}).get('quant', 'nf4_4bit')})"
    lines = [
        "# MSR-VTT 1K-A Single-Shot Benchmark Results (VIDENTRA v2.1 — RTX 3060 6GB Target)\n",
        f"- **Evaluation date (UTC)**: {out_json['evaluated_at_utc']}",
        f"- **Dataset**: {out_json['dataset']}",
        f"- **Active Champion**: `{champion_id}` | **VLM**: `{vlm_desc}` | **Peak VLM VRAM (Kaggle T4)**: `{vlm_test_rec.get('peak_vram_mb')} MB`",
        f"- **Locked Confirm (`valid_confirm`, 2,000 queries) Hit@1**: `{lock_doc['champion_confirm_metrics']['hit_at_1']*100:.2f}%`",
        f"- **Sealed Test (`test_1ka`, 1,000 queries) Hit@1**: `{r9d_m['hit_at_1']*100:.2f}%` (`{r9d_m['hits_at_1']}/1000`), MRR@10 `{r9d_m['mrr_at_10']:.4f}`, R@5 `{r9d_m['hit_at_5']*100:.2f}%`, R@10 `{r9d_m['hit_at_10']*100:.2f}%`\n",
        "| Config ID | Hit@1 (R@1) | R@5 | R@10 | MRR@10 | Δ Hit@1 vs B1 (pp) | 95% Bootstrap CI (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cid, m in test_metrics.items():
        ci = m["ci95_pp_vs_b1"]
        lines.append(
            f"| `{cid}` | {m['hit_at_1']*100:.2f}% | {m['hit_at_5']*100:.2f}% | {m['hit_at_10']*100:.2f}% | "
            f"{m['mrr_at_10']:.4f} | {m['delta_pp_vs_b1']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |"
        )
    (BENCHMARK_DIR / "msrvtt_1ka_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out_json, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VIDENTRA v2.1 (R9) Evaluation Matrix")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_s1 = sub.add_parser("stage1-and-export")
    p_s1.add_argument("--retrain-adapter", action="store_true")
    p_s1.add_argument("--vlm-top-k", type=int, default=8)
    p_s1.add_argument("--vlm-export-quantile", type=float, default=0.70)

    p_abl = sub.add_parser("ablate-llm-chunks")
    p_abl.add_argument("--llm-chunk-dir", type=Path, default=KAGGLE_OUT / "llm_chunk_val_4000")

    p_fuse = sub.add_parser("fuse-vlm-and-lock")
    p_fuse.add_argument("--vlm-tune-dir", type=Path, default=KAGGLE_OUT / "vlm3b_val_tune")
    p_fuse.add_argument("--vlm-confirm-dir", type=Path, default=KAGGLE_OUT / "vlm3b_val_confirm")
    p_fuse.add_argument("--llm-chunk-dir", type=Path, default=None)
    p_fuse.add_argument("--w-llm-chunk", type=float, default=0.0)

    p_test = sub.add_parser("eval-sealed-test")
    p_test.add_argument("--export-vlm-candidates", action="store_true")
    p_test.add_argument("--vlm-dir", type=Path, default=None)
    p_test.add_argument("--llm-chunk-dir", type=Path, default=None)
    p_test.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "stage1-and-export":
        return cmd_stage1_and_export(args)
    if args.cmd == "ablate-llm-chunks":
        return cmd_ablate_llm_chunks(args)
    if args.cmd == "fuse-vlm-and-lock":
        return cmd_fuse_vlm_and_lock(args)
    if args.cmd == "eval-sealed-test":
        return cmd_eval_sealed_test(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
