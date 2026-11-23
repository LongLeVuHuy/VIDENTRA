"""Unified MSR-VTT 1K-A validation/test experiment matrix (B0-B4, R0-R7 + QB-CSLS/DSN).

Enforces:
- Strict separation between gallery/query inference payloads and ground-truth labels.
- Deterministic video_id tie-breaking and gallery rank preservation.
- Frozen Top-20 pool membership invariance across reranker experiments (R0-R7).
- Paired source-group bootstrap 95% CI relative to B1 (VideoPrism-LvT-B standalone).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.compare_msr_runs import paired_source_group_bootstrap_ci
from videntra.ranking.vedje_reranker import (
    TrainQueryBankHubness,
    compute_spaced_top2_frame_score,
    compute_xpool_score,
    l2_normalize_np,
    score_vedje_reranker_batch,
)
from videntra.relations.kalman_tracker import (
    TrackRecord,
    TrackStep,
    classify_motion_relation_query,
    compute_track_query_signals,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OCR_INTENT_RE = re.compile(
    r"\b(?:text|sign|title|caption|word|words|letter|letters|logo|banner|"
    r"screen|written|read|reading|says|showing\s+the\s+word|subtitle|name)\b",
    re.I,
)
_ASR_INTENT_RE = re.compile(
    r"\b(?:say|says|saying|said|speak|speaks|speaking|talk|talks|talking|"
    r"sing|sings|singing|song|lyrics|tell|tells|telling|ask|asks|asking|"
    r"explain|explains|explaining|narrat|interview|speech|voice|announce)\b",
    re.I,
)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "into", "onto", "that", "this", "it", "its",
    "his", "her", "their", "they", "he", "she", "who", "while", "during",
})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_list(items: Sequence[str]) -> str:
    payload = json.dumps(list(items), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenize_sparse(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(str(text or "").lower()) if len(t) > 1 and t not in _STOPWORDS]


def classify_modality_query_intent(query_text: str) -> dict[str, bool]:
    q = str(query_text or "")
    has_ocr_cue = bool(_OCR_INTENT_RE.search(q))
    has_asr_cue = bool(_ASR_INTENT_RE.search(q))
    return {
        "has_ocr_cue": has_ocr_cue,
        "has_asr_cue": has_asr_cue,
        "has_any_lexical_cue": bool(has_ocr_cue or has_asr_cue),
    }


def zscore_normalize_row(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    mu = float(np.mean(a))
    sd = float(np.std(a))
    return (a - mu) / max(eps, sd)


def minmax_normalize_row(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo <= eps:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def rank_gallery_desc(video_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    """Sort gallery video_ids descending by finite score, tie-breaking strictly ascending by video_id."""
    order = sorted(
        range(len(video_ids)),
        key=lambda i: (-float(scores[i]) if math.isfinite(float(scores[i])) else 1e9, str(video_ids[i])),
    )
    return [str(video_ids[i]) for i in order]


def build_bm25_scorer(docs_by_vid: Mapping[str, str], k1: float = 1.2, b: float = 0.75) -> Callable[[str, Sequence[str]], np.ndarray]:
    """Build a deterministic BM25 scorer over sparse text docs keyed by video_id."""
    n_docs = max(1, len(docs_by_vid))
    tokenized: dict[str, Counter[str]] = {}
    doc_len: dict[str, int] = {}
    df: Counter[str] = Counter()
    for vid, txt in docs_by_vid.items():
        toks = tokenize_sparse(txt)
        c = Counter(toks)
        tokenized[vid] = c
        doc_len[vid] = len(toks)
        for tok in c:
            df[tok] += 1
    avgdl = max(1.0, sum(doc_len.values()) / float(n_docs))
    idf = {tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def _score(query_text: str, cand_vids: Sequence[str]) -> np.ndarray:
        q_toks = tokenize_sparse(query_text)
        out = np.zeros(len(cand_vids), dtype=np.float64)
        if not q_toks:
            return out
        for i, vid in enumerate(cand_vids):
            tf_map = tokenized.get(vid)
            if not tf_map:
                continue
            dl = float(doc_len.get(vid, 0))
            norm_len = 1.0 - b + b * (dl / avgdl)
            sc = 0.0
            for t in q_toks:
                tf = tf_map.get(t, 0)
                if tf <= 0:
                    continue
                t_idf = idf.get(t, 0.0)
                sc += t_idf * ((tf * (k1 + 1.0)) / (tf + k1 * norm_len))
            out[i] = sc
        return out

    return _score


@dataclass
class GalleryFeatureBank:
    """Self-contained inference-only feature bank for one 1,000-video gallery (zero GT fields)."""

    gallery_name: str
    video_ids: list[str]
    vid_to_idx: dict[str, int]
    queries: list[dict[str, str]]  # [{query_id, query_text, query_text_sha256}]

    # VideoPrism features
    vp_whole_vecs: np.ndarray | None = None          # (N_vid, 768)
    vp_window_vecs: np.ndarray | None = None         # (N_win, 768)
    vp_win_vid_indices: np.ndarray | None = None     # (N_win,) int32
    vp_video_win_positions: dict[str, np.ndarray] = field(default_factory=dict)
    vp_query_map: dict[str, np.ndarray] = field(default_factory=dict)  # q_sha -> (768,)

    # SigLIP2 features
    sig_frame_vecs: np.ndarray | None = None         # (N_kf, 768) float32/float16
    sig_frame_pts: np.ndarray | None = None          # (N_kf,) float64
    sig_video_kf_pos: dict[str, np.ndarray] = field(default_factory=dict)
    sig_kf_vid_indices: np.ndarray | None = None     # (N_kf,) int32
    sig_clip_mean_vecs: np.ndarray | None = None     # (N_vid, 768)
    sig_query_map: dict[str, np.ndarray] = field(default_factory=dict)

    # Offline Florence-2 captions + E5 dense vectors
    florence_captions: dict[str, str] = field(default_factory=dict)
    cap_e5_vecs: np.ndarray | None = None            # (N_vid, 384)
    e5_query_map: dict[str, np.ndarray] = field(default_factory=dict)

    # OCR / ASR text docs & coverage flags
    ocr_docs: dict[str, str] = field(default_factory=dict)
    ocr_status: dict[str, str] = field(default_factory=dict)
    asr_docs: dict[str, str] = field(default_factory=dict)
    asr_status: dict[str, str] = field(default_factory=dict)

    # YOLO + Kalman SORT/ByteTrack records per video
    kf_detections: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sort_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    bytetrack_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Precomputed Train-9K Query-Bank hubness priors
    vp_hub_priors: dict[str, np.ndarray] | None = None
    sig_hub_priors: dict[str, np.ndarray] | None = None

    # Dual high-resolution vision-language features (PE-Core-L-14-336 & SigLIP2-SO400M-384)
    pe_video_vecs: np.ndarray | None = None          # (N_vid, 1024)
    pe_frame_vecs: np.ndarray | None = None          # (N_vid, F, 1024)
    pe_query_map: dict[str, np.ndarray] = field(default_factory=dict)
    sig400_video_vecs: np.ndarray | None = None      # (N_vid, 1152)
    sig400_frame_vecs: np.ndarray | None = None      # (N_vid, F, 1152)
    sig400_query_map: dict[str, np.ndarray] = field(default_factory=dict)
    pe_hub_priors: dict[str, np.ndarray] | None = None
    sig400_hub_priors: dict[str, np.ndarray] | None = None

    # Dual flagship vision-language features (PE-Core-bigG-14-448 & DFN5B-CLIP-ViT-H-14-378)
    pe_bigg_video_vecs: np.ndarray | None = None     # (N_vid, 1280)
    pe_bigg_frame_vecs: np.ndarray | None = None     # (N_vid, F, 1280)
    pe_bigg_query_map: dict[str, np.ndarray] = field(default_factory=dict)
    dfn_video_vecs: np.ndarray | None = None         # (N_vid, 1024)
    dfn_frame_vecs: np.ndarray | None = None         # (N_vid, F, 1024)
    dfn_query_map: dict[str, np.ndarray] = field(default_factory=dict)
    pe_bigg_hub_priors: dict[str, np.ndarray] | None = None
    dfn_hub_priors: dict[str, np.ndarray] | None = None

    def build_derived_caches(
        self,
        train_vp_bank: np.ndarray | None = None,
        train_sig_bank: np.ndarray | None = None,
        train_pe_bank: np.ndarray | None = None,
        train_sig400_bank: np.ndarray | None = None,
        train_pe_bigg_bank: np.ndarray | None = None,
        train_dfn_bank: np.ndarray | None = None,
    ) -> None:
        if self.sig_frame_vecs is not None and self.sig_clip_mean_vecs is None:
            means = np.zeros((len(self.video_ids), self.sig_frame_vecs.shape[1]), dtype=np.float32)
            for idx, vid in enumerate(self.video_ids):
                pos = self.sig_video_kf_pos.get(vid)
                if pos is not None and len(pos) > 0:
                    m = np.mean(self.sig_frame_vecs[pos].astype(np.float32), axis=0)
                    means[idx] = l2_normalize_np(m)
            self.sig_clip_mean_vecs = means

        for bank_vecs, vid_vecs, attr in (
            (train_vp_bank, self.vp_whole_vecs, "vp_hub_priors"),
            (train_sig_bank, self.sig_clip_mean_vecs, "sig_hub_priors"),
            (train_pe_bank, self.pe_video_vecs, "pe_hub_priors"),
            (train_sig400_bank, self.sig400_video_vecs, "sig400_hub_priors"),
            (train_pe_bigg_bank, self.pe_bigg_video_vecs, "pe_bigg_hub_priors"),
            (train_dfn_bank, self.dfn_video_vecs, "dfn_hub_priors"),
        ):
            if bank_vecs is not None and vid_vecs is not None:
                qb = TrainQueryBankHubness(bank_vecs, k_hub=20, temperature=0.05)
                setattr(self, attr, qb.compute_video_priors(vid_vecs))


def evaluate_predictions(
    config_id: str,
    per_query_preds: Sequence[Mapping[str, Any]],
    labels_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    b1_hits_by_qid: Mapping[str, int] | None = None,
    stage1_pools_by_qid: Mapping[str, Sequence[str]] | None = None,
    latencies_ms: Sequence[float] | None = None,
    seed: int = 20261122,
    extra_slices: Mapping[str, Callable[[str], bool]] | None = None,
    n_boot: int = 10000,
) -> dict[str, Any]:
    """Evaluate predictions against frozen labels with paired source-group bootstrap CI vs B1."""
    n = len(per_query_preds)
    if n == 0:
        raise ValueError(f"Empty predictions for {config_id}")

    h1_sum = h5_sum = h10_sum = s1_r20_sum = 0
    mrr10_sum = 0.0
    paired_rows: list[dict[str, Any]] = []
    slice_hits: dict[str, list[int]] = defaultdict(list)
    b1_slice_hits: dict[str, list[int]] = defaultdict(list)

    for rec in per_query_preds:
        qid = str(rec["query_id"])
        lbl = labels_by_qid[qid]
        target_set = {str(v) for v in lbl["target_videos"]}
        sg = str(lbl["source_group"])
        ranked = [str(v) for v in rec["ranked_video_ids"]]

        h1 = 1 if ranked and ranked[0] in target_set else 0
        h5 = 1 if any(v in target_set for v in ranked[:5]) else 0
        h10 = 1 if any(v in target_set for v in ranked[:10]) else 0
        rr10 = 0.0
        for r_idx, vid in enumerate(ranked[:10], 1):
            if vid in target_set:
                rr10 = 1.0 / r_idx
                break

        s1_pool = stage1_pools_by_qid[qid] if stage1_pools_by_qid and qid in stage1_pools_by_qid else ranked[:20]
        r20 = 1 if any(v in target_set for v in s1_pool[:20]) else 0

        h1_sum += h1
        h5_sum += h5
        h10_sum += h10
        s1_r20_sum += r20
        mrr10_sum += rr10

        b1_hit = int(b1_hits_by_qid[qid]) if b1_hits_by_qid and qid in b1_hits_by_qid else h1
        paired_rows.append({
            "query_id": qid,
            "source_group": sg,
            "base_hit": b1_hit,
            "new_hit": h1,
        })

        if extra_slices:
            for sname, pred_fn in extra_slices.items():
                if pred_fn(qid):
                    slice_hits[sname].append(h1)
                    b1_slice_hits[sname].append(b1_hit)

    wins = sum(1 for r in paired_rows if r["base_hit"] == 0 and r["new_hit"] == 1)
    losses = sum(1 for r in paired_rows if r["base_hit"] == 1 and r["new_hit"] == 0)
    boot = paired_source_group_bootstrap_ci(paired_rows, n_resamples=max(1, int(n_boot)), seed=seed)

    lat_arr = np.asarray(list(latencies_ms), dtype=np.float64) if latencies_ms else np.zeros(1, dtype=np.float64)
    slice_summary = {
        sname: {
            "n": len(vals),
            "hit_at_1": round(sum(vals) / max(1, len(vals)), 6),
            "hits": int(sum(vals)),
            "b1_hit_at_1": round(sum(b1_slice_hits[sname]) / max(1, len(b1_slice_hits[sname])), 6),
        }
        for sname, vals in sorted(slice_hits.items())
    }

    return {
        "config_id": config_id,
        "n_queries": n,
        "hits_at_1": h1_sum,
        "hit_at_1": round(h1_sum / n, 6),
        "hit_at_5": round(h5_sum / n, 6),
        "hit_at_10": round(h10_sum / n, 6),
        "mrr_at_10": round(mrr10_sum / n, 6),
        "stage1_recall_at_20": round(s1_r20_sum / n, 6),
        "rescued_vs_b1": wins,
        "lost_vs_b1": losses,
        "net_vs_b1": wins - losses,
        "delta_pp_vs_b1": boot["delta_pp"],
        "ci95_pp_vs_b1": boot["ci95_pp"],
        "n_source_groups": boot["n_source_groups"],
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "slices": slice_summary,
    }


def compute_b0_siglip_scores(
    bank: GalleryFeatureBank,
    query_text: str,
    *,
    pooling: str = "frame_plus_mean",
    frame_weight: float = 0.65,
) -> np.ndarray:
    """B0: Standalone SigLIP2 cosine + frame/mean pooling across the full gallery."""
    q_sha = sha256_text(query_text)
    q_vec = bank.sig_query_map.get(q_sha)
    if q_vec is None or bank.sig_frame_vecs is None or bank.sig_clip_mean_vecs is None:
        return np.zeros(len(bank.video_ids), dtype=np.float64)

    q_norm = l2_normalize_np(q_vec)
    mean_sims = (bank.sig_clip_mean_vecs @ q_norm).astype(np.float64)
    if pooling == "video_mean":
        return mean_sims

    f_vecs = bank.sig_frame_vecs if bank.sig_frame_vecs.dtype == np.float32 else bank.sig_frame_vecs.astype(np.float32)
    all_f_sims = (f_vecs @ q_norm).astype(np.float64)
    if bank.sig_kf_vid_indices is not None and bank.sig_kf_vid_indices.size == all_f_sims.size:
        max_sims = np.full(len(bank.video_ids), -1e9, dtype=np.float64)
        np.maximum.at(max_sims, bank.sig_kf_vid_indices, all_f_sims)
        missing = max_sims < -1e8
        if np.any(missing):
            max_sims[missing] = mean_sims[missing]
    else:
        max_sims = np.zeros(len(bank.video_ids), dtype=np.float64)
        for idx, vid in enumerate(bank.video_ids):
            pos = bank.sig_video_kf_pos.get(vid)
            if pos is not None and len(pos) > 0:
                max_sims[idx] = float(np.max(all_f_sims[pos]))
            else:
                max_sims[idx] = mean_sims[idx]

    if pooling == "frame_max":
        return max_sims
    return float(frame_weight) * max_sims + (1.0 - float(frame_weight)) * mean_sims


def compute_b1_videoprism_scores(bank: GalleryFeatureBank, query_text: str) -> np.ndarray:
    """B1: Standalone public VideoPrism-LvT-B cosine (1 whole-video vec x 1 original-query vec)."""
    q_sha = sha256_text(query_text)
    q_vec = bank.vp_query_map.get(q_sha)
    if q_vec is None or bank.vp_whole_vecs is None:
        return np.zeros(len(bank.video_ids), dtype=np.float64)
    return (bank.vp_whole_vecs @ l2_normalize_np(q_vec)).astype(np.float64)


def compute_b2_videoprism_window_scores(
    bank: GalleryFeatureBank,
    query_text: str,
    *,
    whole_weight: float = 0.65,
    mode: str = "blend_max",
) -> tuple[np.ndarray, np.ndarray]:
    """B2: VideoPrism whole + multi-window pooling across 100% of clips. Returns (combined, win_max)."""
    whole_sims = compute_b1_videoprism_scores(bank, query_text)
    q_sha = sha256_text(query_text)
    q_vec = bank.vp_query_map.get(q_sha)
    if q_vec is None or bank.vp_window_vecs is None or len(bank.vp_window_vecs) == 0:
        return whole_sims, whole_sims.copy()

    w_all = (bank.vp_window_vecs @ l2_normalize_np(q_vec)).astype(np.float64)
    win_max = whole_sims.copy()

    if bank.vp_win_vid_indices is not None and bank.vp_win_vid_indices.size == w_all.size:
        valid = bank.vp_win_vid_indices >= 0
        np.maximum.at(win_max, bank.vp_win_vid_indices[valid], w_all[valid])

    w = float(whole_weight)
    if mode == "win_max_only":
        return win_max, win_max
    if mode == "blend_top2":
        win_top2 = whole_sims.copy()
        for idx, vid in enumerate(bank.video_ids):
            pos = bank.vp_video_win_positions.get(vid)
            if pos is not None and len(pos) > 0:
                sv = np.sort(w_all[pos])
                win_top2[idx] = float(np.mean(sv[-min(2, len(sv)):]))
        return w * whole_sims + (1.0 - w) * win_top2, win_max
    if mode == "max_gate":
        blend = w * whole_sims + (1.0 - w) * win_max
        return np.maximum(whole_sims, blend), win_max
    return w * whole_sims + (1.0 - w) * win_max, win_max


def compute_frame_bank_scores(
    video_vecs: np.ndarray | None,
    frame_vecs: np.ndarray | None,
    query_vec: np.ndarray | None,
    *,
    mode: str = "mean_plus_xpool",
    xpool_weight: float = 0.40,
    temperature: float = 0.07,
) -> np.ndarray:
    """Compute vectorized video mean + frame X-Pool/max scores over `[N, F, D]` cached frame embeddings."""
    if video_vecs is None:
        return np.zeros(0, dtype=np.float64)
    n_vids = int(video_vecs.shape[0])
    if query_vec is None or n_vids == 0:
        return np.zeros(n_vids, dtype=np.float64)

    q_norm = l2_normalize_np(query_vec)
    mean_sims = (video_vecs.astype(np.float32, copy=False) @ q_norm).astype(np.float64)
    if mode == "mean" or frame_vecs is None:
        return mean_sims

    temp = max(1e-4, float(temperature))
    xpool_sims = np.zeros(n_vids, dtype=np.float64)
    max_sims = np.zeros(n_vids, dtype=np.float64)
    top3_sims = np.zeros(n_vids, dtype=np.float64)
    chunk = 256
    for st in range(0, n_vids, chunk):
        ed = min(n_vids, st + chunk)
        blk = frame_vecs[st:ed].astype(np.float32, copy=False)  # (C, F, D)
        raw_sims = np.einsum("nfd,d->nf", blk, q_norm).astype(np.float64)
        max_row = np.max(raw_sims, axis=1)
        max_sims[st:ed] = max_row
        top_k = min(3, raw_sims.shape[1])
        top3_sims[st:ed] = np.mean(np.sort(raw_sims, axis=1)[:, -top_k:], axis=1)
        if mode in ("xpool", "mean_plus_xpool", "mean_xpool_top3"):
            logits = (raw_sims - max_row[:, None]) / temp
            w = np.exp(logits)
            w /= np.maximum(1e-12, np.sum(w, axis=1, keepdims=True))
            pooled = np.einsum("nf,nfd->nd", w.astype(np.float32), blk)
            p_norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(p_norm, 1e-8)
            xpool_sims[st:ed] = (pooled @ q_norm).astype(np.float64)

    if mode == "frame_max":
        return max_sims
    if mode == "xpool":
        return xpool_sims
    if mode == "mean_xpool_top3":
        return 0.20 * mean_sims + 0.55 * xpool_sims + 0.25 * top3_sims
    if mode == "mean_plus_max":
        return (1.0 - float(xpool_weight)) * mean_sims + float(xpool_weight) * max_sims
    return (1.0 - float(xpool_weight)) * mean_sims + float(xpool_weight) * xpool_sims


def _compute_single_vl2_matrices(
    video_vecs: np.ndarray | None,
    frame_vecs: np.ndarray | None,
    query_map: Mapping[str, np.ndarray],
    queries: Sequence[Mapping[str, str]],
    n_vids: int,
    *,
    prefix: str,
    use_mp: bool,
    temperature: float,
) -> dict[str, np.ndarray]:
    nq = len(queries)
    zeros = np.zeros((nq, n_vids), dtype=np.float64)
    if video_vecs is None or n_vids == 0 or nq == 0:
        return {f"{prefix}_mean": zeros, f"{prefix}_max": zeros, f"{prefix}_xpool": zeros, f"{prefix}_top3": zeros, f"{prefix}_blend": zeros}

    dim = int(video_vecs.shape[1])
    q_mat = np.zeros((nq, dim), dtype=np.float32)
    for i, q in enumerate(queries):
        qt = q["query_text"]
        qv = compute_multiprompt_query_vector(query_map, qt) if use_mp else query_map.get(sha256_text(qt))
        if qv is not None:
            q_mat[i] = l2_normalize_np(qv)

    mean_m = (q_mat @ video_vecs.astype(np.float32, copy=False).T).astype(np.float64)
    if frame_vecs is None or frame_vecs.ndim != 3 or frame_vecs.shape[1] == 0:
        return {f"{prefix}_mean": mean_m, f"{prefix}_max": mean_m, f"{prefix}_xpool": mean_m, f"{prefix}_top3": mean_m, f"{prefix}_blend": mean_m}

    nf = int(frame_vecs.shape[1])
    f32 = frame_vecs.astype(np.float32, copy=False)
    gram = np.einsum("vfd,vsd->vfs", f32, f32).astype(np.float64)  # (N_v, F, F)
    f_sims = (q_mat @ f32.reshape(n_vids * nf, dim).T).reshape(nq, n_vids, nf).astype(np.float64)
    max_m = np.max(f_sims, axis=2)
    top_k = min(3, nf)
    top3_m = np.mean(np.sort(f_sims, axis=2)[:, :, -top_k:], axis=2)

    temp = max(1e-4, float(temperature))
    w = np.exp((f_sims - max_m[:, :, None]) / temp)
    w /= np.maximum(1e-12, np.sum(w, axis=2, keepdims=True))
    xp_num = np.sum(w * f_sims, axis=2)
    xp_den = np.sqrt(np.maximum(1e-12, np.einsum("qvf,vfs,qvs->qv", w, gram, w)))
    xp_m = xp_num / xp_den
    blend_m = 0.20 * mean_m + 0.55 * xp_m + 0.25 * top3_m
    return {
        f"{prefix}_mean": mean_m,
        f"{prefix}_max": max_m,
        f"{prefix}_xpool": xp_m,
        f"{prefix}_top3": top3_m,
        f"{prefix}_blend": blend_m,
    }


def compute_vl2_gallery_score_matrices(
    bank: GalleryFeatureBank,
    *,
    use_mp: bool = False,
    temperature: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute vectorized `(N_q, N_v)` PE-Core and SigLIP2-SO400M score matrices via F×F frame Gram matrices."""
    n_vids = len(bank.video_ids)
    out = _compute_single_vl2_matrices(
        bank.pe_video_vecs, bank.pe_frame_vecs, bank.pe_query_map, bank.queries, n_vids,
        prefix="pe", use_mp=use_mp, temperature=temperature,
    )
    out.update(_compute_single_vl2_matrices(
        bank.sig400_video_vecs, bank.sig400_frame_vecs, bank.sig400_query_map, bank.queries, n_vids,
        prefix="s4", use_mp=use_mp, temperature=temperature,
    ))
    return out


def compute_vl3_gallery_score_matrices(
    bank: GalleryFeatureBank,
    *,
    use_mp: bool = False,
    temperature: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute vectorized `(N_q, N_v)` PE-Core-bigG and DFN5B-H score matrices via F×F frame Gram matrices."""
    n_vids = len(bank.video_ids)
    out = _compute_single_vl2_matrices(
        bank.pe_bigg_video_vecs, bank.pe_bigg_frame_vecs, bank.pe_bigg_query_map, bank.queries, n_vids,
        prefix="pe_bigg", use_mp=use_mp, temperature=temperature,
    )
    out.update(_compute_single_vl2_matrices(
        bank.dfn_video_vecs, bank.dfn_frame_vecs, bank.dfn_query_map, bank.queries, n_vids,
        prefix="dfn", use_mp=use_mp, temperature=temperature,
    ))
    return out


def apply_hubness_csls(
    raw_scores: np.ndarray,
    hub_priors: Mapping[str, np.ndarray] | None,
    *,
    csls_weight: float = 0.45,
) -> np.ndarray:
    """Subtract precomputed train-9K query-bank hubness prior (`mean_topk`) deterministically."""
    sims = np.asarray(raw_scores, dtype=np.float64)
    if hub_priors is None or float(csls_weight) == 0.0:
        return sims
    mean_topk = hub_priors.get("mean_topk")
    if mean_topk is None:
        return sims
    return sims - float(csls_weight) * np.asarray(mean_topk, dtype=np.float64)


def fuse_b3_vp_and_siglip(
    video_ids: Sequence[str],
    vp_scores: np.ndarray,
    sig_scores: np.ndarray,
    *,
    mode: str = "zscore",
    vp_weight: float = 0.70,
    rrf_k: float = 60.0,
) -> np.ndarray:
    """B3: Fusion of VideoPrism and SigLIP2 via weighted RRF or normalized score fusion."""
    if mode == "rrf":
        n = len(video_ids)
        ord_vp = np.argsort(-vp_scores, kind="mergesort")
        r_vp = np.empty(n, dtype=np.float64)
        r_vp[ord_vp] = np.arange(1, n + 1, dtype=np.float64)

        ord_sig = np.argsort(-sig_scores, kind="mergesort")
        r_sig = np.empty(n, dtype=np.float64)
        r_sig[ord_sig] = np.arange(1, n + 1, dtype=np.float64)
        return float(vp_weight) / (rrf_k + r_vp) + (1.0 - float(vp_weight)) / (rrf_k + r_sig)
    if mode == "minmax":
        return float(vp_weight) * minmax_normalize_row(vp_scores) + (1.0 - float(vp_weight)) * minmax_normalize_row(sig_scores)
    return float(vp_weight) * zscore_normalize_row(vp_scores) + (1.0 - float(vp_weight)) * zscore_normalize_row(sig_scores)


def rerank_fixed_pool(
    pool_vids: Sequence[str],
    stage1_scores: Sequence[float],
    signal_scores: Sequence[float],
    *,
    mode: str = "norm_blend",
    alpha: float = 0.25,
    rrf_k: float = 60.0,
) -> tuple[list[str], list[float]]:
    """Rerank a fixed Top-20 pool without changing pool membership."""
    k = len(pool_vids)
    if k == 0:
        return [], []
    if float(alpha) == 0.0:
        return list(pool_vids), list(stage1_scores)

    s1 = np.asarray(stage1_scores, dtype=np.float64)
    s2 = np.asarray(signal_scores, dtype=np.float64)
    if mode == "rrf":
        r1 = np.arange(1, k + 1, dtype=np.float64)
        ord2 = sorted(range(k), key=lambda i: (-s2[i], i))
        r2 = np.empty(k, dtype=np.float64)
        for rank_pos, idx in enumerate(ord2, 1):
            r2[idx] = float(rank_pos)
        comb = (1.0 - float(alpha)) / (rrf_k + r1) + float(alpha) / (rrf_k + r2)
    else:
        n1 = zscore_normalize_row(s1)
        n2 = zscore_normalize_row(s2)
        comb = (1.0 - float(alpha)) * n1 + float(alpha) * n2

    new_order = sorted(range(k), key=lambda i: (-float(comb[i]), i))
    return [str(pool_vids[i]) for i in new_order], [float(comb[i]) for i in new_order]


def apply_confidence_gated_rerank(
    pool_vids: Sequence[str],
    stage1_scores: Sequence[float],
    rerank_scores: Sequence[float],
    *,
    top1_protect_margin: float = 0.0,
) -> list[str]:
    """Protect confident Stage-1 Top-1 when normalized margin exceeds threshold (R7 improvement)."""
    k = len(pool_vids)
    if k <= 1:
        return list(pool_vids)
    s1 = zscore_normalize_row(np.asarray(stage1_scores, dtype=np.float64))
    margin = float(s1[0] - s1[1])
    new_order = sorted(range(k), key=lambda i: (-float(rerank_scores[i]), i))
    if top1_protect_margin > 0.0 and margin >= float(top1_protect_margin) and new_order[0] != 0:
        # Keep rank-1 pinned and sort remaining candidates by rerank_scores
        rest = [i for i in new_order if i != 0]
        return [str(pool_vids[0])] + [str(pool_vids[i]) for i in rest]
    return [str(pool_vids[i]) for i in new_order]


def compute_multiprompt_query_vector(
    query_map: Mapping[str, np.ndarray],
    query_text: str,
    *,
    w_orig: float = 0.65,
    w_clip: float = 0.25,
    w_of: float = 0.10,
) -> np.ndarray | None:
    """Pool normalized query embeddings across original text + video prompt templates (R7)."""
    v_orig = query_map.get(sha256_text(query_text))
    if v_orig is None:
        return None
    acc = float(w_orig) * l2_normalize_np(v_orig)
    v_clip = query_map.get(sha256_text(f"a video clip showing {query_text}"))
    if v_clip is not None:
        acc = acc + float(w_clip) * l2_normalize_np(v_clip)
    v_of = query_map.get(sha256_text(f"a video of {query_text}"))
    if v_of is not None:
        acc = acc + float(w_of) * l2_normalize_np(v_of)
    return l2_normalize_np(acc)


def build_top20_vedje_example(
    bank: GalleryFeatureBank,
    query_text: str,
    pool_vids: Sequence[str],
    stage1_prior: Sequence[float],
    *,
    target_vid: str | None = None,
    use_multiprompt: bool = False,
) -> dict[str, Any]:
    """Construct one Top-20 feature bundle for VEDJEJointRerankerNet training or inference."""
    vp_q = (
        compute_multiprompt_query_vector(bank.vp_query_map, query_text)
        if use_multiprompt
        else bank.vp_query_map.get(sha256_text(query_text))
    )
    sig_q = (
        compute_multiprompt_query_vector(bank.sig_query_map, query_text)
        if use_multiprompt
        else bank.sig_query_map.get(sha256_text(query_text))
    )
    if vp_q is None:
        vp_q = np.zeros(768, dtype=np.float32)
    if sig_q is None:
        sig_q = np.zeros(768, dtype=np.float32)

    vp_q = l2_normalize_np(vp_q)
    sig_q = l2_normalize_np(sig_q)

    k = len(pool_vids)
    vp_whole = np.zeros((k, 768), dtype=np.float32)
    vp_wins: list[np.ndarray] = []
    sig_frames: list[np.ndarray] = []
    scalars = np.zeros((k, 10), dtype=np.float32)

    prior_arr = np.asarray(stage1_prior, dtype=np.float32)
    prior_z = zscore_normalize_row(prior_arr).astype(np.float32)

    target_pos = 0
    for i, vid in enumerate(pool_vids):
        if target_vid is not None and vid == target_vid:
            target_pos = i
        v_idx = bank.vid_to_idx.get(vid, -1)
        w_vec = bank.vp_whole_vecs[v_idx] if (v_idx >= 0 and bank.vp_whole_vecs is not None) else np.zeros(768, dtype=np.float32)
        vp_whole[i] = w_vec

        w_pos = bank.vp_video_win_positions.get(vid)
        if w_pos is not None and len(w_pos) > 0 and bank.vp_window_vecs is not None:
            wins_mat = bank.vp_window_vecs[w_pos].astype(np.float32)
        else:
            wins_mat = w_vec[None, :]
        vp_wins.append(wins_mat)

        f_pos = bank.sig_video_kf_pos.get(vid)
        if f_pos is not None and len(f_pos) > 0 and bank.sig_frame_vecs is not None:
            fr_mat = bank.sig_frame_vecs[f_pos].astype(np.float32)
            fr_pts = bank.sig_frame_pts[f_pos] if bank.sig_frame_pts is not None else np.arange(len(f_pos), dtype=np.float64)
        else:
            fr_mat = np.zeros((1, 768), dtype=np.float32)
            fr_pts = np.zeros(1, dtype=np.float64)

        vp_w_sim = float(np.dot(w_vec, vp_q))
        win_sims = wins_mat @ vp_q
        vp_win_max = float(np.max(win_sims))
        vp_win_mean = float(np.mean(win_sims))

        f_sims = fr_mat @ sig_q
        sig_f_max = float(np.max(f_sims))
        sig_xpool = compute_xpool_score(fr_mat, sig_q, temperature=0.07)
        sig_top2 = compute_spaced_top2_frame_score(fr_mat, fr_pts, sig_q, min_pts_gap=1.5)
        sig_mean = (
            float(np.dot(bank.sig_clip_mean_vecs[v_idx], sig_q))
            if (v_idx >= 0 and bank.sig_clip_mean_vecs is not None)
            else float(np.mean(f_sims))
        )
        if fr_mat.shape[0] > 12:
            top_f = np.argsort(-f_sims, kind="mergesort")[:12]
            fr_mat = fr_mat[top_f]
        sig_frames.append(fr_mat)

        vp_hub = float(bank.vp_hub_priors["mean_topk"][v_idx]) if (v_idx >= 0 and bank.vp_hub_priors is not None) else 0.0
        sig_hub = float(bank.sig_hub_priors["mean_topk"][v_idx]) if (v_idx >= 0 and bank.sig_hub_priors is not None) else 0.0
        scalars[i] = np.array([
            vp_w_sim,
            vp_win_max,
            vp_win_mean,
            sig_f_max,
            sig_xpool,
            sig_top2,
            sig_mean,
            prior_z[i],
            vp_w_sim - 0.5 * vp_hub,
            sig_mean - 0.5 * sig_hub,
        ], dtype=np.float32)

    return {
        "vp_q": vp_q,
        "sig_q": sig_q,
        "vp_whole": vp_whole,
        "vp_wins": vp_wins,
        "sig_frames": sig_frames,
        "scalars": scalars,
        "stage1_prior": prior_z,
        "target_pos": target_pos,
    }


def score_r6_candidates(
    bank: GalleryFeatureBank,
    query_text: str,
    pool_vids: Sequence[str],
    *,
    variant: str = "global_crop_track_bt",
) -> np.ndarray:
    """Compute R6 crop and/or Kalman track scores for a Top-20 pool."""
    out = np.zeros(len(pool_vids), dtype=np.float64)
    if variant == "global":
        return out
    for i, vid in enumerate(pool_vids):
        f_obs = bank.kf_detections.get(vid, ())
        if variant == "global_crop":
            sigs = compute_track_query_signals(query_text, {"tracks": (), "num_frames": max(1, len(f_obs))}, frame_observations=f_obs)
            out[i] = sigs["crop_match_score"]
        elif variant == "global_crop_track_sort":
            t_res = bank.sort_tracks.get(vid, {"tracks": (), "num_frames": max(1, len(f_obs))})
            sigs = compute_track_query_signals(query_text, t_res, frame_observations=f_obs)
            out[i] = 0.45 * sigs["crop_match_score"] + 0.55 * sigs["track_score"]
        else:  # global_crop_track_bt
            t_res = bank.bytetrack_tracks.get(vid, {"tracks": (), "num_frames": max(1, len(f_obs))})
            sigs = compute_track_query_signals(query_text, t_res, frame_observations=f_obs)
            out[i] = 0.45 * sigs["crop_match_score"] + 0.55 * sigs["track_score"]
    return out


def _read_jsonl_any(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    kwargs: dict[str, Any] = {"encoding": "utf-8"}
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", **kwargs) as fh:
            for line in fh:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
    else:
        with path.open("r", **kwargs) as fh:
            for line in fh:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
    return rows


def _deserialize_tracks(raw_list: Sequence[Mapping[str, Any]]) -> list[TrackRecord]:
    out: list[TrackRecord] = []
    for item in raw_list:
        steps = [
            TrackStep(
                frame_ordinal=int(st.get("ordinal", 0)),
                pts_sec=float(st.get("pts_sec", 0.0)),
                shot_id=int(st.get("shot_id", 0)),
                source_kind=str(st.get("source_kind", "observed")),
                class_name=str(st.get("class_name", "")),
                confidence=float(st.get("confidence", 0.0)),
                observed_box=tuple(st["observed_box"]) if st.get("observed_box") is not None else None,
                predicted_box=tuple(st["predicted_box"]) if st.get("predicted_box") is not None else (0.0, 0.0, 0.01, 0.01),
                velocity_xy=tuple(st.get("velocity_xy", (0.0, 0.0))),
            )
            for st in (item.get("steps") or ())
        ]
        out.append(TrackRecord(
            track_id=int(item["track_id"]),
            shot_id=int(item["shot_id"]),
            class_name=str(item["class_name"]),
            steps=steps,
            observed_count=int(item.get("observed_count", 0)),
            predicted_only_count=int(item.get("predicted_only_count", 0)),
            recovered_after_occlusion=int(item.get("recovered_after_occlusion", 0)),
        ))
    return out


def _find_shard_subdir(root: Path) -> Path:
    if (root / "receipt.json").is_file():
        return root
    for p in sorted(root.rglob("receipt.json")):
        return p.parent
    raise FileNotFoundError(f"receipt.json not found under {root}")


def _load_dual_frame_backbone_dirs(
    dirs: Sequence[Path],
    *,
    video_ids: Sequence[str],
    vid_to_idx: Mapping[str, int],
    prefixes: tuple[str, str],
    default_dims: tuple[int, int],
    default_frames: int = 12,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    """Load dual frame+video+query feature shards (reused by both VL2 and VL3)."""
    p_a, p_b = prefixes
    a_vid_map: dict[str, np.ndarray] = {}
    a_fr_map: dict[str, np.ndarray] = {}
    a_q_map: dict[str, np.ndarray] = {}
    b_vid_map: dict[str, np.ndarray] = {}
    b_fr_map: dict[str, np.ndarray] = {}
    b_q_map: dict[str, np.ndarray] = {}
    dim_a, dim_b = default_dims
    n_frames = default_frames

    for raw_d in dirs:
        if not raw_d.exists():
            continue
        sd = _find_shard_subdir(raw_d)
        vr_path = sd / "video_rows.jsonl"
        av_path = sd / f"{p_a}_video_vectors.npy"
        af_path = sd / f"{p_a}_frame_vectors_fp16.npy"
        bv_path = sd / f"{p_b}_video_vectors.npy"
        bf_path = sd / f"{p_b}_frame_vectors_fp16.npy"
        if vr_path.is_file() and av_path.is_file() and bv_path.is_file():
            v_rows = _read_jsonl_any(vr_path)
            if v_rows:
                av_arr = np.load(str(av_path), mmap_mode="r")
                af_arr = np.load(str(af_path), mmap_mode="r") if af_path.is_file() else None
                bv_arr = np.load(str(bv_path), mmap_mode="r")
                bf_arr = np.load(str(bf_path), mmap_mode="r") if bf_path.is_file() else None
                dim_a = int(av_arr.shape[1])
                dim_b = int(bv_arr.shape[1])
                if af_arr is not None and af_arr.ndim == 3:
                    n_frames = int(af_arr.shape[1])
                for r_i, row in enumerate(v_rows):
                    vid = str(row["video_id"])
                    if vid in vid_to_idx:
                        a_vid_map[vid] = np.asarray(av_arr[r_i], dtype=np.float32)
                        b_vid_map[vid] = np.asarray(bv_arr[r_i], dtype=np.float32)
                        if af_arr is not None:
                            a_fr_map[vid] = np.asarray(af_arr[r_i], dtype=np.float16)
                        if bf_arr is not None:
                            b_fr_map[vid] = np.asarray(bf_arr[r_i], dtype=np.float16)

        aq_path = sd / f"{p_a}_query_vectors.npy"
        bq_path = sd / f"{p_b}_query_vectors.npy"
        qr_path = sd / "query_rows.jsonl"
        if qr_path.is_file() and aq_path.is_file() and bq_path.is_file():
            aq_arr = np.load(str(aq_path), mmap_mode="r")
            bq_arr = np.load(str(bq_path), mmap_mode="r")
            q_rows = _read_jsonl_any(qr_path)
            for q_i, row in enumerate(q_rows):
                qsha = str(row["query_text_sha256"])
                a_q_map[qsha] = np.asarray(aq_arr[q_i], dtype=np.float32)
                b_q_map[qsha] = np.asarray(bq_arr[q_i], dtype=np.float32)

    a_v = None
    b_v = None
    if a_vid_map:
        a_v = np.zeros((len(video_ids), dim_a), dtype=np.float32)
        b_v = np.zeros((len(video_ids), dim_b), dtype=np.float32)
        for vid, vec in a_vid_map.items():
            a_v[vid_to_idx[vid]] = l2_normalize_np(vec)
        for vid, vec in b_vid_map.items():
            b_v[vid_to_idx[vid]] = l2_normalize_np(vec)

    a_f = None
    if a_fr_map:
        a_f = np.zeros((len(video_ids), n_frames, dim_a), dtype=np.float16)
        for vid, fmat in a_fr_map.items():
            a_f[vid_to_idx[vid]] = fmat

    b_f = None
    if b_fr_map:
        b_f = np.zeros((len(video_ids), n_frames, dim_b), dtype=np.float16)
        for vid, fmat in b_fr_map.items():
            b_f[vid_to_idx[vid]] = fmat

    return a_v, a_f, a_q_map, b_v, b_f, b_q_map


def load_gallery_feature_bank(
    *,
    gallery_name: str,
    gallery_manifest_path: Path,
    queries_path: Path,
    vp_dirs: Sequence[Path] = (),
    mm_dirs: Sequence[Path] = (),
    asr_dirs: Sequence[Path] = (),
    cap_dirs: Sequence[Path] = (),
    vl2_dirs: Sequence[Path] = (),
    vl3_dirs: Sequence[Path] = (),
    train_vp_bank: np.ndarray | None = None,
    train_sig_bank: np.ndarray | None = None,
    train_pe_bank: np.ndarray | None = None,
    train_sig400_bank: np.ndarray | None = None,
    train_pe_bigg_bank: np.ndarray | None = None,
    train_dfn_bank: np.ndarray | None = None,
) -> GalleryFeatureBank:
    """Assemble an inference-only GalleryFeatureBank for one gallery from Kaggle shard outputs."""
    gal_rows = _read_jsonl_any(gallery_manifest_path)
    video_ids = sorted({str(r["video_id"]) for r in gal_rows})
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    queries = _read_jsonl_any(queries_path)

    bank = GalleryFeatureBank(
        gallery_name=gallery_name,
        video_ids=video_ids,
        vid_to_idx=vid_to_idx,
        queries=queries,
    )

    # 1. Load VideoPrism whole + window + query vectors
    vp_whole_map: dict[str, np.ndarray] = {}
    vp_win_vecs_list: list[np.ndarray] = []
    vp_win_vidx_list: list[int] = []
    vp_win_pos_by_vid: dict[str, list[int]] = defaultdict(list)

    for raw_d in vp_dirs:
        if not raw_d.exists():
            continue
        sd = _find_shard_subdir(raw_d)
        v_vecs = np.load(str(sd / "video_vectors.npy"), mmap_mode="r")
        v_rows = _read_jsonl_any(sd / "video_rows.jsonl")
        for row, vec in zip(v_rows, v_vecs):
            vid = str(row["video_id"])
            if vid in vid_to_idx:
                vp_whole_map[vid] = np.asarray(vec, dtype=np.float32)

        w_path = sd / "window_vectors.npy"
        wr_path = sd / "window_rows.jsonl"
        if w_path.is_file() and wr_path.is_file():
            w_vecs = np.load(str(w_path), mmap_mode="r")
            w_rows = _read_jsonl_any(wr_path)
            for row, vec in zip(w_rows, w_vecs):
                vid = str(row["video_id"])
                if vid in vid_to_idx:
                    g_pos = len(vp_win_vecs_list)
                    vp_win_vecs_list.append(np.asarray(vec, dtype=np.float32))
                    vp_win_vidx_list.append(vid_to_idx[vid])
                    vp_win_pos_by_vid[vid].append(g_pos)

        q_path = sd / "query_vectors.npy"
        qr_path = sd / "query_rows.jsonl"
        if q_path.is_file() and qr_path.is_file():
            q_vecs = np.load(str(q_path), mmap_mode="r")
            q_rows = _read_jsonl_any(qr_path)
            for row, vec in zip(q_rows, q_vecs):
                bank.vp_query_map[str(row["query_text_sha256"])] = np.asarray(vec, dtype=np.float32)

    if vp_whole_map:
        w_mat = np.zeros((len(video_ids), 768), dtype=np.float32)
        for vid, vec in vp_whole_map.items():
            w_mat[vid_to_idx[vid]] = l2_normalize_np(vec)
        bank.vp_whole_vecs = w_mat
    if vp_win_vecs_list:
        bank.vp_window_vecs = np.vstack(vp_win_vecs_list).astype(np.float32)
        bank.vp_win_vid_indices = np.asarray(vp_win_vidx_list, dtype=np.int32)
        bank.vp_video_win_positions = {
            vid: np.asarray(idxs, dtype=np.int32) for vid, idxs in vp_win_pos_by_vid.items()
        }

    # 2. Load SigLIP2 keyframes, YOLO objects, Kalman tracks, OCR
    sig_fr_vecs_list: list[np.ndarray] = []
    sig_fr_pts_list: list[float] = []
    sig_kf_vidx_list: list[int] = []
    sig_pos_by_vid: dict[str, list[int]] = defaultdict(list)
    kfs_by_id: dict[str, dict[str, Any]] = {}
    dets_by_kf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ocr_by_vid: dict[str, list[str]] = defaultdict(list)

    for raw_d in mm_dirs:
        if not raw_d.exists():
            continue
        sd = _find_shard_subdir(raw_d)
        kf_emb = np.load(str(sd / "keyframe_embeddings_fp16.npy"), mmap_mode="r")
        kf_rows = _read_jsonl_any(sd / "keyframes_manifest.jsonl.gz")
        for row in kf_rows:
            vid = str(row["video_id"])
            if vid not in vid_to_idx:
                continue
            fid = int(row["faiss_id"])
            pos = len(sig_fr_vecs_list)
            sig_fr_vecs_list.append(np.asarray(kf_emb[fid], dtype=np.float32))
            sig_fr_pts_list.append(float(row.get("pts_sec", 0.0)))
            sig_kf_vidx_list.append(vid_to_idx[vid])
            sig_pos_by_vid[vid].append(pos)
            kfs_by_id[str(row["keyframe_id"])] = row

        for obj in _read_jsonl_any(sd / "yolo_objects.jsonl.gz"):
            vid = str(obj["video_id"])
            if vid in vid_to_idx:
                dets_by_kf[str(obj["keyframe_id"])].append(obj)

        for tr_row in _read_jsonl_any(sd / "kalman_tracks.jsonl.gz"):
            vid = str(tr_row["video_id"])
            if vid in vid_to_idx:
                nf = int(tr_row.get("num_frames", 1))
                bank.sort_tracks[vid] = {"tracks": _deserialize_tracks(tr_row.get("sort_tracks") or ()), "num_frames": nf}
                bank.bytetrack_tracks[vid] = {"tracks": _deserialize_tracks(tr_row.get("bytetrack_tracks") or ()), "num_frames": nf}

        for ocr in _read_jsonl_any(sd / "ocr_observations.jsonl.gz"):
            vid = str(ocr["video_id"])
            if vid in vid_to_idx and str(ocr.get("raw_text") or "").strip():
                ocr_by_vid[vid].append(str(ocr["raw_text"]).strip())

        ocr_st_path = sd / "ocr_video_status.json"
        if ocr_st_path.is_file():
            st_map = json.loads(ocr_st_path.read_text(encoding="utf-8"))
            for vid, st_val in st_map.items():
                if vid in vid_to_idx:
                    bank.ocr_status[vid] = str(st_val)

        q_path = sd / "siglip_query_vectors.npy"
        qr_path = sd / "query_rows.jsonl"
        if q_path.is_file() and qr_path.is_file():
            q_vecs = np.load(str(q_path), mmap_mode="r")
            q_rows = _read_jsonl_any(qr_path)
            for row, vec in zip(q_rows, q_vecs):
                bank.sig_query_map[str(row["query_text_sha256"])] = np.asarray(vec, dtype=np.float32)

    if sig_fr_vecs_list:
        bank.sig_frame_vecs = np.vstack(sig_fr_vecs_list).astype(np.float32)
        bank.sig_frame_pts = np.asarray(sig_fr_pts_list, dtype=np.float64)
        bank.sig_kf_vid_indices = np.asarray(sig_kf_vidx_list, dtype=np.int32)
        bank.sig_video_kf_pos = {vid: np.asarray(idxs, dtype=np.int32) for vid, idxs in sig_pos_by_vid.items()}

    # Build per-video frame_observations for R6 crop scoring
    kfs_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kf in kfs_by_id.values():
        kfs_grouped[str(kf["video_id"])].append(kf)
    for vid, klist in kfs_grouped.items():
        klist.sort(key=lambda r: (int(r.get("source_frame_index", 0)), float(r.get("pts_sec", 0.0))))
        bank.kf_detections[vid] = [
            {
                "ordinal": int(kf.get("source_frame_index", 0)),
                "pts_sec": float(kf.get("pts_sec", 0.0)),
                "is_shot_cut": bool(kf.get("is_shot_cut", False)),
                "detections": dets_by_kf.get(str(kf["keyframe_id"]), []),
            }
            for kf in klist
        ]

    for vid in video_ids:
        bank.ocr_docs[vid] = " ".join(ocr_by_vid.get(vid, []))

    # 3. Load ASR
    asr_by_vid: dict[str, list[str]] = defaultdict(list)
    for raw_d in asr_dirs:
        if not raw_d.exists():
            continue
        sd = _find_shard_subdir(raw_d)
        for seg in _read_jsonl_any(sd / "asr_segments.jsonl.gz"):
            vid = str(seg["video_id"])
            if vid in vid_to_idx and str(seg.get("raw_text") or "").strip():
                asr_by_vid[vid].append(str(seg["raw_text"]).strip())
        st_path = sd / "asr_video_status.json"
        if st_path.is_file():
            for vid, st_val in json.loads(st_path.read_text(encoding="utf-8")).items():
                if vid in vid_to_idx:
                    bank.asr_status[vid] = str(st_val)
    for vid in video_ids:
        bank.asr_docs[vid] = " ".join(asr_by_vid.get(vid, []))

    # 4. Load Florence-2 captions & E5 vectors
    cap_e5_map: dict[str, np.ndarray] = {}
    for raw_d in cap_dirs:
        if not raw_d.exists():
            continue
        sd = _find_shard_subdir(raw_d)
        c_rows = _read_jsonl_any(sd / "florence2_captions.jsonl.gz")
        c_vecs = np.load(str(sd / "caption_e5_vectors.npy"), mmap_mode="r")
        for row, vec in zip(c_rows, c_vecs):
            vid = str(row["video_id"])
            if vid in vid_to_idx:
                bank.florence_captions[vid] = str(row.get("merged_caption") or "")
                cap_e5_map[vid] = np.asarray(vec, dtype=np.float32)

        qe_path = sd / "query_e5_vectors.npy"
        qr_path = sd / "query_rows.jsonl"
        if qe_path.is_file() and qr_path.is_file():
            q_vecs = np.load(str(qe_path), mmap_mode="r")
            q_rows = _read_jsonl_any(qr_path)
            for row, vec in zip(q_rows, q_vecs):
                bank.e5_query_map[str(row["query_text_sha256"])] = np.asarray(vec, dtype=np.float32)

    if cap_e5_map:
        e5_mat = np.zeros((len(video_ids), 384), dtype=np.float32)
        for vid, vec in cap_e5_map.items():
            e5_mat[vid_to_idx[vid]] = l2_normalize_np(vec)
        bank.cap_e5_vecs = e5_mat

    # 5. Load dual high-res VL2 (PE-Core-L & SigLIP2-SO400M) and VL3 (PE-Core-bigG & DFN5B-H)
    if vl2_dirs:
        pv, pf, pq, s4v, s4f, s4q = _load_dual_frame_backbone_dirs(
            vl2_dirs, video_ids=video_ids, vid_to_idx=vid_to_idx,
            prefixes=("pe", "sig400"), default_dims=(1024, 1152), default_frames=12,
        )
        bank.pe_video_vecs, bank.pe_frame_vecs = pv, pf
        bank.pe_query_map.update(pq)
        bank.sig400_video_vecs, bank.sig400_frame_vecs = s4v, s4f
        bank.sig400_query_map.update(s4q)

    if vl3_dirs:
        bgv, bgf, bgq, dfv, dff, dfq = _load_dual_frame_backbone_dirs(
            vl3_dirs, video_ids=video_ids, vid_to_idx=vid_to_idx,
            prefixes=("pe_bigg", "dfn"), default_dims=(1280, 1024), default_frames=12,
        )
        bank.pe_bigg_video_vecs, bank.pe_bigg_frame_vecs = bgv, bgf
        bank.pe_bigg_query_map.update(bgq)
        bank.dfn_video_vecs, bank.dfn_frame_vecs = dfv, dff
        bank.dfn_query_map.update(dfq)

    bank.build_derived_caches(
        train_vp_bank=train_vp_bank,
        train_sig_bank=train_sig_bank,
        train_pe_bank=train_pe_bank,
        train_sig400_bank=train_sig400_bank,
        train_pe_bigg_bank=train_pe_bigg_bank,
        train_dfn_bank=train_dfn_bank,
    )
    return bank

