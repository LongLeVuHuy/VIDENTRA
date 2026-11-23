"""Lightweight VEDJE-style Joint Reranker & X-Pool / Query-Bank Hubness Scorers (R2, R3, R7).

Paper attribution vs VIDENTRA internal variant:
- Replicated from VEDJE (https://arxiv.org/html/2610.11850v1):
  1. Reranking the fixed Top-20 candidate pool produced by Stage-1 retrieval.
  2. Retaining the Stage-1 score as an explicit residual prior alongside joint reranking.
  3. Query-conditioned frame/window cross-attention aggregation combined with global clip similarity.
  4. Training on retrieval-mined Top-20 hard negatives rather than random negatives.
- VIDENTRA lightweight internal adaptations:
  1. Operates strictly over cached VideoPrism (whole + multi-window) and presentation-ordered
     SigLIP2 keyframe embeddings (<5M params, <2 ms/query CPU inference) without re-encoding pixels.
  2. Applies automated False-Negative Filtering during training on train_9k_clean: Top-20 negatives
     whose own text captions or visual embeddings match the query/positive above threshold are masked
     from negative penalty so multi-video caption ambiguity does not corrupt Top-1 ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize_np(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim == 1:
        norm = max(eps, float(np.linalg.norm(mat)))
        return (mat / norm).astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, eps)).astype(np.float32, copy=False)


def compute_xpool_score(
    frame_vecs: np.ndarray,
    query_vec: np.ndarray,
    *,
    temperature: float = 0.07,
) -> float:
    """Query-conditioned attention pooling over cached frame features (X-Pool, R2)."""
    if frame_vecs.ndim != 2 or frame_vecs.shape[0] == 0:
        return 0.0
    f_norm = l2_normalize_np(frame_vecs)
    q_norm = l2_normalize_np(query_vec)
    sims = (f_norm @ q_norm).astype(np.float64)
    logits = (sims - float(np.max(sims))) / max(1e-4, float(temperature))
    weights = np.exp(logits)
    weights /= max(1e-12, float(np.sum(weights)))
    pooled = l2_normalize_np((weights[:, None].astype(np.float32) * f_norm).sum(axis=0))
    return float(np.dot(pooled, q_norm))


def compute_spaced_top2_frame_score(
    frame_vecs: np.ndarray,
    frame_pts: Sequence[float],
    query_vec: np.ndarray,
    *,
    min_pts_gap: float = 1.5,
) -> float:
    """Mean of top-2 query-matching frames separated by >= min_pts_gap seconds (R2)."""
    if frame_vecs.ndim != 2 or frame_vecs.shape[0] == 0:
        return 0.0
    f_norm = l2_normalize_np(frame_vecs)
    q_norm = l2_normalize_np(query_vec)
    sims = (f_norm @ q_norm).astype(np.float64)
    if sims.size == 1:
        return float(sims[0])
    pts = np.asarray(frame_pts, dtype=np.float64)
    order = np.argsort(-sims, kind="mergesort")
    idx1 = int(order[0])
    s1 = float(sims[idx1])
    t1 = float(pts[idx1]) if idx1 < pts.size else 0.0
    for idx2_raw in order[1:]:
        idx2 = int(idx2_raw)
        t2 = float(pts[idx2]) if idx2 < pts.size else 0.0
        if abs(t2 - t1) >= float(min_pts_gap):
            return 0.5 * (s1 + float(sims[idx2]))
    return 0.5 * (s1 + float(sims[int(order[1])]))


class TrainQueryBankHubness:
    """Zero-leakage Query-Bank CSLS & Dual-Softmax hubness prior built ONLY from train_9k_clean.

    Every gallery video gets an offline/gallery-level mean top-K similarity against a fixed bank
    of independent training captions (zero test/validation query interaction), so inference on a
    single validation/test query remains 100% deterministic, batch-size invariant, and query-independent.
    """

    def __init__(
        self,
        train_query_vecs: np.ndarray,
        *,
        k_hub: int = 20,
        temperature: float = 0.05,
    ) -> None:
        self.train_bank = l2_normalize_np(np.asarray(train_query_vecs, dtype=np.float32))
        self.k_hub = max(1, int(k_hub))
        self.temperature = max(1e-3, float(temperature))

    def compute_video_priors(self, video_vecs: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-video mean top-K, bank stats (QB-Z-Norm), and logsumexp priors (QB-DIS)."""
        v_norm = l2_normalize_np(np.asarray(video_vecs, dtype=np.float32))
        n_vids = v_norm.shape[0]
        if n_vids == 0 or self.train_bank.shape[0] == 0:
            z = np.zeros(n_vids, dtype=np.float32)
            return {
                "mean_topk": z.copy(),
                "mean_top50": z.copy(),
                "bank_mean": z.copy(),
                "bank_std": np.ones(n_vids, dtype=np.float32),
                "lse_prior": z.copy(),
                "lse_sum_003": z.copy(),
                "lse_sum_005": z.copy(),
                "lse_sum_007": z.copy(),
            }

        # Chunked matrix multiply to bound peak RAM
        chunk = 1024
        k_eff = min(self.k_hub, self.train_bank.shape[0])
        k_50 = min(50, self.train_bank.shape[0])
        mean_topk = np.zeros(n_vids, dtype=np.float32)
        mean_top50 = np.zeros(n_vids, dtype=np.float32)
        bank_mean = np.zeros(n_vids, dtype=np.float32)
        bank_std = np.ones(n_vids, dtype=np.float32)
        lse_prior = np.zeros(n_vids, dtype=np.float32)
        lse_sum_003 = np.zeros(n_vids, dtype=np.float32)
        lse_sum_005 = np.zeros(n_vids, dtype=np.float32)
        lse_sum_007 = np.zeros(n_vids, dtype=np.float32)
        bank_t = self.train_bank.T

        def _lse_sum(mat: np.ndarray, tau: float) -> np.ndarray:
            sc = mat / tau
            mx = np.max(sc, axis=1, keepdims=True)
            return (mx[:, 0] + np.log(np.sum(np.exp(sc - mx), axis=1))).astype(np.float32)

        for st in range(0, n_vids, chunk):
            ed = min(n_vids, st + chunk)
            sims = (v_norm[st:ed] @ bank_t).astype(np.float32)
            part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
            mean_topk[st:ed] = np.mean(part, axis=1)
            part50 = np.partition(sims, -k_50, axis=1)[:, -k_50:]
            mean_top50[st:ed] = np.mean(part50, axis=1)
            bank_mean[st:ed] = np.mean(sims, axis=1)
            bank_std[st:ed] = np.maximum(1e-4, np.std(sims, axis=1))
            scaled = sims / self.temperature
            m_row = np.max(scaled, axis=1, keepdims=True)
            lse = m_row[:, 0] + np.log(np.mean(np.exp(scaled - m_row), axis=1))
            lse_prior[st:ed] = lse * self.temperature
            lse_sum_003[st:ed] = _lse_sum(sims, 0.03)
            lse_sum_005[st:ed] = _lse_sum(sims, 0.05)
            lse_sum_007[st:ed] = _lse_sum(sims, 0.07)

        return {
            "mean_topk": mean_topk,
            "mean_top50": mean_top50,
            "bank_mean": bank_mean,
            "bank_std": bank_std,
            "lse_prior": lse_prior,
            "lse_sum_003": lse_sum_003,
            "lse_sum_005": lse_sum_005,
            "lse_sum_007": lse_sum_007,
        }

    def score_query_against_gallery(
        self,
        query_vec: np.ndarray,
        raw_sims: np.ndarray,
        video_priors: Mapping[str, np.ndarray],
        *,
        csls_weight: float = 0.5,
        dsn_weight: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Score one query against all gallery videos using precomputed train-bank priors."""
        sims = np.asarray(raw_sims, dtype=np.float32)
        csls_scores = sims - float(csls_weight) * video_priors["mean_topk"]
        dsn_logits = (sims - float(dsn_weight) * video_priors["lse_prior"]) / self.temperature
        return {
            "csls": csls_scores.astype(np.float32),
            "dsn_logits": dsn_logits.astype(np.float32),
        }


def apply_qb_dis_matrix(
    raw_mat: np.ndarray,
    video_priors: Mapping[str, np.ndarray] | None,
    *,
    tau: float = 0.05,
) -> np.ndarray:
    """Apply exact single-query Inductive Dynamic Inverted Softmax (QB-DIS) per gallery column."""
    sims = np.asarray(raw_mat, dtype=np.float64)
    if video_priors is None:
        return sims
    key = {0.03: "lse_sum_003", 0.05: "lse_sum_005", 0.07: "lse_sum_007"}.get(round(float(tau), 2), "lse_sum_005")
    lse_bank = video_priors.get(key)
    if lse_bank is None:
        return sims
    t = max(1e-3, float(tau))
    scaled = sims / t
    denom = np.logaddexp(scaled, np.asarray(lse_bank, dtype=np.float64)[None, :])
    return scaled - denom


def apply_qb_znorm_matrix(
    raw_mat: np.ndarray,
    video_priors: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    """Standardize each video column using its offline train-9K query bank mean & std (QB-Z-Norm)."""
    sims = np.asarray(raw_mat, dtype=np.float64)
    if video_priors is None:
        return sims
    mu = video_priors.get("bank_mean")
    sd = video_priors.get("bank_std")
    if mu is None or sd is None:
        return sims
    return (sims - np.asarray(mu, dtype=np.float64)[None, :]) / np.maximum(1e-4, np.asarray(sd, dtype=np.float64)[None, :])


class VEDJEJointRerankerNet(nn.Module):
    """Compact PyTorch cross-modal Top-20 reranker combining VideoPrism + SigLIP2 frames + prior."""

    def __init__(
        self,
        vp_dim: int = 768,
        sig_dim: int = 768,
        proj_dim: int = 128,
        scalar_dim: int = 10,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.vp_dim = int(vp_dim)
        self.sig_dim = int(sig_dim)
        self.proj_dim = int(proj_dim)
        self.scalar_dim = int(scalar_dim)

        self.vp_q_proj = nn.Linear(self.vp_dim, self.proj_dim, bias=False)
        self.vp_v_proj = nn.Linear(self.vp_dim, self.proj_dim, bias=False)
        self.sig_q_proj = nn.Linear(self.sig_dim, self.proj_dim, bias=False)
        self.sig_f_proj = nn.Linear(self.sig_dim, self.proj_dim, bias=False)

        self.attn_temp = nn.Parameter(torch.tensor(0.07))
        self.win_temp = nn.Parameter(torch.tensor(0.07))
        self.drop = nn.Dropout(dropout)

        # Scalar features + projected bilateral interaction features -> scalar residual delta
        in_mlp = self.scalar_dim + 4
        self.head = nn.Sequential(
            nn.Linear(in_mlp, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        vp_q: torch.Tensor,         # (B, 768)
        vp_whole: torch.Tensor,     # (B, K, 768)
        vp_wins: torch.Tensor,      # (B, K, W, 768)
        vp_win_mask: torch.Tensor,  # (B, K, W) bool
        sig_q: torch.Tensor,        # (B, 768)
        sig_frames: torch.Tensor,   # (B, K, M, 768)
        sig_mask: torch.Tensor,     # (B, K, M) bool
        scalars: torch.Tensor,      # (B, K, scalar_dim)
        stage1_prior: torch.Tensor, # (B, K)
    ) -> torch.Tensor:
        # 1. Projected VideoPrism whole & query-attended windows
        vq_p = F.normalize(self.vp_q_proj(self.drop(vp_q)), dim=-1)               # (B, P)
        vw_p = F.normalize(self.vp_v_proj(self.drop(vp_whole)), dim=-1)           # (B, K, P)
        vwin_p = F.normalize(self.vp_v_proj(self.drop(vp_wins)), dim=-1)          # (B, K, W, P)

        vp_whole_proj_sim = (vw_p * vq_p.unsqueeze(1)).sum(dim=-1)                # (B, K)
        win_raw_sim = (vwin_p * vq_p.unsqueeze(1).unsqueeze(2)).sum(dim=-1)       # (B, K, W)
        win_raw_sim = win_raw_sim.masked_fill(~vp_win_mask, -1e4)
        w_attn = torch.softmax(win_raw_sim / self.win_temp.clamp(min=0.02, max=0.5), dim=-1)
        vwin_agg = F.normalize((w_attn.unsqueeze(-1) * vwin_p).sum(dim=2), dim=-1)
        vp_win_proj_sim = (vwin_agg * vq_p.unsqueeze(1)).sum(dim=-1)              # (B, K)

        # 2. Projected SigLIP2 frame query-conditioned attention (X-Pool inside joint model)
        sq_p = F.normalize(self.sig_q_proj(self.drop(sig_q)), dim=-1)             # (B, P)
        sf_p = F.normalize(self.sig_f_proj(self.drop(sig_frames)), dim=-1)        # (B, K, M, P)
        frame_sim = (sf_p * sq_p.unsqueeze(1).unsqueeze(2)).sum(dim=-1)           # (B, K, M)
        frame_sim = frame_sim.masked_fill(~sig_mask, -1e4)
        f_attn = torch.softmax(frame_sim / self.attn_temp.clamp(min=0.02, max=0.5), dim=-1)
        sf_agg = F.normalize((f_attn.unsqueeze(-1) * sf_p).sum(dim=2), dim=-1)
        sig_xpool_sim = (sf_agg * sq_p.unsqueeze(1)).sum(dim=-1)                  # (B, K)
        sig_max_sim = frame_sim.max(dim=-1).values                                # (B, K)

        joint_feats = torch.cat(
            [
                scalars,
                vp_whole_proj_sim.unsqueeze(-1),
                vp_win_proj_sim.unsqueeze(-1),
                sig_xpool_sim.unsqueeze(-1),
                sig_max_sim.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.head(joint_feats).squeeze(-1)                             # (B, K)
        return self.prior_weight * stage1_prior + residual


@dataclass
class VEDJETrainingResult:
    """Summary of VEDJE joint reranker fitting on train_9k_clean."""

    epochs_ran: int
    best_val_hit1: float
    train_loss_final: float
    false_negatives_masked: int
    total_negatives_seen: int
    checkpoint_path: str
    checkpoint_sha256: str


def detect_false_negative_mask(
    query_vec: np.ndarray,
    cand_caption_vecs: Sequence[np.ndarray],
    pos_video_vec: np.ndarray | None,
    cand_video_vecs: np.ndarray,
    is_positive: Sequence[bool],
    *,
    text_sim_thresh: float = 0.82,
    vis_sim_thresh: float = 0.94,
) -> np.ndarray:
    """Identify Top-20 negatives that legitimately match the query/positive clip.

    When multiple MSR-VTT clips show nearly identical scenes or share matching captions,
    masking clear false negatives from negative loss prevents hurting calibration.
    """
    k = len(is_positive)
    keep_mask = np.ones(k, dtype=bool)
    q = l2_normalize_np(query_vec)
    v_pos = l2_normalize_np(pos_video_vec) if pos_video_vec is not None else None
    c_vecs = l2_normalize_np(cand_video_vecs)

    for idx in range(k):
        if is_positive[idx]:
            continue
        if idx < len(cand_caption_vecs) and cand_caption_vecs[idx].size > 0:
            cap_mat = l2_normalize_np(cand_caption_vecs[idx])
            if cap_mat.ndim == 1:
                cap_mat = cap_mat[None, :]
            if float(np.max(cap_mat @ q)) >= float(text_sim_thresh):
                keep_mask[idx] = False
                continue
        if v_pos is not None and float(np.dot(c_vecs[idx], v_pos)) >= float(vis_sim_thresh):
            keep_mask[idx] = False
    return keep_mask


def train_vedje_reranker(
    train_examples: Sequence[Mapping[str, Any]],
    val_examples: Sequence[Mapping[str, Any]],
    *,
    checkpoint_out: Path,
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 8e-4,
    weight_decay: float = 1e-2,
    seed: int = 20261122,
    device: str = "cpu",
) -> VEDJETrainingResult:
    """Fit VEDJEJointRerankerNet strictly on train_9k_clean hard-negative Top-20 pools."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not train_examples:
        raise ValueError("train_examples cannot be empty")

    scalar_dim = int(len(train_examples[0]["scalars"][0]))
    vp_dim = int(len(train_examples[0]["vp_q"]))
    sig_dim = int(len(train_examples[0]["sig_q"]))
    model = VEDJEJointRerankerNet(vp_dim=vp_dim, sig_dim=sig_dim, scalar_dim=scalar_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _collate(batch: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, ...]:
        bsz = len(batch)
        k_max = max(len(ex["vp_whole"]) for ex in batch)
        w_max = max(max(len(w) for w in ex["vp_wins"]) for ex in batch)
        m_max = max(max(len(f) for f in ex["sig_frames"]) for ex in batch)
        w_max = max(1, w_max)
        m_max = max(1, m_max)

        vp_q = np.zeros((bsz, vp_dim), dtype=np.float32)
        vp_whole = np.zeros((bsz, k_max, vp_dim), dtype=np.float32)
        vp_wins = np.zeros((bsz, k_max, w_max, vp_dim), dtype=np.float32)
        vp_win_mask = np.zeros((bsz, k_max, w_max), dtype=bool)
        sig_q = np.zeros((bsz, sig_dim), dtype=np.float32)
        sig_frames = np.zeros((bsz, k_max, m_max, sig_dim), dtype=np.float32)
        sig_mask = np.zeros((bsz, k_max, m_max), dtype=bool)
        scalars = np.zeros((bsz, k_max, scalar_dim), dtype=np.float32)
        prior = np.zeros((bsz, k_max), dtype=np.float32)
        target_idx = np.zeros(bsz, dtype=np.int64)
        valid_neg_mask = np.ones((bsz, k_max), dtype=bool)

        for b_i, ex in enumerate(batch):
            k_cur = len(ex["vp_whole"])
            vp_q[b_i] = ex["vp_q"]
            sig_q[b_i] = ex["sig_q"]
            vp_whole[b_i, :k_cur] = np.asarray(ex["vp_whole"], dtype=np.float32)
            scalars[b_i, :k_cur] = np.asarray(ex["scalars"], dtype=np.float32)
            prior[b_i, :k_cur] = np.asarray(ex["stage1_prior"], dtype=np.float32)
            target_idx[b_i] = int(ex["target_pos"])
            if "neg_keep_mask" in ex and ex["neg_keep_mask"] is not None:
                valid_neg_mask[b_i, :k_cur] = np.asarray(ex["neg_keep_mask"], dtype=bool)
            if k_cur < k_max:
                valid_neg_mask[b_i, k_cur:] = False

            for k_i in range(k_cur):
                wins_k = np.asarray(ex["vp_wins"][k_i], dtype=np.float32)
                if wins_k.ndim == 2 and wins_k.shape[0] > 0:
                    nw = min(w_max, wins_k.shape[0])
                    vp_wins[b_i, k_i, :nw] = wins_k[:nw]
                    vp_win_mask[b_i, k_i, :nw] = True
                fr_k = np.asarray(ex["sig_frames"][k_i], dtype=np.float32)
                if fr_k.ndim == 2 and fr_k.shape[0] > 0:
                    nm = min(m_max, fr_k.shape[0])
                    sig_frames[b_i, k_i, :nm] = fr_k[:nm]
                    sig_mask[b_i, k_i, :nm] = True

        return (
            torch.from_numpy(vp_q).to(device),
            torch.from_numpy(vp_whole).to(device),
            torch.from_numpy(vp_wins).to(device),
            torch.from_numpy(vp_win_mask).to(device),
            torch.from_numpy(sig_q).to(device),
            torch.from_numpy(sig_frames).to(device),
            torch.from_numpy(sig_mask).to(device),
            torch.from_numpy(scalars).to(device),
            torch.from_numpy(prior).to(device),
            torch.from_numpy(target_idx).to(device),
            torch.from_numpy(valid_neg_mask).to(device),
        )

    false_negs_masked = 0
    total_negs_seen = 0
    for ex in train_examples:
        k_cur = len(ex["vp_whole"])
        total_negs_seen += max(0, k_cur - 1)
        if "neg_keep_mask" in ex and ex["neg_keep_mask"] is not None:
            m = np.asarray(ex["neg_keep_mask"], dtype=bool)
            false_negs_masked += int(np.sum(~m))

    best_state = None
    best_val_hit1 = -1.0
    final_loss = 0.0
    rng = np.random.default_rng(seed)

    all_tr = _collate(train_examples)
    eval_pool = val_examples if val_examples else train_examples
    all_va = _collate(eval_pool)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_examples))
        ep_losses: list[float] = []
        for st in range(0, len(perm), batch_size):
            b_idx = torch.as_tensor(perm[st : st + batch_size], dtype=torch.long, device=device)
            vp_q = all_tr[0][b_idx]
            vp_whole = all_tr[1][b_idx]
            vp_wins = all_tr[2][b_idx]
            vp_win_mask = all_tr[3][b_idx]
            sig_q = all_tr[4][b_idx]
            sig_frames = all_tr[5][b_idx]
            sig_mask = all_tr[6][b_idx]
            scalars = all_tr[7][b_idx]
            prior = all_tr[8][b_idx]
            target_idx = all_tr[9][b_idx]
            valid_neg_mask = all_tr[10][b_idx]

            logits = model(
                vp_q, vp_whole, vp_wins, vp_win_mask,
                sig_q, sig_frames, sig_mask, scalars, prior,
            )
            # Mask false negatives out of denominator while always keeping target_idx active
            row_idx = torch.arange(logits.size(0), device=device)
            effective_mask = valid_neg_mask.clone()
            effective_mask[row_idx, target_idx] = True
            masked_logits = logits.masked_fill(~effective_mask, -1e4)

            loss_ce = F.cross_entropy(masked_logits * 12.0, target_idx)
            pos_logits = logits[row_idx, target_idx].unsqueeze(1)
            margin_violations = F.relu(0.15 - (pos_logits - logits))
            neg_only_mask = effective_mask.clone()
            neg_only_mask[row_idx, target_idx] = False
            loss_margin = (margin_violations * neg_only_mask.float()).sum() / neg_only_mask.float().sum().clamp(min=1.0)
            loss = loss_ce + 0.5 * loss_margin

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        final_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Evaluate on internal train-holdout split (no test/validation leakage!)
        model.eval()
        with torch.no_grad():
            scores = model(
                all_va[0], all_va[1], all_va[2], all_va[3],
                all_va[4], all_va[5], all_va[6], all_va[7], all_va[8],
            )
            preds = torch.argmax(scores, dim=1)
            hits = int((preds == all_va[9]).sum().item())
        val_hit1 = hits / float(max(1, len(eval_pool)))
        if val_hit1 > best_val_hit1:
            best_val_hit1 = val_hit1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "vp_dim": vp_dim,
        "sig_dim": sig_dim,
        "proj_dim": model.proj_dim,
        "scalar_dim": scalar_dim,
        "seed": seed,
        "best_val_hit1": best_val_hit1,
        "false_negatives_masked": false_negs_masked,
    }
    torch.save(payload, str(checkpoint_out))
    sha = hashlib.sha256(checkpoint_out.read_bytes()).hexdigest()
    return VEDJETrainingResult(
        epochs_ran=epochs,
        best_val_hit1=round(best_val_hit1, 6),
        train_loss_final=round(final_loss, 6),
        false_negatives_masked=false_negs_masked,
        total_negatives_seen=total_negs_seen,
        checkpoint_path=str(checkpoint_out),
        checkpoint_sha256=sha,
    )


def score_vedje_reranker_batch(
    checkpoint_path: Path,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
) -> list[np.ndarray]:
    """Score candidate Top-20 pools with a trained VEDJEJointRerankerNet checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model = VEDJEJointRerankerNet(
        vp_dim=int(ckpt["vp_dim"]),
        sig_dim=int(ckpt["sig_dim"]),
        proj_dim=int(ckpt["proj_dim"]),
        scalar_dim=int(ckpt["scalar_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_scores: list[np.ndarray] = []
    batch_size = 128
    vp_dim = int(ckpt["vp_dim"])
    sig_dim = int(ckpt["sig_dim"])
    with torch.no_grad():
        for st in range(0, len(examples), batch_size):
            batch = examples[st : st + batch_size]
            bsz = len(batch)
            non_empty = [i for i, ex in enumerate(batch) if len(ex["vp_whole"]) > 0]
            if not non_empty:
                out_scores.extend(np.zeros(0, dtype=np.float32) for _ in batch)
                continue
            k_max = max(len(batch[i]["vp_whole"]) for i in non_empty)
            w_max = max(1, max(max(len(w) for w in batch[i]["vp_wins"]) for i in non_empty))
            m_max = max(1, max(max(len(f) for f in batch[i]["sig_frames"]) for i in non_empty))

            vp_q_arr = np.zeros((bsz, vp_dim), dtype=np.float32)
            sig_q_arr = np.zeros((bsz, sig_dim), dtype=np.float32)
            vp_whole_arr = np.zeros((bsz, k_max, vp_dim), dtype=np.float32)
            scalars_arr = np.zeros((bsz, k_max, int(ckpt["scalar_dim"])), dtype=np.float32)
            prior_arr = np.zeros((bsz, k_max), dtype=np.float32)
            vp_wins_arr = np.zeros((bsz, k_max, w_max, vp_dim), dtype=np.float32)
            vp_win_mask_arr = np.zeros((bsz, k_max, w_max), dtype=bool)
            sig_frames_arr = np.zeros((bsz, k_max, m_max, sig_dim), dtype=np.float32)
            sig_mask_arr = np.zeros((bsz, k_max, m_max), dtype=bool)

            for b_i, ex in enumerate(batch):
                k_cur = len(ex["vp_whole"])
                if k_cur == 0:
                    continue
                vp_q_arr[b_i] = ex["vp_q"]
                sig_q_arr[b_i] = ex["sig_q"]
                vp_whole_arr[b_i, :k_cur] = np.asarray(ex["vp_whole"], dtype=np.float32)
                scalars_arr[b_i, :k_cur] = np.asarray(ex["scalars"], dtype=np.float32)
                prior_arr[b_i, :k_cur] = np.asarray(ex["stage1_prior"], dtype=np.float32)
                for k_i in range(k_cur):
                    wins_k = np.asarray(ex["vp_wins"][k_i], dtype=np.float32)
                    if wins_k.ndim == 2 and wins_k.shape[0] > 0:
                        nw = min(w_max, wins_k.shape[0])
                        vp_wins_arr[b_i, k_i, :nw] = wins_k[:nw]
                        vp_win_mask_arr[b_i, k_i, :nw] = True
                    fr_k = np.asarray(ex["sig_frames"][k_i], dtype=np.float32)
                    if fr_k.ndim == 2 and fr_k.shape[0] > 0:
                        nm = min(m_max, fr_k.shape[0])
                        sig_frames_arr[b_i, k_i, :nm] = fr_k[:nm]
                        sig_mask_arr[b_i, k_i, :nm] = True

            sc = model(
                torch.from_numpy(vp_q_arr).to(device),
                torch.from_numpy(vp_whole_arr).to(device),
                torch.from_numpy(vp_wins_arr).to(device),
                torch.from_numpy(vp_win_mask_arr).to(device),
                torch.from_numpy(sig_q_arr).to(device),
                torch.from_numpy(sig_frames_arr).to(device),
                torch.from_numpy(sig_mask_arr).to(device),
                torch.from_numpy(scalars_arr).to(device),
                torch.from_numpy(prior_arr).to(device),
            )
            sc_np = sc.detach().cpu().numpy().astype(np.float32)
            for b_i, ex in enumerate(batch):
                out_scores.append(sc_np[b_i, :len(ex["vp_whole"])].copy())
    return out_scores
