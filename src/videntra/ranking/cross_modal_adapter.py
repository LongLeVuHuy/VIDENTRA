"""Train-9K Residual Cross-Modal & Multi-Backbone Adapter with False-Negative Masking.

Fits lightweight residual heads strictly on `train_9k_clean` (`source_group`-disjoint
internal validation split for early stopping, zero touch of `Tune`, `Confirm`, or
`test_1ka`). Each backbone preserves pretrained cosine geometry at init while adapting
caption-to-video alignment AND learning a fused cross-backbone common latent space.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from videntra.ranking.vedje_reranker import l2_normalize_np


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ResidualModalityTower(nn.Module):
    """Residual bottleneck projection preserving pretrained L2 unit sphere geometry."""

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.20) -> None:
        super().__init__()
        self.dim = int(dim)
        h = max(32, min(int(hidden_dim), int(dim)))
        self.norm = nn.LayerNorm(self.dim)
        self.down = nn.Linear(self.dim, h)
        self.up = nn.Linear(h, self.dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.down(self.drop(self.norm(x))))
        delta = self.up(self.drop(h))
        return F.normalize(F.normalize(x, dim=-1) + delta, dim=-1)


class MultiBackboneCrossModalAdapter(nn.Module):
    """Per-backbone residual domain adapter + joint cross-backbone latent projector."""

    def __init__(
        self,
        backbone_dims: Mapping[str, int],
        *,
        joint_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.backbone_names = tuple(sorted(backbone_dims.keys()))
        self.backbone_dims = {k: int(backbone_dims[k]) for k in self.backbone_names}
        self.joint_dim = int(joint_dim)
        self.hidden_dim = int(hidden_dim)

        self.video_heads = nn.ModuleDict({
            b: ResidualModalityTower(self.backbone_dims[b], hidden_dim=hidden_dim, dropout=dropout)
            for b in self.backbone_names
        })
        self.query_heads = nn.ModuleDict({
            b: ResidualModalityTower(self.backbone_dims[b], hidden_dim=hidden_dim, dropout=dropout)
            for b in self.backbone_names
        })

        total_dim = sum(self.backbone_dims[b] for b in self.backbone_names)
        self.joint_video = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, self.joint_dim),
            nn.GELU(),
            nn.LayerNorm(self.joint_dim),
            nn.Linear(self.joint_dim, self.joint_dim),
        )
        self.joint_query = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, self.joint_dim),
            nn.GELU(),
            nn.LayerNorm(self.joint_dim),
            nn.Linear(self.joint_dim, self.joint_dim),
        )
        self.backbone_logits = nn.Parameter(torch.zeros(len(self.backbone_names)))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / 0.05))))
        self.joint_logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / 0.05))))

    def encode_videos(self, video_by_backbone: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        adapted: dict[str, torch.Tensor] = {}
        cat_parts: list[torch.Tensor] = []
        for b in self.backbone_names:
            x = F.normalize(video_by_backbone[b].float(), dim=-1)
            ax = self.video_heads[b](x)
            adapted[b] = ax
            cat_parts.append(ax)
        v_cat = torch.cat(cat_parts, dim=-1)
        v_joint = F.normalize(self.joint_video(v_cat), dim=-1)
        return adapted, v_joint

    def encode_queries(self, query_by_backbone: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        adapted: dict[str, torch.Tensor] = {}
        cat_parts: list[torch.Tensor] = []
        for b in self.backbone_names:
            x = F.normalize(query_by_backbone[b].float(), dim=-1)
            ax = self.query_heads[b](x)
            adapted[b] = ax
            cat_parts.append(ax)
        q_cat = torch.cat(cat_parts, dim=-1)
        q_joint = F.normalize(self.joint_query(q_cat), dim=-1)
        return adapted, q_joint


@dataclass
class CrossModalAdapterTrainingResult:
    epochs_ran: int
    best_val_hit1: float
    train_loss_final: float
    train_videos: int
    train_pairs: int
    val_videos: int
    checkpoint_path: str
    checkpoint_sha256: str


def _symmetric_fn_masked_infonce(
    q_vecs: torch.Tensor,
    v_vecs: torch.Tensor,
    logit_scale: torch.Tensor,
    fn_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE masking known same-video / high-cosine false negatives cleanly."""
    scale = logit_scale.exp().clamp(min=1.0, max=60.0)
    logits = scale * (q_vecs @ v_vecs.T)
    bsz = logits.size(0)
    targets = torch.arange(bsz, device=logits.device)

    # Zero out off-diagonal false-negative positions from denominator (keep diagonal active)
    diag = torch.eye(bsz, dtype=torch.bool, device=logits.device)
    suppressed = fn_mask & (~diag)
    logits_masked = logits.masked_fill(suppressed, -1e4)
    logits_t_masked = logits.T.masked_fill(suppressed.T, -1e4)
    return 0.5 * (F.cross_entropy(logits_masked, targets) + F.cross_entropy(logits_t_masked, targets))


def train_cross_modal_adapter(
    *,
    backbone_dims: Mapping[str, int],
    train_video_feats: Mapping[str, np.ndarray],
    train_query_feats: Mapping[str, np.ndarray],
    pair_video_idx: np.ndarray,
    pair_source_groups: Sequence[str],
    video_source_groups: Sequence[str],
    checkpoint_out: Path,
    joint_dim: int = 512,
    hidden_dim: int = 256,
    epochs: int = 22,
    batch_size: int = 256,
    lr: float = 4e-4,
    weight_decay: float = 0.04,
    fn_cos_threshold: float = 0.88,
    seed: int = 20261123,
    device: str = "cpu",
) -> CrossModalAdapterTrainingResult:
    """Fit MultiBackboneCrossModalAdapter strictly on train_9k_clean pairs (source-group disjoint val)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    names = tuple(sorted(backbone_dims.keys()))
    if not names:
        raise ValueError("backbone_dims cannot be empty")
    n_pairs = len(pair_video_idx)
    if n_pairs == 0:
        raise ValueError("No training pairs provided")

    # Deterministic 90/10 internal validation split strictly by video source_group
    unique_groups = sorted(set(str(g) for g in video_source_groups))
    perm_groups = rng.permutation(len(unique_groups))
    n_val_groups = max(1, int(round(0.10 * len(unique_groups))))
    val_group_set = {unique_groups[int(i)] for i in perm_groups[:n_val_groups]}

    val_pair_mask = np.asarray([str(g) in val_group_set for g in pair_source_groups], dtype=bool)
    tr_pair_idx = np.flatnonzero(~val_pair_mask)
    val_pair_idx = np.flatnonzero(val_pair_mask)
    if tr_pair_idx.size == 0 or val_pair_idx.size == 0:
        raise ValueError("Internal source_group split produced empty train or val pairs")

    val_vid_mask = np.asarray([str(g) in val_group_set for g in video_source_groups], dtype=bool)
    val_vid_indices = np.flatnonzero(val_vid_mask)
    val_vid_pos_map = {int(gv): i for i, gv in enumerate(val_vid_indices)}
    val_pair_local_targets = torch.as_tensor(
        [val_vid_pos_map[int(pair_video_idx[p])] for p in val_pair_idx],
        dtype=torch.long,
        device=device,
    )

    v_tensors = {
        b: torch.from_numpy(l2_normalize_np(np.asarray(train_video_feats[b], dtype=np.float32))).to(device)
        for b in names
    }
    q_tensors = {
        b: torch.from_numpy(l2_normalize_np(np.asarray(train_query_feats[b], dtype=np.float32))).to(device)
        for b in names
    }
    raw_vid_cat = F.normalize(torch.cat([v_tensors[b] for b in names], dim=-1), dim=-1)
    raw_qry_cat = F.normalize(torch.cat([q_tensors[b] for b in names], dim=-1), dim=-1)

    model = MultiBackboneCrossModalAdapter(
        backbone_dims=backbone_dims,
        joint_dim=joint_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_hit1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        # Sample up to 1 caption per distinct video per batch to keep in-batch negatives diverse & zero duplicate videos
        perm = tr_pair_idx[rng.permutation(len(tr_pair_idx))]
        ep_losses: list[float] = []
        for st in range(0, len(perm), batch_size):
            batch_p = perm[st : st + batch_size]
            # Deduplicate videos within the same batch so diagonal = unique positive video
            seen_v: set[int] = set()
            uniq_p: list[int] = []
            for p_i in batch_p:
                vid_i = int(pair_video_idx[p_i])
                if vid_i not in seen_v:
                    seen_v.add(vid_i)
                    uniq_p.append(int(p_i))
            if len(uniq_p) < 8:
                continue

            p_arr = np.asarray(uniq_p, dtype=np.int64)
            v_arr = np.asarray([int(pair_video_idx[p]) for p in p_arr], dtype=np.int64)
            p_t = torch.from_numpy(p_arr).to(device)
            v_t = torch.from_numpy(v_arr).to(device)

            bq = {b: q_tensors[b][p_t] for b in names}
            bv = {b: v_tensors[b][v_t] for b in names}

            # False-negative mask: same source_group OR high raw cosine similarity
            sgs = [str(pair_source_groups[p]) for p in p_arr]
            sg_same = torch.tensor(
                [[sgs[i] == sgs[j] for j in range(len(sgs))] for i in range(len(sgs))],
                dtype=torch.bool,
                device=device,
            )
            with torch.no_grad():
                raw_sim = raw_qry_cat[p_t] @ raw_vid_cat[v_t].T
                fn_mask = sg_same | (raw_sim >= float(fn_cos_threshold))

            aq, q_joint = model.encode_queries(bq)
            av, v_joint = model.encode_videos(bv)

            loss_joint = _symmetric_fn_masked_infonce(q_joint, v_joint, model.joint_logit_scale, fn_mask)
            loss_per_b = torch.zeros((), device=device)
            w_soft = torch.softmax(model.backbone_logits, dim=0)
            for idx_b, b in enumerate(names):
                loss_per_b = loss_per_b + w_soft[idx_b] * _symmetric_fn_masked_infonce(
                    aq[b], av[b], model.logit_scale, fn_mask
                )
            loss = loss_joint + 0.85 * loss_per_b

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        scheduler.step()
        final_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Early stopping evaluated against internal disjoint train_9k val videos
        model.eval()
        with torch.no_grad():
            vq_in = {b: q_tensors[b][torch.from_numpy(val_pair_idx).to(device)] for b in names}
            vv_in = {b: v_tensors[b][torch.from_numpy(val_vid_indices).to(device)] for b in names}
            aq_val, qj_val = model.encode_queries(vq_in)
            av_val, vj_val = model.encode_videos(vv_in)
            sim_mat = qj_val @ vj_val.T
            w_soft = torch.softmax(model.backbone_logits, dim=0)
            for idx_b, b in enumerate(names):
                sim_mat = sim_mat + w_soft[idx_b] * (aq_val[b] @ av_val[b].T)
            preds = torch.argmax(sim_mat, dim=1)
            val_hit1 = float((preds == val_pair_local_targets).float().mean().item())
        if val_hit1 > best_hit1:
            best_hit1 = val_hit1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_out.with_name(f"{checkpoint_out.name}.tmp")
    torch.save(
        {
            "backbone_dims": dict(model.backbone_dims),
            "joint_dim": model.joint_dim,
            "hidden_dim": model.hidden_dim,
            "state_dict": model.state_dict(),
            "best_val_hit1": float(best_hit1),
            "epochs_ran": epochs,
            "seed": seed,
        },
        tmp,
    )
    tmp.replace(checkpoint_out)

    return CrossModalAdapterTrainingResult(
        epochs_ran=epochs,
        best_val_hit1=round(float(best_hit1), 6),
        train_loss_final=round(float(final_loss), 6),
        train_videos=int(len(video_source_groups) - len(val_vid_indices)),
        train_pairs=int(len(tr_pair_idx)),
        val_videos=int(len(val_vid_indices)),
        checkpoint_path=str(checkpoint_out),
        checkpoint_sha256=_sha256_file(checkpoint_out),
    )


def _load_adapter_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> MultiBackboneCrossModalAdapter:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model = MultiBackboneCrossModalAdapter(
        backbone_dims=ckpt["backbone_dims"],
        joint_dim=int(ckpt.get("joint_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def adapt_vectors_with_checkpoint(
    checkpoint_path: Path,
    *,
    video_feats: Mapping[str, np.ndarray] | None = None,
    query_feats: Mapping[str, np.ndarray] | None = None,
    device: str = "cpu",
    batch_size: int = 2048,
) -> dict[str, dict[str, np.ndarray]]:
    """Project video and/or query embeddings through a trained MultiBackboneCrossModalAdapter."""
    model = _load_adapter_checkpoint(checkpoint_path, device=device)
    names = model.backbone_names
    out: dict[str, dict[str, np.ndarray]] = {}

    def _encode_split(feats: Mapping[str, np.ndarray], encode_fn: Any) -> dict[str, np.ndarray]:
        n_rows = int(feats[names[0]].shape[0])
        res: dict[str, list[np.ndarray]] = {b: [] for b in names}
        res["joint"] = []
        with torch.no_grad():
            for st in range(0, n_rows, batch_size):
                ed = min(n_rows, st + batch_size)
                batch = {
                    b: torch.from_numpy(l2_normalize_np(np.asarray(feats[b][st:ed], dtype=np.float32))).to(device)
                    for b in names
                }
                adapted, joint = encode_fn(batch)
                for b in names:
                    res[b].append(adapted[b].cpu().numpy().astype(np.float32))
                res["joint"].append(joint.cpu().numpy().astype(np.float32))
        return {k: (np.vstack(v) if v else np.zeros((0, 1), dtype=np.float32)) for k, v in res.items()}

    if video_feats is not None:
        out["videos"] = _encode_split(video_feats, model.encode_videos)
    if query_feats is not None:
        out["queries"] = _encode_split(query_feats, model.encode_queries)
    return out


def score_cross_modal_adapter_gallery(
    checkpoint_path: Path,
    *,
    video_feats: Mapping[str, np.ndarray],
    query_feats: Mapping[str, np.ndarray],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Compute adapted per-backbone and joint `(N_q, N_v)` similarity matrices for one gallery."""
    proj = adapt_vectors_with_checkpoint(
        checkpoint_path,
        video_feats=video_feats,
        query_feats=query_feats,
        device=device,
    )
    v_map, q_map = proj["videos"], proj["queries"]
    out: dict[str, np.ndarray] = {
        "joint": (q_map["joint"] @ v_map["joint"].T).astype(np.float64),
    }
    for b in v_map:
        if b != "joint":
            out[b] = (q_map[b] @ v_map[b].T).astype(np.float64)
    return out
