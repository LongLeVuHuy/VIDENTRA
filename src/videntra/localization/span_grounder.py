"""Cross-event temporal span grounding models (`CrossEventSpanNet` & `SpanGrounderLGBM`).

Resolves multi-event paragraph localization across arbitrary PTS grids (`M=48` coarse or
`M=128` coarse-to-fine top candidates) without forced `D/M` uniform length bias and
supports bounded chronological overlap via Pairwise Soft-Overlap Semi-Markov DP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from videntra.retrieval.alignment import (
    EventChainLocalizationResult,
    _voronoi_frame_edges,
    localize_soft_overlap_segment_dp,
)

DEFAULT_BACKBONES: tuple[str, ...] = ("pe_bigg", "pe_l", "sig400")
DEFAULT_BACKBONE_WEIGHTS: dict[str, float] = {
    "pe_bigg": 1.40,
    "pe_l": 1.10,
    "sig400": 0.75,
}
NUM_FRAME_FEATURES = 24
NUM_SPAN_FEATURES = 47


def _znorm_rows(mat: np.ndarray) -> np.ndarray:
    mu = np.mean(mat, axis=1, keepdims=True)
    sd = np.maximum(1e-5, np.std(mat, axis=1, keepdims=True))
    return ((mat - mu) / sd).astype(np.float32)


def _compute_visual_boundary_jumps(
    frame_embeddings_by_backbone: Mapping[str, np.ndarray] | None,
    m_v: int,
    backbones: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute resolution-invariant 1-step & 2-step normalized visual cosine jumps."""
    if not frame_embeddings_by_backbone or m_v < 3:
        return np.zeros(m_v, dtype=np.float32), np.zeros(m_v, dtype=np.float32)

    s1 = max(1, int(round(m_v / 48.0)))
    s2 = max(2, int(round(m_v / 24.0)))
    vd1 = np.zeros(m_v, dtype=np.float64)
    vd2 = np.zeros(m_v, dtype=np.float64)
    used = 0
    for b in backbones:
        emb = frame_embeddings_by_backbone.get(b)
        if emb is None or emb.ndim != 2 or emb.shape[0] != m_v:
            continue
        f = emb.astype(np.float64)
        f /= np.maximum(1e-9, np.linalg.norm(f, axis=1, keepdims=True))
        vd1[s1:] += 1.0 - np.sum(f[s1:] * f[:-s1], axis=1)
        if s2 < m_v:
            vd2[s2:] += 1.0 - np.sum(f[s2:] * f[:-s2], axis=1)
        used += 1
    if used == 0:
        return np.zeros(m_v, dtype=np.float32), np.zeros(m_v, dtype=np.float32)
    vd1 /= float(used)
    vd2 /= float(used)
    vd1_z = (vd1 - np.mean(vd1)) / max(1e-5, float(np.std(vd1)))
    vd2_z = (vd2 - np.mean(vd2)) / max(1e-5, float(np.std(vd2)))
    return vd1_z.astype(np.float32), vd2_z.astype(np.float32)


def extract_multievent_frame_features(
    timestamps_sec: Sequence[float],
    duration_sec: float,
    curve_by_backbone: Mapping[str, np.ndarray],
    *,
    frame_embeddings_by_backbone: Mapping[str, np.ndarray] | None = None,
    weights: Mapping[str, float] = DEFAULT_BACKBONE_WEIGHTS,
) -> dict[str, np.ndarray]:
    """Compute `(K, M, 24)` resolution-invariant frame features + intermediate curves."""
    b_list = [b for b in DEFAULT_BACKBONES if b in curve_by_backbone]
    if not b_list:
        raise ValueError("curve_by_backbone must include at least one backbone curve matrix")

    first = np.asarray(curve_by_backbone[b_list[0]], dtype=np.float32)
    if first.ndim == 1:
        first = first.reshape(1, -1)
    k_ev, m_v = first.shape
    ts_list = [float(t) for t in timestamps_sec]
    dur = max(1.0, float(duration_sec), ts_list[-1] if len(ts_list) > 0 else 1.0)

    w_sum = 0.0
    raw = np.zeros((k_ev, m_v), dtype=np.float32)
    raw_b: dict[str, np.ndarray] = {}
    for b in DEFAULT_BACKBONES:
        mat = (
            np.asarray(curve_by_backbone[b], dtype=np.float32).reshape(k_ev, m_v)
            if b in curve_by_backbone
            else first.copy()
        )
        raw_b[b] = mat
        wb = float(weights.get(b, 1.0))
        raw += wb * mat
        w_sum += wb
    raw /= max(1e-6, w_sum)

    z_t = _znorm_rows(raw)
    z_peb = _znorm_rows(raw_b["pe_bigg"])
    z_pel = _znorm_rows(raw_b["pe_l"])
    z_sig = _znorm_rows(raw_b["sig400"])

    if k_ev > 1:
        other_mean = (np.sum(z_t, axis=0, keepdims=True) - z_t) / float(k_ev - 1)
        cc = (z_t - 0.85 * other_mean).astype(np.float32)
        mx = np.max(z_t, axis=0, keepdims=True)
        p_cl07 = np.exp((z_t - mx) / 0.7)
        p_cl07 /= np.maximum(1e-6, np.sum(p_cl07, axis=0, keepdims=True))
        p_cl15 = np.exp((z_t - mx) / 1.5)
        p_cl15 /= np.maximum(1e-6, np.sum(p_cl15, axis=0, keepdims=True))
        # Strict margin over strongest OTHER event at frame t
        other_max = np.empty_like(z_t)
        for i in range(k_ev):
            mask = np.ones(k_ev, dtype=bool)
            mask[i] = False
            other_max[i] = np.max(z_t[mask], axis=0)
        exp_margin = (z_t - other_max).astype(np.float32)
        rank_c = np.argsort(np.argsort(z_t, axis=0), axis=0).astype(np.float32) / float(k_ev - 1)
    else:
        cc = z_t.copy()
        p_cl07 = np.ones_like(z_t)
        p_cl15 = np.ones_like(z_t)
        exp_margin = np.zeros_like(z_t)
        rank_c = np.ones_like(z_t)

    prev_z = np.vstack([np.zeros((1, m_v), dtype=np.float32), z_t[:-1]])
    next_z = np.vstack([z_t[1:], np.zeros((1, m_v), dtype=np.float32)])

    s1 = max(1, int(round(m_v / 48.0)))
    s2 = max(2, min(m_v - 1, int(round(m_v / 24.0)))) if m_v > 2 else 1
    dz1_l = np.zeros_like(z_t)
    dz1_r = np.zeros_like(z_t)
    dz1_l[:, s1:] = z_t[:, s1:] - z_t[:, :-s1]
    dz1_r[:, :-s1] = z_t[:, :-s1] - z_t[:, s1:]

    z_sm = z_t.copy()
    if m_v >= 3:
        z_sm[:, 1:-1] = 0.25 * z_t[:, :-2] + 0.50 * z_t[:, 1:-1] + 0.25 * z_t[:, 2:]
    dz2_l = np.zeros_like(z_sm)
    dz2_r = np.zeros_like(z_sm)
    if s2 < m_v:
        dz2_l[:, s2:] = z_sm[:, s2:] - z_sm[:, :-s2]
        dz2_r[:, :-s2] = z_sm[:, :-s2] - z_sm[:, s2:]

    rank_t = np.argsort(np.argsort(z_t, axis=1), axis=1).astype(np.float32) / max(1.0, float(m_v - 1))
    vd1, vd2 = _compute_visual_boundary_jumps(frame_embeddings_by_backbone, m_v, DEFAULT_BACKBONES)
    vd1_m = np.broadcast_to(vd1[None, :], (k_ev, m_v)).copy()
    vd2_m = np.broadcast_to(vd2[None, :], (k_ev, m_v)).copy()

    ts_arr = np.asarray(ts_list, dtype=np.float32)
    t_pos = np.broadcast_to((ts_arr / dur)[None, :], (k_ev, m_v)).copy()
    k_pos = (
        np.broadcast_to(np.linspace(0.0, 1.0, k_ev, dtype=np.float32)[:, None], (k_ev, m_v)).copy()
        if k_ev > 1
        else np.zeros((k_ev, m_v), dtype=np.float32)
    )
    inv_k = np.full((k_ev, m_v), 1.0 / float(k_ev), dtype=np.float32)
    ldur = np.full((k_ev, m_v), np.log1p(dur) / 6.0, dtype=np.float32)

    seq_x = np.stack(
        [
            raw, z_t, z_peb, z_pel, z_sig,
            cc, p_cl07, p_cl15, prev_z, next_z,
            dz1_l, dz1_r, dz2_l, dz2_r,
            rank_t, rank_c, vd1_m, vd2_m,
            t_pos, k_pos, inv_k, ldur,
            z_sm, exp_margin,
        ],
        axis=-1,
    ).astype(np.float32)

    return {
        "seq_x": seq_x,
        "z_t": z_t,
        "z_peb": z_peb,
        "z_pel": z_pel,
        "z_sig": z_sig,
        "cc": cc,
        "p_cl07": p_cl07,
        "exp_margin": exp_margin,
        "vd1": vd1,
    }


def extract_multievent_span_features(
    timestamps_sec: Sequence[float],
    duration_sec: float,
    frame_ctx: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact vectorized `(K, S, 47)` span features via prefix sums (`S = M*(M+1)//2`)."""
    ts = [float(t) for t in timestamps_sec]
    dur = max(1.0, float(duration_sec), ts[-1] if ts else 1.0)
    le, re = _voronoi_frame_edges(ts, dur)
    le_a = np.asarray(le, dtype=np.float64)
    re_a = np.asarray(re, dtype=np.float64)

    z_t = frame_ctx["z_t"].astype(np.float64)
    z_peb = frame_ctx["z_peb"].astype(np.float64)
    z_pel = frame_ctx["z_pel"].astype(np.float64)
    z_sig = frame_ctx["z_sig"].astype(np.float64)
    cc = frame_ctx["cc"].astype(np.float64)
    p_cl = frame_ctx["p_cl07"].astype(np.float64)
    exp_m = frame_ctx["exp_margin"].astype(np.float64)
    vd1 = frame_ctx["vd1"].astype(np.float64)

    k_ev, m_v = z_t.shape
    ta, tb = np.triu_indices(m_v)
    cnt = (tb - ta + 1).astype(np.float64)
    out_cnt = np.maximum(1.0, float(m_v) - cnt)
    n_spans = len(ta)

    pref_vd = np.concatenate([[0.0], np.cumsum(vd1)])
    vd_mean_in = (pref_vd[tb + 1] - pref_vd[ta]) / cnt
    vd_l = vd1[ta]
    vd_r = np.where(tb + 1 < m_v, vd1[np.minimum(m_v - 1, tb + 1)], 0.0)
    step = max(1, int(round(m_v / 48.0)))

    sa_r = le_a[ta] / dur
    eb_r = re_a[tb] / dur
    len_r = eb_r - sa_r
    mid_r = 0.5 * (sa_r + eb_r)
    is_st = (ta == 0).astype(np.float64)
    is_ed = (tb == m_v - 1).astype(np.float64)
    inv_k = np.full(n_spans, 1.0 / float(k_ev), dtype=np.float64)
    k_val = np.full(n_spans, float(k_ev), dtype=np.float64)
    ldur = np.full(n_spans, np.log1p(dur), dtype=np.float64)

    def _pref(arr: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros((arr.shape[0], 1), dtype=np.float64), np.cumsum(arr, axis=1)], axis=1)

    p_z = _pref(z_t)
    p_cc = _pref(cc)
    p_pb = _pref(z_peb)
    p_pl = _pref(z_pel)
    p_sg = _pref(z_sig)
    p_pr = _pref(p_cl)
    p_em = _pref(exp_m)

    X_out = np.empty((k_ev, n_spans, NUM_SPAN_FEATURES), dtype=np.float32)
    idx_left_prev = np.maximum(0, ta - step)
    idx_right_next = np.minimum(m_v - 1, tb + step)

    for i in range(k_ev):
        s_cc = p_cc[i, tb + 1] - p_cc[i, ta]
        cc_in = s_cc / cnt
        cc_out = np.where(cnt < m_v, (p_cc[i, -1] - s_cc) / out_cnt, 0.0)

        s_z = p_z[i, tb + 1] - p_z[i, ta]
        z_in = s_z / cnt
        z_out = np.where(cnt < m_v, (p_z[i, -1] - s_z) / out_cnt, 0.0)

        s_p = p_pr[i, tb + 1] - p_pr[i, ta]
        p_in = s_p / cnt
        p_out = np.where(cnt < m_v, (p_pr[i, -1] - s_p) / out_cnt, 0.0)
        p_share = s_p / max(1e-6, float(p_pr[i, -1]))

        pb_in = (p_pb[i, tb + 1] - p_pb[i, ta]) / cnt
        pb_out = np.where(cnt < m_v, (p_pb[i, -1] - (p_pb[i, tb + 1] - p_pb[i, ta])) / out_cnt, 0.0)
        pl_in = (p_pl[i, tb + 1] - p_pl[i, ta]) / cnt
        pl_out = np.where(cnt < m_v, (p_pl[i, -1] - (p_pl[i, tb + 1] - p_pl[i, ta])) / out_cnt, 0.0)
        sg_in = (p_sg[i, tb + 1] - p_sg[i, ta]) / cnt
        sg_out = np.where(cnt < m_v, (p_sg[i, -1] - (p_sg[i, tb + 1] - p_sg[i, ta])) / out_cnt, 0.0)

        cc_lp = np.where(ta >= step, cc[i, idx_left_prev], 0.0)
        cc_rn = np.where(tb + step < m_v, cc[i, idx_right_next], 0.0)
        z_lp = np.where(ta >= step, z_t[i, idx_left_prev], 0.0)
        z_rn = np.where(tb + step < m_v, z_t[i, idx_right_next], 0.0)

        am_cc = int(np.argmax(cc[i]))
        am_z = int(np.argmax(z_t[i]))
        has_am_cc = ((ta <= am_cc) & (am_cc <= tb)).astype(np.float64)
        has_am_z = ((ta <= am_z) & (am_z <= tb)).astype(np.float64)

        w_com = np.exp(np.clip(z_t[i] * 2.0, -20.0, 20.0))
        com_pos = float(np.sum(w_com * (np.arange(m_v, dtype=np.float64) / max(1.0, float(m_v - 1)))) / max(1e-9, float(np.sum(w_com))))

        max_mat_z = np.full((m_v, m_v), 0.0, dtype=np.float64)
        for a in range(m_v):
            max_mat_z[a, a:] = np.maximum.accumulate(z_t[i, a:])
        z_max_in = max_mat_z[ta, tb]
        z_peak_ratio = z_max_in / max(1e-6, float(np.max(z_t[i])))

        prev_in = (
            (p_z[i - 1, tb + 1] - p_z[i - 1, ta]) / cnt
            if i > 0
            else np.zeros(n_spans, dtype=np.float64)
        )
        next_in = (
            (p_z[i + 1, tb + 1] - p_z[i + 1, ta]) / cnt
            if i + 1 < k_ev
            else np.zeros(n_spans, dtype=np.float64)
        )
        margin_in = (p_em[i, tb + 1] - p_em[i, ta]) / cnt
        k_rel = np.full(n_spans, float(i) / max(1.0, float(k_ev - 1)), dtype=np.float64)
        sa_minus_ik = sa_r - float(i) / float(k_ev)

        X_out[i] = np.stack(
            [
                cc_in, cc_out, cc_in - cc_out, s_cc / float(m_v),
                z_in, z_out, z_in - z_out,
                p_in, p_out, p_in - p_out, p_share,
                pb_in, pl_in, sg_in,
                pb_in - pb_out, pl_in - pl_out, sg_in - sg_out,
                cc[i, ta] - cc_lp, cc[i, tb] - cc_rn,
                z_t[i, ta] - z_lp, z_t[i, tb] - z_rn,
                vd_l, vd_r, vd_mean_in, z_max_in,
                has_am_cc, has_am_z, z_peak_ratio,
                sa_r, eb_r, len_r, mid_r,
                np.abs(mid_r - com_pos), np.abs(sa_r - com_pos), np.abs(eb_r - com_pos), sa_minus_ik,
                k_rel, inv_k, k_val, ldur, is_st, is_ed,
                prev_in, next_in, cc_in - prev_in, cc_in - next_in, margin_in,
            ],
            axis=1,
        ).astype(np.float32)

    return X_out, ta, tb, le_a, re_a


class _ResConvBlock(nn.Module):
    def __init__(self, d_hid: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.c3 = nn.Conv1d(d_hid, d_hid, kernel_size=3, padding=1)
        self.c5 = nn.Conv1d(d_hid, d_hid, kernel_size=5, padding=2)
        self.proj = nn.Linear(d_hid * 2, d_hid)
        self.norm = nn.LayerNorm(d_hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)  # (K, d, M)
        h3 = F.gelu(self.c3(xt)).transpose(1, 2)
        h5 = F.gelu(self.c5(xt)).transpose(1, 2)
        h = self.proj(torch.cat([h3, h5], dim=-1))
        return self.norm(x + self.drop(h))


class CrossEventSpanNet(nn.Module):
    """Multi-scale BiGRU + Cross-Clause Transformer neural grounder predicting soft tIoU, P(tIoU>=0.5), and transitions."""

    def __init__(
        self,
        d_in: int = NUM_FRAME_FEATURES,
        d_hid: int = 96,
        d_sp: int = 64,
        base_m: int = 48,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_hid = int(d_hid)
        self.d_sp = int(d_sp)
        self.base_m = int(base_m)

        self.in_proj = nn.Sequential(nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU())
        self.conv_blk = _ResConvBlock(d_hid, dropout=dropout)
        self.t_gru1 = nn.GRU(d_hid, d_hid // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_hid,
            nhead=4,
            dim_feedforward=d_hid * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.c_attn = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.t_gru2 = nn.GRU(d_hid, d_hid // 2, num_layers=1, batch_first=True, bidirectional=True)

        # Frame-level auxiliary supervision (inside, boundary start/end)
        self.frame_heads = nn.Linear(d_hid, 3)

        # Factored O(M) span representations -> dual outputs: [expected_tiou_logit, p05_logit]
        self.pa_proj = nn.Linear(d_hid, d_sp)
        self.pb_proj = nn.Linear(d_hid, d_sp)
        self.pin_proj = nn.Linear(d_hid, d_sp)
        self.pout_proj = nn.Linear(d_hid, d_sp)
        self.ppos_proj = nn.Linear(4, d_sp)
        self.sp_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_sp, d_sp),
            nn.GELU(),
            nn.Linear(d_sp, 2),
        )

        # Learned soft-overlap pairwise transition potential `(K-1, M, M)` for `(prev_b -> curr_a)`
        self.tr_prev_proj = nn.Linear(d_hid, 32)
        self.tr_curr_proj = nn.Linear(d_hid, 32)

        ta, tb = np.triu_indices(base_m)
        self.register_buffer("ta", torch.from_numpy(ta).long())
        self.register_buffer("tb", torch.from_numpy(tb).long())
        cnt_np = (tb - ta + 1).astype(np.float32)
        self.register_buffer("cnt", torch.from_numpy(cnt_np).view(1, -1, 1))
        pos4 = np.stack(
            [
                ta / float(base_m),
                tb / float(base_m),
                cnt_np / float(base_m),
                np.log1p(cnt_np) / np.log1p(float(base_m)),
            ],
            axis=-1,
        ).astype(np.float32)
        self.register_buffer("pos4", torch.from_numpy(pos4))

    def _resolve_span_geometry(self, m_v: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if m_v == self.base_m:
            return self.ta, self.tb, self.cnt, self.pos4
        ta_np, tb_np = np.triu_indices(m_v)
        ta = torch.from_numpy(ta_np).long().to(device)
        tb = torch.from_numpy(tb_np).long().to(device)
        cnt_np = (tb_np - ta_np + 1).astype(np.float32)
        cnt = torch.from_numpy(cnt_np).view(1, -1, 1).to(device)
        pos4 = torch.from_numpy(
            np.stack(
                [
                    ta_np / float(m_v),
                    tb_np / float(m_v),
                    cnt_np / float(m_v),
                    np.log1p(cnt_np) / np.log1p(float(m_v)),
                ],
                axis=-1,
            ).astype(np.float32)
        ).to(device)
        return ta, tb, cnt, pos4

    def forward_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Batched forward pass for `(B, K, M, d_in)` frame features -> `(iou_l, p05_l, fr_l, pair_bonus)`."""
        b_sz, k_ev, m_v, d_in = x.shape
        h = self.in_proj(x.reshape(b_sz * k_ev, m_v, d_in))
        h = self.conv_blk(h)
        h_g1, _ = self.t_gru1(h)
        h = h + h_g1
        h_cross = h.view(b_sz, k_ev, m_v, self.d_hid).permute(0, 2, 1, 3).reshape(b_sz * m_v, k_ev, self.d_hid)
        h_cross = self.c_attn(h_cross).view(b_sz, m_v, k_ev, self.d_hid).permute(0, 2, 1, 3).reshape(b_sz * k_ev, m_v, self.d_hid)
        h = h + h_cross
        h_g2, _ = self.t_gru2(h)
        h = h + h_g2  # (B*K, M, d_hid)

        frame_logits = self.frame_heads(h).view(b_sz, k_ev, m_v, 3)
        ta, tb, cnt, pos4 = self._resolve_span_geometry(m_v, h.device)

        fa = self.pa_proj(h)[:, ta, :]
        fb = self.pb_proj(h)[:, tb, :]
        fin_h = self.pin_proj(h)
        fout_h = self.pout_proj(h)
        zeros = torch.zeros((b_sz * k_ev, 1, self.d_sp), device=h.device, dtype=h.dtype)
        pref_in = torch.cat([zeros, torch.cumsum(fin_h, dim=1)], dim=1)
        pref_out = torch.cat([zeros, torch.cumsum(fout_h, dim=1)], dim=1)
        s_in = pref_in[:, tb + 1, :] - pref_in[:, ta, :]
        m_in = s_in / cnt
        s_out = pref_out[:, -1:, :] - (pref_out[:, tb + 1, :] - pref_out[:, ta, :])
        m_out = torch.where(cnt < m_v, s_out / torch.clamp(float(m_v) - cnt, min=1.0), torch.zeros_like(s_out))

        z = fa + fb + m_in + m_out + self.ppos_proj(pos4).unsqueeze(0)
        logits = self.sp_head(z).view(b_sz, k_ev, -1, 2)
        pair_bonus: torch.Tensor | None = None
        if k_ev > 1:
            h_b = h.view(b_sz, k_ev, m_v, self.d_hid)
            tp = F.normalize(self.tr_prev_proj(h_b[:, :-1]).reshape(b_sz * (k_ev - 1), m_v, 32), dim=-1)
            tc = F.normalize(self.tr_curr_proj(h_b[:, 1:]).reshape(b_sz * (k_ev - 1), m_v, 32), dim=-1)
            pair_bonus = torch.bmm(tp, tc.transpose(1, 2)).view(b_sz, k_ev - 1, m_v, m_v)
        return logits[..., 0], logits[..., 1], frame_logits, pair_bonus

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass for `(K, M, d_in)` frame features -> `(iou_logit, p05_logit, frame_logits, pair_bonus)`."""
        iou_b, p05_b, fr_b, pb_b = self.forward_batch(x.unsqueeze(0))
        return iou_b[0], p05_b[0], fr_b[0], (pb_b[0] if pb_b is not None else None)

    @torch.no_grad()
    def predict_maps(self, seq_x: np.ndarray, *, target_m: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Predict `(K, M, M)` probability maps (`pred_iou`, `pred_p05`) and optional `(K-1, M, M)` pair bonus."""
        self.eval()
        k_ev, m_v, _ = seq_x.shape
        out_m = int(target_m or m_v)
        dev = next(self.parameters()).device
        xt = torch.from_numpy(np.asarray(seq_x, dtype=np.float32)).to(dev)

        # If input length != base_m, evaluate on pooled base_m grid & bilinearly upsample (symmetrized to prevent diagonal bleed).
        if m_v != self.base_m:
            x_res = F.interpolate(
                xt.transpose(1, 2), size=self.base_m, mode="linear", align_corners=True
            ).transpose(1, 2)
            iou_l, p05_l, _, pb_t = self.forward(x_res)
            iou_grid = torch.zeros((k_ev, self.base_m, self.base_m), device=dev, dtype=iou_l.dtype)
            p05_grid = torch.zeros((k_ev, self.base_m, self.base_m), device=dev, dtype=p05_l.dtype)
            iou_grid[:, self.ta, self.tb] = torch.sigmoid(iou_l)
            p05_grid[:, self.ta, self.tb] = torch.sigmoid(p05_l)
            diag_iou = torch.diag_embed(torch.diagonal(iou_grid, dim1=1, dim2=2))
            diag_p05 = torch.diag_embed(torch.diagonal(p05_grid, dim1=1, dim2=2))
            iou_grid = iou_grid + iou_grid.transpose(1, 2) - diag_iou
            p05_grid = p05_grid + p05_grid.transpose(1, 2) - diag_p05
            iou_out = F.interpolate(iou_grid.unsqueeze(1), size=(out_m, out_m), mode="bilinear", align_corners=True).squeeze(1)
            p05_out = F.interpolate(p05_grid.unsqueeze(1), size=(out_m, out_m), mode="bilinear", align_corners=True).squeeze(1)
            pb_out = (
                F.interpolate(pb_t.unsqueeze(1), size=(out_m, out_m), mode="bilinear", align_corners=True).squeeze(1).cpu().numpy()
                if pb_t is not None
                else None
            )
            return iou_out.cpu().numpy(), p05_out.cpu().numpy(), pb_out

        iou_l, p05_l, _, pb_t = self.forward(xt)
        iou_grid = np.full((k_ev, out_m, out_m), -np.inf, dtype=np.float64)
        p05_grid = np.full((k_ev, out_m, out_m), -np.inf, dtype=np.float64)
        ta_np = self.ta.cpu().numpy()
        tb_np = self.tb.cpu().numpy()
        iou_grid[:, ta_np, tb_np] = torch.sigmoid(iou_l).cpu().numpy()
        p05_grid[:, ta_np, tb_np] = torch.sigmoid(p05_l).cpu().numpy()
        pb_np = pb_t.cpu().numpy().astype(np.float64) if pb_t is not None else None
        return iou_grid, p05_grid, pb_np


def build_span_net_from_checkpoint(ckpt_path: Path | str, device: str = "cpu") -> CrossEventSpanNet:
    """Restore a saved `CrossEventSpanNet` state dict strictly from disk."""
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    cfg = payload["config"]
    mdl = CrossEventSpanNet(
        d_in=int(cfg["d_in"]),
        d_hid=int(cfg["d_hid"]),
        d_sp=int(cfg["d_sp"]),
        base_m=int(cfg["base_m"]),
        dropout=float(cfg.get("dropout", 0.12)),
    )
    mdl.load_state_dict(payload["state_dict"])
    return mdl.to(device).eval()


def save_span_net_checkpoint(model: CrossEventSpanNet, ckpt_path: Path | str) -> None:
    """Persist `CrossEventSpanNet` config & weights safely with atomic write."""
    p = Path(ckpt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save(
        {
            "config": {
                "d_in": model.d_in,
                "d_hid": model.d_hid,
                "d_sp": model.d_sp,
                "base_m": model.base_m,
            },
            "state_dict": model.state_dict(),
        },
        str(tmp),
    )
    tmp.replace(p)


class SpanGrounderLGBM:
    """Native LightGBM dual-head span grounder (`regression tIoU` + `binary P(tIoU>=0.5)`)."""

    def __init__(self, reg_booster: Any | None = None, cls_booster: Any | None = None) -> None:
        self.reg_booster = reg_booster
        self.cls_booster = cls_booster

    @classmethod
    def fit(
        cls,
        X_tr: np.ndarray,
        y_tr_iou: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val_iou: np.ndarray | None = None,
        num_boost_round: int = 450,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        seed: int = 20261125,
        num_threads: int = 4,
    ) -> "SpanGrounderLGBM":
        import lightgbm as lgb  # type: ignore

        tr_reg = lgb.Dataset(X_tr, label=y_tr_iou.astype(np.float32), free_raw_data=False)
        tr_cls = lgb.Dataset(X_tr, label=(y_tr_iou >= 0.5).astype(np.float32), free_raw_data=False)

        va_reg = (
            [lgb.Dataset(X_val, label=y_val_iou.astype(np.float32), reference=tr_reg)]
            if (X_val is not None and y_val_iou is not None)
            else None
        )
        va_cls = (
            [lgb.Dataset(X_val, label=(y_val_iou >= 0.5).astype(np.float32), reference=tr_cls)]
            if (X_val is not None and y_val_iou is not None)
            else None
        )

        cb_es = [lgb.early_stopping(40, verbose=False)] if va_reg is not None else []
        params_common = {
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_data_in_leaf": 40,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "lambda_l2": 2.0,
            "seed": int(seed),
            "num_threads": int(num_threads),
            "verbose": -1,
        }
        reg_b = lgb.train(
            {**params_common, "objective": "regression", "metric": "l2"},
            tr_reg,
            num_boost_round=int(num_boost_round),
            valid_sets=va_reg,
            callbacks=cb_es,
        )
        cls_b = lgb.train(
            {**params_common, "objective": "binary", "metric": "binary_logloss"},
            tr_cls,
            num_boost_round=int(num_boost_round),
            valid_sets=va_cls,
            callbacks=cb_es,
        )
        return cls(reg_booster=reg_b, cls_booster=cls_b)

    @staticmethod
    def _upsample_strided_span_maps(
        reg_raw: np.ndarray,
        cls_raw: np.ndarray,
        ta_sub: np.ndarray,
        tb_sub: np.ndarray,
        step: int,
        m_v: int,
        k_ev: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct `(K, M, M)` upper-triangular score maps from strided `(K, N_sub)` predictions."""
        from scipy.ndimage import zoom  # type: ignore

        sub_m = len(range(0, m_v, step))
        su = ta_sub // step
        sv = tb_sub // step
        sub_iou = np.zeros((k_ev, sub_m, sub_m), dtype=np.float64)
        sub_p05 = np.zeros((k_ev, sub_m, sub_m), dtype=np.float64)
        sub_iou[:, su, sv] = np.clip(reg_raw, 0.0, 1.0)
        sub_p05[:, su, sv] = np.clip(cls_raw, 1e-6, 1.0 - 1e-6)

        idx_sub = np.arange(sub_m)
        sub_iou = sub_iou + np.swapaxes(sub_iou, 1, 2)
        sub_iou[:, idx_sub, idx_sub] *= 0.5
        sub_p05 = sub_p05 + np.swapaxes(sub_p05, 1, 2)
        sub_p05[:, idx_sub, idx_sub] *= 0.5

        z_f = float(m_v) / float(sub_m)
        up_iou = np.clip(zoom(sub_iou, (1.0, z_f, z_f), order=1, mode="nearest"), 0.0, 1.0)
        up_p05 = np.clip(zoom(sub_p05, (1.0, z_f, z_f), order=1, mode="nearest"), 1e-6, 1.0 - 1e-6)

        ta_full, tb_full = np.triu_indices(m_v)
        g_iou = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
        g_p05 = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
        g_iou[:, ta_full, tb_full] = up_iou[:, ta_full, tb_full]
        g_p05[:, ta_full, tb_full] = up_p05[:, ta_full, tb_full]
        return g_iou, g_p05

    def predict_span_scores(
        self,
        X_spans: np.ndarray,
        ta: np.ndarray,
        tb: np.ndarray,
        m_v: int,
        *,
        num_threads: int = 4,
        stride_above_m: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict `(K, M, M)` matrices (`pred_iou`, `pred_p05`) over upper-triangular spans."""
        k_ev, n_sp, _ = X_spans.shape
        if m_v <= int(stride_above_m):
            flat = X_spans.reshape(-1, X_spans.shape[-1])
            reg_raw = np.clip(self.reg_booster.predict(flat, num_threads=int(num_threads)), 0.0, 1.0).reshape(k_ev, n_sp)
            cls_raw = np.clip(self.cls_booster.predict(flat, num_threads=int(num_threads)), 1e-6, 1.0 - 1e-6).reshape(
                k_ev, n_sp
            )
            g_iou = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
            g_p05 = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
            g_iou[:, ta, tb] = reg_raw
            g_p05[:, ta, tb] = cls_raw
            return g_iou, g_p05

        step = max(2, int(np.ceil(float(m_v) / float(stride_above_m))))
        mask = (ta % step == 0) & (tb % step == 0)
        x_sub = X_spans[:, mask, :]
        n_sub = int(x_sub.shape[1])
        flat_sub = x_sub.reshape(-1, x_sub.shape[-1])
        reg_raw = self.reg_booster.predict(flat_sub, num_threads=int(num_threads)).reshape(k_ev, n_sub)
        cls_raw = self.cls_booster.predict(flat_sub, num_threads=int(num_threads)).reshape(k_ev, n_sub)
        return self._upsample_strided_span_maps(reg_raw, cls_raw, ta[mask], tb[mask], step, m_v, k_ev)

    def save(self, model_dir: Path | str) -> None:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.reg_booster.save_model(str(d / "lgbm_iou_reg.txt"))
        self.cls_booster.save_model(str(d / "lgbm_p05_cls.txt"))
        (d / "lgbm_meta.json").write_text(
            json.dumps({"num_span_features": NUM_SPAN_FEATURES}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_dir: Path | str) -> "SpanGrounderLGBM":
        import lightgbm as lgb  # type: ignore

        d = Path(model_dir)
        reg_b = lgb.Booster(model_file=str(d / "lgbm_iou_reg.txt"))
        cls_b = lgb.Booster(model_file=str(d / "lgbm_p05_cls.txt"))
        return cls(reg_booster=reg_b, cls_booster=cls_b)


class JointMultiEventGrounder:
    """Ensemble scorer combining `CrossEventSpanNet` + `SpanGrounderLGBM` + Pairwise Soft-Overlap DP."""

    def __init__(
        self,
        *,
        span_nets: Sequence[CrossEventSpanNet] = (),
        lgbm: SpanGrounderLGBM | None = None,
        weights: Mapping[str, float] = DEFAULT_BACKBONE_WEIGHTS,
    ) -> None:
        self.span_nets = list(span_nets)
        self.lgbm = lgbm
        self.weights = dict(weights)

    @classmethod
    def from_dir(cls, model_dir: Path | str, device: str = "cpu") -> "JointMultiEventGrounder":
        d = Path(model_dir)
        nets = [build_span_net_from_checkpoint(p, device=device) for p in sorted(d.glob("cross_event_span_net*.pt"))]
        lgbm = SpanGrounderLGBM.load(d) if (d / "lgbm_iou_reg.txt").is_file() else None
        return cls(span_nets=nets, lgbm=lgbm)

    def predict_emissions(
        self,
        timestamps_sec: Sequence[float],
        duration_sec: float,
        curve_by_backbone: Mapping[str, np.ndarray],
        *,
        frame_embeddings_by_backbone: Mapping[str, np.ndarray] | None = None,
        w_net_iou: float = 0.55,
        w_net_logp: float = 0.15,
        w_lgb_iou: float = 0.45,
        w_lgb_logp: float = 0.12,
        w_pair_net: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Compute `(emit_cube, pair_bonus_cube, fused_curve_z)` on any PTS grid resolution `M`."""
        fctx = extract_multievent_frame_features(
            timestamps_sec,
            duration_sec,
            curve_by_backbone,
            frame_embeddings_by_backbone=frame_embeddings_by_backbone,
            weights=self.weights,
        )
        seq_x = fctx["seq_x"]
        k_ev, m_v, _ = seq_x.shape
        ta, tb = np.triu_indices(m_v)
        emit = np.full((k_ev, m_v, m_v), -np.inf, dtype=np.float64)
        emit[:, ta, tb] = 0.0
        pair_bonus: np.ndarray | None = None

        if self.span_nets:
            net_iou_acc = np.zeros((k_ev, m_v, m_v), dtype=np.float64)
            net_p05_acc = np.zeros((k_ev, m_v, m_v), dtype=np.float64)
            pb_acc = np.zeros((max(0, k_ev - 1), m_v, m_v), dtype=np.float64) if k_ev > 1 else None
            pb_cnt = 0
            for net in self.span_nets:
                g_iou, g_p05, pb = net.predict_maps(seq_x, target_m=m_v)
                net_iou_acc += np.nan_to_num(g_iou, neginf=0.0)
                net_p05_acc += np.nan_to_num(g_p05, neginf=0.0)
                if pb is not None and pb_acc is not None:
                    pb_acc += pb
                    pb_cnt += 1
            n_nets = float(len(self.span_nets))
            net_iou = net_iou_acc / n_nets
            net_p05 = np.clip(net_p05_acc / n_nets, 1e-5, 1.0 - 1e-5)
            emit[:, ta, tb] += (
                float(w_net_iou) * net_iou[:, ta, tb]
                + float(w_net_logp) * np.log(net_p05[:, ta, tb])
            )
            if pb_acc is not None and pb_cnt > 0 and float(w_pair_net) != 0.0:
                pair_bonus = float(w_pair_net) * (pb_acc / float(pb_cnt))

        if self.lgbm is not None:
            x_sp, ta_sp, tb_sp, _, _ = extract_multievent_span_features(timestamps_sec, duration_sec, fctx)
            lgb_iou, lgb_p05 = self.lgbm.predict_span_scores(x_sp, ta_sp, tb_sp, m_v)
            lgb_p05_clip = np.clip(lgb_p05, 1e-5, 1.0 - 1e-5)
            emit[:, ta, tb] += (
                float(w_lgb_iou) * lgb_iou[:, ta, tb]
                + float(w_lgb_logp) * np.log(lgb_p05_clip[:, ta, tb])
            )

        return emit, pair_bonus, fctx["z_t"]

    def localize(
        self,
        timestamps_sec: Sequence[float],
        duration_sec: float,
        curve_by_backbone: Mapping[str, np.ndarray],
        *,
        clause_ids: Sequence[str] | None = None,
        clause_texts: Sequence[str] | None = None,
        frame_ordinals: Sequence[int] | None = None,
        frame_embeddings_by_backbone: Mapping[str, np.ndarray] | None = None,
        w_net_iou: float = 0.55,
        w_net_logp: float = 0.15,
        w_lgb_iou: float = 0.45,
        w_lgb_logp: float = 0.12,
        w_pair_net: float = 0.08,
        w_gap: float = 0.80,
        w_ov: float = 0.60,
        w_edge: float = 0.20,
        max_ov_ratio: float = 0.50,
        enforce_end_order: bool = True,
    ) -> EventChainLocalizationResult | None:
        emit, pair_bonus, z_t = self.predict_emissions(
            timestamps_sec,
            duration_sec,
            curve_by_backbone,
            frame_embeddings_by_backbone=frame_embeddings_by_backbone,
            w_net_iou=w_net_iou,
            w_net_logp=w_net_logp,
            w_lgb_iou=w_lgb_iou,
            w_lgb_logp=w_lgb_logp,
            w_pair_net=w_pair_net,
        )
        return localize_soft_overlap_segment_dp(
            timestamps_sec,
            emit,
            clause_ids=clause_ids,
            clause_texts=clause_texts,
            frame_ordinals=frame_ordinals,
            v_dur_sec=duration_sec,
            pair_transition_bonus=pair_bonus,
            w_gap=w_gap,
            w_ov=w_ov,
            w_edge=w_edge,
            max_ov_ratio=max_ov_ratio,
            enforce_end_order=enforce_end_order,
            phi_curves=z_t,
        )
