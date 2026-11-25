"""Unit tests for Soft-Overlap Semi-Markov DP, CrossEventSpanNet, and SpanGrounderLGBM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from videntra.localization.span_grounder import (
    NUM_FRAME_FEATURES,
    NUM_SPAN_FEATURES,
    CrossEventSpanNet,
    JointMultiEventGrounder,
    SpanGrounderLGBM,
    build_span_net_from_checkpoint,
    extract_multievent_frame_features,
    extract_multievent_span_features,
    save_span_net_checkpoint,
)
from videntra.retrieval.alignment import (
    build_zeroshot_soft_dp_emissions,
    localize_exact_segment_chain_dp,
    localize_soft_overlap_segment_dp,
)


def test_soft_overlap_dp_recovers_overlapping_and_nested_chains() -> None:
    ts = np.linspace(0.5, 47.5, 48, dtype=np.float64)
    k_events, m_v = 2, 48
    emit = np.full((k_events, m_v, m_v), -10.0, dtype=np.float64)
    for a in range(m_v):
        emit[:, a, a:] = -0.2

    # Overlapping pair: event 0 at [4, 24], event 1 at [14, 34] (overlap = 11 frames)
    emit[0, 4, 24] = 3.0
    emit[1, 14, 34] = 3.0

    # Strict non-overlapping DP cannot select both [4, 24] and [14, 34] together
    vis = np.zeros((k_events, m_v), dtype=np.float64)
    vis[0, 4:25] = 0.8
    vis[1, 14:35] = 0.8
    strict_res = localize_exact_segment_chain_dp(ts, vis, relations_to_prev=["start", "before"], v_dur_sec=48.0)
    assert strict_res is not None
    assert strict_res.spans[0].end_sec <= strict_res.spans[1].start_sec + 1e-3

    # Soft-Overlap DP recovers exact overlapping pair
    soft_res = localize_soft_overlap_segment_dp(
        ts, emit, clause_ids=["c0", "c1"], v_dur_sec=48.0, w_gap=0.1, w_ov=0.05, w_edge=0.0, max_ov_ratio=0.5
    )
    assert soft_res is not None
    assert soft_res.spans[0].start_frame_index == 4
    assert soft_res.spans[0].end_frame_index == 24
    assert soft_res.spans[1].start_frame_index == 14
    assert soft_res.spans[1].end_frame_index == 34

    # Nested pair: event 0 at [2, 30], event 1 nested at [10, 22] (`enforce_end_order=False`)
    emit2 = np.full((k_events, m_v, m_v), -5.0, dtype=np.float64)
    for a in range(m_v):
        emit2[:, a, a:] = -0.2
    emit2[0, 2, 30] = 4.0
    emit2[1, 10, 22] = 4.0
    nested_res = localize_soft_overlap_segment_dp(
        ts, emit2, clause_ids=["c0", "c1"], v_dur_sec=48.0, w_gap=0.05, w_ov=0.02, w_edge=0.0, max_ov_ratio=0.5, enforce_end_order=False
    )
    assert nested_res is not None
    assert nested_res.spans[0].start_frame_index == 2
    assert nested_res.spans[0].end_frame_index == 30
    assert nested_res.spans[1].start_frame_index == 10
    assert nested_res.spans[1].end_frame_index == 22


def test_multievent_features_and_zeroshot_emissions_resolution_invariant() -> None:
    rng = np.random.default_rng(7)
    for m_v in (48, 128):
        ts = np.linspace(0.5, 99.5, m_v)
        curves = {
            "pe_bigg": rng.normal(0.2, 0.05, size=(3, m_v)).astype(np.float32),
            "pe_l": rng.normal(0.2, 0.05, size=(3, m_v)).astype(np.float32),
            "sig400": rng.normal(0.15, 0.04, size=(3, m_v)).astype(np.float32),
        }
        embs = {"pe_bigg": rng.normal(size=(m_v, 16)).astype(np.float32)}
        fctx = extract_multievent_frame_features(ts, 100.0, curves, frame_embeddings_by_backbone=embs)
        assert fctx["seq_x"].shape == (3, m_v, NUM_FRAME_FEATURES)
        assert np.all(np.isfinite(fctx["seq_x"]))

        x_sp, ta, tb, le_a, re_a = extract_multievent_span_features(ts, 100.0, fctx)
        expected_s = (m_v * (m_v + 1)) // 2
        assert x_sp.shape == (3, expected_s, NUM_SPAN_FEATURES)
        assert len(ta) == expected_s and len(tb) == expected_s
        assert len(le_a) == m_v and len(re_a) == m_v
        assert np.all(np.isfinite(x_sp))

        zs = build_zeroshot_soft_dp_emissions(ts, curves["pe_bigg"], frame_embeddings=embs["pe_bigg"])
        assert zs.shape == (3, m_v, m_v)
        assert np.all(np.isfinite(zs[:, ta, tb]))


def test_cross_event_span_net_forward_backward_save_load(tmp_path: Path) -> None:
    net = CrossEventSpanNet(d_in=NUM_FRAME_FEATURES, d_hid=32, d_sp=24, base_m=48)
    x = torch.randn(3, 48, NUM_FRAME_FEATURES, requires_grad=True)
    iou_l, p05_l, fr_l, pb = net(x)
    assert iou_l.shape == (3, (48 * 49) // 2)
    assert p05_l.shape == iou_l.shape
    assert fr_l.shape == (3, 48, 3)
    assert pb is not None and pb.shape == (2, 48, 48)

    loss = iou_l.mean() + p05_l.mean() + fr_l.mean() + pb.mean()
    loss.backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))

    ckpt = tmp_path / "net.pt"
    save_span_net_checkpoint(net, ckpt)
    loaded = build_span_net_from_checkpoint(ckpt)
    net.eval()
    g1, p1, b1 = net.predict_maps(x.detach().numpy())
    g2, p2, b2 = loaded.predict_maps(x.detach().numpy())
    assert np.allclose(g1, g2, atol=1e-6)
    assert np.allclose(p1, p2, atol=1e-6)
    assert b1 is not None and b2 is not None and np.allclose(b1, b2, atol=1e-6)

    # Also test M=128 bilinear upsampling path
    x128 = np.random.default_rng(3).normal(size=(3, 128, NUM_FRAME_FEATURES)).astype(np.float32)
    g128, p128, b128 = loaded.predict_maps(x128, target_m=128)
    assert g128.shape == (3, 128, 128)
    assert p128.shape == (3, 128, 128)
    assert b128 is not None and b128.shape == (2, 128, 128)


def test_cross_event_span_net_batch_equivalence() -> None:
    net = CrossEventSpanNet(d_in=NUM_FRAME_FEATURES, d_hid=32, d_sp=24, base_m=48)
    net.eval()
    x0 = torch.randn(4, 48, NUM_FRAME_FEATURES)
    x1 = x0 * 0.5
    xb = torch.stack([x0, x1], dim=0)
    with torch.no_grad():
        bi, bp, bf, bpb = net.forward_batch(xb)
        s0 = net(x0)
        s1 = net(x1)
    assert torch.allclose(bi[0], s0[0], atol=1e-4)
    assert torch.allclose(bp[1], s1[1], atol=1e-4)
    assert torch.allclose(bf[0], s0[2], atol=1e-4)
    assert bpb is not None and s0[3] is not None and torch.allclose(bpb[0], s0[3], atol=1e-4)


def test_lgbm_span_grounder_and_joint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n_rows = 400
    x_tr = rng.normal(size=(n_rows, NUM_SPAN_FEATURES)).astype(np.float32)
    y_tr = (1.0 / (1.0 + np.exp(-x_tr[:, 0]))).astype(np.float32)
    lgb_m = SpanGrounderLGBM.fit(x_tr, y_tr, num_boost_round=15, num_leaves=15, seed=7)
    lgb_m.save(tmp_path)
    loaded_lgb = SpanGrounderLGBM.load(tmp_path)

    ta128, tb128 = np.triu_indices(128)
    x128_sp = rng.normal(size=(2, len(ta128), NUM_SPAN_FEATURES)).astype(np.float32)
    gi128, gp128 = loaded_lgb.predict_span_scores(x128_sp, ta128, tb128, 128)
    assert gi128.shape == (2, 128, 128) and gp128.shape == (2, 128, 128)
    assert np.all((gi128[:, ta128, tb128] >= 0.0) & (gi128[:, ta128, tb128] <= 1.0))
    assert np.all((gp128[:, ta128, tb128] >= 1e-6) & (gp128[:, ta128, tb128] <= 1.0))
    la128, lb128 = np.tril_indices(128, k=-1)
    assert np.all(np.isneginf(gi128[:, la128, lb128])) and np.all(np.isneginf(gp128[:, la128, lb128]))

    net = CrossEventSpanNet(d_in=NUM_FRAME_FEATURES, d_hid=24, d_sp=16, base_m=48)
    save_span_net_checkpoint(net, tmp_path / "cross_event_span_net_s0.pt")

    joint = JointMultiEventGrounder.from_dir(tmp_path)
    assert len(joint.span_nets) == 1
    assert joint.lgbm is not None

    ts = np.linspace(0.5, 47.5, 48)
    curves = {
        "pe_bigg": rng.normal(size=(2, 48)).astype(np.float32),
        "pe_l": rng.normal(size=(2, 48)).astype(np.float32),
        "sig400": rng.normal(size=(2, 48)).astype(np.float32),
    }
    res = joint.localize(ts, 48.0, curves, clause_ids=["c0", "c1"], clause_texts=["a", "b"])
    assert res is not None and len(res.spans) == 2
    assert all(sp.end_sec > sp.start_sec for sp in res.spans)
