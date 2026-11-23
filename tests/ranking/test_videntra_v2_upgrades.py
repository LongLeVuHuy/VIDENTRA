import json
from pathlib import Path

import numpy as np
import pytest

from videntra.ranking.cross_modal_adapter import (
    adapt_vectors_with_checkpoint,
    score_cross_modal_adapter_gallery,
    train_cross_modal_adapter,
)
from videntra.ranking.vedje_reranker import (
    TrainQueryBankHubness,
    apply_qb_dis_matrix,
    apply_qb_znorm_matrix,
)
from videntra.retrieval.msrvtt_matrix import (
    compute_vl3_gallery_score_matrices,
    load_gallery_feature_bank,
    sha256_text,
)


def _norm(shape: tuple[int, ...], seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    return arr / np.maximum(1e-12, np.linalg.norm(arr, axis=-1, keepdims=True))


def test_qb_dis_and_qb_znorm_reduce_hubness_inductively() -> None:
    rng = np.random.default_rng(42)
    bank = _norm((120, 32), seed=42)
    hub_dir = bank[:20].mean(axis=0)
    hub_dir = hub_dir / np.linalg.norm(hub_dir)

    vids = _norm((5, 32), seed=99)
    # Force video index 0 to be a massive generic hub close to bank direction
    vids[0] = 0.92 * hub_dir + 0.08 * vids[0]
    vids[0] /= np.linalg.norm(vids[0])

    qb = TrainQueryBankHubness(bank, k_hub=10, temperature=0.05)
    priors = qb.compute_video_priors(vids)
    assert "lse_sum_005" in priors and "bank_mean" in priors and "bank_std" in priors
    assert priors["bank_mean"][0] > priors["bank_mean"][2]

    # Query aligns genuinely with video 2, while hub video 0 has inflated raw cosine
    q = 0.85 * vids[2] + 0.15 * hub_dir
    q /= np.linalg.norm(q)
    raw = (q[None, :] @ vids.T).astype(np.float64)
    dis = apply_qb_dis_matrix(raw, priors, tau=0.05)
    zn = apply_qb_znorm_matrix(raw, priors)

    assert int(np.argmax(dis[0])) == 2
    assert int(np.argmax(zn[0])) == 2
    # Must be strictly query-independent across columns: single query vs 2-query batch identical for query 0
    raw2 = np.vstack([raw, (vids[3][None, :] @ vids.T).astype(np.float64)])
    dis2 = apply_qb_dis_matrix(raw2, priors, tau=0.05)
    np.testing.assert_allclose(dis[0], dis2[0], atol=1e-12)


def test_residual_cross_modal_adapter_trains_and_projects_cleanly(tmp_path: Path) -> None:
    n_vid = 40
    dims = {"vp": 16, "pe_bigg": 24, "dfn": 16}
    v_feats = {b: _norm((n_vid, d), seed=10 + i) for i, (b, d) in enumerate(dims.items())}
    # Build 2 noisy domain-shifted captions per video
    pair_vid = np.repeat(np.arange(n_vid), 2)
    shift_vec = {b: _norm((d,), seed=200 + i) for i, (b, d) in enumerate(dims.items())}
    q_feats = {}
    for i, (b, d) in enumerate(dims.items()):
        base = v_feats[b][pair_vid] + 0.35 * shift_vec[b][None, :] + 0.15 * _norm((len(pair_vid), d), seed=300 + i)
        q_feats[b] = (base / np.maximum(1e-12, np.linalg.norm(base, axis=1, keepdims=True))).astype(np.float32)

    vid_groups = [f"sg_{i}" for i in range(n_vid)]
    pair_groups = [vid_groups[int(v)] for v in pair_vid]
    ckpt = tmp_path / "adapter.pt"
    res = train_cross_modal_adapter(
        backbone_dims=dims,
        train_video_feats=v_feats,
        train_query_feats=q_feats,
        pair_video_idx=pair_vid,
        pair_source_groups=pair_groups,
        video_source_groups=vid_groups,
        checkpoint_out=ckpt,
        joint_dim=32,
        hidden_dim=16,
        epochs=6,
        batch_size=16,
        seed=7,
    )
    assert res.epochs_ran == 6
    assert ckpt.is_file()

    sims = score_cross_modal_adapter_gallery(ckpt, video_feats=v_feats, query_feats=q_feats)
    assert sims["joint"].shape == (len(pair_vid), n_vid)
    assert set(sims.keys()) == {"joint", "vp", "pe_bigg", "dfn"}

    proj = adapt_vectors_with_checkpoint(ckpt, video_feats=v_feats, query_feats=q_feats)
    v_norms = np.linalg.norm(proj["videos"]["joint"], axis=1)
    np.testing.assert_allclose(v_norms, np.ones(n_vid), atol=1e-5)


def test_load_gallery_feature_bank_supports_vl3_shards(tmp_path: Path) -> None:
    mpath = tmp_path / "gallery.jsonl"
    mpath.write_text('{"video_id": "v1"}\n{"video_id": "v2"}\n', encoding="utf-8")
    qpath = tmp_path / "queries.jsonl"
    qpath.write_text('{"query_id": "q1", "query_text": "cat on sofa"}\n', encoding="utf-8")

    vl3_dir = tmp_path / "vl3_shard"
    vl3_dir.mkdir()
    (vl3_dir / "receipt.json").write_text('{"status": "SUCCESS"}', encoding="utf-8")
    np.save(vl3_dir / "pe_bigg_video_vectors.npy", _norm((2, 16), seed=1))
    np.save(vl3_dir / "pe_bigg_frame_vectors_fp16.npy", _norm((2, 4, 16), seed=2).astype(np.float16))
    np.save(vl3_dir / "pe_bigg_query_vectors.npy", _norm((1, 16), seed=3))
    np.save(vl3_dir / "dfn_video_vectors.npy", _norm((2, 12), seed=4))
    np.save(vl3_dir / "dfn_frame_vectors_fp16.npy", _norm((2, 4, 12), seed=5).astype(np.float16))
    np.save(vl3_dir / "dfn_query_vectors.npy", _norm((1, 12), seed=6))
    (vl3_dir / "video_rows.jsonl").write_text('{"video_id": "v1"}\n{"video_id": "v2"}\n', encoding="utf-8")
    q_sha = sha256_text("cat on sofa")
    (vl3_dir / "query_rows.jsonl").write_text(json.dumps({"query_text": "cat on sofa", "query_text_sha256": q_sha}) + "\n", encoding="utf-8")

    bank = load_gallery_feature_bank(
        gallery_name="g_test",
        gallery_manifest_path=mpath,
        queries_path=qpath,
        vl3_dirs=[vl3_dir],
        train_pe_bigg_bank=_norm((10, 16), seed=8),
        train_dfn_bank=_norm((10, 12), seed=9),
    )
    assert bank.pe_bigg_video_vecs is not None and bank.pe_bigg_video_vecs.shape == (2, 16)
    assert bank.dfn_video_vecs is not None and bank.dfn_video_vecs.shape == (2, 12)
    assert bank.pe_bigg_hub_priors is not None and "lse_sum_005" in bank.pe_bigg_hub_priors
    mats = compute_vl3_gallery_score_matrices(bank)
    assert mats["pe_bigg_xpool"].shape == (1, 2)
    assert mats["dfn_blend"].shape == (1, 2)


def test_margin_gated_vlm_blending_preserves_confident_queries() -> None:
    from scripts.run_msrvtt_v2_matrix import _blend_stage1_and_vlm_preds

    base_preds = [
        {"query_id": "q_conf", "gallery": "g", "ranked_video_ids": ["v1", "v2", "v3", "v4"]},
        {"query_id": "q_unc", "gallery": "g", "ranked_video_ids": ["v1", "v2", "v3", "v4"]},
    ]
    top20 = {
        "q_conf": {"v1": 2.5, "v2": 0.4, "v3": 0.1, "v4": -0.2},  # margin = 2.1 >= 1.0
        "q_unc": {"v1": 1.05, "v2": 1.00, "v3": 0.2, "v4": -0.5},  # margin = 0.05 < 1.0
    }
    vlm_scores = {
        "q_conf": {"v1": -2.0, "v2": 3.0, "v3": 0.0},
        "q_unc": {"v1": -1.5, "v2": 3.2, "v3": 0.1},
    }
    out = _blend_stage1_and_vlm_preds(
        base_preds, top20, vlm_scores, margin_gate=1.0, w_vlm=0.8, top_k_rerank=3,
    )
    assert out[0]["ranked_video_ids"] == ["v1", "v2", "v3", "v4"]
    assert out[1]["ranked_video_ids"][0] == "v2"
    assert set(out[1]["ranked_video_ids"]) == {"v1", "v2", "v3", "v4"}


def test_vlm_debiasing_and_margin_raw_fusion() -> None:
    from scripts.run_msrvtt_v2_matrix import _blend_stage1_and_vlm_preds, _debias_vlm_scores

    raw_vlm = {
        "q1": {"v1": 2.0, "v2": 1.0, "v3": 0.0},
        "q2": {"v1": 2.0, "v2": 0.5, "v3": 1.5},
    }
    deb = _debias_vlm_scores(raw_vlm, alpha=1.0, shrink=0.0)
    # v1 appears high across queries so its debiased margin decreases relative to unbiased candidates
    assert deb["q1"]["v1"] < deb["q1"]["v2"]
    preds = [{"query_id": "q1", "gallery": "g", "ranked_video_ids": ["v1", "v2", "v3"]}]
    top20 = {"q1": {"v1": 1.1, "v2": 1.05, "v3": 0.2}}
    out = _blend_stage1_and_vlm_preds(
        preds, top20, raw_vlm, margin_gate=0.5, w_vlm=0.8, top_k_rerank=3, mode="margin_raw", debias_alpha=1.0, debias_shrink=0.0,
    )
    assert out[0]["ranked_video_ids"][0] == "v2"
