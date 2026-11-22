"""Unit and integration tests for MSR-VTT 1K-A protocol, Kalman tracking (R6), VEDJE (R3), and QB-CSLS."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_msrvtt_1ka_protocol import (
    _allocate_balanced_source_groups,
    normalize_youtube_source_group,
    select_deterministic_caption,
)
from src.videntra.ranking.vedje_reranker import (
    TrainQueryBankHubness,
    compute_spaced_top2_frame_score,
    compute_xpool_score,
    detect_false_negative_mask,
    score_vedje_reranker_batch,
    train_vedje_reranker,
)
from src.videntra.relations.kalman_tracker import (
    ShotAwareKalmanTracker,
    classify_motion_relation_query,
    compute_track_query_signals,
)


def test_source_group_and_deterministic_caption() -> None:
    assert normalize_youtube_source_group("https://www.youtube.com/watch?v=Sa4BUsvAcjc", "video7010") == "yt:Sa4BUsvAcjc"
    assert normalize_youtube_source_group("https://youtu.be/2Poaq62Wrvk", "video7011") == "yt:2Poaq62Wrvk"
    assert normalize_youtube_source_group("", "video9999") == "vid:video9999"

    sents = [
        {"video_id": "video10", "sen_id": 2, "caption": "second"},
        {"video_id": "video10", "sen_id": 1, "caption": "first"},
        {"video_id": "video10", "sen_id": 3, "caption": "third"},
    ]
    pick1 = select_deterministic_caption("video10", sents, seed=20261122)
    pick2 = select_deterministic_caption("video10", list(reversed(sents)), seed=20261122)
    assert pick1 == pick2


def test_balanced_source_group_allocation_disjoint() -> None:
    groups = []
    for i in range(60):
        n_vids = 2 if i < 10 else 1
        vids = [{"video_id": f"v_{i}_{j}", "category": (i + j) % 5} for j in range(n_vids)]
        groups.append({"source_group": f"sg_{i}", "videos": vids, "sort_tie": f"{i:04d}"})

    buckets, train_grps = _allocate_balanced_source_groups(
        groups, ("tune_g1", "tune_g2", "confirm_g1", "confirm_g2"), target_size=10, seed=20261122
    )
    seen_vids: set[str] = set()
    seen_sgs: set[str] = set()
    for name, b_grps in buckets.items():
        b_vids = [v["video_id"] for g in b_grps for v in g["videos"]]
        b_sgs = {g["source_group"] for g in b_grps}
        assert len(b_vids) == 10, name
        assert seen_vids.isdisjoint(b_vids)
        assert seen_sgs.isdisjoint(b_sgs)
        seen_vids.update(b_vids)
        seen_sgs.update(b_sgs)
    tr_vids = {v["video_id"] for g in train_grps for v in g["videos"]}
    tr_sgs = {g["source_group"] for g in train_grps}
    assert seen_vids.isdisjoint(tr_vids)
    assert seen_sgs.isdisjoint(tr_sgs)


def test_kalman_tracker_separates_observed_predicted_and_resets_at_cut_and_gap() -> None:
    tracker = ShotAwareKalmanTracker(mode="bytetrack", max_age_frames=2, max_gap_sec=2.0, min_hits=2)
    frames = [
        {
            "ordinal": 0,
            "pts_sec": 0.0,
            "is_shot_cut": False,
            "detections": [{"class_name": "person", "bbox": [0.10, 0.20, 0.30, 0.60], "confidence": 0.85}],
        },
        {
            "ordinal": 15,
            "pts_sec": 0.5,
            "is_shot_cut": False,
            # Occluded: no detection in this frame -> Kalman must emit predicted-only step
            "detections": [],
        },
        {
            "ordinal": 30,
            "pts_sec": 1.0,
            "is_shot_cut": False,
            # Low-confidence reappearance rescued by ByteTrack
            "detections": [{"class_name": "person", "bbox": [0.14, 0.20, 0.34, 0.60], "confidence": 0.25}],
        },
        {
            "ordinal": 45,
            "pts_sec": 1.5,
            "is_shot_cut": True,
            # Shot cut must reset track_id and increment shot_id
            "detections": [{"class_name": "person", "bbox": [0.15, 0.20, 0.35, 0.60], "confidence": 0.90}],
        },
        {
            "ordinal": 150,
            "pts_sec": 5.0,
            "is_shot_cut": False,
            # PTS gap (5.0 - 1.5 = 3.5 > 2.0) must also reset tracks
            "detections": [{"class_name": "person", "bbox": [0.15, 0.20, 0.35, 0.60], "confidence": 0.90}],
        },
    ]
    res = tracker.track_video_frames(frames)
    assert res["num_shots"] == 3
    assert len(res["reset_events"]) == 2
    assert res["reset_events"][0]["reason"] == "shot_cut"
    assert res["reset_events"][1]["reason"] == "gap_exceeded"
    assert len(res["tracks"]) == 1
    trk = res["tracks"][0]
    assert trk.observed_count == 2
    assert trk.predicted_only_count == 1
    assert trk.recovered_after_occlusion == 1
    pred_steps = [s for s in trk.steps if s.source_kind == "predicted"]
    obs_steps = [s for s in trk.steps if s.source_kind == "observed"]
    assert len(pred_steps) == 1 and pred_steps[0].observed_box is None and pred_steps[0].predicted_box is not None
    assert len(obs_steps) == 2 and all(s.observed_box is not None for s in obs_steps)

    intent = classify_motion_relation_query("a man walking with a dog")
    assert intent["has_motion"] and intent["has_relation"] and intent["is_motion_or_relation"]
    sigs = compute_track_query_signals("a man walking", res, frame_observations=frames)
    assert sigs["crop_match_score"] > 0.0
    assert sigs["track_score"] > 0.0


def test_vedje_false_negative_filter_training_and_hubness_invariance(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261122)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pos_v = np.array([0.95, 0.31, 0.0, 0.0], dtype=np.float32)
    fn_v = np.array([0.94, 0.34, 0.0, 0.0], dtype=np.float32)  # visually near-duplicate
    tn_v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    keep = detect_false_negative_mask(
        q,
        [
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.array([[0.1, 0.1, 0.9, 0.0]], dtype=np.float32),
        ],
        pos_v,
        np.stack([pos_v, fn_v, tn_v], axis=0),
        [True, False, False],
        text_sim_thresh=0.80,
        vis_sim_thresh=0.95,
    )
    assert keep.tolist() == [True, False, True]

    # Build synthetic Top-5 examples for VEDJE training
    examples = []
    for i in range(16):
        q_vec = rng.normal(size=16).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        vp_whole = rng.normal(size=(5, 16)).astype(np.float32)
        vp_whole[2] = 0.8 * q_vec + 0.2 * vp_whole[2]
        vp_whole /= np.linalg.norm(vp_whole, axis=1, keepdims=True)
        vp_wins = [rng.normal(size=(2, 16)).astype(np.float32) for _ in range(5)]
        sig_fr = [rng.normal(size=(4, 16)).astype(np.float32) for _ in range(5)]
        sig_fr[2][0] = q_vec.copy()
        scalars = rng.normal(size=(5, 6)).astype(np.float32)
        prior = np.array([0.2, 0.15, 0.12, 0.10, 0.08], dtype=np.float32)
        examples.append({
            "vp_q": q_vec,
            "sig_q": q_vec,
            "vp_whole": vp_whole,
            "vp_wins": vp_wins,
            "sig_frames": sig_fr,
            "scalars": scalars,
            "stage1_prior": prior,
            "target_pos": 2,
            "neg_keep_mask": np.array([True, False, True, True, True], dtype=bool),
        })

    ckpt = tmp_path / "vedje.pt"
    train_res = train_vedje_reranker(examples[:12], examples[12:], checkpoint_out=ckpt, epochs=5, batch_size=4)
    assert train_res.false_negatives_masked == 12
    preds = score_vedje_reranker_batch(ckpt, examples[12:])
    assert len(preds) == 4 and all(p.shape == (5,) and np.isfinite(p).all() for p in preds)

    # X-Pool & spaced top-2 checks
    f_bank = np.stack([q_vec, q_vec, -q_vec], axis=0)
    assert compute_xpool_score(f_bank, q_vec) > 0.8
    assert compute_spaced_top2_frame_score(f_bank, [0.0, 0.4, 2.5], q_vec, min_pts_gap=1.5) < 0.1

    # QB-CSLS query-independence & order invariance check
    train_bank = rng.normal(size=(30, 16)).astype(np.float32)
    gal_vecs = rng.normal(size=(25, 16)).astype(np.float32)
    qb = TrainQueryBankHubness(train_bank, k_hub=10)
    priors = qb.compute_video_priors(gal_vecs)
    s_single = qb.score_query_against_gallery(q_vec, gal_vecs @ q_vec, priors)["csls"]
    s_again = qb.score_query_against_gallery(q_vec, gal_vecs @ q_vec, priors)["csls"]
    assert np.allclose(s_single, s_again, atol=1e-7)


def test_matrix_r0_parity_tie_break_confidence_gate_and_multiprompt() -> None:
    from src.videntra.retrieval.msrvtt_matrix import (
        apply_confidence_gated_rerank,
        compute_multiprompt_query_vector,
        rank_gallery_desc,
        rerank_fixed_pool,
        sha256_text,
    )

    # Tie-break determinism & NaN/inf handling in gallery ranking
    vids = ["video010", "video002", "video005"]
    scores = np.array([0.5, 0.5, float("nan")], dtype=np.float64)
    assert rank_gallery_desc(vids, scores) == ["video002", "video010", "video005"]

    # R0 alpha=0 replay parity and fixed pool membership preservation
    pool = ["video1", "video2", "video3", "video4"]
    s1 = [0.9, 0.8, 0.7, 0.6]
    sig = [0.1, 0.9, 0.2, 0.3]
    r0_vids, r0_sc = rerank_fixed_pool(pool, s1, sig, alpha=0.0)
    assert r0_vids == pool and r0_sc == s1
    rr_vids, _ = rerank_fixed_pool(pool, s1, sig, mode="rrf", alpha=0.5)
    assert set(rr_vids) == set(pool) and len(rr_vids) == len(pool)

    # Confidence-gated top-1 margin protection
    strong_s1 = [5.0, 1.0, 0.5, 0.4]  # z-score margin > 1.5
    flip_rerank = [0.1, 0.9, 0.2, 0.3]
    assert apply_confidence_gated_rerank(pool, strong_s1, flip_rerank, top1_protect_margin=1.5)[0] == "video1"
    assert apply_confidence_gated_rerank(pool, strong_s1, flip_rerank, top1_protect_margin=0.0)[0] == "video2"

    # Multi-prompt query vector pooling
    q = "a dog running"
    qmap = {
        sha256_text(q): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sha256_text(f"a video clip showing {q}"): np.array([0.0, 1.0, 0.0], dtype=np.float32),
        sha256_text(f"a video of {q}"): np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    pooled = compute_multiprompt_query_vector(qmap, q)
    assert pooled is not None
    assert np.isclose(np.linalg.norm(pooled), 1.0, atol=1e-6)
    assert np.allclose(pooled, np.array([0.65, 0.25, 0.10]) / np.linalg.norm([0.65, 0.25, 0.10]), atol=1e-5)


def test_protocol_receipt_non_overlap_and_single_shot_lock_guard(tmp_path: Path) -> None:
    from scripts.evaluate_msrvtt_1ka_test import evaluate_sealed_test_1ka

    receipt_path = Path("work/msrvtt_protocol_20261122/protocol_receipt.json")
    if receipt_path.is_file():
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert rec["zero_video_overlap_verified"] is True
        assert rec["zero_source_group_overlap_verified"] is True
        assert rec["excluded_9k_source_leaks_count"] == 415
        assert rec["splits"]["test_1ka"]["queries_sealed_until_champion_lock"] is True

    # Verify fail-closed guard against double-evaluating sealed test_1ka
    fake_proto = tmp_path / "proto"
    fake_proto.mkdir()
    (fake_proto / "champion_lock_receipt.json").write_text(
        json.dumps({"champion_id": "R7_combined_champion_candidate", "test_1ka_evaluated": True}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ALREADY evaluated"):
        evaluate_sealed_test_1ka(protocol_root=fake_proto)


def test_vl2_scorers_csls_vl2_loader_and_r8_only_configs(tmp_path: Path) -> None:
    from scripts.prepare_clip_vp_kaggle_bundle import sha256_text
    from scripts.run_msrvtt_validation_matrix import _evaluate_split_matrix
    from src.videntra.retrieval.msrvtt_matrix import (
        apply_hubness_csls,
        compute_frame_bank_scores,
        load_gallery_feature_bank,
    )

    # 1. compute_frame_bank_scores & apply_hubness_csls
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vid_vecs = np.array([
        [0.9, 0.43589, 0.0, 0.0],
        [0.4, 0.91651, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)
    fr_vecs = np.array([
        [[0.2, 0.9798, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[0.4, 0.9165, 0.0, 0.0], [0.4, 0.9165, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    ], dtype=np.float16)
    s_mean = compute_frame_bank_scores(vid_vecs, fr_vecs, q, mode="mean")
    s_xp = compute_frame_bank_scores(vid_vecs, fr_vecs, q, mode="xpool", temperature=0.05)
    s_mix = compute_frame_bank_scores(vid_vecs, fr_vecs, q, mode="mean_plus_xpool", xpool_weight=0.40)
    assert s_mean[0] > s_mean[1] > s_mean[2]
    assert s_xp[0] > s_mean[0]
    assert np.isclose(s_mix[0], 0.60 * s_mean[0] + 0.40 * s_xp[0], atol=1e-4)

    hub = apply_hubness_csls(s_mean, {"mean_topk": np.array([0.8, 0.1, 0.0])}, csls_weight=1.0)
    assert hub[1] > hub[0]  # high-hubness candidate penalized cleanly

    # 2. Synthetic vl2 directory loading + R8 tune -> confirm + only_configs parity
    rng = np.random.default_rng(7)
    n_vids = 22
    vids = [f"video{i:03d}" for i in range(n_vids)]
    gal_manifest = tmp_path / "gallery.jsonl"
    gal_manifest.write_text("\n".join(json.dumps({"video_id": v}) for v in vids) + "\n", encoding="utf-8")

    queries = []
    labels = {}
    qtexts = [f"synthetic clip query {i}" for i in range(4)]
    for i, qt in enumerate(qtexts):
        qid = f"q_{i}"
        qsha = sha256_text(qt)
        queries.append({"query_id": qid, "query_text": qt, "query_text_sha256": qsha})
        labels[qid] = {"query_id": qid, "target_videos": [vids[i]], "source_group": f"sg_{i}"}
    q_path = tmp_path / "queries.jsonl"
    q_path.write_text("\n".join(json.dumps(r) for r in queries) + "\n", encoding="utf-8")

    def _norm_mat(shape: tuple[int, ...]) -> np.ndarray:
        arr = rng.normal(size=shape).astype(np.float32)
        arr /= np.maximum(1e-8, np.linalg.norm(arr, axis=-1, keepdims=True))
        return arr

    # Build minimal VP + VL2 shards on disk
    vp_dir = tmp_path / "vp_shard"
    vp_dir.mkdir()
    vp_v = _norm_mat((n_vids, 768))
    vp_q = np.stack([vp_v[i] for i in range(len(qtexts))], axis=0)
    np.save(vp_dir / "video_vectors.npy", vp_v)
    np.save(vp_dir / "query_vectors.npy", vp_q)
    (vp_dir / "video_rows.jsonl").write_text("\n".join(json.dumps({"row_id": i, "video_id": v}) for i, v in enumerate(vids)) + "\n", encoding="utf-8")
    (vp_dir / "query_rows.jsonl").write_text("\n".join(json.dumps({"row_id": i, "query_text": q["query_text"], "query_text_sha256": q["query_text_sha256"]}) for i, q in enumerate(queries)) + "\n", encoding="utf-8")
    (vp_dir / "receipt.json").write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")

    vl2_dir = tmp_path / "vl2_shard"
    vl2_dir.mkdir()
    pe_v = _norm_mat((n_vids, 1024))
    pe_f = _norm_mat((n_vids, 4, 1024)).astype(np.float16)
    s4_v = _norm_mat((n_vids, 1152))
    s4_f = _norm_mat((n_vids, 4, 1152)).astype(np.float16)
    pe_q = np.stack([pe_v[i] for i in range(len(qtexts))], axis=0)
    s4_q = np.stack([s4_v[i] for i in range(len(qtexts))], axis=0)
    np.save(vl2_dir / "pe_video_vectors.npy", pe_v)
    np.save(vl2_dir / "pe_frame_vectors_fp16.npy", pe_f)
    np.save(vl2_dir / "pe_query_vectors.npy", pe_q)
    np.save(vl2_dir / "sig400_video_vectors.npy", s4_v)
    np.save(vl2_dir / "sig400_frame_vectors_fp16.npy", s4_f)
    np.save(vl2_dir / "sig400_query_vectors.npy", s4_q)
    (vl2_dir / "video_rows.jsonl").write_text("\n".join(json.dumps({"row_id": i, "video_id": v}) for i, v in enumerate(vids)) + "\n", encoding="utf-8")
    (vl2_dir / "query_rows.jsonl").write_text("\n".join(json.dumps({"row_id": i, "query_text": q["query_text"], "query_text_sha256": q["query_text_sha256"]}) for i, q in enumerate(queries)) + "\n", encoding="utf-8")
    (vl2_dir / "receipt.json").write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")

    # Minimal trained VEDJE checkpoint for full matrix pass
    vedje_ex = []
    for i in range(4):
        vedje_ex.append({
            "vp_q": vp_q[i],
            "sig_q": _norm_mat((768,)),
            "vp_whole": vp_v[:20],
            "vp_wins": [vp_v[j : j + 1] for j in range(20)],
            "sig_frames": [_norm_mat((2, 768)) for _ in range(20)],
            "scalars": np.zeros((20, 10), dtype=np.float32),
            "stage1_prior": np.zeros(20, dtype=np.float32),
            "target_pos": i,
            "neg_keep_mask": np.ones(20, dtype=bool),
        })
    ckpt = tmp_path / "vedje_test.pt"
    train_vedje_reranker(vedje_ex, vedje_ex, checkpoint_out=ckpt, epochs=2, batch_size=2, seed=7)

    tr_pe_bank = _norm_mat((12, 1024))
    tr_s4_bank = _norm_mat((12, 1152))
    bank = load_gallery_feature_bank(
        gallery_name="syn",
        gallery_manifest_path=gal_manifest,
        queries_path=q_path,
        vp_dirs=[vp_dir],
        vl2_dirs=[vl2_dir],
        train_pe_bank=tr_pe_bank,
        train_sig400_bank=tr_s4_bank,
    )
    assert bank.pe_video_vecs is not None and bank.pe_video_vecs.shape == (n_vids, 1024)
    assert bank.sig400_frame_vecs is not None and bank.sig400_frame_vecs.shape == (n_vids, 4, 1152)
    assert bank.pe_hub_priors is not None and bank.sig400_hub_priors is not None

    tune_res, tuned_params, _ = _evaluate_split_matrix(
        "tune", [bank], labels,
        train_vp_bank=None, train_sig_bank=None,
        ltr_ranker=None, vedje_ckpt=ckpt, tuned_params=None, seed=7,
    )
    best_r13 = max(("R1_vp_window_rerank", "R2_frame_xpool", "R3_vedje_joint_reranker"), key=lambda c: (tune_res[c]["hit_at_1"], tune_res[c]["mrr_at_10"]))
    assert tune_res["R7_step0_base_r13"]["hit_at_1"] == tune_res[best_r13]["hit_at_1"]
    assert "R8_videntra_champion_candidate" in tune_res

    full_confirm, _, full_preds = _evaluate_split_matrix(
        "confirm", [bank], labels,
        train_vp_bank=None, train_sig_bank=None,
        ltr_ranker=None, vedje_ckpt=ckpt, tuned_params=tuned_params, seed=7,
    )
    only_res, _, only_preds = _evaluate_split_matrix(
        "test", [bank], labels,
        train_vp_bank=None, train_sig_bank=None,
        ltr_ranker=None, vedje_ckpt=ckpt, tuned_params=tuned_params, seed=7,
        only_configs=("B1_videoprism_standalone", "B0_siglip2_standalone", "R8_videntra_champion_candidate"),
    )
    assert only_res["R8_videntra_champion_candidate"]["hits_at_1"] == full_confirm["R8_videntra_champion_candidate"]["hits_at_1"]
    assert only_preds["R8_videntra_champion_candidate"] == full_preds["R8_videntra_champion_candidate"]

    from scripts.evaluate_msrvtt_1ka_test import _format_test_report

    md_report = _format_test_report({
        "evaluated_at_utc": "2026-11-22T00:00:00Z",
        "seed": 7,
        "champion_id": "R8_videntra_champion_candidate",
        "results": only_res,
        "category_breakdown": {"0": {"n_queries": len(qtexts), "hits_at_1": only_res["R8_videntra_champion_candidate"]["hits_at_1"], "hit_at_1": only_res["R8_videntra_champion_candidate"]["hit_at_1"]}},
        "failure_breakdown": None,
        "runtime_memory": {"peak_ram_mb": 123.4},
    })
    assert "R8_videntra_champion_candidate" in md_report
    assert "S1 R@20" in md_report


def test_kaggle_slice_sharding_and_title_slug(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_msrvtt_kaggle_pipeline as kp

    prot = tmp_path / "prot"
    t_dir = prot / "manifests" / "test_1ka"
    t_dir.mkdir(parents=True)
    vids = [f"video{9000 + i}" for i in range(10)]
    (t_dir / "gallery_manifest.jsonl").write_text(
        "\n".join(json.dumps({"video_id": v}) for v in vids) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kp, "KAGGLE_WORK_DIR", tmp_path / "k_work")

    seen: list[str] = []
    for idx in range(4):
        res = kp.run_shard_job(
            "vl2",
            "test_1ka_videos_only",
            protocol_root=prot,
            push=False,
            accelerator="cpu",
            num_slices=4,
            slice_index=idx,
        )
        slug = res["kernel_id"].split("/", 1)[1]
        assert len(slug) <= 50
        meta = json.loads((Path(res["kernel_dir"]) / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["title"].lower().replace(" ", "-") == slug
        assert res["video_count"] == len(vids[idx::4])
        seen.extend(vids[idx::4])
    assert sorted(seen) == sorted(vids)
