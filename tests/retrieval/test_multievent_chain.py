"""Tests for multi-sentence >=4-event parsing, Semi-Markov DP spans & Directional Order Margin."""
from __future__ import annotations

import numpy as np

from videntra.query.rules import decompose_query_clauses, decompose_query_event_chain
from videntra.retrieval.alignment import localize_event_chain_dp, score_videocomp_order
from videntra.retrieval.visual import RankedMoment


def test_parser_handles_four_plus_sentences_simultaneous_and_reverse_after() -> None:
    q5 = (
        "A person opens the cardboard box. "
        "Then he grabs a white mug from inside. "
        "Meanwhile a dog sits near the table. "
        "Next he pours hot water into the mug. "
        "Finally he stirs the coffee slowly."
    )
    chain = decompose_query_event_chain(q5)
    assert len(chain) == 5
    assert [c.clause_id for c in chain] == ["e0", "e1", "e2", "e3", "e4"]
    assert chain[0].relation_to_prev == "start"
    assert chain[1].relation_to_prev == "before"
    assert chain[2].relation_to_prev == "simultaneous"
    assert chain[3].relation_to_prev == "before"
    assert chain[4].relation_to_prev == "before"

    clauses = decompose_query_clauses(q5)
    assert clauses[0] == q5
    assert len(clauses) == 6  # original query + all 5 events (no [:3] truncation!)

    q_after = "The chef serves the steak after grilling the meat on fire"
    chain_after = decompose_query_event_chain(q_after)
    assert len(chain_after) == 2
    assert "grilling the meat" in chain_after[0].text
    assert "serves the steak" in chain_after[1].text
    assert chain_after[1].relation_to_prev == "before"


def test_localize_event_chain_dp_four_events_simultaneous_and_missing() -> None:
    ts = [float(t) for t in range(0, 42, 2)]  # 21 frames: 0, 2, ..., 40
    ords = [i * 10 for i in range(len(ts))]  # non-contiguous presentation-order ordinals

    def _gauss(center: float, peak: float = 0.9, width: float = 2.5) -> np.ndarray:
        arr = np.asarray([0.05 + peak * np.exp(-0.5 * ((t - center) / width) ** 2) for t in ts], dtype=np.float32)
        return arr

    # e0 at 4s, e1 & e2 SIMULTANEOUS at 16s & 18s, e3 at 32s
    curves = np.vstack([
        _gauss(4.0),
        _gauss(16.0),
        _gauss(18.0),
        _gauss(32.0),
    ])
    res = localize_event_chain_dp(
        timestamps_sec=ts,
        event_vis_curves=curves,
        clause_ids=["e0", "e1", "e2", "e3"],
        clause_texts=["open box", "take cup", "dog barks", "pour water"],
        relations_to_prev=["start", "before", "simultaneous", "before"],
        frame_ordinals=ords,
        v_dur_sec=42.0,
    )
    assert res is not None
    assert len(res.spans) == 4
    assert res.all_supported is True

    s0, s1, s2, s3 = res.spans
    # Strict before order preserved cleanly
    assert s0.end_sec <= s1.start_sec + 1e-6
    assert s2.end_sec <= s3.start_sec + 1e-6
    # Simultaneous pair permitted to overlap
    overlap_12 = min(s1.end_sec, s2.end_sec) - max(s1.start_sec, s2.start_sec)
    assert overlap_12 > 0.0
    # 0-based frame ordinals mapped cleanly
    assert s0.start_frame_index <= s0.end_frame_index
    assert s3.start_frame_index > s0.end_frame_index

    # Missing evidence event curve (max=0.01 < 0.05) gets status="missing"
    missing_curves = curves.copy()
    missing_curves[2] = 0.01
    res_miss = localize_event_chain_dp(
        timestamps_sec=ts,
        event_vis_curves=missing_curves,
        clause_ids=["e0", "e1", "e2", "e3"],
        relations_to_prev=["start", "before", "before", "before"],
        v_dur_sec=42.0,
        missing_evidence_threshold=0.05,
    )
    assert res_miss is not None
    assert res_miss.all_supported is False
    assert res_miss.spans[2].status == "missing"


def test_directional_order_margin_discriminates_videocomp_reordering() -> None:
    ts = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    # True sequence in video: A peaks at 4s, B peaks at 12s, C peaks at 16s
    curve_a = np.array([0.05, 0.95, 0.15, 0.05, 0.05, 0.05], dtype=np.float32)
    curve_b = np.array([0.05, 0.05, 0.20, 0.92, 0.10, 0.05], dtype=np.float32)
    curve_c = np.array([0.05, 0.05, 0.05, 0.15, 0.94, 0.20], dtype=np.float32)

    pos_scores = score_videocomp_order(
        ts,
        np.vstack([curve_a, curve_b, curve_c]),
        relations_to_prev=["start", "before", "before"],
        beta_order=1.0,
    )
    # Disrupted temporal reordering swaps A <-> C
    neg_scores = score_videocomp_order(
        ts,
        np.vstack([curve_c, curve_b, curve_a]),
        relations_to_prev=["start", "before", "before"],
        beta_order=1.0,
    )
    # Static max pooling cannot tell them apart!
    assert abs(pos_scores["event_max_mean"] - neg_scores["event_max_mean"]) < 1e-6
    assert pos_scores["directional_order_margin"] > 0.0
    assert neg_scores["directional_order_margin"] < 0.0
    assert pos_scores["combined_score"] > neg_scores["combined_score"]


def test_ranked_moment_serializes_event_spans_in_result_dict() -> None:
    sp = {
        "clause_id": "e0",
        "start_sec": 1.0,
        "end_sec": 4.0,
        "status": "supported",
    }
    mom_with = RankedMoment(
        moment_id="v1_1_4",
        video_id="v1",
        moment_start_seconds=1.0,
        moment_end_seconds=4.0,
        start_original_frame_index=2,
        end_original_frame_index=8,
        representative_original_frame_index=5,
        representative_timestamp_seconds=2.5,
        ranking_score=0.88,
        event_spans=(sp,),
    )
    d_with = mom_with.to_result_dict()
    assert d_with["event_spans"] == [sp]

    mom_plain = RankedMoment(
        moment_id="v1_1_4",
        video_id="v1",
        moment_start_seconds=1.0,
        moment_end_seconds=4.0,
        start_original_frame_index=2,
        end_original_frame_index=8,
        representative_original_frame_index=5,
        representative_timestamp_seconds=2.5,
        ranking_score=0.88,
    )
    assert "event_spans" not in mom_plain.to_result_dict()


def test_unified_pipeline_end_to_end_four_event_spans() -> None:
    import faiss
    from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline

    q4 = (
        "A person opens the wooden box. "
        "Then he lifts a ceramic cup. "
        "Next he pours steaming coffee. "
        "Finally he drinks from the cup."
    )
    chain = decompose_query_event_chain(q4)
    assert len(chain) == 4

    # Construct orthogonal synthetic 768d vectors matching each of the 4 clauses along timestamps 2s, 8s, 14s, 20s
    rng = np.random.RandomState(7)
    q_mat = np.zeros((4, 768), dtype=np.float32)
    for idx in range(4):
        q_mat[idx, idx * 4 : (idx + 1) * 4] = 0.5
        q_mat[idx] /= np.linalg.norm(q_mat[idx])

    raw_768 = np.zeros((4, 768), dtype=np.float32)
    raw_768[:] = q_mat
    aug_864 = np.zeros((4, 864), dtype=np.float32)
    aug_864[:, :768] = raw_768

    idx_768 = faiss.IndexFlatIP(768)
    idx_768.add(raw_768)
    idx_864 = faiss.IndexFlatIP(864)
    idx_864.add(aug_864)

    ts = [2.0, 8.0, 14.0, 20.0]
    kf_map = {
        i: {
            "faiss_id": i,
            "keyframe_id": f"vid_4ev:kf_{i}",
            "video_id": "vid_4ev",
            "pts_time": ts[i],
            "source_frame_index": i * 10,
            "keyframe_path": "",
        }
        for i in range(4)
    }

    with UnifiedMultimodalSearchPipeline(
        lazy_load=True,
        use_disk_cache=False,
        rules_only=True,
        enable_videoprism=False,
        ltr_model_path=None,
    ) as pipe:
        pipe.all_embeddings = raw_768
        pipe.augmented_embeddings = aug_864
        pipe.index = idx_768
        pipe.aug_index = idx_864
        pipe.keyframe_map = kf_map
        pipe.video_keyframes = {"vid_4ev": [kf_map[i] for i in range(4)]}
        pipe.video_durations = {"vid_4ev": 24.0}

        def _fake_encode(texts, **kw):
            out = []
            for t in texts:
                t_low = str(t).lower()
                if "opens" in t_low and "drinks" not in t_low:
                    out.append(q_mat[0])
                elif "lifts" in t_low:
                    out.append(q_mat[1])
                elif "pours" in t_low:
                    out.append(q_mat[2])
                elif "drinks" in t_low:
                    out.append(q_mat[3])
                else:
                    v = np.mean(q_mat, axis=0)
                    out.append(v / np.linalg.norm(v))
            return np.asarray(out, dtype=np.float32)

        pipe.encode_texts = _fake_encode  # type: ignore[method-assign]
        res = pipe.search_with_metadata(snapshot_id="snap_4ev", query=q4, top_k=2)

    assert len(res["ranked_moments"]) >= 1
    top_m = res["ranked_moments"][0]
    assert top_m.video_id == "vid_4ev"
    assert len(top_m.event_spans) == 4
    assert [sp["clause_id"] for sp in top_m.event_spans] == ["e0", "e1", "e2", "e3"]
    for i in range(3):
        assert top_m.event_spans[i]["end_sec"] <= top_m.event_spans[i + 1]["start_sec"] + 1e-6
