"""Tests for P0 post-VLM rank propagation and P1 storyboard/pair-cache/OOM upgrades."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from videntra.encoders.siglip2 import MockSigLIP2Encoder
from videntra.ranking.vlm_reranker import (
    Qwen25VLReranker,
    VLMFusionConfig,
    build_chronological_storyboard_with_meta,
)
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
from videntra.storage.database import OfflineMetadataStore


def _make_synthetic_mp4(path: Path, *, num_frames: int = 24, fps: int = 2) -> None:
    """Create a deterministic synthetic MP4 with clean presentation-order PTS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for i in range(num_frames):
            arr = np.full((48, 64, 3), fill_value=(i * 10) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def _setup_mini_pipeline_db_and_faiss(tmp_path: Path):
    import faiss

    db_path = tmp_path / "mini.db"
    faiss_path = tmp_path / "mini.faiss"
    enc = MockSigLIP2Encoder()

    vA = np.zeros(768, dtype=np.float32)
    vA[0] = 1.0
    vB = np.zeros(768, dtype=np.float32)
    vB[0] = 0.96
    vB[1] = 0.28
    vC = np.zeros(768, dtype=np.float32)
    vC[2] = 1.0

    q_vec = enc.encode_texts(["a chef cutting vegetables in kitchen"])[0]
    vA = (0.99 * q_vec + 0.01 * vA).astype(np.float32)
    vA /= np.linalg.norm(vA)
    vB = (0.97 * q_vec + 0.15 * vB).astype(np.float32)
    vB /= np.linalg.norm(vB)
    vC = (0.50 * q_vec + 0.50 * vC).astype(np.float32)
    vC /= np.linalg.norm(vC)

    embs_768 = np.ascontiguousarray(np.vstack([vA, vB, vC]).astype(np.float32))
    idx = faiss.IndexFlatIP(768)
    idx.add(embs_768)
    faiss.write_index(idx, str(faiss_path))

    embs_864 = np.zeros((3, 864), dtype=np.float32)
    embs_864[:, :768] = embs_768
    embs_864 = np.ascontiguousarray(embs_864)
    aug_idx = faiss.IndexFlatIP(864)
    aug_idx.add(embs_864)

    kf_map = {}
    vid_kfs = {}
    vid_durs = {}
    store = OfflineMetadataStore(db_path)
    with store.transaction() as conn:
        for fid, (vid, pts_ms) in enumerate([("vid_A", 1000), ("vid_B", 1000), ("vid_C", 1000)]):
            mp4 = tmp_path / f"{vid}.mp4"
            _make_synthetic_mp4(mp4, num_frames=10, fps=2)
            conn.execute(
                """
                INSERT INTO Videos(
                    video_id, dataset, file_path, duration_ms, fps,
                    time_base_num, time_base_den, width, height, total_frames, has_audio, status, created_at, updated_at
                ) VALUES (?, 'msrvtt', ?, 5000, 2.0, 1, 1000, 64, 48, 10, 0, 'ingested', '2026-01-01', '2026-01-01');
                """,
                (vid, str(mp4)),
            )
            kid = f"{vid}:kf_0"
            conn.execute(
                """
                INSERT INTO Keyframes(
                    keyframe_id, video_id, keyframe_index, source_frame_index, pts, time_base,
                    timestamp_ms, keyframe_path, faiss_id, phash, sampling_reason, created_at
                ) VALUES (?, ?, 0, 0, ?, '1/1000', ?, '', ?, '0', 'test', '2026-01-01');
                """,
                (kid, vid, pts_ms, pts_ms, fid),
            )
            conn.execute(
                """
                INSERT INTO KeyframeMap(faiss_id, keyframe_id, video_id, pts_time, source_frame_index, keyframe_path)
                VALUES (?, ?, ?, ?, 0, '');
                """,
                (fid, kid, vid, pts_ms / 1000.0),
            )
            kf_rec = {
                "faiss_id": fid,
                "keyframe_id": kid,
                "video_id": vid,
                "pts_time": pts_ms / 1000.0,
                "source_frame_index": 0,
                "keyframe_path": "",
            }
            kf_map[fid] = kf_rec
            vid_kfs[vid] = [kf_rec]
            vid_durs[vid] = 5.0
    return db_path, idx, aug_idx, embs_768, embs_864, kf_map, vid_kfs, vid_durs, enc


def test_p0_vlm_rerank_updates_ranked_moments_and_ranked_videos(tmp_path: Path) -> None:
    db_path, idx, aug_idx, embs_768, embs_864, kf_map, vid_kfs, vid_durs, enc = (
        _setup_mini_pipeline_db_and_faiss(tmp_path)
    )

    cache_file = tmp_path / "pair_cache.jsonl"
    from videntra.ranking.vlm_reranker import _sha256_text

    query = "a chef cutting vegetables in kitchen"
    qsha = _sha256_text(query)
    rows = [
        {"query_text_sha256": qsha, "video_id": "vid_A", "score": -2.5},
        {"query_text_sha256": qsha, "video_id": "vid_B", "score": 3.5},
        {"query_text_sha256": qsha, "video_id": "vid_C", "score": -1.0},
    ]
    cache_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    reranker = Qwen25VLReranker(
        enabled=True,
        score_cache_path=cache_file,
        fusion_config=VLMFusionConfig(margin_gate=2.0, w_vlm=1.5, top_k_rerank=3, mode="margin_raw"),
    )
    with UnifiedMultimodalSearchPipeline(
        db_path=db_path,
        vlm_reranker=reranker,
        use_disk_cache=False,
        lazy_load=True,
        rules_only=True,
        enable_videoprism=False,
        ltr_model_path=None,
    ) as pipe:
        pipe.all_embeddings = embs_768
        pipe.augmented_embeddings = embs_864
        pipe.index = idx
        pipe.aug_index = aug_idx
        pipe.keyframe_map = kf_map
        pipe.video_keyframes = vid_kfs
        pipe.video_durations = vid_durs
        pipe.encode_texts = lambda texts, **kw: enc.encode_texts(texts)  # type: ignore[method-assign]
        res = pipe.search_with_metadata(snapshot_id="test_snap", query=query, top_k=3)

    assert res["diagnostics"]["vlm_rerank_stats"]["applied"] is True
    assert res["ranked_moments"][0].video_id == "vid_B"
    assert res["ranked_videos"][0].video_id == "vid_B"
    assert res["diagnostics"]["ranked_video_ids"][0] == "vid_B"
    assert res["ranked_videos"][0].ranking_score > res["ranked_videos"][1].ranking_score


def test_p1_pair_cache_partial_hit_and_provenance_invalidation(tmp_path: Path) -> None:
    vid_a = tmp_path / "vid_A.mp4"
    vid_b = tmp_path / "vid_B.mp4"
    _make_synthetic_mp4(vid_a, num_frames=16, fps=2)
    _make_synthetic_mp4(vid_b, num_frames=16, fps=2)

    cache_file = tmp_path / "pairs.jsonl"
    _, sb_meta_a = build_chronological_storyboard_with_meta(vid_a)
    query = "a dog catches a red ball"

    rr = Qwen25VLReranker(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_revision="rev_1",
        processor_revision="proc_1",
        enabled=True,
        score_cache_path=cache_file,
        persist_pair_cache=True,
        batch_size=1,
    )
    from videntra.ranking.vlm_reranker import _sha256_text

    pre_row = {
        "query_id": "q1",
        "query_text_sha256": _sha256_text(query),
        "video_id": "vid_A",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "rev_1",
        "processor_revision": "proc_1",
        "prompt_sha256": rr.prompt_sha256,
        "storyboard_sha256": sb_meta_a["storyboard_sha256"],
        "score": 1.25,
    }
    cache_file.write_text(json.dumps(pre_row) + "\n", encoding="utf-8")
    rr._load_score_cache(cache_file)

    scored_calls: list[int] = []
    rr._processor = SimpleNamespace(
        apply_chat_template=lambda *a, **kw: "prompt",
    )
    rr._model = object()
    rr._yes_ids = [1]
    rr._no_ids = [2]

    def _fake_score(sub_imgs, sub_prompts):
        scored_calls.append(len(sub_imgs))
        return [2.75 for _ in sub_imgs]

    rr._score_images = _fake_score  # type: ignore[method-assign]

    res1 = rr.score_query_candidates(
        query,
        ["vid_A", "vid_B"],
        {"vid_A": vid_a, "vid_B": vid_b},
        query_id="q1",
    )
    assert res1 == {"vid_A": 1.25, "vid_B": 2.75}
    assert scored_calls == [1]  # Only vid_B ran inference!
    assert rr.last_score_stats["cache_hits"] == 1
    assert rr.last_score_stats["live_scored"] == 1

    # Rerun on exact same shard: 0 additional inference calls
    res2 = rr.score_query_candidates(
        query,
        ["vid_A", "vid_B"],
        {"vid_A": vid_a, "vid_B": vid_b},
        query_id="q1",
    )
    assert res2 == {"vid_A": 1.25, "vid_B": 2.75}
    assert scored_calls == [1]
    assert rr.last_score_stats["cache_hits"] == 2
    assert rr.last_score_stats["live_scored"] == 0

    # Provenance invalidation on revision change reloads cache & re-scores
    rr_new_rev = Qwen25VLReranker(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_revision="rev_2",
        processor_revision="proc_1",
        enabled=True,
        score_cache_path=cache_file,
        batch_size=1,
    )
    rr_new_rev._processor = rr._processor
    rr_new_rev._model = object()
    rr_new_rev._yes_ids = [1]
    rr_new_rev._no_ids = [2]
    rr_new_rev._score_images = lambda imgs, prompts: [0.5 for _ in imgs]  # type: ignore[method-assign]
    res_rev2 = rr_new_rev.score_query_candidates(
        query,
        ["vid_A"],
        {"vid_A": vid_a},
        query_id="q1",
    )
    assert res_rev2 == {"vid_A": 0.5}
    assert rr_new_rev.last_score_stats["cache_hits"] == 0
    assert rr_new_rev.last_score_stats["live_scored"] == 1


def test_p1_streaming_storyboard_low_ram_clip_slice_timestamps_and_oom_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp4 = tmp_path / "long_clip.mp4"
    _make_synthetic_mp4(mp4, num_frames=24, fps=2)  # 12 seconds total

    # Count full RGB materializations: must materialize ONLY selected target ordinals (<= 8), NOT 24
    orig_to_ndarray = av.VideoFrame.to_ndarray
    decode_rgb_calls = 0

    def _counting_to_ndarray(self, *args, **kwargs):
        nonlocal decode_rgb_calls
        decode_rgb_calls += 1
        return orig_to_ndarray(self, *args, **kwargs)

    monkeypatch.setattr(av.VideoFrame, "to_ndarray", _counting_to_ndarray)
    jb_slice, meta_slice = build_chronological_storyboard_with_meta(
        mp4,
        num_frames=8,
        clip_start_sec=6.0,
        clip_end_sec=11.0,
        annotate_timestamps=False,
    )
    assert len(jb_slice) > 100
    assert decode_rgb_calls <= 8
    assert all(6.0 <= pts < 11.0 for pts in meta_slice["frame_pts_sec"])

    _, meta_annot = build_chronological_storyboard_with_meta(
        mp4,
        num_frames=8,
        clip_start_sec=6.0,
        clip_end_sec=11.0,
        annotate_timestamps=True,
    )
    assert meta_annot["storyboard_sha256"] != meta_slice["storyboard_sha256"]
    assert meta_annot["annotate_timestamps"] is True

    # Test OOM at batch_size=1: shrink once to (160, 120), then graceful fallback if still OOM
    rr = Qwen25VLReranker(enabled=True, batch_size=1)
    rr._processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
    rr._model = object()
    rr._yes_ids = [1]
    rr._no_ids = [2]

    observed_widths: list[int] = []

    def _oom_on_large(imgs, prompts):
        observed_widths.append(imgs[0].width)
        if imgs[0].width >= 800:  # 4 * 224 = 896
            raise RuntimeError("CUDA out of memory")
        return [1.9]

    rr._score_images = _oom_on_large  # type: ignore[method-assign]
    out_shrunk = rr.score_query_candidates(
        "test oom shrink", ["v1"], {"v1": mp4}, query_id="q_oom_1"
    )
    assert observed_widths == [4 * 224, 4 * 160]
    assert out_shrunk == {"v1": 1.9}
    assert rr.last_score_stats["oom_shrunk"] == 1
    assert rr.last_score_stats["oom_fallback"] == 0

    def _always_oom(imgs, prompts):
        raise RuntimeError("CUDA out of memory")

    rr._score_images = _always_oom  # type: ignore[method-assign]
    order, rerank_st = rr.rerank_video_scores(
        "test oom fallback",
        [("v1", 2.0), ("v2", 1.9)],
        {"v1": mp4, "v2": mp4},
        query_id="q_oom_2",
    )
    assert order == ["v1", "v2"]
    assert rerank_st["applied"] is False
    assert rerank_st["reason"] == "oom_fallback"
    assert rerank_st["oom_fallback"] == 2
