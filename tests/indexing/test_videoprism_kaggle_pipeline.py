"""CPU-safe tests for VideoPrism-LvT-B Kaggle bundle preparation and offline worker."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from scripts.kaggle_videoprism_worker import (
    VIDEOPRISM_CHECKPOINT_FILENAME,
    VIDEOPRISM_CHECKPOINT_SHA256,
    VIDEOPRISM_COMMIT,
    VIDEOPRISM_CONFIG,
    VIDEOPRISM_REPO,
    VIDEOPRISM_TOKENIZER_FILENAME,
    VIDEOPRISM_TOKENIZER_SHA256,
    disable_tf_gpu_before_jax,
    normalize_rows,
    prepare_clip_batch,
    resize_and_pad_288,
    run_videoprism_worker,
    verify_shard_outputs,
    verify_videoprism_pins,
)
from scripts.prepare_clip_vp_kaggle_bundle import (
    assert_no_gt_leakage,
    compute_video_recipe_from_ledger,
    prepare_kaggle_bundle,
    select_bin_center_frames,
    sha256_text,
)
from videntra.ingestion.decode import DecodedFrame
from videntra.timeline.ledger import write_timeline_ledger


def _write_synthetic_mp4(path: Path, num_frames: int = 8, width: int = 160, height: int = 90, fps: int = 4) -> None:
    """Create a tiny deterministic RGB MP4 in tmp_path using PyAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(num_frames):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = (i * 30) % 255
            rgb[:, :, 1] = 120
            rgb[:, :, 2] = 200
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def test_pinned_metadata_and_sha_verification(tmp_path: Path):
    assert VIDEOPRISM_REPO == "google-deepmind/videoprism"
    assert VIDEOPRISM_COMMIT == "4f831e332b80d6e77bdd43882206f23aba2a67a7"
    assert VIDEOPRISM_CONFIG == "videoprism_lvt_public_v1_base"
    assert VIDEOPRISM_CHECKPOINT_SHA256 == "dcc9bd31ba063221ca952e5a22370236018460faa09aa1a5649c6ccfaf071180"
    assert VIDEOPRISM_TOKENIZER_SHA256 == "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"

    assets = tmp_path / "pins"
    assets.mkdir()

    # Missing files fail closed
    with pytest.raises(FileNotFoundError):
        verify_videoprism_pins(assets)

    ckpt = assets / VIDEOPRISM_CHECKPOINT_FILENAME
    tok = assets / VIDEOPRISM_TOKENIZER_FILENAME
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    tok.write_bytes(b"fake-tokenizer-bytes")

    # Mismatched SHA fails closed
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        verify_videoprism_pins(assets)

    ckpt_sha = hashlib.sha256(b"fake-checkpoint-bytes").hexdigest()
    tok_sha = hashlib.sha256(b"fake-tokenizer-bytes").hexdigest()

    # Matching expected hashes succeeds
    res = verify_videoprism_pins(assets, checkpoint_sha256=ckpt_sha, tokenizer_sha256=tok_sha)
    assert res["checkpoint_sha256"] == ckpt_sha
    assert res["tokenizer_sha256"] == tok_sha

    # Check exact byte-size guard fails when size doesn't match 990527124
    with pytest.raises(ValueError, match="Checkpoint size mismatch"):
        verify_videoprism_pins(assets, checkpoint_sha256=ckpt_sha, tokenizer_sha256=tok_sha, check_size=True)


def test_inference_bundle_leakage_guard_and_missing_media(tmp_path: Path):
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    media_root = tmp_path / "media"
    vid1_path = media_root / "vid_001.mp4"
    _write_synthetic_mp4(vid1_path, num_frames=6)

    # Input queries/pools might contain extra evaluation fields in upstream files;
    # bundle export MUST strip them and export strictly inference-safe schemas.
    (pools_dir / "queries.jsonl").write_text(
        json.dumps({
            "query_id": "q1",
            "query_text": "a person slicing vegetables",
            "target_videos": ["vid_001"],
            "split": "dev",
        }) + "\n",
        encoding="utf-8",
    )
    (pools_dir / "a5_pool.jsonl").write_text(
        json.dumps({"query_id": "q1", "ranked_video_ids": ["vid_001", "vid_missing"]}) + "\n",
        encoding="utf-8",
    )
    (pools_dir / "clip_pool.jsonl").write_text(
        json.dumps({"query_id": "q1", "ranked_video_ids": ["vid_001", "vid_002"]}) + "\n",
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({
            "records": [
                {
                    "video_id": "vid_001",
                    "relative_path": "vid_001.mp4",
                    "timeline_revision": "tl_rev_1",
                    "clip_start_sec": 0.0,
                    "clip_end_sec": 1.5,
                    "basis": "full_clip",
                }
            ]
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "bundle"
    manifest = prepare_kaggle_bundle(
        queries_path=pools_dir / "queries.jsonl",
        pool_paths=[pools_dir / "a5_pool.jsonl", pools_dir / "clip_pool.jsonl"],
        output_dir=out_dir,
        video_registry_path=registry_path,
        media_root=media_root,
    )

    assert manifest["inference_only"] is True
    assert manifest["num_queries"] == 1
    assert manifest["num_candidate_videos"] == 3
    assert set(manifest["missing_media_video_ids"]) == {"vid_missing", "vid_002"}

    q_row = json.loads((out_dir / "queries.jsonl").read_text(encoding="utf-8").strip())
    assert set(q_row.keys()) == {"query_id", "query_text", "query_text_sha256"}
    assert q_row["query_text_sha256"] == sha256_text("a person slicing vegetables")

    c_row = json.loads((out_dir / "candidates.jsonl").read_text(encoding="utf-8").strip())
    assert c_row == {"query_id": "q1", "candidate_video_ids": ["vid_001", "vid_missing", "vid_002"]}

    m_rows = [json.loads(line) for line in (out_dir / "media_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(m_rows) == 3
    assert set(m_rows[0].keys()) == {
        "video_id",
        "relative_path",
        "media_sha256",
        "timeline_revision",
        "clip_start_sec",
        "clip_end_sec",
        "basis",
    }

    # Guard itself must raise on any GT field
    with pytest.raises(ValueError, match="Inference leakage guard"):
        assert_no_gt_leakage([{"query_id": "q1", "target_windows": [[0.0, 5.0]]}])


def test_16_frame_bin_center_sampling_tie_break_and_short_clip_repeat(tmp_path: Path):
    # 1. Exact tie-breaking: smaller ordinal wins
    # Interval [0, 16) -> bin 0 center = 0.5, frames at 0.0 and 1.0 are equidistant (dist=0.5)
    tie_frames = [(5, 1.0), (2, 0.0)]
    res_tie = select_bin_center_frames(tie_frames, clip_start_sec=0.0, clip_end_sec=16.0, num_frames=16)
    assert res_tie["selected_ordinals"][0] == 2  # smaller ordinal wins tie at bin 0 (t=0.5)
    assert res_tie["unique_frames"] == 2
    assert len(res_tie["selected_ordinals"]) == 16

    # 2. Short clip (<16 frames) deterministic repeat + missing PTS mapping count
    short_frames = [(0, 0.0), (1, None), (2, 0.5), (3, 1.0)]
    res_short_a = select_bin_center_frames(short_frames, clip_start_sec=0.0, clip_end_sec=1.5, num_frames=16)
    res_short_b = select_bin_center_frames(short_frames, clip_start_sec=0.0, clip_end_sec=1.5, num_frames=16)
    assert res_short_a["selected_ordinals"] == res_short_b["selected_ordinals"]
    assert res_short_a["recipe_hash"] == res_short_b["recipe_hash"]
    assert len(res_short_a["selected_ordinals"]) == 16
    assert res_short_a["unique_frames"] <= 3
    assert res_short_a["missing_mapping"] == 1

    # 3. Integration with Parquet timeline ledger
    ledger_path = tmp_path / "frames.parquet"
    decoded = [
        DecodedFrame(
            ordinal=i,
            pts=i * 250,
            dts=i * 250,
            time_base_num=1,
            time_base_den=1000,
            origin=Fraction(0, 1),
            is_keyframe=(i == 0),
        )
        for i in range(4)
    ]
    write_timeline_ledger(ledger_path, decoded, video_id="vid_short", timeline_revision="tl_1")
    ledger_recipe = compute_video_recipe_from_ledger(ledger_path, clip_start_sec=0.0, clip_end_sec=1.0)
    assert ledger_recipe["status"] == "ok"
    assert len(ledger_recipe["selected_ordinals"]) == 16
    assert ledger_recipe["unique_frames"] == 4


def test_aspect_preserving_288_black_pad_and_clip_batch():
    # Landscape 640x360 (16:9) -> scaled to 288x162, padded top/bottom (63px each)
    landscape = np.full((360, 640, 3), 255, dtype=np.uint8)
    out_land = resize_and_pad_288(landscape)
    assert out_land.shape == (288, 288, 3)
    assert out_land.dtype == np.float32
    assert out_land.min() >= 0.0 and out_land.max() <= 1.0
    assert np.all(out_land[:63, :, :] == 0.0)
    assert np.all(out_land[225:, :, :] == 0.0)
    assert np.allclose(out_land[144, 144, :], 1.0)

    # Portrait 360x640 -> scaled to 162x288, padded left/right (63px each)
    portrait = np.full((640, 360, 3), 255, dtype=np.uint8)
    out_port = resize_and_pad_288(portrait)
    assert out_port.shape == (288, 288, 3)
    assert np.all(out_port[:, :63, :] == 0.0)
    assert np.all(out_port[:, 225:, :] == 0.0)
    assert np.allclose(out_port[144, 144, :], 1.0)

    batch = prepare_clip_batch([[landscape] * 16, [portrait] * 16])
    assert batch.shape == (2, 16, 288, 288, 3)
    assert batch.dtype == np.float32


def test_finite_and_norm_guards_and_tf_gpu_disable(monkeypatch):
    # Valid normalization
    valid = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normed = normalize_rows(valid)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-6)

    # NaN / Inf / zero-norm / wrong shape rejected
    with pytest.raises(ValueError, match="non-finite"):
        normalize_rows(np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_rows(np.array([[np.inf, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="zero or near-zero norm"):
        normalize_rows(np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="Expected"):
        normalize_rows(np.array([1.0, 2.0], dtype=np.float32))

    # TF GPU disable helper
    calls: list[tuple[list[Any], str]] = []
    fake_tf = types.SimpleNamespace(
        config=types.SimpleNamespace(
            set_visible_devices=lambda devices, dev_type: calls.append((devices, dev_type))
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    assert disable_tf_gpu_before_jax() is True
    assert calls == [([], "GPU")]


def test_dry_run_shard_roundtrip_parity_and_resume(tmp_path: Path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    media_dir = bundle_dir / "media"
    v_path = media_dir / "vid_a.mp4"
    _write_synthetic_mp4(v_path, num_frames=8, width=160, height=90, fps=4)

    q_text = "a colorful synthetic video clip"
    (bundle_dir / "queries.jsonl").write_text(
        json.dumps({
            "query_id": "q_001",
            "query_text": q_text,
            "query_text_sha256": sha256_text(q_text),
        }) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "media_manifest.jsonl").write_text(
        json.dumps({
            "video_id": "vid_a",
            "relative_path": "media/vid_a.mp4",
            "media_sha256": hashlib.sha256(v_path.read_bytes()).hexdigest(),
            "timeline_revision": "tl_1",
            "clip_start_sec": 0.0,
            "clip_end_sec": 2.0,
            "basis": "full_clip",
        }) + "\n",
        encoding="utf-8",
    )

    out_1 = tmp_path / "shard_1"
    out_2 = tmp_path / "shard_2"

    receipt_1 = run_videoprism_worker(bundle_dir=bundle_dir, output_dir=out_1, dry_run=True)
    receipt_2 = run_videoprism_worker(bundle_dir=bundle_dir, output_dir=out_2, dry_run=True)

    assert receipt_1["status"] == "SUCCESS"
    assert receipt_1["video_count"] == 1
    assert receipt_1["query_count"] == 1
    assert receipt_1["embedding_dim"] == 768
    assert receipt_1["errors"] == []

    # Deterministic parity across runs
    v1 = np.load(str(out_1 / "video_vectors.npy"))
    v2 = np.load(str(out_2 / "video_vectors.npy"))
    q1 = np.load(str(out_1 / "query_vectors.npy"))
    q2 = np.load(str(out_2 / "query_vectors.npy"))
    assert np.array_equal(v1, v2)
    assert np.array_equal(q1, q2)

    verified = verify_shard_outputs(out_1, expected_dim=768)
    assert verified["video_count"] == 1
    assert verified["query_count"] == 1

    # Resume on completed shard succeeds without re-encoding
    resumed = run_videoprism_worker(bundle_dir=bundle_dir, output_dir=out_1, dry_run=True, resume=True)
    assert resumed["resumed"] is True
    assert resumed["files"] == receipt_1["files"]

    # Non-resume into existing output dir fails closed
    with pytest.raises(FileExistsError):
        run_videoprism_worker(bundle_dir=bundle_dir, output_dir=out_1, dry_run=True, resume=False)

    # Corrupted shard fails closed on resume
    (out_1 / "query_vectors.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        run_videoprism_worker(bundle_dir=bundle_dir, output_dir=out_1, dry_run=True, resume=True)
