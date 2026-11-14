"""Unit tests for QAT-LateInteraction, Temporal Motion Delta, Multi-Scale Context, and Kaggle 3x3/E5 helpers."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
import torch

from scripts.kaggle_multimodal_offline_worker import (
    E5_EMBEDDING_DIM,
    build_e5_text_faiss_index,
    encode_e5_passages,
    extract_spatial_patches_3x3,
)
from videntra.indexing.augmented_vector import (
    compute_late_interaction_maxsim,
    compute_multiscale_temporal_context,
    compute_temporal_motion_delta,
)


def test_temporal_motion_delta_and_video_isolation():
    # Two videos interleaved in keyframe_map, plus a single-keyframe video
    vecs = np.zeros((5, 768), dtype=np.float32)
    vecs[0, 0] = 1.0  # vid_a t=0.0
    vecs[1, 1] = 1.0  # vid_b t=0.0
    vecs[2, 2] = 1.0  # vid_a t=2.0
    vecs[3, 3] = 1.0  # vid_c t=0.0 (single keyframe)
    vecs[4, 4] = 1.0  # vid_a t=4.0

    kf_map = {
        0: {"video_id": "vid_a", "pts_time": 0.0},
        1: {"video_id": "vid_b", "pts_time": 0.0},
        2: {"video_id": "vid_a", "pts_time": 2.0},
        3: {"video_id": "vid_c", "pts_time": 0.0},
        4: {"video_id": "vid_a", "pts_time": 4.0},
    }

    deltas = compute_temporal_motion_delta(vecs, kf_map)
    assert deltas.shape == (5, 768)
    assert deltas.dtype == np.float32

    # Single-keyframe videos must be zero (no cross-video bleeding)
    assert np.allclose(deltas[1], 0.0)
    assert np.allclose(deltas[3], 0.0)

    # Interior keyframe of vid_a (idx 2): (vecs[4] - vecs[0]) / sqrt(2)
    expected_mid = (vecs[4] - vecs[0]) / np.linalg.norm(vecs[4] - vecs[0])
    assert np.allclose(deltas[2], expected_mid, atol=1e-6)

    # Boundary keyframes of vid_a
    expected_first = (vecs[2] - vecs[0]) / np.linalg.norm(vecs[2] - vecs[0])
    expected_last = (vecs[4] - vecs[2]) / np.linalg.norm(vecs[4] - vecs[2])
    assert np.allclose(deltas[0], expected_first, atol=1e-6)
    assert np.allclose(deltas[4], expected_last, atol=1e-6)


def test_multiscale_temporal_context_gaussian_window():
    rng = np.random.default_rng(11)
    vecs = rng.standard_normal((4, 768)).astype(np.float32)
    kf_map = [
        {"video_id": "v1", "pts_time": 0.0},
        {"video_id": "v1", "pts_time": 1.0},
        {"video_id": "v1", "pts_time": 10.0},  # outside +-3.0s window from t=0,1
        {"video_id": "v2", "pts_time": 0.5},   # different video
    ]
    ctx = compute_multiscale_temporal_context(vecs, kf_map, window_sec=3.0, sigma_sec=1.5)
    assert ctx.shape == (4, 768)
    assert ctx.dtype == np.float32

    # All non-zero rows should be unit L2-norm
    norms = np.linalg.norm(ctx, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Keyframe 2 (t=10.0) has no neighbors in [7, 13], so its context equals its own unit vector
    v2_unit = vecs[2] / np.linalg.norm(vecs[2])
    assert np.allclose(ctx[2], v2_unit, atol=1e-5)


def test_late_interaction_maxsim_math_and_fp16():
    # Empty inputs return 0.0
    assert compute_late_interaction_maxsim(np.zeros((0, 768)), None, np.zeros((2, 9, 768))) == 0.0
    assert compute_late_interaction_maxsim(np.ones((2, 768)), [1.0, 2.0], np.zeros((0, 9, 768))) == 0.0

    q = np.zeros((2, 768), dtype=np.float32)
    q[0, 0] = 1.0
    q[1, 1] = 1.0
    idf = [1.0, 3.0]

    patches = np.zeros((2, 9, 768), dtype=np.float16)
    patches[0, 4, 0] = 1.0  # exact match for token 0 -> maxsim = 1.0
    patches[1, 2, 1] = 0.8
    patches[1, 2, 2] = 0.6  # unit vector [0, 0.8, 0.6] -> dot with q[1] = 0.8

    score = compute_late_interaction_maxsim(q, idf, patches)
    expected = (1.0 * 1.0 + 3.0 * 0.8) / (1.0 + 3.0)
    assert pytest.approx(score, abs=1e-3) == expected


def test_extract_spatial_patches_3x3_siglip_and_cls():
    # SigLIP 2: 14x14 = 196 patches, batch=2, dim=768
    hidden = torch.randn(2, 196, 768)
    patches = extract_spatial_patches_3x3(hidden)
    assert patches.shape == (2, 9, 768)
    assert patches.dtype == np.float16
    norms = np.linalg.norm(patches.astype(np.float32), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-2)

    # ViT with CLS token: 197 tokens
    hidden_cls = torch.randn(2, 197, 768)
    patches_cls = extract_spatial_patches_3x3(hidden_cls)
    assert patches_cls.shape == (2, 9, 768)
    assert patches_cls.dtype == np.float16


class _StubTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        bsz = len(texts)
        seq_len = 4
        return {
            "input_ids": torch.ones((bsz, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((bsz, seq_len), dtype=torch.long),
        }


class _StubE5Model:
    def __call__(self, **enc):
        bsz, seq_len = enc["input_ids"].shape
        torch.manual_seed(0)
        return SimpleNamespace(last_hidden_state=torch.randn(bsz, seq_len, E5_EMBEDDING_DIM))


def test_build_e5_text_faiss_index(tmp_path: Path):
    tok = _StubTokenizer()
    model = _StubE5Model()

    vecs = encode_e5_passages(["hello world", "passage: already prefixed"], tok, model)
    assert vecs.shape == (2, E5_EMBEDDING_DIM)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    asr_segs = [
        {"segment_id": "v1__asr_0", "video_id": "v1", "start_ms": 0, "end_ms": 2000, "raw_text": "hello world"},
        {"segment_id": "v1__asr_1", "video_id": "v1", "start_ms": 2000, "end_ms": 3000, "raw_text": "   "},
    ]
    ocr_obs = [
        {"observation_id": "v1__kf_0__ocr_0", "video_id": "v1", "timestamp_ms": 1500, "raw_text": "OPEN"},
    ]

    info = build_e5_text_faiss_index(
        asr_segments=asr_segs,
        ocr_observations=ocr_obs,
        out_dir=tmp_path,
        tokenizer=tok,
        model=model,
    )
    assert info["num_entries"] == 2
    assert info["ntotal"] == 2

    idx = faiss.read_index(str(tmp_path / "faiss_text_e5_384.bin"))
    assert idx.d == E5_EMBEDDING_DIM
    assert idx.ntotal == 2

    manifest = json.loads((tmp_path / "e5_text_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    assert manifest[0]["source_type"] == "asr"
    assert manifest[1]["source_type"] == "ocr"
