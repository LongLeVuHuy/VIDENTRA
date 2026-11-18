"""3-Tier Keyframe Extraction Strategy Benchmark (Synthetic Pathologies, Real 30-Video Corpus, Validation SigLIP2 Retrieval).

Strictly isolated from test_queries/; Tier 3 reads only validation/ queries matched to local data/raw/ media.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.config.ingestion import SamplingConfig, VALID_SAMPLING_STRATEGIES
from videntra.encoders.siglip2 import SigLIP2Config, SigLIP2Encoder
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import DecodedFrame, decode_frames
from videntra.ingestion.probe import probe_media
from videntra.sampling.compare import (
    QualityBenchmarkMetrics,
    compare_sampling_strategies,
    evaluate_keyframe_quality,
    generate_uniform_baseline,
    precompute_eval_signals,
)
from videntra.sampling.hybrid import (
    compute_hist_distance,
    compute_luma_difference,
    resize_lowres,
    sample_frames_hybrid,
)


def _configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _make_decoded_frame(ordinal: int, t_sec: float, pixels: np.ndarray, *, fps_den: int = 1000) -> DecodedFrame:
    pts = int(round(t_sec * fps_den))
    return DecodedFrame(
        ordinal=ordinal,
        pts=pts,
        dts=pts,
        time_base_num=1,
        time_base_den=fps_den,
        origin=Fraction(0),
        is_keyframe=(ordinal == 0),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        pixels=pixels,
    )


def _textured_scene(seed: int, h: int = 96, w: int = 128, *, tint: tuple[int, int, int] = (20, 40, 60)) -> np.ndarray:
    y_grad = np.linspace(35, 200, h, dtype=np.float32)[:, None]
    x_grad = np.linspace(30, 185, w, dtype=np.float32)[None, :]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip((y_grad + seed * 7) % 210 + 20, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip((x_grad + seed * 13) % 210 + 20, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip((tint[2] + (y_grad * 0.3)) % 220 + 20, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (18, 16), (56, 56), tint, -1)
    cv2.rectangle(img, (22, 20), (52, 52), (245, 245, 245), 2)
    cv2.line(img, (68, 16), (118, 82), (40, 230, 220), 3)
    cv2.putText(img, "KEY", (66, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
    return img


def build_synthetic_pathology_streams(tmp_fixture_dir: Path) -> list[dict[str, Any]]:
    """Build 5 M1 container fixtures + 4 synthetic pathology streams."""
    if not (tmp_fixture_dir / "receipt.json").exists():
        build_fixtures(tmp_fixture_dir, include_extended=False)

    streams: list[dict[str, Any]] = []
    for name in ("cfr.mkv", "vfr.mkv", "nonzero_pts.mkv", "repeated_pts.mkv", "b_frames.mkv"):
        p = tmp_fixture_dir / name
        if p.exists():
            pr = probe_media(p)
            frames = list(decode_frames(p, pr, include_pixels=True))
            streams.append({"name": f"m1_{p.stem}", "frames": frames, "kind": "m1_fixture", "ref_cuts": []})

    # 1. Motion blur at exact 2.0s and 4.0s trigger points, sharp 2 frames later (2.13s, 4.13s)
    blur_frames: list[DecodedFrame] = []
    for i in range(90):  # 6.0s at 15fps
        t = i / 15.0
        base = _textured_scene(10 + (i // 30), tint=(30 + (i // 30) * 60, 80, 140))
        if abs(t - 2.0) <= 0.08 or abs(t - 4.0) <= 0.08:
            img = cv2.GaussianBlur(base, (9, 9), 3.0)
        else:
            img = base
        blur_frames.append(_make_decoded_frame(i, t, img))
    streams.append({"name": "motion_blur_whip_pan", "frames": blur_frames, "kind": "pathology", "ref_cuts": [30, 60]})

    # 2. Fade-in (0.0..0.4s) & fade-out tail (5.3..5.9s)
    fade_frames: list[DecodedFrame] = []
    for i in range(60):  # 6.0s at 10fps
        t = i / 10.0
        if i <= 4 or i >= 53:
            img = np.zeros((96, 128, 3), dtype=np.uint8)
        else:
            img = _textured_scene(200 + i // 15, tint=(60, 170, 90))
        fade_frames.append(_make_decoded_frame(i, t, img))
    streams.append({"name": "fade_in_fade_out", "frames": fade_frames, "kind": "pathology", "ref_cuts": []})

    # 3. Gradual dissolve / slow pan (t=1.0..2.2s adjacent diff < 0.02, cumulative drift > 0.6)
    scene_a = _textured_scene(301, tint=(200, 30, 30)).astype(np.float32)
    scene_b = _textured_scene(909, tint=(20, 40, 230)).astype(np.float32)
    dissolve_frames: list[DecodedFrame] = []
    for i in range(90):  # 6.0s at 15fps
        if i < 12:
            alpha = 0.0
        elif i <= 42:
            alpha = (i - 12) / 30.0
        else:
            alpha = 1.0
        img = np.clip((1.0 - alpha) * scene_a + alpha * scene_b, 0, 255).astype(np.uint8)
        dissolve_frames.append(_make_decoded_frame(i, i / 15.0, img))
    streams.append({"name": "gradual_dissolve_pan", "frames": dissolve_frames, "kind": "pathology", "ref_cuts": []})

    # 4. Flash strobe flicker (single white frames at t=1.2s and t=2.6s in steady scene)
    steady = _textured_scene(404, tint=(80, 140, 60))
    flash_img = np.full((96, 128, 3), 255, dtype=np.uint8)
    flash_frames: list[DecodedFrame] = []
    for i in range(60):  # 4.0s at 15fps
        img = flash_img if i in (18, 39) else steady.copy()
        flash_frames.append(_make_decoded_frame(i, i / 15.0, img))
    streams.append({"name": "flash_strobe_flicker", "frames": flash_frames, "kind": "pathology", "ref_cuts": []})

    return streams


def _detect_reference_cuts(frames: list[DecodedFrame], c_thresh: float = 0.35, d_thresh: float = 0.35) -> list[int]:
    """Independent reference shot-cut detector used solely to evaluate Cut Recall across samplers."""
    cuts: list[int] = []
    prev_f: DecodedFrame | None = None
    for f in frames:
        if prev_f is not None and prev_f.pixels is not None and f.pixels is not None:
            d = compute_luma_difference(prev_f.pixels, f.pixels)
            c = compute_hist_distance(prev_f.pixels, f.pixels)
            if c > c_thresh or d > d_thresh:
                cuts.append(f.ordinal)
        prev_f = f
    return cuts


def _cut_recall_fraction(frames: list[DecodedFrame], selected_ordinals: list[int], ref_cut_ordinals: list[int]) -> float | None:
    if not ref_cut_ordinals:
        return None
    ord_to_t = {f.ordinal: (f.timestamp_seconds or 0.0) for f in frames}
    sel_times = [ord_to_t[o] for o in selected_ordinals if o in ord_to_t]
    if not sel_times:
        return 0.0
    hits = 0
    for c_ord in ref_cut_ordinals:
        c_t = ord_to_t.get(c_ord, 0.0)
        if any(abs(st - c_t) <= 0.16 for st in sel_times):
            hits += 1
    return hits / float(len(ref_cut_ordinals))


def run_tier1_synthetic(tmp_fixture_dir: Path, base_cfg: SamplingConfig) -> dict[str, Any]:
    print("\n=== TẦNG 1: Synthetic Pathology & M1 Contract Suite ===")
    streams = build_synthetic_pathology_streams(tmp_fixture_dir)
    all_strats = [*VALID_SAMPLING_STRATEGIES, "uniform_matched_budget"]
    by_strat: dict[str, list[dict[str, Any]]] = {s: [] for s in all_strats}
    per_stream_checks: list[dict[str, Any]] = []

    for st in streams:
        name = st["name"]
        frames = st["frames"]
        metrics = compare_sampling_strategies(
            frames,
            base_config=base_cfg,
            strategies=VALID_SAMPLING_STRATEGIES,
            video_id=name,
            include_uniform_matched_budget=True,
        )
        m_map = {m.strategy: m for m in metrics}
        row = {"stream": name, "kind": st["kind"], "total_frames": len(frames)}
        for s_name, m in m_map.items():
            by_strat[s_name].append(asdict(m))
            row[s_name] = {
                "n": m.selected_count,
                "rep": m.mean_representativeness,
                "sharp_gain": m.sharpness_gain_ratio,
                "blank_rate": m.blank_frame_rate,
                "gap_viol": m.max_gap_violations,
            }
        per_stream_checks.append(row)

    summary = {}
    for s_name, rows in by_strat.items():
        n = max(1, len(rows))
        summary[s_name] = {
            "streams": len(rows),
            "mean_selected": round(sum(r["selected_count"] for r in rows) / n, 2),
            "mean_rep": round(sum(r["mean_representativeness"] for r in rows) / n, 4),
            "mean_p10_rep": round(sum(r["p10_representativeness"] for r in rows) / n, 4),
            "mean_sharpness_gain": round(sum(r["sharpness_gain_ratio"] for r in rows) / n, 4),
            "mean_blank_rate": round(sum(r["blank_frame_rate"] for r in rows) / n, 4),
            "mean_redundancy": round(sum(r["soft_redundancy_rate"] for r in rows) / n, 4),
            "total_gap_violations": int(sum(r["max_gap_violations"] for r in rows)),
        }
        print(
            f"  [{s_name:<24}] sel={summary[s_name]['mean_selected']:5.1f} | "
            f"rep={summary[s_name]['mean_rep']:.4f} | sharp_gain={summary[s_name]['mean_sharpness_gain']:.3f}x | "
            f"blank={summary[s_name]['mean_blank_rate']*100:4.1f}% | viol={summary[s_name]['total_gap_violations']}"
        )
    return {"summary": summary, "streams": per_stream_checks}


def _decode_lowres_frames(video_path: Path, max_dim: int = 256) -> tuple[list[DecodedFrame], float]:
    """Decode media sequentially using PyAV, downscaling pixels immediately to keep RAM bounded."""
    t0 = time.perf_counter()
    frames: list[DecodedFrame] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return [], 0.0
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        tb_num, tb_den = tb.numerator, tb.denominator
        origin: Fraction | None = None
        prev_pts: int | None = None
        epoch_idx = 0
        pts_occ = 0
        for ordinal, frame in enumerate(container.decode(stream)):
            raw_pts = frame.pts
            if origin is None:
                origin = Fraction(raw_pts * tb_num, tb_den) if raw_pts is not None else Fraction(0)
            if raw_pts is not None and prev_pts is not None:
                if raw_pts == prev_pts:
                    pts_occ += 1
                elif raw_pts < prev_pts:
                    epoch_idx += 1
                    pts_occ = 0
                else:
                    pts_occ = 0
            prev_pts = raw_pts
            rgb = resize_lowres(frame.to_ndarray(format="rgb24"), max_dim=max_dim)
            frames.append(
                DecodedFrame(
                    ordinal=ordinal,
                    pts=raw_pts,
                    dts=frame.dts,
                    time_base_num=tb_num,
                    time_base_den=tb_den,
                    origin=origin,
                    pts_occurrence=pts_occ,
                    epoch_id=f"epoch_{epoch_idx}",
                    is_keyframe=bool(frame.key_frame),
                    width=int(rgb.shape[1]),
                    height=int(rgb.shape[0]),
                    pixels=rgb,
                )
            )
    elapsed_sec = time.perf_counter() - t0
    return frames, elapsed_sec


def _select_real_videos(raw_dir: Path, per_dataset: int = 10, seed: int = 42) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    chosen: list[tuple[str, Path]] = []
    for ds in ("msrvtt", "avc", "qvhighlights"):
        ds_dir = raw_dir / ds
        if not ds_dir.is_dir():
            continue
        files = sorted(ds_dir.rglob("*.mp4"))
        rng.shuffle(files)
        for p in files[:per_dataset]:
            chosen.append((ds, p))
    return chosen


def run_tier2_real_corpus(
    raw_dir: Path,
    per_dataset: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    print(f"\n=== TẦNG 2: Real Multi-Dataset Corpus Benchmark ({per_dataset}/dataset, seed={seed}) ===")
    videos = _select_real_videos(raw_dir, per_dataset=per_dataset, seed=seed)
    all_strats = [*VALID_SAMPLING_STRATEGIES, "uniform_matched_budget"]
    records_by_strat: dict[str, list[dict[str, Any]]] = {s: [] for s in all_strats}
    total_decoded_frames = 0
    total_decode_wall = 0.0

    for v_idx, (ds, v_path) in enumerate(videos):
        frames, dec_sec = _decode_lowres_frames(v_path, max_dim=256)
        if not frames:
            continue
        total_decoded_frames += len(frames)
        total_decode_wall += dec_sec

        dur_sec = float(frames[-1].timestamp_seconds or 0.0) if len(frames) > 1 else 0.0
        cfg = SamplingConfig.for_duration(dur_sec)
        max_gap_limit = cfg.max_static_gap_seconds or cfg.max_gap_seconds
        precomputed = precompute_eval_signals(frames, lowres_max_dim=cfg.lowres_max_dim, min_luma_std=cfg.min_luma_std)
        ref_cuts = _detect_reference_cuts(frames, c_thresh=cfg.change_threshold_c)

        adaptive_budget: int | None = None
        for strat in VALID_SAMPLING_STRATEGIES:
            s_cfg = replace(cfg, strategy=strat)
            t0 = time.perf_counter()
            manifest = sample_frames_hybrid(frames, s_cfg, video_id=v_path.stem, timeline_revision="bench_t2")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            q_met = evaluate_keyframe_quality(
                frames,
                manifest,
                strategy_name=strat,
                max_allowed_gap_seconds=max_gap_limit,
                min_luma_std=cfg.min_luma_std,
                elapsed_ms=elapsed_ms,
                precomputed=precomputed,
                max_eval_frames=250,
            )
            row = asdict(q_met)
            sel_ords = [s.original_decoded_frame_index for s in manifest.samples]
            row["cut_recall"] = _cut_recall_fraction(frames, sel_ords, ref_cuts)
            row["dataset"] = ds
            row["video_id"] = v_path.stem
            records_by_strat[strat].append(row)
            if strat == "hybrid_v2_adaptive":
                adaptive_budget = q_met.selected_count

        if adaptive_budget and adaptive_budget > 0:
            t0 = time.perf_counter()
            u_ords = generate_uniform_baseline(frames, budget_count=adaptive_budget)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            q_met_u = evaluate_keyframe_quality(
                frames,
                u_ords,
                strategy_name="uniform_matched_budget",
                max_allowed_gap_seconds=max_gap_limit,
                min_luma_std=cfg.min_luma_std,
                elapsed_ms=elapsed_ms,
                precomputed=precomputed,
                max_eval_frames=250,
            )
            row_u = asdict(q_met_u)
            row_u["cut_recall"] = _cut_recall_fraction(frames, u_ords, ref_cuts)
            row_u["dataset"] = ds
            row_u["video_id"] = v_path.stem
            records_by_strat["uniform_matched_budget"].append(row_u)

        if (v_idx + 1) % 10 == 0 or (v_idx + 1) == len(videos):
            print(f"  Tiến độ Tier 2: {v_idx + 1}/{len(videos)} videos ({total_decoded_frames} frames decoded)")

    summary: dict[str, dict[str, Any]] = {}
    for strat, rows in records_by_strat.items():
        n = max(1, len(rows))
        cut_vals = [r["cut_recall"] for r in rows if r["cut_recall"] is not None]
        tot_ms = sum(r["elapsed_ms"] for r in rows)
        summary[strat] = {
            "videos": len(rows),
            "mean_selected": round(sum(r["selected_count"] for r in rows) / n, 2),
            "mean_density_fps": round(sum(r["density_fps"] for r in rows) / n, 4),
            "mean_rep": round(sum(r["mean_representativeness"] for r in rows) / n, 4),
            "mean_p10_rep": round(sum(r["p10_representativeness"] for r in rows) / n, 4),
            "mean_sharpness_gain": round(sum(r["sharpness_gain_ratio"] for r in rows) / n, 4),
            "mean_blank_rate": round(sum(r["blank_frame_rate"] for r in rows) / n, 4),
            "mean_redundancy": round(sum(r["soft_redundancy_rate"] for r in rows) / n, 4),
            "cut_recall": round(sum(cut_vals) / len(cut_vals), 4) if cut_vals else None,
            "total_gap_violations": int(sum(r["max_gap_violations"] for r in rows)),
            "sampler_fps": round(total_decoded_frames / (tot_ms / 1000.0), 1) if tot_ms > 0 else 0.0,
        }
        s = summary[strat]
        cut_str = f"{s['cut_recall']*100:5.1f}%" if s["cut_recall"] is not None else "  N/A "
        print(
            f"  [{strat:<24}] sel={s['mean_selected']:5.1f} | rep={s['mean_rep']:.4f} (p10={s['mean_p10_rep']:.4f}) | "
            f"cut_rec={cut_str} | sharp={s['mean_sharpness_gain']:.3f}x | "
            f"blank={s['mean_blank_rate']*100:4.1f}% | redun={s['mean_redundancy']*100:4.1f}% | viol={s['total_gap_violations']} | {s['sampler_fps']:.0f} fps"
        )

    return {
        "total_videos": len(videos),
        "total_decoded_frames": total_decoded_frames,
        "decode_wall_seconds": round(total_decode_wall, 2),
        "summary": summary,
    }


def _load_validation_retrieval_items(
    val_dir: Path,
    raw_dir: Path,
    *,
    n_msrvtt: int = 6,
    n_avc: int = 6,
    n_qvh: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load clean validation queries whose target raw MP4 exists on disk (ZERO test_queries access)."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    raw_by_ds = {
        ds: {p.stem: p for p in (raw_dir / ds).rglob("*.mp4")}
        for ds in ("msrvtt", "avc", "qvhighlights")
        if (raw_dir / ds).is_dir()
    }

    # MSR-VTT
    msr_p = val_dir / "msrvtt" / "test_videodatainfo.json"
    if msr_p.is_file():
        msr_doc = json.loads(msr_p.read_text("utf-8"))
        by_vid: dict[str, dict[str, Any]] = {}
        for s in msr_doc.get("sentences", []):
            vid = s.get("video_id")
            if vid in raw_by_ds.get("msrvtt", {}) and vid not in by_vid:
                by_vid[vid] = s
        cand_msr = list(by_vid.values())
        rng.shuffle(cand_msr)
        for s in cand_msr[:n_msrvtt]:
            items.append({
                "dataset": "msrvtt",
                "query_id": f"msr_{s.get('sen_id', s['video_id'])}",
                "query_text": s["caption"],
                "video_id": s["video_id"],
                "video_path": raw_by_ds["msrvtt"][s["video_id"]],
                "relevant_windows": [],
            })

    # AVC
    avc_p = val_dir / "avc" / "candidate_plan_inference.json"
    pilot_p = raw_dir / "avc" / "provenance" / "data" / "processed" / "avc_1.8m" / "manifests" / "pilot_queries_eval_only.json"
    if avc_p.is_file() and pilot_p.is_file():
        pilot_tgt = {x["query_id"]: x["known_target_window_id"] for x in json.loads(pilot_p.read_text("utf-8"))}
        avc_doc = json.loads(avc_p.read_text("utf-8"))
        seen_avc: set[str] = set()
        cand_avc: list[dict[str, Any]] = []
        for q in avc_doc.get("queries", []):
            vid = pilot_tgt.get(q["query_id"])
            if vid and vid in raw_by_ds.get("avc", {}) and vid not in seen_avc:
                seen_avc.add(vid)
                cand_avc.append({
                    "dataset": "avc",
                    "query_id": f"avc_{q['query_id']}",
                    "query_text": q["query_text"],
                    "video_id": vid,
                    "video_path": raw_by_ds["avc"][vid],
                    "relevant_windows": [],
                })
        rng.shuffle(cand_avc)
        items.extend(cand_avc[:n_avc])

    # QVHighlights
    qvh_p = val_dir / "qvhighlights" / "highlight_val_release.jsonl"
    if qvh_p.is_file():
        seen_qvh: set[str] = set()
        cand_qvh: list[dict[str, Any]] = []
        for line in qvh_p.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            vid = q.get("vid")
            if vid in raw_by_ds.get("qvhighlights", {}) and vid not in seen_qvh:
                seen_qvh.add(vid)
                cand_qvh.append({
                    "dataset": "qvhighlights",
                    "query_id": f"qvh_{q['qid']}",
                    "query_text": q["query"],
                    "video_id": vid,
                    "video_path": raw_by_ds["qvhighlights"][vid],
                    "relevant_windows": q.get("relevant_windows", []),
                })
        rng.shuffle(cand_qvh)
        items.extend(cand_qvh[:n_qvh])

    return items


def _extract_native_rgb_map(video_path: Path, ordinals: set[int]) -> dict[int, Image.Image]:
    """Decode exact native-resolution RGB frames with PyAV for selected union ordinals."""
    out: dict[int, Image.Image] = {}
    if not ordinals:
        return out
    max_ord = max(ordinals)
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return out
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx in ordinals:
                out[idx] = Image.fromarray(frame.to_ndarray(format="rgb24"))
                if len(out) >= len(ordinals):
                    break
            if idx >= max_ord:
                break
    return out


def run_tier3_siglip2_validation(
    val_dir: Path,
    raw_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    print("\n=== TẦNG 3: Downstream SigLIP2 Validation Retrieval Benchmark (validation/ only) ===", flush=True)
    items = _load_validation_retrieval_items(val_dir, raw_dir, n_msrvtt=8, n_avc=8, n_qvh=2, seed=seed)
    if not items:
        raise RuntimeError("No matched validation queries/media found for Tier 3.")

    all_strats = [*VALID_SAMPLING_STRATEGIES, "uniform_matched_budget"]
    selections: dict[str, dict[str, list[tuple[int, float]]]] = {s: {} for s in all_strats}
    union_ordinals_by_vid: dict[str, set[int]] = {}
    vid_paths: dict[str, Path] = {}

    for it in items:
        vid = it["video_id"]
        v_path = it["video_path"]
        vid_paths[vid] = v_path
        frames, _ = _decode_lowres_frames(v_path, max_dim=256)
        ord_to_t = {f.ordinal: float(f.timestamp_seconds or 0.0) for f in frames}
        dur_sec = float(frames[-1].timestamp_seconds or 0.0) if len(frames) > 1 else 0.0
        cfg = SamplingConfig.for_duration(dur_sec)

        u_set = union_ordinals_by_vid.setdefault(vid, set())
        adaptive_budget: int | None = None
        for strat in VALID_SAMPLING_STRATEGIES:
            s_cfg = replace(cfg, strategy=strat)
            man = sample_frames_hybrid(frames, s_cfg, video_id=vid, timeline_revision="bench_t3")
            pairs = [(s.original_decoded_frame_index, float(s.timestamp_seconds or 0.0)) for s in man.samples]
            selections[strat][vid] = pairs
            u_set.update(o for o, _ in pairs)
            if strat == "hybrid_v2_adaptive":
                adaptive_budget = len(pairs)

        u_ords = generate_uniform_baseline(frames, budget_count=max(1, adaptive_budget or 1))
        selections["uniform_matched_budget"][vid] = [(o, ord_to_t.get(o, 0.0)) for o in u_ords]
        u_set.update(u_ords)

    total_unique_imgs = sum(len(v) for v in union_ordinals_by_vid.values())
    print(f"  Encoding {total_unique_imgs} unique native-res union frames + {len(items)} queries with local SigLIP2 (CPU)...", flush=True)
    t_enc = time.perf_counter()
    encoder = SigLIP2Encoder(config=SigLIP2Config(device="cpu", batch_size=16))

    pil_list: list[Image.Image] = []
    pil_keys: list[tuple[str, int]] = []
    for vid, ords in union_ordinals_by_vid.items():
        rgb_map = _extract_native_rgb_map(vid_paths[vid], ords)
        for o in sorted(rgb_map.keys()):
            pil_list.append(rgb_map[o])
            pil_keys.append((vid, o))

    img_chunks: list[np.ndarray] = []
    chunk_sz = 64
    for st in range(0, len(pil_list), chunk_sz):
        ed = min(len(pil_list), st + chunk_sz)
        img_chunks.append(encoder.encode_images(pil_list[st:ed]))
        print(f"    SigLIP2 encoded {ed}/{len(pil_list)} frames ({time.perf_counter()-t_enc:.1f}s)...", flush=True)
    img_vecs = np.concatenate(img_chunks, axis=0) if img_chunks else np.empty((0, 768), dtype=np.float32)
    img_emb_lookup: dict[tuple[str, int], np.ndarray] = {k: img_vecs[i] for i, k in enumerate(pil_keys)}
    q_vecs = encoder.encode_texts([it["query_text"] for it in items])
    enc_elapsed = time.perf_counter() - t_enc
    print(f"  SigLIP2 encoding completed in {enc_elapsed:.2f}s.")

    # Evaluate Video Hit@1, Hit@3, MRR, Mean Peak Cosine & QVH In-Window Peak Cosine
    summary: dict[str, dict[str, Any]] = {}
    for strat in all_strats:
        hit1 = 0
        hit3 = 0
        mrr_sum = 0.0
        peak_cos_list: list[float] = []
        qvh_in_win_cos: list[float] = []
        total_sel_cnt = 0

        # Pre-stack each video's selected frame embeddings under this strategy
        vid_mats: dict[str, np.ndarray] = {}
        for vid in vid_paths:
            ords = [o for o, _ in selections[strat][vid] if (vid, o) in img_emb_lookup]
            total_sel_cnt += len(ords)
            vid_mats[vid] = np.stack([img_emb_lookup[(vid, o)] for o in ords], axis=0) if ords else np.zeros((1, 768), dtype=np.float32)

        for q_idx, it in enumerate(items):
            q_vec = q_vecs[q_idx]
            tgt_vid = it["video_id"]
            scores: list[tuple[str, float]] = []
            for vid, mat in vid_mats.items():
                max_sim = float(np.max(mat @ q_vec))
                scores.append((vid, max_sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_vids = [v for v, _ in scores]
            rank_pos = ranked_vids.index(tgt_vid) + 1
            if rank_pos == 1:
                hit1 += 1
            if rank_pos <= 3:
                hit3 += 1
            mrr_sum += 1.0 / rank_pos

            tgt_score = float(np.max(vid_mats[tgt_vid] @ q_vec))
            peak_cos_list.append(tgt_score)

            if it["dataset"] == "qvhighlights" and it["relevant_windows"]:
                wins = it["relevant_windows"]
                in_win_vecs = [
                    img_emb_lookup[(tgt_vid, o)]
                    for o, t_s in selections[strat][tgt_vid]
                    if (tgt_vid, o) in img_emb_lookup and any(float(ws) <= t_s <= float(we) for ws, we in wins)
                ]
                qvh_in_win_cos.append(
                    float(np.max(np.stack(in_win_vecs, axis=0) @ q_vec)) if in_win_vecs else 0.0
                )

        n_q = len(items)
        summary[strat] = {
            "queries": n_q,
            "mean_keyframes_per_video": round(total_sel_cnt / n_q, 2),
            "video_hit_at_1": round(hit1 / n_q, 4),
            "video_hit_at_3": round(hit3 / n_q, 4),
            "mrr": round(mrr_sum / n_q, 4),
            "mean_peak_cosine": round(float(np.mean(peak_cos_list)), 4),
            "qvh_in_window_peak_cosine": round(float(np.mean(qvh_in_win_cos)), 4) if qvh_in_win_cos else None,
        }
        s = summary[strat]
        qvh_s = f"{s['qvh_in_window_peak_cosine']:.4f}" if s["qvh_in_window_peak_cosine"] is not None else "N/A"
        print(
            f"  [{strat:<24}] kf/vid={s['mean_keyframes_per_video']:5.1f} | Hit@1={s['video_hit_at_1']*100:5.1f}% | "
            f"Hit@3={s['video_hit_at_3']*100:5.1f}% | MRR={s['mrr']:.4f} | PeakCos={s['mean_peak_cosine']:.4f} | QVH-WinCos={qvh_s}"
        )

    return {
        "queries": len(items),
        "unique_union_images": total_unique_imgs,
        "siglip2_encode_seconds": round(enc_elapsed, 2),
        "summary": summary,
    }


def choose_winner(t1: dict[str, Any], t2: dict[str, Any], t3: dict[str, Any]) -> dict[str, Any]:
    """Select the objectively best strategy honoring strict contract compliance + SigLIP2 + visual quality."""
    candidates = list(VALID_SAMPLING_STRATEGIES) + ["uniform_matched_budget"]
    scored: list[dict[str, Any]] = []
    for strat in candidates:
        s1 = t1["summary"].get(strat, {})
        s2 = t2["summary"].get(strat, {})
        s3 = t3["summary"].get(strat, {})
        violations = int(s1.get("total_gap_violations", 0)) + int(s2.get("total_gap_violations", 0))
        cut_rec = float(s2.get("cut_recall") or 0.0)
        eligible = (strat in VALID_SAMPLING_STRATEGIES) and violations == 0 and strat != "uniform_matched"
        # Composite decision index: 0.45 * SigLIP2 MRR + 0.25 * PeakCos + 0.15 * Rep + 0.10 * CutRec - 0.05 * BlankRate
        composite = (
            0.45 * float(s3.get("mrr", 0.0))
            + 0.25 * float(s3.get("mean_peak_cosine", 0.0))
            + 0.15 * float(s2.get("mean_rep", 0.0))
            + 0.10 * cut_rec
            - 0.05 * float(s2.get("mean_blank_rate", 0.0))
        )
        scored.append({
            "strategy": strat,
            "eligible_default": eligible,
            "composite_score": round(composite, 5),
            "total_gap_violations": violations,
            "t3_mrr": s3.get("mrr"),
            "t3_hit1": s3.get("video_hit_at_1"),
            "t3_peak_cos": s3.get("mean_peak_cosine"),
            "t2_rep": s2.get("mean_rep"),
            "t2_sharp_gain": s2.get("mean_sharpness_gain"),
            "t2_cut_recall": s2.get("cut_recall"),
            "t2_blank_rate": s2.get("mean_blank_rate"),
            "t2_redundancy": s2.get("mean_redundancy"),
        })

    eligible_rows = [r for r in scored if r["eligible_default"]]
    eligible_rows.sort(key=lambda r: (r["composite_score"], float(r["t3_peak_cos"] or 0.0), float(r["t2_rep"] or 0.0)), reverse=True)
    winner = eligible_rows[0] if eligible_rows else scored[0]
    print(f"\n>>> WINNER SELECTION: '{winner['strategy']}' (composite={winner['composite_score']}, MRR={winner['t3_mrr']}, PeakCos={winner['t3_peak_cos']}, Rep={winner['t2_rep']}, Violations={winner['total_gap_violations']})")
    return {"winner": winner, "ranking": scored}


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Benchmark all 5 VIDENTRA keyframe sampling strategies.")
    parser.add_argument("--out-dir", type=Path, default=Path("work/experiments/keyframe_strategy_benchmark"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_fixtures = out_dir / "_tmp_fixtures"

    base_cfg = SamplingConfig()
    t1 = run_tier1_synthetic(tmp_fixtures, base_cfg)
    t2 = run_tier2_real_corpus(args.raw_dir, per_dataset=args.per_dataset, seed=args.seed)
    t3 = run_tier3_siglip2_validation(args.val_dir, args.raw_dir, seed=args.seed)
    decision = choose_winner(t1, t2, t3)

    # Clean up temporary synthetic fixture directory inside benchmark dir
    if tmp_fixtures.exists():
        import shutil
        shutil.rmtree(tmp_fixtures, ignore_errors=True)

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
        },
        "tier1_synthetic_pathology": t1,
        "tier2_real_corpus": t2,
        "tier3_siglip2_validation": t3,
        "decision": decision,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write concise Vietnamese + tabular Markdown report
    lines = [
        "# Benchmark Đa phương án Trích xuất Keyframes — VIDENTRA",
        "",
        f"- **Thời gian UTC:** `{payload['timestamp_utc']}`",
        f"- **Phần cứng:** `{payload['hardware']['platform']}`, `cpu_count={payload['hardware']['cpu_count']}`, `device=cpu`",
        f"- **Phương án thắng cuộc:** **`{decision['winner']['strategy']}`** (`composite={decision['winner']['composite_score']}`, `0` vi phạm max-gap)",
        "",
        "## Bảng Tổng hợp Quyết định (Tầng 2 Corpus + Tầng 3 SigLIP2 Validation)",
        "",
        "| Strategy | T3 Hit@1 | T3 MRR | T3 Peak Cosine | T2 Rep | T2 Cut Recall | T2 Sharpness Gain | T2 Blank Rate | T2 Redundancy | Violations |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in decision["ranking"]:
        cr = f"{r['t2_cut_recall']*100:.1f}%" if r["t2_cut_recall"] is not None else "N/A"
        lines.append(
            f"| `{r['strategy']}` | {r['t3_hit1']*100:.1f}% | {r['t3_mrr']:.4f} | {r['t3_peak_cos']:.4f} | "
            f"{r['t2_rep']:.4f} | {cr} | {r['t2_sharp_gain']:.3f}x | "
            f"{r['t2_blank_rate']*100:.2f}% | {r['t2_redundancy']*100:.2f}% | {r['total_gap_violations']} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark artifacts to {out_dir / 'results.json'} and {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
