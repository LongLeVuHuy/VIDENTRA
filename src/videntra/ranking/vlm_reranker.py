"""Stage-2 Margin-Gated VLM Cross-Encoder Reranker (Qwen2.5-VL-3B/7B) & Storyboard Builder."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np

from videntra.ingestion.probe import probe_media

VLM_3B_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_3B_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
VLM_3B_LICENSE = "apache-2.0"

VLM_7B_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_7B_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
VLM_7B_LICENSE = "apache-2.0"

PINNED_VLM_REVISIONS: dict[str, dict[str, str]] = {
    VLM_3B_MODEL_ID: {"revision": VLM_3B_REVISION, "license": VLM_3B_LICENSE},
    VLM_7B_MODEL_ID: {"revision": VLM_7B_REVISION, "license": VLM_7B_LICENSE},
}

DEFAULT_VLM_QUERY_TEMPLATE = (
    'These {num_frames} chronological frames are sampled from a video clip. '
    'Does this video match the description: "{query_text}"? Answer strictly Yes or No.'
)

COMPOSITIONAL_VLM_QUERY_TEMPLATE = (
    'These {num_frames} chronological frames with timestamps #1..#{num_frames} show events from a video clip in time order. '
    'Check every event order, exact action verbs, and whether the full description matches these frames: "{query_text}". '
    'Answer strictly Yes or No.'
)


def _sha256_text(text: str) -> str:
    norm = unicodedata.normalize("NFC", text).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VLMFusionConfig:
    """Hyperparameters governing margin-gated Stage-1 + Stage-2 VLM score fusion."""

    margin_gate: float = 1.2
    w_vlm: float = 0.35
    top_k_rerank: int = 5
    mode: str = "margin_raw"
    debias_alpha: float = 0.3
    debias_shrink: float = 2.0


def ensure_vlm_lm_head_tied(mdl: Any) -> Any:
    """Tie `lm_head.weight` to input embeddings ONLY when model config specifies `tie_word_embeddings=True`.

    Never overwrites independent untied `lm_head.weight` (e.g., Qwen2.5-VL-7B-Instruct where
    `tie_word_embeddings=False`), avoiding output logit corruption while fixing uninitialized
    tied heads on `Qwen2.5-VL-3B-Instruct` when loaded from FP16/NF4 shards.
    """
    cfg = getattr(mdl, "config", None)
    tie_top = getattr(cfg, "tie_word_embeddings", None)
    text_cfg = getattr(cfg, "text_config", None)
    tie_text = getattr(text_cfg, "tie_word_embeddings", None) if text_cfg is not None else None
    should_tie = bool(tie_top) if tie_top is not None else bool(tie_text)
    if not should_tie:
        return mdl

    in_emb = mdl.get_input_embeddings() if hasattr(mdl, "get_input_embeddings") else None
    lm_head = getattr(mdl, "lm_head", None)
    if (
        in_emb is not None
        and lm_head is not None
        and hasattr(in_emb, "weight")
        and hasattr(lm_head, "weight")
        and tuple(lm_head.weight.shape) == tuple(in_emb.weight.shape)
    ):
        lm_head.weight = in_emb.weight
    if hasattr(mdl, "tie_weights"):
        try:
            mdl.tie_weights()
        except Exception:
            pass
    return mdl


def select_bin_center_ordinals(
    frame_records: Sequence[tuple[int, float | None]],
    clip_start_sec: float,
    clip_end_sec: float,
    *,
    num_frames: int = 8,
) -> list[int]:
    """Select `num_frames` presentation-order frame ordinals at uniform PTS bin centers."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if clip_end_sec <= clip_start_sec:
        raise ValueError(f"Invalid clip interval [{clip_start_sec}, {clip_end_sec})")

    valid_frames = [
        (int(o), float(ts))
        for o, ts in frame_records
        if ts is not None and clip_start_sec <= float(ts) < clip_end_sec
    ]
    if not valid_frames:
        valid_frames = [(int(o), float(ts)) for o, ts in frame_records if ts is not None]
    if not valid_frames:
        return []

    valid_frames.sort(key=lambda item: item[0])
    duration = clip_end_sec - clip_start_sec
    selected: list[int] = []
    for idx in range(num_frames):
        target_ts = clip_start_sec + ((idx + 0.5) * duration) / float(num_frames)
        best_ord, _ = min(valid_frames, key=lambda item: (abs(item[1] - target_ts), item[0]))
        selected.append(int(best_ord))
    return selected


def _format_mmss(sec: float) -> str:
    sec_i = max(0, int(round(float(sec))))
    return f"{sec_i // 60:02d}:{sec_i % 60:02d}"


def select_event_aligned_ordinals(
    frame_records: Sequence[tuple[int, float | None]],
    clip_start_sec: float,
    clip_end_sec: float,
    focus_intervals: Sequence[tuple[float, float]] | None = None,
    *,
    num_frames: int = 8,
) -> list[int]:
    """Select chronological frame ordinals biased toward localized event intervals when available."""
    if not focus_intervals:
        return select_bin_center_ordinals(
            frame_records, clip_start_sec, clip_end_sec, num_frames=num_frames
        )

    valid_records = sorted(
        (
            (int(o), float(ts))
            for o, ts in frame_records
            if ts is not None and clip_start_sec <= float(ts) < clip_end_sec
        ),
        key=lambda item: item[0],
    )
    if not valid_records:
        return select_bin_center_ordinals(
            frame_records, clip_start_sec, clip_end_sec, num_frames=num_frames
        )

    clipped: list[tuple[float, float]] = []
    for a, b in focus_intervals:
        s_t = max(float(clip_start_sec), float(min(a, b)))
        e_t = min(float(clip_end_sec), float(max(a, b)))
        if e_t <= s_t:
            e_t = min(float(clip_end_sec), s_t + 0.5)
        if e_t > s_t:
            clipped.append((s_t, e_t))
    if not clipped:
        return select_bin_center_ordinals(
            frame_records, clip_start_sec, clip_end_sec, num_frames=num_frames
        )

    if len(clipped) > num_frames:
        pick_idx = np.linspace(0, len(clipped) - 1, num_frames, dtype=int)
        clipped = [clipped[int(i)] for i in pick_idx]

    n_int = len(clipped)
    base_q, rem_q = divmod(num_frames, n_int)
    quotas = [base_q + (1 if idx < rem_q else 0) for idx in range(n_int)]
    chosen: list[int] = []
    seen_ords: set[int] = set()
    for (s_t, e_t), q_cnt in zip(clipped, quotas):
        if q_cnt <= 0:
            continue
        sub_ords = select_bin_center_ordinals(valid_records, s_t, e_t, num_frames=q_cnt)
        for o in sub_ords:
            if o not in seen_ords:
                chosen.append(o)
                seen_ords.add(o)

    if len(chosen) < num_frames:
        fallback = select_bin_center_ordinals(
            valid_records, clip_start_sec, clip_end_sec, num_frames=num_frames
        )
        for o in fallback:
            if len(chosen) >= num_frames:
                break
            if o not in seen_ords:
                chosen.append(o)
                seen_ords.add(o)

    chosen.sort()
    return chosen[:num_frames]


def build_chronological_storyboard_with_meta(
    video_path: Path,
    *,
    num_frames: int = 8,
    tile_size: tuple[int, int] = (224, 168),
    jpeg_quality: int = 88,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
    frame_ledger: Sequence[tuple[int, float | None]] | None = None,
    focus_intervals: Sequence[tuple[float, float]] | None = None,
    annotate_timestamps: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Stream-decode ONLY selected presentation-order ordinals into a bounded-RAM 2x4 JPEG + metadata."""
    import av  # type: ignore
    from PIL import Image, ImageDraw  # type: ignore

    pr = probe_media(video_path)
    records: list[tuple[int, float | None]] = []
    if frame_ledger is not None:
        records = [
            (int(o), float(ts) if ts is not None else None)
            for o, ts in frame_ledger
        ]
    else:
        with av.open(str(video_path)) as container:
            stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass
            tb = float(stream.time_base) if stream.time_base is not None else 0.0
            for cur_ord, frame in enumerate(container.decode(stream)):
                ts = float(frame.pts * tb) if (frame.pts is not None and tb > 0.0) else None
                records.append((cur_ord, ts))

    pts_by_ord: dict[int, float] = {int(o): float(ts) for o, ts in records if ts is not None}
    if not pts_by_ord:
        raise ValueError(f"Video lacks mapped PTS timestamps: {video_path}")

    min_ts = min(pts_by_ord.values())
    max_ts = max(pts_by_ord.values()) + 1e-3
    eff_start = float(clip_start_sec) if clip_start_sec is not None else min_ts
    eff_end = float(clip_end_sec) if clip_end_sec is not None else max_ts
    if eff_end <= eff_start:
        eff_end = eff_start + 1e-3

    ords = select_event_aligned_ordinals(
        records,
        eff_start,
        eff_end,
        focus_intervals=focus_intervals,
        num_frames=num_frames,
    )
    if not ords:
        raise ValueError(f"Unable to select chronological storyboard frames for: {video_path}")

    target_set = set(ords)
    max_target = max(target_set)
    tw, th = int(tile_size[0]), int(tile_size[1])
    tiles_by_ord: dict[int, Any] = {}

    with av.open(str(video_path)) as container:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        for cur_ord, frame in enumerate(container.decode(stream)):
            if cur_ord in target_set and cur_ord not in tiles_by_ord:
                rgb = frame.to_ndarray(format="rgb24")
                tiles_by_ord[cur_ord] = Image.fromarray(rgb, mode="RGB").resize((tw, th), Image.BILINEAR)
            if cur_ord >= max_target and len(tiles_by_ord) >= len(target_set):
                break

    missing_ords = [o for o in ords if o not in tiles_by_ord]
    if missing_ords:
        raise ValueError(f"Missing decoded storyboard ordinal(s) {missing_ords} in {video_path}")

    cols = 4
    rows = max(1, (len(ords) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * tw, rows * th))
    draw = ImageDraw.Draw(canvas) if annotate_timestamps else None
    for i_o, ord_idx in enumerate(ords):
        x0 = (i_o % cols) * tw
        y0 = (i_o // cols) * th
        canvas.paste(tiles_by_ord[ord_idx], (x0, y0))
        if draw is not None:
            label = f"#{i_o + 1} {_format_mmss(pts_by_ord.get(ord_idx, 0.0))}"
            box_w = min(tw - 4, max(44, 6 * len(label) + 8))
            draw.rectangle([x0 + 2, y0 + 2, x0 + box_w, y0 + 16], fill=(12, 14, 18))
            draw.text((x0 + 5, y0 + 4), label, fill=(255, 255, 255))
    tiles_by_ord.clear()

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=int(jpeg_quality))
    jb = buf.getvalue()
    meta: dict[str, Any] = {
        "frame_ordinals": [int(o) for o in ords],
        "frame_pts_sec": [round(float(pts_by_ord[o]), 4) for o in ords],
        "storyboard_sha256": hashlib.sha256(jb).hexdigest(),
        "tile_size": [tw, th],
        "clip_start_sec": round(eff_start, 4),
        "clip_end_sec": round(eff_end, 4),
        "annotate_timestamps": bool(annotate_timestamps),
    }
    return jb, meta


def build_chronological_storyboard_jpeg(
    video_path: Path,
    *,
    num_frames: int = 8,
    tile_size: tuple[int, int] = (224, 168),
    jpeg_quality: int = 88,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
    frame_ledger: Sequence[tuple[int, float | None]] | None = None,
    focus_intervals: Sequence[tuple[float, float]] | None = None,
    annotate_timestamps: bool = False,
) -> bytes:
    """Decode only selected presentation-order frames via PyAV and pack `num_frames` PTS bin-centers into a 2x4 JPEG."""
    jb, _ = build_chronological_storyboard_with_meta(
        video_path,
        num_frames=num_frames,
        tile_size=tile_size,
        jpeg_quality=jpeg_quality,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        frame_ledger=frame_ledger,
        focus_intervals=focus_intervals,
        annotate_timestamps=annotate_timestamps,
    )
    return jb


def debias_vlm_scores(
    vlm_scores: Mapping[str, Mapping[str, float]],
    *,
    alpha: float = 0.0,
    shrink: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Shrink per-video VLM logit prior across a split's candidate pools (zero GT)."""
    if float(alpha) <= 0.0 or not vlm_scores:
        return {str(q): {str(v): float(s) for v, s in sm.items()} for q, sm in vlm_scores.items()}
    by_vid: dict[str, list[float]] = defaultdict(list)
    total_sum = 0.0
    total_cnt = 0
    for sm in vlm_scores.values():
        for vid, sc in sm.items():
            f_sc = float(sc)
            by_vid[str(vid)].append(f_sc)
            total_sum += f_sc
            total_cnt += 1
    g_mean = total_sum / max(1, total_cnt)
    bias_by_vid = {
        vid: ((len(vals) * float(np.mean(vals)) + float(shrink) * g_mean) / (len(vals) + float(shrink)))
        if float(shrink) > 0.0
        else float(np.mean(vals))
        for vid, vals in by_vid.items()
    }
    a = float(alpha)
    return {
        str(qid): {str(vid): float(sc) - a * bias_by_vid.get(str(vid), g_mean) for vid, sc in sm.items()}
        for qid, sm in vlm_scores.items()
    }


def combine_stage1_and_vlm_pool(
    s1_arr: np.ndarray,
    vlm_arr: np.ndarray,
    *,
    w_vlm: float,
    mode: str = "z_pool",
) -> np.ndarray:
    """Combine Stage-1 and VLM scores over a candidate pool."""
    if mode == "z_pool":
        s1_norm = (s1_arr - float(np.mean(s1_arr))) / max(1e-9, float(np.std(s1_arr)))
        v_norm = (vlm_arr - float(np.mean(vlm_arr))) / max(1e-9, float(np.std(vlm_arr)))
        return s1_norm + float(w_vlm) * v_norm
    if mode == "margin_raw":
        return s1_arr + float(w_vlm) * (vlm_arr - float(np.mean(vlm_arr)))
    if mode == "margin_vz":
        v_norm = (vlm_arr - float(np.mean(vlm_arr))) / max(1e-9, float(np.std(vlm_arr)))
        return s1_arr + float(w_vlm) * v_norm
    raise ValueError(f"Unsupported VLM blend mode: {mode}")


def blend_stage1_and_vlm_predictions(
    base_preds: Sequence[Mapping[str, Any]],
    top20_scores: Mapping[str, Mapping[str, float]],
    vlm_scores: Mapping[str, Mapping[str, float]],
    *,
    margin_gate: float,
    w_vlm: float,
    top_k_rerank: int,
    mode: str = "z_pool",
    debias_alpha: float = 0.0,
    debias_shrink: float = 2.0,
) -> list[dict[str, Any]]:
    """Apply Stage-2 Margin-Gated VLM Reranking strictly per query on frozen Stage-1 Top-K."""
    eff_vlm = debias_vlm_scores(vlm_scores, alpha=debias_alpha, shrink=debias_shrink) if float(w_vlm) != 0.0 else {}
    out: list[dict[str, Any]] = []
    for p in base_preds:
        qid = str(p["query_id"])
        ranked = [str(v) for v in p["ranked_video_ids"]]
        s1_map = top20_scores.get(qid, {})
        if float(w_vlm) == 0.0 or qid not in eff_vlm or len(ranked) < 2:
            out.append({"query_id": qid, "gallery": p["gallery"], "ranked_video_ids": ranked})
            continue
        m1_z = float(s1_map.get(ranked[0], 0.0) - s1_map.get(ranked[1], 0.0))
        if m1_z >= float(margin_gate):
            out.append({"query_id": qid, "gallery": p["gallery"], "ranked_video_ids": ranked})
            continue

        pool = [v for v in ranked[:top_k_rerank] if v in eff_vlm[qid]]
        if len(pool) < 2:
            out.append({"query_id": qid, "gallery": p["gallery"], "ranked_video_ids": ranked})
            continue

        s1_arr = np.asarray([s1_map[v] for v in pool], dtype=np.float64)
        vlm_arr = np.asarray([eff_vlm[qid][v] for v in pool], dtype=np.float64)
        comb = combine_stage1_and_vlm_pool(s1_arr, vlm_arr, w_vlm=w_vlm, mode=mode)
        blended_scores_by_vid = {v: float(sc) for v, sc in zip(pool, comb)}
        reordered = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-comb[i], i))]
        pool_set = set(reordered)
        new_ranked = reordered + [v for v in ranked if v not in pool_set]
        out.append({
            "query_id": qid,
            "gallery": p["gallery"],
            "ranked_video_ids": new_ranked,
            "blended_scores_by_vid": blended_scores_by_vid,
        })
    return out


class Qwen25VLReranker:
    """6GB-VRAM-compatible Margin-Gated Qwen2.5-VL cross-encoder reranker (default: 3B NF4)."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
        processor_revision: str | None = None,
        quant: str | None = None,
        enabled: bool | None = None,
        score_cache_path: Path | str | None = None,
        persist_pair_cache: bool = False,
        fusion_config: VLMFusionConfig | None = None,
        batch_size: int | None = None,
        device: str = "auto",
        num_frames: int = 8,
        annotate_timestamps: bool = False,
        query_template: str | None = None,
        tile_size: tuple[int, int] = (224, 168),
        shrunk_tile_size: tuple[int, int] = (160, 120),
    ) -> None:
        env_model = os.environ.get("VIDENTRA_VLM_MODEL_ID", "").strip()
        self.model_id = str(model_id or env_model or VLM_3B_MODEL_ID)
        pinned = PINNED_VLM_REVISIONS.get(self.model_id, {})
        self.model_revision = model_revision if model_revision is not None else (pinned.get("revision") or "")
        self.processor_revision = (
            processor_revision if processor_revision is not None else self.model_revision
        )
        self.num_frames = max(1, int(num_frames))
        self.query_template = str(query_template or DEFAULT_VLM_QUERY_TEMPLATE)
        tmpl_norm = self.query_template.format(query_text="{query_text}", num_frames=self.num_frames)
        self.prompt_sha256 = _sha256_text(tmpl_norm)
        self.annotate_timestamps = bool(annotate_timestamps)
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        self.shrunk_tile_size = (int(shrunk_tile_size[0]), int(shrunk_tile_size[1]))
        self.persist_pair_cache = bool(persist_pair_cache)

        env_quant = os.environ.get("VIDENTRA_VLM_QUANT", "").strip().lower()
        self.quant = str(quant or env_quant or "nf4_4bit")
        self.device_pref = str(device).strip().lower() or "auto"
        self.fusion_config = fusion_config or VLMFusionConfig()

        env_cache = os.environ.get("VIDENTRA_VLM_CACHE_PATH", "").strip()
        resolved_cache = score_cache_path if score_cache_path is not None else (env_cache or None)
        self.score_cache_path: Path | None = Path(resolved_cache) if resolved_cache else None
        self._cached_scores_by_qid: dict[str, tuple[dict[str, float], dict[str, str]]] = {}
        self._cached_scores_by_sha: dict[str, tuple[dict[str, float], dict[str, str]]] = {}
        self._pair_cache: dict[tuple[str, str, str, str, str, str, str], float] = {}
        self._pair_config_entries: dict[tuple[str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)
        if self.score_cache_path is not None and self.score_cache_path.is_file():
            self._load_score_cache(self.score_cache_path)

        has_any_cache = bool(self._cached_scores_by_qid or self._cached_scores_by_sha or self._pair_config_entries)
        if enabled is not None:
            self.enabled = bool(enabled)
        else:
            env_en = os.environ.get("VIDENTRA_VLM_RERANK_ENABLED", "").strip().lower()
            if env_en in ("1", "true", "yes", "on"):
                self.enabled = True
            elif env_en in ("0", "false", "no", "off"):
                self.enabled = False
            elif env_en == "auto":
                self.enabled = self._has_cuda() or has_any_cache
            else:
                self.enabled = has_any_cache

        self.batch_size = int(batch_size) if batch_size is not None else None
        self._processor: Any = None
        self._model: Any = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []
        self._loaded_device: str | None = None
        self.last_score_stats: dict[str, Any] = {}
        self._storyboard_cache: dict[tuple[Any, ...], tuple[bytes, dict[str, Any]]] = {}

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _pair_config_key(self, query_sha256: str, video_id: str) -> tuple[str, str, str, str, str, str]:
        return (
            str(query_sha256),
            str(video_id),
            str(self.model_id),
            str(self.model_revision or ""),
            str(self.processor_revision or ""),
            str(self.prompt_sha256),
        )

    def _legacy_row_matches_provenance(self, prov: Mapping[str, str]) -> bool:
        mid = prov.get("model_id", "")
        if mid and mid != self.model_id:
            return False
        mrev = prov.get("model_revision", "")
        if mrev and self.model_revision and mrev != self.model_revision:
            return False
        prev = prov.get("processor_revision", "")
        if prev and self.processor_revision and prev != self.processor_revision:
            return False
        psha = prov.get("prompt_sha256", "")
        if psha and psha != self.prompt_sha256:
            return False
        return True

    def _load_score_cache(self, path: Path) -> None:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as stream:  # type: ignore[arg-type]
            for line in stream:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if "video_id" in row and "score" in row:
                    qsha = str(row.get("query_text_sha256") or "").strip()
                    if not qsha and row.get("query_text"):
                        qsha = _sha256_text(str(row["query_text"]))
                    if not qsha:
                        continue
                    vid = str(row["video_id"])
                    sc = float(row["score"])
                    mid = str(row.get("model_id") or self.model_id)
                    mrev = str(row.get("model_revision") if row.get("model_revision") is not None else (self.model_revision or ""))
                    prev = str(row.get("processor_revision") if row.get("processor_revision") is not None else (self.processor_revision or ""))
                    psha = str(row.get("prompt_sha256") or self.prompt_sha256)
                    sb_sha = str(row.get("storyboard_sha256") or "")
                    full_key = (qsha, vid, mid, mrev, prev, psha, sb_sha)
                    cfg_key = (qsha, vid, mid, mrev, prev, psha)
                    self._pair_cache[full_key] = sc
                    self._pair_config_entries[cfg_key][sb_sha] = sc
                    continue

                cids = row.get("candidate_video_ids") or []
                scores = row.get("vlm_scores") or []
                if not cids or len(cids) != len(scores):
                    continue
                score_map = {str(v): float(sc) for v, sc in zip(cids, scores)}
                prov = {
                    "model_id": str(row.get("model_id") or ""),
                    "model_revision": str(row.get("model_revision") or ""),
                    "processor_revision": str(row.get("processor_revision") or ""),
                    "prompt_sha256": str(row.get("prompt_sha256") or ""),
                }
                qid = str(row.get("query_id") or "").strip()
                if qid:
                    self._cached_scores_by_qid[qid] = (score_map, prov)
                qsha = str(row.get("query_text_sha256") or "").strip()
                if qsha:
                    self._cached_scores_by_sha[qsha] = (score_map, prov)

    def _append_pair_cache_record(
        self,
        *,
        query_id: str | None,
        query_sha256: str,
        video_id: str,
        storyboard_sha256: str,
        score: float,
    ) -> None:
        if not self.persist_pair_cache or self.score_cache_path is None:
            return
        if self.score_cache_path.suffix == ".gz":
            return
        self.score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "query_id": query_id,
            "query_text_sha256": query_sha256,
            "video_id": str(video_id),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "prompt_sha256": self.prompt_sha256,
            "storyboard_sha256": str(storyboard_sha256),
            "score": float(score),
        }
        with self.score_cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def lookup_cached_scores(
        self,
        query_text: str,
        *,
        query_id: str | None = None,
        candidate_video_ids: Sequence[str] | None = None,
        storyboard_sha256_by_vid: Mapping[str, str] | None = None,
    ) -> dict[str, float] | None:
        """Return pre-scored VLM map if cached locally with matching provenance (zero GPU/VRAM cost)."""
        qsha = _sha256_text(query_text)
        merged: dict[str, float] = {}

        if query_id and str(query_id) in self._cached_scores_by_qid:
            sm, prov = self._cached_scores_by_qid[str(query_id)]
            if self._legacy_row_matches_provenance(prov):
                merged.update(sm)
        if qsha in self._cached_scores_by_sha:
            sm, prov = self._cached_scores_by_sha[qsha]
            if self._legacy_row_matches_provenance(prov):
                for vid, sc in sm.items():
                    merged.setdefault(vid, sc)

        probe_vids = list(candidate_video_ids) if candidate_video_ids is not None else sorted({
            k[1] for k in self._pair_config_entries if k[0] == qsha
        })
        for vid in probe_vids:
            vid_s = str(vid)
            cfg_key = self._pair_config_key(qsha, vid_s)
            entries = self._pair_config_entries.get(cfg_key)
            if not entries:
                continue
            sb_req = storyboard_sha256_by_vid.get(vid_s) if storyboard_sha256_by_vid else None
            if sb_req is not None:
                if sb_req in entries:
                    merged[vid_s] = float(entries[sb_req])
                elif "" in entries:
                    merged[vid_s] = float(entries[""])
            else:
                # Prefer deterministic latest storyboard entry when caller does not specify storyboard hash
                last_sc = next(reversed(entries.values()))
                merged[vid_s] = float(last_sc)

        return merged or None

    def _resolve_batch_size(self) -> int:
        if self.batch_size is not None:
            return max(1, int(self.batch_size))
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
                return 1 if vram_gb <= 8.5 else 4
        except Exception:
            pass
        return 1

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if not self.enabled:
            return False

        import torch  # type: ignore
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

        use_cuda = (self.device_pref in ("auto", "cuda")) and torch.cuda.is_available()
        dev = "cuda:0" if use_cuda else "cpu"
        rev_kw: dict[str, Any] = {"revision": self.model_revision} if self.model_revision else {}

        self._processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False, **rev_kw)
        if use_cuda and self.quant == "nf4_4bit":
            from transformers import BitsAndBytesConfig  # type: ignore

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=bnb_cfg,
                device_map=dev,
                **rev_kw,
            )
        else:
            dtype = torch.float16 if use_cuda else torch.float32
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                **rev_kw,
            ).to(dev)

        mdl = ensure_vlm_lm_head_tied(mdl).eval()
        tok = self._processor.tokenizer
        tok.padding_side = "left"
        self._yes_ids = [t[0] for w in ("Yes", "yes") if len(t := tok.encode(w, add_special_tokens=False)) == 1]
        self._no_ids = [t[0] for w in ("No", "no") if len(t := tok.encode(w, add_special_tokens=False)) == 1]
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(f"Missing single-token Yes/No IDs for VLM tokenizer: {self.model_id}")

        self._model = mdl
        self._loaded_device = dev
        return True

    def unload(self) -> None:
        """Release model weights and reclaim CUDA VRAM (critical for 6GB GPUs)."""
        import gc

        self._model = None
        self._processor = None
        self._loaded_device = None
        self._storyboard_cache.clear()
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        try:
            import torch  # type: ignore

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_candidate_storyboard(
        self,
        vpath: Path,
        *,
        tile_size: tuple[int, int],
        clip_window: tuple[float, float] | None = None,
        frame_ledger: Sequence[tuple[int, float | None]] | None = None,
        focus_intervals: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        from PIL import Image  # type: ignore

        cache_key = (
            str(vpath),
            (int(tile_size[0]), int(tile_size[1])),
            (float(clip_window[0]), float(clip_window[1])) if clip_window is not None else None,
            tuple((float(a), float(b)) for a, b in focus_intervals) if focus_intervals else None,
            int(self.num_frames),
            bool(self.annotate_timestamps),
            len(frame_ledger) if frame_ledger is not None else None,
        )
        cached = self._storyboard_cache.get(cache_key)
        if cached is not None:
            jb, meta = cached
            return Image.open(io.BytesIO(jb)).convert("RGB"), dict(meta)

        jb, meta = build_chronological_storyboard_with_meta(
            vpath,
            num_frames=self.num_frames,
            tile_size=tile_size,
            clip_start_sec=clip_window[0] if clip_window is not None else None,
            clip_end_sec=clip_window[1] if clip_window is not None else None,
            frame_ledger=frame_ledger,
            focus_intervals=focus_intervals,
            annotate_timestamps=self.annotate_timestamps,
        )
        self._storyboard_cache[cache_key] = (jb, dict(meta))
        return Image.open(io.BytesIO(jb)).convert("RGB"), meta

    def _score_images(self, sub_imgs: list[Any], sub_prompts: list[str]) -> list[float]:
        import torch  # type: ignore

        dev = self._loaded_device or "cpu"
        inputs = self._processor(text=sub_prompts, images=sub_imgs, padding=True, return_tensors="pt").to(dev)
        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes_lp = torch.logsumexp(lp[:, self._yes_ids], dim=-1)
            no_lp = torch.logsumexp(lp[:, self._no_ids], dim=-1)
            return (yes_lp - no_lp).cpu().numpy().astype(float).tolist()

    def score_query_candidates(
        self,
        query_text: str,
        candidate_video_ids: Sequence[str],
        video_paths_by_id: Mapping[str, Path],
        *,
        query_id: str | None = None,
        clip_windows_by_id: Mapping[str, tuple[float, float]] | None = None,
        frame_ledgers_by_id: Mapping[str, Sequence[tuple[int, float | None]]] | None = None,
        focus_intervals_by_id: Mapping[str, Sequence[tuple[float, float]]] | None = None,
        storyboard_sha256_by_id: Mapping[str, str] | None = None,
    ) -> dict[str, float]:
        """Compute Yes-vs-No log-odds per pair (pair provenance cache-first, incremental live scoring + OOM fallback)."""
        qsha = _sha256_text(query_text)
        pre_cached = (
            self.lookup_cached_scores(
                query_text,
                query_id=query_id,
                candidate_video_ids=candidate_video_ids,
                storyboard_sha256_by_vid=storyboard_sha256_by_id,
            )
            or {}
        )

        scores_out: dict[str, float] = {}
        missing_vids: list[str] = []
        cache_hits = 0
        for raw_vid in candidate_video_ids:
            vid = str(raw_vid)
            vpath = video_paths_by_id.get(vid)
            cfg_key = self._pair_config_key(qsha, vid)
            cfg_entries = self._pair_config_entries.get(cfg_key)

            # Fast path when no local clip overrides or when legacy cache satisfies CPU precomputed evaluation
            if vid in pre_cached and (
                vpath is None
                or not Path(vpath).is_file()
                or storyboard_sha256_by_id is not None
                or not cfg_entries
            ):
                scores_out[vid] = float(pre_cached[vid])
                cache_hits += 1
                continue

            # If local video file exists AND strict pair cache exists, validate storyboard hash match
            if vpath is not None and Path(vpath).is_file() and cfg_entries:
                try:
                    _, sb_meta = self._build_candidate_storyboard(
                        Path(vpath),
                        tile_size=self.tile_size,
                        clip_window=clip_windows_by_id.get(vid) if clip_windows_by_id else None,
                        frame_ledger=frame_ledgers_by_id.get(vid) if frame_ledgers_by_id else None,
                        focus_intervals=focus_intervals_by_id.get(vid) if focus_intervals_by_id else None,
                    )
                    sb_sha = str(sb_meta["storyboard_sha256"])
                    if sb_sha in cfg_entries:
                        scores_out[vid] = float(cfg_entries[sb_sha])
                        cache_hits += 1
                        continue
                except Exception:
                    pass

            missing_vids.append(vid)

        stats: dict[str, Any] = {
            "cache_hits": cache_hits,
            "cache_misses": len(missing_vids),
            "live_scored": 0,
            "oom_shrunk": 0,
            "oom_fallback": 0,
        }
        self.last_score_stats = stats
        if not missing_vids or not self.enabled or not self._ensure_loaded():
            return scores_out

        prepared: list[tuple[str, Path, Any, str]] = []
        for vid in missing_vids:
            vpath = video_paths_by_id.get(vid)
            if vpath is None or not Path(vpath).is_file():
                continue
            img, sb_meta = self._build_candidate_storyboard(
                Path(vpath),
                tile_size=self.tile_size,
                clip_window=clip_windows_by_id.get(vid) if clip_windows_by_id else None,
                frame_ledger=frame_ledgers_by_id.get(vid) if frame_ledgers_by_id else None,
                focus_intervals=focus_intervals_by_id.get(vid) if focus_intervals_by_id else None,
            )
            sb_sha = str(sb_meta["storyboard_sha256"])
            cfg_key = self._pair_config_key(qsha, vid)
            if sb_sha in self._pair_config_entries.get(cfg_key, {}):
                scores_out[vid] = float(self._pair_config_entries[cfg_key][sb_sha])
                stats["cache_hits"] += 1
                stats["cache_misses"] = max(0, int(stats["cache_misses"]) - 1)
                continue
            prepared.append((vid, Path(vpath), img, sb_sha))

        if not prepared:
            return scores_out

        msg_txt = self.query_template.format(query_text=query_text, num_frames=self.num_frames)
        prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msg_txt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        def _store_score(vid: str, sb_sha: str, sc_val: float) -> None:
            sc_f = float(sc_val)
            scores_out[vid] = sc_f
            cfg_key = self._pair_config_key(qsha, vid)
            self._pair_cache[cfg_key + (sb_sha,)] = sc_f
            self._pair_config_entries[cfg_key][sb_sha] = sc_f
            self._append_pair_cache_record(
                query_id=query_id,
                query_sha256=qsha,
                video_id=vid,
                storyboard_sha256=sb_sha,
                score=sc_f,
            )

        def _retry_shrunk(vid: str, vpath: Path) -> None:
            try:
                shrunk_img, shrunk_meta = self._build_candidate_storyboard(
                    vpath,
                    tile_size=self.shrunk_tile_size,
                    clip_window=clip_windows_by_id.get(vid) if clip_windows_by_id else None,
                    frame_ledger=frame_ledgers_by_id.get(vid) if frame_ledgers_by_id else None,
                    focus_intervals=focus_intervals_by_id.get(vid) if focus_intervals_by_id else None,
                )
                sc_shrunk = self._score_images([shrunk_img], [prompt])[0]
                stats["oom_shrunk"] += 1
                stats["live_scored"] += 1
                _store_score(vid, str(shrunk_meta["storyboard_sha256"]), sc_shrunk)
            except Exception as exc2:
                if self._is_oom_error(exc2):
                    self._clear_cuda_cache()
                    stats["oom_fallback"] += 1
                else:
                    raise

        bs = self._resolve_batch_size()
        for st in range(0, len(prepared), bs):
            chunk = prepared[st : st + bs]
            sub_imgs = [item[2] for item in chunk]
            try:
                batch_scores = self._score_images(sub_imgs, [prompt] * len(chunk))
                for (vid, _vp, _img, sb_sha), sc in zip(chunk, batch_scores):
                    stats["live_scored"] += 1
                    _store_score(vid, sb_sha, sc)
            except Exception as exc:
                if not self._is_oom_error(exc):
                    raise
                self._clear_cuda_cache()
                if len(chunk) > 1:
                    for vid, vpath, img, sb_sha in chunk:
                        try:
                            sc_single = self._score_images([img], [prompt])[0]
                            stats["live_scored"] += 1
                            _store_score(vid, sb_sha, sc_single)
                        except Exception as exc_single:
                            if not self._is_oom_error(exc_single):
                                raise
                            self._clear_cuda_cache()
                            _retry_shrunk(vid, vpath)
                else:
                    vid, vpath, _img, _sb_sha = chunk[0]
                    _retry_shrunk(vid, vpath)

        return scores_out

    def rerank_video_scores(
        self,
        query_text: str,
        ordered_video_scores: Sequence[tuple[str, float]],
        video_paths_by_id: Mapping[str, Path],
        *,
        query_id: str | None = None,
        clip_windows_by_id: Mapping[str, tuple[float, float]] | None = None,
        frame_ledgers_by_id: Mapping[str, Sequence[tuple[int, float | None]]] | None = None,
        focus_intervals_by_id: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Apply margin-gated VLM reranking over ranked `(video_id, stage1_score)` pairs."""
        if not self.enabled or len(ordered_video_scores) < 2 or float(self.fusion_config.w_vlm) == 0.0:
            return [str(v) for v, _ in ordered_video_scores], {"applied": False, "reason": "disabled_or_insufficient"}

        cfg = self.fusion_config
        top0 = float(ordered_video_scores[0][1])
        top1 = float(ordered_video_scores[1][1])
        margin = top0 - top1
        if margin >= float(cfg.margin_gate):
            return [str(v) for v, _ in ordered_video_scores], {
                "applied": False,
                "reason": "margin_confident",
                "margin": round(margin, 6),
                "margin_gate": float(cfg.margin_gate),
            }

        k_lim = max(2, int(cfg.top_k_rerank))
        cand_vids = [str(v) for v, _ in ordered_video_scores[:k_lim]]
        raw_vlm = self.score_query_candidates(
            query_text,
            cand_vids,
            video_paths_by_id,
            query_id=query_id,
            clip_windows_by_id=clip_windows_by_id,
            frame_ledgers_by_id=frame_ledgers_by_id,
            focus_intervals_by_id=focus_intervals_by_id,
        )
        score_stats = dict(self.last_score_stats)
        if len(raw_vlm) < 2:
            return [str(v) for v, _ in ordered_video_scores], {
                "applied": False,
                "reason": "oom_fallback" if score_stats.get("oom_fallback", 0) > 0 else "insufficient_vlm_scores",
                "scored_count": len(raw_vlm),
                **score_stats,
            }

        preds_in = [{"query_id": str(query_id or "q0"), "gallery": "runtime", "ranked_video_ids": [str(v) for v, _ in ordered_video_scores]}]
        top20 = {str(query_id or "q0"): {str(v): float(s) for v, s in ordered_video_scores[:20]}}
        vlm_in = {str(query_id or "q0"): raw_vlm}
        blended = blend_stage1_and_vlm_predictions(
            preds_in,
            top20,
            vlm_in,
            margin_gate=cfg.margin_gate,
            w_vlm=cfg.w_vlm,
            top_k_rerank=k_lim,
            mode=cfg.mode,
            debias_alpha=0.0,
        )
        new_order = blended[0]["ranked_video_ids"]
        return new_order, {
            "applied": True,
            "model_id": self.model_id,
            "quant": self.quant,
            "margin": round(margin, 6),
            "scored_count": len(raw_vlm),
            "top_k_rerank": k_lim,
            "blended_scores_by_vid": dict(blended[0].get("blended_scores_by_vid") or {}),
            **score_stats,
        }
