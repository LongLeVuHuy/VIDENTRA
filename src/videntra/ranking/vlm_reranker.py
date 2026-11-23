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
    'These 8 chronological frames are sampled from a video clip. '
    'Does this video match the description: "{query_text}"? Answer strictly Yes or No.'
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


def build_chronological_storyboard_jpeg(
    video_path: Path,
    *,
    num_frames: int = 8,
    tile_size: tuple[int, int] = (224, 168),
    jpeg_quality: int = 88,
) -> bytes:
    """Decode presentation-order frames via PyAV and pack `num_frames` PTS bin-centers into a 2x4 JPEG."""
    import av  # type: ignore
    from PIL import Image  # type: ignore

    pr = probe_media(video_path)
    records: list[tuple[int, float | None]] = []
    valid_ts: list[float] = []
    raw_frames: list[Any] = []
    with av.open(str(video_path)) as container:
        stream = next(s for s in container.streams if s.index == pr.selected_video_stream_index)
        tb = float(stream.time_base) if stream.time_base is not None else 0.0
        for cur_ord, frame in enumerate(container.decode(stream)):
            ts = float(frame.pts * tb) if (frame.pts is not None and tb > 0.0) else None
            records.append((cur_ord, ts))
            if ts is not None:
                valid_ts.append(ts)
            raw_frames.append(frame)

    if not valid_ts:
        raise ValueError(f"Video lacks mapped PTS timestamps: {video_path}")
    ords = select_bin_center_ordinals(records, min(valid_ts), max(valid_ts) + 1e-3, num_frames=num_frames)
    if not ords:
        raise ValueError(f"Unable to select chronological storyboard frames for: {video_path}")

    cols = 4
    rows = max(1, (len(ords) + cols - 1) // cols)
    tw, th = int(tile_size[0]), int(tile_size[1])
    canvas = Image.new("RGB", (cols * tw, rows * th))
    for i_o, ord_idx in enumerate(ords):
        tile = Image.fromarray(raw_frames[ord_idx].to_ndarray(format="rgb24"), mode="RGB").resize(
            (tw, th), Image.BILINEAR
        )
        canvas.paste(tile, ((i_o % cols) * tw, (i_o // cols) * th))
    raw_frames.clear()

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=int(jpeg_quality))
    return buf.getvalue()


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
        reordered = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-comb[i], i))]
        pool_set = set(reordered)
        new_ranked = reordered + [v for v in ranked if v not in pool_set]
        out.append({"query_id": qid, "gallery": p["gallery"], "ranked_video_ids": new_ranked})
    return out


class Qwen25VLReranker:
    """6GB-VRAM-compatible Margin-Gated Qwen2.5-VL cross-encoder reranker (default: 3B NF4)."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
        quant: str | None = None,
        enabled: bool | None = None,
        score_cache_path: Path | str | None = None,
        fusion_config: VLMFusionConfig | None = None,
        batch_size: int | None = None,
        device: str = "auto",
    ) -> None:
        env_model = os.environ.get("VIDENTRA_VLM_MODEL_ID", "").strip()
        self.model_id = str(model_id or env_model or VLM_3B_MODEL_ID)
        pinned = PINNED_VLM_REVISIONS.get(self.model_id, {})
        self.model_revision = model_revision if model_revision is not None else pinned.get("revision")

        env_quant = os.environ.get("VIDENTRA_VLM_QUANT", "").strip().lower()
        self.quant = str(quant or env_quant or "nf4_4bit")
        self.device_pref = str(device).strip().lower() or "auto"
        self.fusion_config = fusion_config or VLMFusionConfig()

        env_cache = os.environ.get("VIDENTRA_VLM_CACHE_PATH", "").strip()
        resolved_cache = score_cache_path if score_cache_path is not None else (env_cache or None)
        self.score_cache_path: Path | None = Path(resolved_cache) if resolved_cache else None
        self._cached_scores_by_qid: dict[str, dict[str, float]] = {}
        self._cached_scores_by_sha: dict[str, dict[str, float]] = {}
        if self.score_cache_path is not None and self.score_cache_path.is_file():
            self._load_score_cache(self.score_cache_path)

        if enabled is not None:
            self.enabled = bool(enabled)
        else:
            env_en = os.environ.get("VIDENTRA_VLM_RERANK_ENABLED", "").strip().lower()
            if env_en in ("1", "true", "yes", "on"):
                self.enabled = True
            elif env_en in ("0", "false", "no", "off"):
                self.enabled = False
            elif env_en == "auto":
                self.enabled = self._has_cuda() or bool(self._cached_scores_by_qid or self._cached_scores_by_sha)
            else:
                self.enabled = bool(self._cached_scores_by_qid or self._cached_scores_by_sha)

        self.batch_size = int(batch_size) if batch_size is not None else None
        self._processor: Any = None
        self._model: Any = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []
        self._loaded_device: str | None = None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _load_score_cache(self, path: Path) -> None:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as stream:  # type: ignore[arg-type]
            for line in stream:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                cids = row.get("candidate_video_ids") or []
                scores = row.get("vlm_scores") or []
                if not cids or len(cids) != len(scores):
                    continue
                score_map = {str(v): float(sc) for v, sc in zip(cids, scores)}
                qid = str(row.get("query_id") or "").strip()
                if qid:
                    self._cached_scores_by_qid[qid] = score_map
                qsha = str(row.get("query_text_sha256") or "").strip()
                if qsha:
                    self._cached_scores_by_sha[qsha] = score_map

    def lookup_cached_scores(
        self,
        query_text: str,
        *,
        query_id: str | None = None,
    ) -> dict[str, float] | None:
        """Return pre-scored VLM map if cached locally (zero GPU/VRAM cost)."""
        if query_id and str(query_id) in self._cached_scores_by_qid:
            return dict(self._cached_scores_by_qid[str(query_id)])
        qsha = _sha256_text(query_text)
        if qsha in self._cached_scores_by_sha:
            return dict(self._cached_scores_by_sha[qsha])
        return None

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
        self._model = None
        self._processor = None
        self._loaded_device = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def score_query_candidates(
        self,
        query_text: str,
        candidate_video_ids: Sequence[str],
        video_paths_by_id: Mapping[str, Path],
        *,
        query_id: str | None = None,
    ) -> dict[str, float]:
        """Compute Yes-vs-No log-odds for candidate videos (cache-first, live GPU fallback)."""
        cached = self.lookup_cached_scores(query_text, query_id=query_id)
        if cached is not None:
            return {str(v): cached[str(v)] for v in candidate_video_ids if str(v) in cached}

        if not self.enabled or not self._ensure_loaded():
            return {}

        from PIL import Image  # type: ignore
        import torch  # type: ignore

        imgs: list[Any] = []
        valid_vids: list[str] = []
        for vid in candidate_video_ids:
            vpath = video_paths_by_id.get(str(vid))
            if vpath is None or not Path(vpath).is_file():
                continue
            jb = build_chronological_storyboard_jpeg(Path(vpath))
            imgs.append(Image.open(io.BytesIO(jb)).convert("RGB"))
            valid_vids.append(str(vid))
        if not valid_vids:
            return {}

        msg_txt = DEFAULT_VLM_QUERY_TEMPLATE.format(query_text=query_text)
        prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msg_txt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        bs = self._resolve_batch_size()
        dev = self._loaded_device or "cpu"

        def _forward(sub_imgs: list[Any], sub_prompts: list[str]) -> list[float]:
            inputs = self._processor(text=sub_prompts, images=sub_imgs, padding=True, return_tensors="pt").to(dev)
            with torch.inference_mode():
                logits = self._model(**inputs).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                yes_lp = torch.logsumexp(lp[:, self._yes_ids], dim=-1)
                no_lp = torch.logsumexp(lp[:, self._no_ids], dim=-1)
                return (yes_lp - no_lp).cpu().numpy().astype(float).tolist()

        scores: list[float] = []
        for st in range(0, len(imgs), bs):
            try:
                scores.extend(_forward(imgs[st : st + bs], [prompt] * min(bs, len(imgs) - st)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for idx in range(st, min(len(imgs), st + bs)):
                    scores.extend(_forward([imgs[idx]], [prompt]))
        return {vid: float(sc) for vid, sc in zip(valid_vids, scores)}

    def rerank_video_scores(
        self,
        query_text: str,
        ordered_video_scores: Sequence[tuple[str, float]],
        video_paths_by_id: Mapping[str, Path],
        *,
        query_id: str | None = None,
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
            query_text, cand_vids, video_paths_by_id, query_id=query_id
        )
        if len(raw_vlm) < 2:
            return [str(v) for v, _ in ordered_video_scores], {
                "applied": False,
                "reason": "insufficient_vlm_scores",
                "scored_count": len(raw_vlm),
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
        }
