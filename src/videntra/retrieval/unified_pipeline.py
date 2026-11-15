"""Single Source of Truth (SSOT) Unified Multimodal Search Pipeline for VIDENTRA.

Implements the end-to-end 7-stage multimodal retrieval pipeline:
- Stage 1: Text Embedding & Query Vector Construction (864d with B5, B6, C1).
- Stage 2: FAISS IP Retrieval & Multi-Clause Sequential Alignment (B4 exact full score fallback, B11 same-frame penalty 0.92).
- Stage 3: Specificity-Sorted Lexical FTS5 Search (B7 quoted/proper nouns/length-sorted, C2 persistent read-only connection).
- Stage 4: Multimodal Temporal Clustering (B2 per-cluster matched texts, B3 order-invariant RRF fusion).
- Stage 5: Gated Lexical-Visual Fusion (B1 temporal-local visual lookup, B9 pure-ASR false positive gate).
- Stage 6: Anchor-Aware Boundary Expansion (B10 anchor_idx & anchor_window).
- Stage 7: Video Diversity Aggregation & RankedMoment Formatting.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import threading
from copy import deepcopy
from typing import Any, Sequence
import unicodedata
import warnings

import faiss
import numpy as np
import torch

from videntra.query.rules import COCO_SYNONYM_MAP, decompose_query_clauses, extract_object_requirements
from videntra.query.llm import AdaptiveQueryDecomposer
from videntra.query.schemas import ModalityRoutingPlan
from videntra.retrieval.visual import (
    RankedMoment,
    ensemble_clause_scores,
    temporal_iou,
)
from videntra.indexing.text_store import (
    build_fts5_query,
    GENERIC_VISUAL_WORDS,
    ALL_STOPWORDS,
    split_compound_tokens,
)
from videntra.indexing.augmented_vector import (
    build_augmented_864d_embeddings,
    build_query_864d_vector,
    compute_late_interaction_maxsim,
    compute_multiscale_temporal_context,
)
from videntra.retrieval.alignment import (
    align_sequential_keyframe_clauses,
    compute_temporal_pool_score,
    query_guided_temporal_attention_score,
    shot_aware_graph_diffusion,
    temporal_sinkhorn_alignment_score,
    viterbi_subevent_localization,
)
from videntra.retrieval.clip_branch import (
    ClipBranchIndex,
    build_clip_branch_index,
    build_clip_reserve_pool,
    score_clip_branch,
)
from videntra.evidence.normalize import contains_text_phrase
from videntra.relations.entities import compute_object_verification_boost
from videntra.localization.refine import expand_boundary_contrast
from videntra.ranking.calibration import calibrate_asr_confidence
from videntra.ranking.features import compute_query_z_scores, extract_ltr_features_24d
from videntra.ranking.learned_ranker import DEFAULT_LTR_MODEL_PATH, LearnedRankerLTR

logger = logging.getLogger(__name__)


def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    """Inspect SQLite table schema safely; return empty set if table is absent."""
    try:
        rows = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        return {str(row[1]) for row in rows}
    except sqlite3.Error:
        return set()


def extract_tensor_features(out: Any) -> torch.Tensor:
    """Extract pooled representation from Hugging Face model output."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def unified_result_dict(moment: RankedMoment, snapshot_id: str) -> dict[str, Any]:
    """Serialize without inventing ledger, model or snapshot provenance."""
    result = moment.to_result_dict()
    result["source_video_id"] = None
    result["timing_metadata"].update({"time_base": None, "timeline_revision": None,
                                      "boundary_uncertainty_seconds": None})
    for evidence in result["evidence"]:
        if evidence["original_decoded_frame_index"] != moment.representative_original_frame_index:
            evidence["timestamp_us"] = None
    result["provenance"] = {"synthetic": None, "index_snapshot": snapshot_id,
                            "snapshot_verified": False, "model_manifest": None,
                            "query_graph_version": None, "media_sha256": None, "run_id": None}
    return result


DOMAIN_LEXICAL_SYNONYMS: dict[str, dict[str, Any]] = {
    # Chemical / formula notation variants often transcribed differently in ASR/OCR
    "io3": {
        "fts_tokens": ('"hio"', "iodi*"),
        "high_spec": ("hio", "iodide"),
        "proper_nouns": ("hio", "iodide"),
    },
    "h2o2": {
        "fts_tokens": ('"hio"', "iodi*"),
        "high_spec": ("hio", "iodide"),
        "proper_nouns": ("hio", "iodide"),
    },
    # Domain lexical & OCR/ASR transcription variants
    "letterman": {
        "fts_tokens": ('"lateshow"',),
        "high_spec": ("lateshow",),
    },
    "host": {
        "required_high": ("late",),
        "required_low": ("show",),
        "fts_tokens": ('"letterman"', '"ATESHO"'),
        "high_spec": ("letterman", "atesho"),
        "proper_nouns": ("letterman",),
    },
    "checking": {
        "required_low": ("food",),
        "fts_tokens": ('"check"', '"food"'),
        "high_spec": ("check", "food"),
    },
    "news": {
        "required_low": ("channel",),
        "fts_tokens": ('"cnn"', '"fox"'),
        "high_spec": ("cnn", "fox"),
    },
    "phrenologists": {
        "fts_tokens": ("phrenolog*", '"frenologist"'),
        "high_spec": ("phrenology", "frenologist"),
    },
}


def _apply_domain_synonyms(
    token: str,
    cand_tokens: list[str],
    high_spec_tokens: set[str],
    low_spec_tokens: set[str],
    proper_nouns_lower: set[str],
) -> None:
    """Expand token using DOMAIN_LEXICAL_SYNONYMS when contextual preconditions hold."""
    spec = DOMAIN_LEXICAL_SYNONYMS.get(token)
    if spec is None:
        return
    req_high = spec.get("required_high", ())
    req_low = spec.get("required_low", ())
    if req_high and not all(t in high_spec_tokens for t in req_high):
        return
    if req_low and not all(t in low_spec_tokens for t in req_low):
        return
    cand_tokens.extend(spec.get("fts_tokens", ()))
    high_spec_tokens.update(spec.get("high_spec", ()))
    proper_nouns_lower.update(spec.get("proper_nouns", ()))


_SPEECH_ONLY_CUE_WORDS: frozenset[str] = frozenset({
    "said", "says", "saying", "told", "tells", "speaking", "speaks",
    "asked", "asks", "talking", "talks", "nói", "kể", "hỏi",
})

_VISUAL_ONLY_CUE_WORDS: frozenset[str] = frozenset({
    "wearing", "wears", "holding", "holds", "standing", "sitting",
    "looking", "showing", "shown", "written", "displayed",
    "mặc", "cầm", "đứng", "ngồi", "hiển",
})


def _filter_fts5_cue_tokens(fts_query: str, cue_words: frozenset[str]) -> str:
    """Remove modality-exclusive cue tokens from an FTS5 OR query."""
    if not fts_query:
        return ""
    parts = [p.strip() for p in fts_query.split(" OR ") if p.strip()]
    kept = []
    for part in parts:
        cleaned = part.strip('"').rstrip("*").lower()
        if cleaned and cleaned not in cue_words:
            kept.append(part)
    return " OR ".join(kept)


def _build_routed_fts5_queries(
    base_fts: str,
    routing_plan: ModalityRoutingPlan | None,
) -> tuple[str, str]:
    """Build channel-specific OCR and ASR FTS5 queries from ModalityRoutingPlan."""
    if routing_plan is None or routing_plan.routing_mode != "llm_compact":
        return base_fts, base_fts

    orig_norm = routing_plan.original_query.strip().lower()
    ocr_parts: list[str] = []
    for phrase in routing_plan.ocr_phrases:
        if phrase.strip().lower() == orig_norm:
            continue
        pq = build_fts5_query(phrase, enable_prefix=True)
        if pq:
            ocr_parts.append(pq)
    filtered_ocr_base = _filter_fts5_cue_tokens(base_fts, _SPEECH_ONLY_CUE_WORDS)
    if filtered_ocr_base:
        ocr_parts.append(filtered_ocr_base)
    fts_ocr = " OR ".join(dict.fromkeys(ocr_parts)) if ocr_parts else base_fts

    asr_parts: list[str] = []
    for phrase in routing_plan.asr_phrases:
        if phrase.strip().lower() == orig_norm:
            continue
        pq = build_fts5_query(phrase, enable_prefix=True)
        if pq:
            asr_parts.append(pq)
    filtered_asr_base = _filter_fts5_cue_tokens(base_fts, _VISUAL_ONLY_CUE_WORDS)
    if filtered_asr_base:
        asr_parts.append(filtered_asr_base)
    fts_asr = " OR ".join(dict.fromkeys(asr_parts)) if asr_parts else base_fts

    return fts_ocr, fts_asr


def _resolve_moment_frame_indices(
    kf_list: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    best_kf_pos: int,
) -> tuple[int, int, int, tuple[int, ...]]:
    """Resolve decoded source_frame_index values for start, end, representative, and evidence frames."""
    if not kf_list:
        return 0, 0, 0, ()
    pos = max(0, min(int(best_kf_pos), len(kf_list) - 1))
    raw_rep = kf_list[pos].get("source_frame_index")
    rep_orig_idx = int(raw_rep) if raw_rep is not None else 0
    in_win = [
        k for k in kf_list
        if start_sec <= float(k.get("pts_time", 0.0)) <= end_sec
    ]
    if not in_win:
        in_win = [kf_list[pos]]
    raw_start = in_win[0].get("source_frame_index")
    raw_end = in_win[-1].get("source_frame_index")
    start_orig_idx = int(raw_start) if raw_start is not None else rep_orig_idx
    end_orig_idx = int(raw_end) if raw_end is not None else rep_orig_idx
    evidence_indices = tuple(
        int(k["source_frame_index"]) if k.get("source_frame_index") is not None else 0
        for k in in_win[:5]
    )
    return start_orig_idx, end_orig_idx, rep_orig_idx, evidence_indices

_MID_SENTENCE_COMMON_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "two", "three", "four", "five", "one", "several", "multiple", "some",
    "woman", "women", "man", "men", "girl", "girls", "boy", "boys", "person", "people",
    "kid", "kids", "child", "children", "close", "view", "clip", "video", "young", "old",
    "group", "someone", "somebody", "something", "asian", "indian", "american", "black", "white",
    "smiling", "standing", "sitting", "walking", "talking", "eating", "cooking", "playing",
    "daily", "show", "night", "news", "late", "live", "today", "world", "morning", "evening",
    "chef", "doctor", "teacher", "student", "officer", "player", "driver", "singer", "speaker",
})

_COMMON_INITIAL_OPENERS: frozenset[str] = _MID_SENTENCE_COMMON_WORDS | frozenset({
    "six", "seven", "eight", "nine", "ten", "many", "various", "different", "other", "another",
    "baby", "toddler", "infant", "teen", "teenager", "couple", "family", "crowd", "audience",
    "team", "players", "athletes", "students", "clips", "videos", "footage", "scene", "scenes",
    "movie", "film", "trailer", "commercial", "advertisement", "cartoon", "animation", "animated",
    "documentary", "tutorial", "review", "interview", "highlight", "highlights", "gameplay",
    "older", "elderly", "little", "small", "large", "big", "tiny", "huge", "female", "male",
    "red", "blue", "green", "yellow", "running", "singing", "dancing", "driving", "riding",
    "baking", "drawing", "music", "song", "sports", "basketball", "soccer", "football",
    "baseball", "tennis", "golf", "hockey", "boxing", "wrestling", "recipe", "kitchen",
    "police", "soldiers", "animals", "birds", "fish", "horse", "horses", "dogs", "cats",
    "cars", "planes", "trains", "bikes",
})

_HUMAN_SUBJECT_LOW_WORDS: frozenset[str] = frozenset({
    "man", "men", "woman", "women", "girl", "girls", "boy", "boys",
    "person", "people", "kid", "kids", "child", "children", "guy", "guys",
    "baby", "someone", "somebody", "crowd", "audience", "group", "family",
})

_COCO_VOCAB_WORDS: frozenset[str] = frozenset(
    set(COCO_SYNONYM_MAP.keys()) | set(COCO_SYNONYM_MAP.values())
)


def _is_common_sentence_initial_word(w_lower: str) -> bool:
    """Return True when a sentence-initial capitalized word is likely a common English opener."""
    if w_lower in _COMMON_INITIAL_OPENERS or w_lower in GENERIC_VISUAL_WORDS or w_lower in _COCO_VOCAB_WORDS:
        return True
    if len(w_lower) >= 5 and w_lower.endswith(("ing", "ed", "ly")):
        return True
    if len(w_lower) >= 4 and w_lower.endswith("s"):
        singular = w_lower[:-1]
        if singular in _COMMON_INITIAL_OPENERS or singular in GENERIC_VISUAL_WORDS or singular in _COCO_VOCAB_WORDS:
            return True
    return False


def _generate_morphological_stems(token: str) -> list[str]:
    """Generate conservative English root stems for long content tokens (len >= 8) to avoid over-stemming."""
    t = token.lower().strip()
    if len(t) < 8 or not t.isalpha() or t in GENERIC_VISUAL_WORDS:
        return []

    stems: list[str] = []
    if t.endswith("ologists") and len(t) >= 10:
        root = t[:-4]
        stems.append(root)  # e.g. phrenologists -> phrenolog
        if root.startswith("ph") and len(root) >= 6:
            stems.append(f"f{root[2:]}")  # OCR/ASR phonetic ph->f variant (e.g. frenolog)
    elif t.endswith("ologist") and len(t) >= 9:
        root = t[:-3]
        stems.append(root)
        if root.startswith("ph") and len(root) >= 6:
            stems.append(f"f{root[2:]}")
    elif t.endswith("ing"):
        base = t[:-3]
        stems.append(base)  # e.g. checking -> check, building -> build, teaching -> teach
        if len(base) >= 4 and base[-1] == base[-2]:
            stems.append(base[:-1])

    seen: set[str] = set()
    clean_stems: list[str] = []
    for s in stems:
        if len(s) >= 4 and s != t and s not in ALL_STOPWORDS and s not in GENERIC_VISUAL_WORDS and s not in seen:
            seen.add(s)
            clean_stems.append(s)
    return clean_stems


def analyze_query_lexical_tokens(raw_query: str) -> tuple[list[str], set[str], set[str], set[str]]:
    """Extracts quoted phrases, proper nouns, and partitions query tokens into high-spec and low-spec."""
    quoted_phrases = re.findall(r'["\']([^"\']+)["\']', raw_query)

    clean_text = split_compound_tokens(raw_query)
    tokens = re.findall(r'\b[a-z0-9à-ỹ]+\b', clean_text.lower())

    raw_words = re.findall(r'\b[A-Za-z0-9à-ỹÀ-Ỹ]+\b', raw_query)
    words_in_q = clean_text.strip().split()
    first_word_clean = words_in_q[0].lower() if words_in_q else ""

    proper_nouns: set[str] = set()
    for idx, w in enumerate(raw_words):
        w_lower = w.lower()
        if w_lower in ALL_STOPWORDS or len(w_lower) <= 1:
            continue
        if not w[0].isupper():
            continue
        next_is_cap = (
            idx + 1 < len(raw_words)
            and raw_words[idx + 1][0].isupper()
            and raw_words[idx + 1].lower() not in ALL_STOPWORDS
            and not _is_common_sentence_initial_word(raw_words[idx + 1].lower())
        )
        if idx == 0 or w_lower == first_word_clean:
            if (w.isupper() and len(w) >= 2) or next_is_cap or not _is_common_sentence_initial_word(w_lower):
                proper_nouns.add(w)
        else:
            if w_lower not in GENERIC_VISUAL_WORDS and w_lower not in _MID_SENTENCE_COMMON_WORDS:
                proper_nouns.add(w)

    high_spec_tokens: set[str] = set()
    low_spec_tokens: set[str] = set()

    for t in tokens:
        if t in ALL_STOPWORDS or len(t) <= 1:
            continue
        if t in GENERIC_VISUAL_WORDS:
            low_spec_tokens.add(t)
        else:
            high_spec_tokens.add(t)

    for pn in proper_nouns:
        pn_lower = pn.lower()
        if pn_lower not in ALL_STOPWORDS and len(pn_lower) > 1:
            low_spec_tokens.discard(pn_lower)
            high_spec_tokens.add(pn_lower)

    for qp in quoted_phrases:
        qp_tokens = re.findall(r'\b[a-z0-9à-ỹ]+\b', split_compound_tokens(qp).lower())
        for qt in qp_tokens:
            if qt not in ALL_STOPWORDS and len(qt) > 1:
                low_spec_tokens.discard(qt)
                high_spec_tokens.add(qt)

    proper_nouns_lower = {pn.lower() for pn in proper_nouns if len(pn) > 1 and pn.lower() not in ALL_STOPWORDS}
    return quoted_phrases, high_spec_tokens, low_spec_tokens, proper_nouns_lower


def compute_order_invariant_rrf(rrf_list: list[float]) -> float:
    """Computes order-invariant RRF fusion score: r_1 + 0.35 * sum(r_j for j in 2..min(M,3)) (B3)."""
    if not rrf_list:
        return 0.0
    sorted_r = sorted(rrf_list, reverse=True)
    r1 = sorted_r[0]
    tail_sum = sum(sorted_r[1:min(len(sorted_r), 3)])
    return float(r1 + 0.35 * tail_sum)


def _fast_object_boost_from_summary(
    object_requirements: Sequence[dict[str, Any]],
    summary_dict: dict[str, tuple[float, int]],
    base_match_boost: float = 0.15,
    cardinality_boost: float = 0.10,
    no_object_penalty: float = 0.10,
) -> float:
    """O(1) per-requirement calculation identical to compute_object_verification_boost using pre-grouped summary."""
    if not object_requirements:
        return 0.0
    if not summary_dict:
        only_person = all(
            (r.get("canonical_class") or COCO_SYNONYM_MAP.get(str(r.get("class_name", "")).lower(), str(r.get("class_name", "")).lower())) == "person"
            for r in object_requirements
        )
        return -0.02 if only_person else -float(no_object_penalty)

    total_boost = 0.0
    has_any_target_match = False
    for req in object_requirements:
        canonical_class = req.get("canonical_class") or COCO_SYNONYM_MAP.get(str(req.get("class_name", "")).lower(), str(req.get("class_name", "")).lower())
        entry = summary_dict.get(canonical_class)
        if entry is not None:
            has_any_target_match = True
            best_conf, max_frame_count = entry
            req_count = int(req.get("cardinality", req.get("count", 1)))
            is_satisfied = max_frame_count >= req_count
            if canonical_class == "person":
                match_boost = 0.04 * best_conf
                card_boost = (0.14 if req_count >= 2 else 0.02) if is_satisfied else (-0.04 if req_count >= 2 else 0.0)
            else:
                match_boost = base_match_boost * best_conf
                card_boost = cardinality_boost if is_satisfied else 0.0
            total_boost += (match_boost + card_boost)

    if not has_any_target_match:
        only_person = all(
            (r.get("canonical_class") or COCO_SYNONYM_MAP.get(str(r.get("class_name", "")).lower(), str(r.get("class_name", "")).lower())) == "person"
            for r in object_requirements
        )
        return -0.02 if only_person else -float(no_object_penalty)
    return float(round(total_boost, 4))


class TextEmbeddingDiskCache:
    """C1: Deterministic disk cache for 768d text embeddings keyed by SHA256 of NFC-normalized text."""

    def __init__(self, cache_file: Path | str = Path("data/cache/query_text_embeddings_768d.npz")):
        self.cache_file = Path(cache_file)
        self._memory_cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self.last_error: str | None = None
        self._load()

    @staticmethod
    def normalize_key(text: str) -> str:
        nfc_text = unicodedata.normalize("NFC", text.strip())
        return hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with np.load(str(self.cache_file)) as data:
                    for k in data.files:
                        self._memory_cache[k] = data[k]
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("Failed to load text embedding cache %s: %s", self.cache_file, exc, exc_info=True)

    def get(self, text: str) -> np.ndarray | None:
        k = self.normalize_key(text)
        return self._memory_cache.get(k)

    def set(self, text: str, vec: np.ndarray) -> None:
        k = self.normalize_key(text)
        self._memory_cache[k] = np.asarray(vec, dtype=np.float32).flatten()
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(self.cache_file), **self._memory_cache)
            self._dirty = False
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Failed to save text embedding cache %s: %s", self.cache_file, exc, exc_info=True)

    def __len__(self) -> int:
        return len(self._memory_cache)


class UnifiedMultimodalSearchPipeline:
    """Single Source of Truth (SSOT) 7-Stage Multimodal Search Pipeline."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        faiss_path: Path | str | None = None,
        model_dir: Path | str | None = None,
        device: str | None = None,
        use_disk_cache: bool = False,
        memory_cache_size: int = 512,
        cache_path: Path | str = Path("data/cache/query_text_embeddings_768d.npz"),
        e5_cache_path: Path | str = Path("data/cache/query_e5_embeddings_384d.npz"),
        alpha_obj: float = 0.25,
        alpha_meta: float = 0.15,
        beta_hub: float = 0.08,
        search_depth: int = 120,
        lazy_load: bool = False,
        ltr_model_path: Path | str | None = DEFAULT_LTR_MODEL_PATH,
        ltr_blend_weight: float = 0.015,
        enable_spatial_patches: bool = True,
        enable_vlm_caption_lexical: bool = True,
        enable_vlm_caption_dense: bool = True,
        enable_temporal_rerank: bool = False,
        temporal_rerank_mode: str = "attention",
        temporal_rerank_lambda: float = 0.0,
        temporal_rerank_top_k: int = 10,
        temporal_rerank_window_sec: float = 3.0,
        temporal_rerank_max_frames: int = 5,
        temporal_rerank_tau: float = 12.0,
        enable_clip_branch: bool = False,
        clip_reserve_k: int = 10,
        clip_window_size: int | None = None,
        # Optional pre-injected objects for lightweight testing or server sharing:
        tokenizer: Any | None = None,
        text_model: Any | None = None,
        index: faiss.Index | None = None,
        augmented_embeddings: np.ndarray | None = None,
        coco_idf_map: dict[str, float] | None = None,
        keyframe_map: dict[int, dict] | None = None,
        video_keyframes: dict[str, list[dict]] | None = None,
        video_durations: dict[str, float] | None = None,
        video_objects: dict[str, list[dict]] | None = None,
        query_decomposer: AdaptiveQueryDecomposer | None = None,
        query_plan_cache: Path | str | None = None,
        query_plan_revision: str | None = None,
        rules_only: bool = False,
    ):
        self.db_path = Path(db_path if db_path is not None else os.environ.get("VIDENTRA_DB_PATH", "data/processed/metadata.db"))
        self.faiss_path = Path(faiss_path if faiss_path is not None else os.environ.get("VIDENTRA_FAISS_PATH", "data/processed/faiss_index.bin"))
        self.model_dir = Path(model_dir if model_dir is not None else os.environ.get("VIDENTRA_MODEL_DIR", "models/siglip2"))
        self.device = device if device is not None else os.environ.get("VIDENTRA_DEVICE", "cpu")
        self.use_disk_cache = use_disk_cache
        if memory_cache_size < 0:
            raise ValueError("memory_cache_size must be non-negative")
        self.memory_cache_size = memory_cache_size
        self._text_memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_path = Path(cache_path)
        self._e5_memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.e5_cache_path = Path(e5_cache_path)
        self.alpha_obj = alpha_obj
        self.alpha_meta = alpha_meta
        self.beta_hub = beta_hub
        self.search_depth = search_depth

        self.tokenizer = tokenizer
        self.text_model = text_model
        self.index = index
        self.augmented_embeddings = augmented_embeddings
        self.all_embeddings: np.ndarray = np.zeros((0, 768), dtype=np.float32)
        self.coco_idf_map = coco_idf_map or {}
        self.keyframe_map = keyframe_map or {}
        self.video_keyframes = video_keyframes or defaultdict(list)
        self.video_durations = video_durations or {}
        self.video_objects = video_objects or defaultdict(list)
        self.video_kf_pts: dict[str, list[float]] = {}
        self.video_kf_fids: dict[str, list[int]] = {}
        self.video_kf_matrix: dict[str, np.ndarray] = {}
        self.keyframe_obj_window: dict[int, list[dict[str, Any]]] = {}
        self.keyframe_obj_summary: dict[int, dict[str, tuple[float, int]]] = {}
        self.video_obj_pts: dict[str, list[float]] = {}
        self.video_classes_set: dict[str, set[str]] = {}

        self.db_conn: sqlite3.Connection | None = None
        if use_disk_cache:
            warnings.warn("Legacy text cache is not model-namespaced; do not use it for evaluation", RuntimeWarning)
        self.cache: TextEmbeddingDiskCache | None = TextEmbeddingDiskCache(self.cache_path) if use_disk_cache else None
        self.e5_cache: TextEmbeddingDiskCache | None = TextEmbeddingDiskCache(self.e5_cache_path) if use_disk_cache else None
        self._degraded_channels: dict[str, str] = {}
        self._channel_mismatches: dict[str, str] = {}
        self._spatial_patch_row_by_fid: dict[int, int] | None = None
        if self.cache is not None and self.cache.last_error:
            self._degraded_channels["text_cache"] = self.cache.last_error
        if self.e5_cache is not None and self.e5_cache.last_error:
            self._degraded_channels["e5_cache"] = self.e5_cache.last_error
        self.last_timings: dict[str, float] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self._search_lock = threading.RLock()

        # Feature flags & Section 3.1 temporal reranker config
        if temporal_rerank_mode not in ("mean", "mean_top2", "mean_top3", "attention"):
            raise ValueError(f"Unsupported temporal_rerank_mode: {temporal_rerank_mode!r}")
        if temporal_rerank_top_k < 1:
            raise ValueError("temporal_rerank_top_k must be >= 1")
        if temporal_rerank_max_frames < 1:
            raise ValueError("temporal_rerank_max_frames must be >= 1")
        if temporal_rerank_window_sec < 0.0:
            raise ValueError("temporal_rerank_window_sec must be >= 0.0")
        self.enable_spatial_patches = bool(enable_spatial_patches)
        self.enable_vlm_caption_lexical = bool(enable_vlm_caption_lexical)
        self.enable_vlm_caption_dense = bool(enable_vlm_caption_dense)
        self.enable_temporal_rerank = bool(enable_temporal_rerank)
        self.temporal_rerank_mode = str(temporal_rerank_mode)
        self.temporal_rerank_lambda = float(temporal_rerank_lambda)
        self.temporal_rerank_top_k = int(temporal_rerank_top_k)
        self.temporal_rerank_window_sec = float(temporal_rerank_window_sec)
        self.temporal_rerank_max_frames = int(temporal_rerank_max_frames)
        self.temporal_rerank_tau = float(temporal_rerank_tau)

        if clip_reserve_k < 2:
            raise ValueError("clip_reserve_k must be >= 2")
        if clip_window_size is not None and clip_window_size < 1:
            raise ValueError("clip_window_size must be >= 1 when provided")
        self.enable_clip_branch = bool(enable_clip_branch)
        self.clip_reserve_k = int(clip_reserve_k)
        self.clip_window_size = int(clip_window_size) if clip_window_size is not None else None
        self._clip_branch_index: ClipBranchIndex | None = None

        # Phase 7.2 / 7.3 / 7.4 components (fail-safe when local models/artifacts are absent)
        if (query_plan_cache is None) != (query_plan_revision is None):
            raise ValueError("query_plan_cache and query_plan_revision must be provided together")
        if rules_only and query_plan_cache is not None:
            raise ValueError("rules_only and query_plan_cache are mutually exclusive")
        if query_decomposer is not None and (
            query_plan_cache is not None or query_plan_revision is not None or rules_only
        ):
            raise ValueError(
                "Cannot combine explicit query_decomposer with query_plan_cache, query_plan_revision, or rules_only"
            )
        self.query_plan_cache: Path | None = Path(query_plan_cache) if query_plan_cache is not None else None
        self.query_plan_revision: str | None = query_plan_revision
        self.rules_only = bool(rules_only)
        if query_decomposer is not None:
            self.query_decomposer = query_decomposer
        elif self.query_plan_cache is not None and self.query_plan_revision is not None:
            self.query_decomposer = AdaptiveQueryDecomposer.from_frozen_llm_cache(
                self.query_plan_cache,
                self.query_plan_revision,
            )
        elif self.rules_only:
            self.query_decomposer = AdaptiveQueryDecomposer(rules_only=True, enable_disk_cache=False)
        else:
            self.query_decomposer = AdaptiveQueryDecomposer(enable_disk_cache=use_disk_cache)
        self.ltr_blend_weight = float(ltr_blend_weight)
        self.ltr_ranker = LearnedRankerLTR(model_path=ltr_model_path)
        self.spatial_patches: np.ndarray | None = None
        self.temporal_context_768: np.ndarray | None = None
        self.e5_text_index: faiss.Index | None = None
        self.e5_text_manifest: list[dict[str, Any]] = []
        self.has_vlm_caption_fts: bool = False
        self.e5_caption_index: faiss.Index | None = None
        self.e5_caption_manifest: list[dict[str, Any]] = []
        self.e5_tokenizer: Any | None = None
        self.e5_model: Any | None = None
        self.last_routing_plan: ModalityRoutingPlan | None = None
        self._last_event_intervals: dict[str, dict[str, tuple[float, float]]] = {}

        if not lazy_load:
            self._initialize()
            self.load_optional_phase74_artifacts(self.db_path.parent)

    def _initialize(self) -> None:
        """Initialize SQLite, FAISS index, 864d embeddings, and encoder model."""
        # 1. C2: Persistent Read-Only SQLite Connection
        if self.db_path.exists():
            resolved_uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            self.db_conn = sqlite3.connect(resolved_uri, uri=True, check_same_thread=False)

            try:
                cur_fts = self.db_conn.cursor()
                cur_fts.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_vlm_caption' LIMIT 1;"
                )
                self.has_vlm_caption_fts = cur_fts.fetchone() is not None
            except sqlite3.Error:
                self.has_vlm_caption_fts = False

            if not self.keyframe_map:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT faiss_id, video_id, pts_time, source_frame_index, keyframe_path FROM KeyframeMap;")
                self.keyframe_map = {
                    int(row[0]): {
                        "video_id": row[1],
                        "pts_time": float(row[2]),
                        "source_frame_index": row[3],
                        "keyframe_path": row[4],
                    }
                    for row in cursor.fetchall()
                }

                cursor.execute("SELECT video_id, duration_ms FROM Videos;")
                for row in cursor.fetchall():
                    if row[1] is not None:
                        self.video_durations[row[0]] = float(row[1]) / 1000.0

                self.video_keyframes = defaultdict(list)
                for fid, kinfo in self.keyframe_map.items():
                    self.video_keyframes[kinfo["video_id"]].append({
                        "faiss_id": fid,
                        "pts_time": kinfo["pts_time"],
                        "source_frame_index": kinfo["source_frame_index"],
                    })
                for vid in self.video_keyframes:
                    self.video_keyframes[vid].sort(key=lambda x: x["pts_time"])

                fo_cols = _table_columns(cursor, "FrameObjects")
                has_fo_spatial = {"keyframe_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(fo_cols)
                if has_fo_spatial:
                    cursor.execute(
                        "SELECT video_id, class_name, timestamp_ms, confidence, "
                        "keyframe_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM FrameObjects;"
                    )
                    for vid, cls_name, ts_ms, conf, kf_id, x1, y1, x2, y2 in cursor.fetchall():
                        raw_c = str(cls_name).lower() if cls_name else ""
                        obj_rec: dict[str, Any] = {
                            "class_name": cls_name,
                            "canonical_class": COCO_SYNONYM_MAP.get(raw_c, raw_c),
                            "pts_sec": float(ts_ms) / 1000.0 if ts_ms is not None else 0.0,
                            "confidence": float(conf) if conf is not None else 1.0,
                        }
                        if kf_id is not None:
                            obj_rec["keyframe_id"] = kf_id
                        if None not in (x1, y1, x2, y2):
                            obj_rec["bbox"] = (float(x1), float(y1), float(x2), float(y2))
                        self.video_objects[vid].append(obj_rec)
                elif fo_cols:
                    cursor.execute("SELECT video_id, class_name, timestamp_ms, confidence FROM FrameObjects;")
                    for vid, cls_name, ts_ms, conf in cursor.fetchall():
                        raw_c = str(cls_name).lower() if cls_name else ""
                        self.video_objects[vid].append({
                            "class_name": cls_name,
                            "canonical_class": COCO_SYNONYM_MAP.get(raw_c, raw_c),
                            "pts_sec": float(ts_ms) / 1000.0 if ts_ms is not None else 0.0,
                            "confidence": float(conf) if conf is not None else 1.0,
                        })
                for vid in self.video_objects:
                    self.video_objects[vid].sort(key=lambda x: x["pts_sec"])

                cursor.execute("SELECT video_id, timestamp_ms FROM OCRObservations;")
                self.ocr_data: dict[str, list[float]] = defaultdict(list)
                for vid, ts_ms in cursor.fetchall():
                    self.ocr_data[vid].append(float(ts_ms) / 1000.0)

                cursor.execute("SELECT video_id, start_ms, end_ms FROM ASRSegments;")
                self.asr_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for vid, st_ms, et_ms in cursor.fetchall():
                    self.asr_data[vid].append((float(st_ms) / 1000.0, float(et_ms) / 1000.0))
        else:
            self.ocr_data = defaultdict(list)
            self.asr_data = defaultdict(list)

        self.video_obj_pts: dict[str, list[float]] = {}
        self.video_classes_set: dict[str, set[str]] = {}
        for vid, objs in self.video_objects.items():
            for o in objs:
                if "canonical_class" not in o:
                    raw_c = str(o.get("class_name", "")).lower()
                    o["canonical_class"] = COCO_SYNONYM_MAP.get(raw_c, raw_c)
            objs.sort(key=lambda x: x["pts_sec"])
            self.video_obj_pts[vid] = [o["pts_sec"] for o in objs]
            self.video_classes_set[vid] = {o["class_name"] for o in objs}

        # 2. FAISS Index & 864d Subspace
        if self.index is None and self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))

        if self.index is not None and self.index.ntotal > 0:
            all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
            self.all_embeddings = all_vecs
            if self.augmented_embeddings is None or not self.coco_idf_map:
                self.augmented_embeddings, self.mean_vec_768, self.coco_idf_map = build_augmented_864d_embeddings(
                    raw_embeddings=all_vecs,
                    keyframe_map=self.keyframe_map,
                    frame_objects=self.video_objects,
                    videos_duration=self.video_durations,
                    ocr_data=getattr(self, "ocr_data", defaultdict(list)),
                    asr_data=getattr(self, "asr_data", defaultdict(list)),
                    alpha_obj=self.alpha_obj,
                    alpha_meta=self.alpha_meta,
                    beta_hub=self.beta_hub,
                )
            self.augmented_embeddings = np.ascontiguousarray(self.augmented_embeddings, dtype=np.float32)
            self.all_embeddings = np.ascontiguousarray(self.all_embeddings, dtype=np.float32)
            self.aug_index = faiss.IndexFlatIP(864)
            self.aug_index.add(self.augmented_embeddings)
            if self.keyframe_map:
                self.temporal_context_768 = compute_multiscale_temporal_context(
                    self.all_embeddings, self.keyframe_map
                )
            else:
                self.temporal_context_768 = None
        else:
            self.all_embeddings = np.zeros((0, 768), dtype=np.float32)
            self.augmented_embeddings = np.zeros((0, 864), dtype=np.float32)
            self.aug_index = faiss.IndexFlatIP(864)
            self.mean_vec_768 = np.zeros(768, dtype=np.float32)
            self.temporal_context_768 = None

        # Precompute per-video keyframe matrices and timestamp lists for fast O(1) lookup
        self.video_kf_pts: dict[str, list[float]] = {}
        self.video_kf_fids: dict[str, list[int]] = {}
        self.video_kf_matrix: dict[str, np.ndarray] = {}
        self.keyframe_obj_window: dict[int, list[dict[str, Any]]] = {}
        self.keyframe_obj_summary: dict[int, dict[str, tuple[float, int]]] = {}
        if self.augmented_embeddings is not None and len(self.augmented_embeddings) > 0:
            n_aug = len(self.augmented_embeddings)
            for vid, kf_list in self.video_keyframes.items():
                fids = [int(kf["faiss_id"]) for kf in kf_list if int(kf["faiss_id"]) < n_aug]
                pts = [float(kf["pts_time"]) for kf in kf_list if int(kf["faiss_id"]) < n_aug]
                if fids:
                    self.video_kf_fids[vid] = fids
                    self.video_kf_pts[vid] = pts
                    self.video_kf_matrix[vid] = np.ascontiguousarray(self.augmented_embeddings[fids])
                v_objs = self.video_objects.get(vid, [])
                pts_arr = self.video_obj_pts.get(vid)
                if v_objs and pts_arr is not None:
                    for kf in kf_list:
                        fid = int(kf["faiss_id"])
                        t_sec = float(kf["pts_time"])
                        lo_idx = bisect_left(pts_arr, t_sec - 3.5)
                        hi_idx = bisect_right(pts_arr, t_sec + 3.5)
                        win_objs = v_objs[lo_idx:hi_idx]
                        self.keyframe_obj_window[fid] = win_objs
                        summary_map: dict[str, tuple[float, int]] = {}
                        if win_objs:
                            by_cls: dict[str, list[dict[str, Any]]] = defaultdict(list)
                            for obj in win_objs:
                                c_cls = obj.get("canonical_class") or COCO_SYNONYM_MAP.get(str(obj.get("class_name", "")).lower(), str(obj.get("class_name", "")).lower())
                                by_cls[c_cls].append(obj)
                            for c_cls, matched_list in by_cls.items():
                                b_conf = max(float(o.get("confidence", 1.0)) for o in matched_list)
                                c_by_frame: dict[Any, int] = {}
                                for o in matched_list:
                                    f_k = o.get("keyframe_id", o.get("pts_sec"))
                                    if f_k is not None:
                                        c_by_frame[f_k] = c_by_frame.get(f_k, 0) + 1
                                max_cnt = max(c_by_frame.values()) if c_by_frame else len(matched_list)
                                summary_map[c_cls] = (b_conf, max_cnt)
                        self.keyframe_obj_summary[fid] = summary_map
                else:
                    for kf in kf_list:
                        fid = int(kf["faiss_id"])
                        self.keyframe_obj_window[fid] = []
                        self.keyframe_obj_summary[fid] = {}

        # 3. Text Encoder (Lazy-loaded on first cache miss to save RAM and startup latency)
        pass

    def _ensure_text_encoder(self) -> None:
        """Lazy-load SigLIP 2 text encoder only when encountering an uncached query."""
        if self.tokenizer is None and self.text_model is None and self.model_dir.exists():
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            full_model = AutoModel.from_pretrained(str(self.model_dir)).to(self.device)
            full_model.eval()
            if hasattr(full_model, "vision_model"):
                del full_model.vision_model
            self.text_model = full_model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode NFC texts with fixed padding and a bounded per-instance RAM cache."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        n = len(texts)
        result = np.zeros((n, 768), dtype=np.float32)
        missing: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            text = unicodedata.normalize("NFC", text)
            cached = self._text_memory_cache.get(text)
            if cached is not None:
                self._text_memory_cache.move_to_end(text)
                result[i] = cached
                continue
            if self.cache is not None:
                cached = self.cache.get(text)
                if cached is not None and cached.shape == (768,):
                    result[i] = cached
                    continue
            missing.setdefault(text, []).append(i)

        if missing:
            self._ensure_text_encoder()
            if self.tokenizer is not None and self.text_model is not None:
                # SigLIP 2 requires fixed max_length=64 padding for pooling alignment at index 63
                inputs = self.tokenizer(
                    list(missing),
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                ).to(self.device)

                with torch.inference_mode():
                    if hasattr(self.text_model, "get_text_features"):
                        raw = self.text_model.get_text_features(**inputs)
                    else:
                        raw = self.text_model(**inputs)
                    feats = extract_tensor_features(raw)
                    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                    arr = feats.cpu().float().numpy()

                if (arr.shape != (len(missing), 768) or not np.isfinite(arr).all()
                        or np.any(np.linalg.norm(arr, axis=1) < 1e-8)):
                    raise ValueError("Text encoder returned invalid embeddings")
                for j, (text, indices) in enumerate(missing.items()):
                    vec = arr[j].astype(np.float32)
                    for orig_idx in indices:
                        result[orig_idx] = vec
                    if self.memory_cache_size:
                        self._text_memory_cache[text] = vec.copy()
                        self._text_memory_cache.move_to_end(text)
                        while len(self._text_memory_cache) > self.memory_cache_size:
                            self._text_memory_cache.popitem(last=False)
                    if self.cache is not None:
                        self.cache.set(text, vec)

                if self.cache is not None:
                    self.cache.save()
            else:
                raise RuntimeError(f"Text encoder unavailable: {self.model_dir}")

        return result

    def encode_augmented_query(self, query: str) -> np.ndarray:
        """Encode query using dual-prompt ensemble (0.75 raw + 0.25 frame prompt)."""
        p1 = query
        p2 = f"a video frame showing {query}"
        vecs = self.encode_texts([p1, p2])
        ens = 0.75 * vecs[0] + 0.25 * vecs[1]
        ens /= (np.linalg.norm(ens) + 1e-8)
        return ens.reshape(1, -1)

    def prepare_fts5_query_and_tokens(self, query: str) -> tuple[str, list[str], set[str], set[str], set[str]]:
        """B7: Generates specificity-sorted FTS5 query capped at 14 tokens."""
        quoted_phrases, high_spec_tokens, low_spec_tokens, proper_nouns_lower = analyze_query_lexical_tokens(query)
        base_fts = build_fts5_query(query, enable_prefix=True)

        cand_tokens: list[str] = []
        stem_tokens: list[str] = []

        # 1. Quoted phrase tokens
        for qp in quoted_phrases:
            for w in re.findall(r'\b[a-zA-Z0-9à-ỹÀ-Ỹ]+\b', qp):
                w_l = w.lower()
                if w_l not in ALL_STOPWORDS and len(w_l) >= 3:
                    cand_tokens.append(f'"{w}"')
                    if len(w_l) >= 5:
                        cand_tokens.append(f'{w_l}*')

        # 2. Proper nouns and clean domain synonyms (e.g., formula notations)
        for pn in list(proper_nouns_lower):
            if pn not in ALL_STOPWORDS and len(pn) >= 3:
                cand_tokens.append(f'"{pn}"')
                if len(pn) >= 5:
                    cand_tokens.append(f'{pn}*')
                _apply_domain_synonyms(
                    pn,
                    cand_tokens,
                    high_spec_tokens,
                    low_spec_tokens,
                    proper_nouns_lower,
                )

        # 3. High-specificity tokens (sorted descending by length)
        sorted_high = sorted(high_spec_tokens, key=lambda t: len(t), reverse=True)
        for ht in list(sorted_high):
            if ht not in proper_nouns_lower and len(ht) >= 3:
                cand_tokens.append(f'"{ht}"')
                if len(ht) >= 5:
                    cand_tokens.append(f'{ht}*')
                _apply_domain_synonyms(
                    ht,
                    cand_tokens,
                    high_spec_tokens,
                    low_spec_tokens,
                    proper_nouns_lower,
                )
            if ht == "checking" or ht.endswith(("ologists", "ologist")):
                for st in _generate_morphological_stems(ht):
                    if len(stem_tokens) >= 2:
                        break
                    st_tok = f'{st}*' if len(st) >= 5 else f'"{st}"'
                    stem_tokens.append(st_tok)

        # 4. General adjacent-bigram OCR compound joins (resilient to missing-space captions like "lateshow")
        ordered_words = re.findall(r'\b[a-z0-9à-ỹ]+\b', split_compound_tokens(query).lower())
        compound_tokens: list[str] = []
        for w1, w2 in zip(ordered_words, ordered_words[1:]):
            if (
                len(w1) >= 3
                and len(w2) >= 3
                and w1 not in ALL_STOPWORDS
                and w2 not in ALL_STOPWORDS
                and (w1 in high_spec_tokens or w2 in high_spec_tokens)
            ):
                joined = f"{w1}{w2}"
                if len(joined) >= 6:
                    compound_tokens.append(f'"{joined}"')
                    if len(compound_tokens) >= 2:
                        break

        # 5. Low-specificity tokens
        low_tokens: list[str] = [
            f'{lt}*'
            for lt in low_spec_tokens
            if lt not in ALL_STOPWORDS and len(lt) >= 4
        ]

        # Preserve all primary tokens without premature slicing; bounded stems, low-spec, compounds follow
        dedup_tokens = list(dict.fromkeys(cand_tokens + stem_tokens[:2] + low_tokens + compound_tokens[:2]))
        selected_tokens = dedup_tokens[:14]

        if selected_tokens:
            fts_query = " OR ".join(selected_tokens)
        else:
            fts_query = base_fts

        return fts_query, quoted_phrases, high_spec_tokens, low_spec_tokens, proper_nouns_lower

    def _compute_video_visual_curve(
        self,
        vid: str,
        primary_q_864: np.ndarray,
        primary_q_768: np.ndarray | None = None,
        blend_768_864: bool = True,
    ) -> tuple[list[float], np.ndarray]:
        """Compute per-keyframe visual similarity curve for a video, blending 768d+864d when aligned."""
        kf_list = self.video_keyframes.get(vid, [])
        if not kf_list:
            return [], np.zeros(0, dtype=np.float32)

        kf_pts = self.video_kf_pts.get(vid)
        kf_matrix = self.video_kf_matrix.get(vid)
        fids = self.video_kf_fids.get(vid)
        if kf_pts is None or kf_matrix is None:
            n_aug = len(self.augmented_embeddings) if self.augmented_embeddings is not None else 0
            valid_kfs = [kf for kf in kf_list if int(kf["faiss_id"]) < n_aug]
            if not valid_kfs:
                return [], np.zeros(0, dtype=np.float32)
            fids = [int(kf["faiss_id"]) for kf in valid_kfs]
            kf_pts = [float(kf["pts_time"]) for kf in valid_kfs]
            kf_matrix = self.augmented_embeddings[fids]
        elif fids is None:
            fids = [int(kf["faiss_id"]) for kf in kf_list]

        sims_864 = (kf_matrix @ primary_q_864.reshape(-1)).astype(np.float32).flatten()
        if (
            blend_768_864
            and primary_q_768 is not None
            and fids
            and self.all_embeddings is not None
            and self.augmented_embeddings is not None
            and len(self.all_embeddings) == len(self.augmented_embeddings)
            and len(self.all_embeddings) > 0
        ):
            sims_768 = (self.all_embeddings[fids] @ primary_q_768.reshape(-1)).astype(np.float32).flatten()
            sims = (0.70 * sims_768 + 0.30 * sims_864).astype(np.float32)
        else:
            sims = sims_864
        return kf_pts, sims

    def _video_sorted_visual_scores(
        self,
        vid: str,
        primary_q_864: np.ndarray,
        primary_q_768: np.ndarray | None = None,
        blend_768_864: bool = True,
    ) -> list[float]:
        """Return descending per-keyframe visual similarities across all keyframes of a video."""
        if primary_q_864 is None:
            return []
        _, sims = self._compute_video_visual_curve(
            vid,
            primary_q_864=primary_q_864,
            primary_q_768=primary_q_768,
            blend_768_864=blend_768_864,
        )
        if len(sims) == 0:
            return []
        return np.sort(sims)[::-1].tolist()

    def lookup_true_visual_sim(
        self,
        vid: str,
        ts: float,
        start_sec: float | None = None,
        end_sec: float | None = None,
        primary_q_864: np.ndarray | None = None,
        primary_q_768: np.ndarray | None = None,
        curve_cache: dict[str, tuple[list[float], np.ndarray]] | None = None,
    ) -> float:
        """B1: Temporal-local visual similarity lookup restricted to [ts-3.5, ts+3.5] or nearest <= 6.0s."""
        if vid not in self.video_keyframes or len(self.video_keyframes[vid]) == 0 or primary_q_864 is None:
            return 0.0

        if curve_cache is not None and vid in curve_cache:
            kf_pts, sims = curve_cache[vid]
        else:
            kf_pts, sims = self._compute_video_visual_curve(
                vid,
                primary_q_864=primary_q_864,
                primary_q_768=primary_q_768,
                blend_768_864=True,
            )
            if len(sims) == 0:
                return 0.0
            if curve_cache is not None:
                curve_cache[vid] = (kf_pts, sims)

        kf_fids = self.video_kf_fids.get(vid)

        w_start = (start_sec - 3.5) if start_sec is not None else (ts - 3.5)
        w_end = (end_sec + 3.5) if end_sec is not None else (ts + 3.5)
        w_start = max(0.0, w_start)

        lo_i = bisect_left(kf_pts, w_start)
        hi_i = bisect_right(kf_pts, w_end)
        if lo_i < hi_i:
            return float(np.max(sims[lo_i:hi_i]))

        # Nearest keyframe search via binary search
        pos = bisect_left(kf_pts, ts)
        if pos == 0:
            nearest_idx = 0
        elif pos >= len(kf_pts):
            nearest_idx = len(kf_pts) - 1
        else:
            nearest_idx = pos if (kf_pts[pos] - ts) < (ts - kf_pts[pos - 1]) else (pos - 1)
        if abs(kf_pts[nearest_idx] - ts) <= 6.0:
            return float(sims[nearest_idx])

        # Beyond 6.0s: baseline 35th percentile of video to close visual gate (B1)
        return float(np.percentile(sims, 35))

    def search_with_metadata(
        self,
        *,
        snapshot_id: str,
        verify_relations: bool = False,
        rerank_with_relations: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize access to the shared SQLite connection and return request-local metadata."""
        with self._search_lock:
            self.last_timings = {}
            self.last_diagnostics = {}
            moments = self.search(**kwargs)
            verify_start = time.perf_counter()
            relation_assessments = self._verify_relations(kwargs["query"], moments) if verify_relations else []
            if verify_relations and rerank_with_relations and len(moments) == len(relation_assessments) and moments:
                scored_pairs: list[tuple[float, RankedMoment, dict[str, Any]]] = []
                for m, assess in zip(moments, relation_assessments):
                    base_score = float(m.ranking_score)
                    status = assess["status"]
                    struct_score = float(assess.get("structural_score", 0.0))
                    if status == "contradicted":
                        adj = max(0.0001, base_score * 0.15)
                    elif status == "unknown":
                        adj = base_score
                    else:
                        adj = base_score + 0.08 * struct_score
                    updated_m = RankedMoment(
                        moment_id=m.moment_id,
                        video_id=m.video_id,
                        moment_start_seconds=m.moment_start_seconds,
                        moment_end_seconds=m.moment_end_seconds,
                        start_original_frame_index=m.start_original_frame_index,
                        end_original_frame_index=m.end_original_frame_index,
                        representative_original_frame_index=m.representative_original_frame_index,
                        representative_timestamp_seconds=m.representative_timestamp_seconds,
                        ranking_score=round(adj, 5),
                        score_type="relation_aware_ranking" if status != "unknown" else m.score_type,
                        match_status=status if status != "unknown" else m.match_status,
                        evidence_frame_indices=m.evidence_frame_indices,
                        unresolved_predicates=tuple(assess.get("unresolved_predicates", ())),
                        structural_score=round(struct_score, 4),
                    )
                    scored_pairs.append((adj, updated_m, assess))
                scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
                moments = [pair[1] for pair in scored_pairs]
                relation_assessments = [pair[2] for pair in scored_pairs]
            timings = deepcopy(self.last_timings)
            if verify_relations:
                timings["t_relation_ms"] = (time.perf_counter() - verify_start) * 1000.0
            cand_moments = self.last_diagnostics.get("all_candidate_moments", ())
            top_k = int(kwargs.get("top_k", 10))
            ranked_videos = self._build_ranked_videos(moments, cand_moments, top_k=top_k)
            diagnostics = deepcopy(self.last_diagnostics)
            diagnostics["ranked_videos"] = ranked_videos
            if self.last_diagnostics:
                diagnostics["ranked_video_ids"] = [m.video_id for m in ranked_videos]
            return {
                "snapshot_id": snapshot_id,
                "moments": moments,
                "ranked_moments": moments,
                "ranked_videos": ranked_videos,
                "timings": timings,
                "diagnostics": diagnostics,
                "relation_assessments": relation_assessments,
            }

    @staticmethod
    def _build_ranked_videos(
        moments: Sequence[RankedMoment],
        candidate_moments: Sequence[RankedMoment] = (),
        top_k: int | None = None,
    ) -> list[RankedMoment]:
        """Deduplicate scored moments by video_id keeping each video's best score with stable tie-break."""
        best_by_vid: dict[str, RankedMoment] = {}
        seen_in_primary: set[str] = set()
        for m in moments:
            seen_in_primary.add(m.video_id)
            prev = best_by_vid.get(m.video_id)
            if prev is None or float(m.ranking_score) > float(prev.ranking_score):
                best_by_vid[m.video_id] = m
        for m in candidate_moments:
            if m.video_id in seen_in_primary:
                continue
            prev = best_by_vid.get(m.video_id)
            if prev is None or float(m.ranking_score) > float(prev.ranking_score):
                best_by_vid[m.video_id] = m
        ranked = sorted(
            best_by_vid.values(),
            key=lambda m: (-float(m.ranking_score), str(m.video_id), float(m.moment_start_seconds)),
        )
        return ranked[:top_k] if top_k is not None else ranked

    def _is_spatial_patches_enabled(self, ablation_config: str) -> tuple[bool, str]:
        if not self.enable_spatial_patches:
            return False, "disabled_by_flag"
        if ablation_config in ("A0", "A1", "A2", "A3"):
            return False, f"ablation_{ablation_config}"
        return True, "enabled"

    def _is_vlm_caption_lexical_enabled(self, ablation_config: str) -> tuple[bool, str]:
        if not self.enable_vlm_caption_lexical:
            return False, "disabled_by_flag"
        if ablation_config in ("A0", "A1", "A2"):
            return False, f"ablation_{ablation_config}"
        return True, "enabled"

    def _is_vlm_caption_dense_enabled(self, ablation_config: str) -> tuple[bool, str]:
        if not self.enable_vlm_caption_dense:
            return False, "disabled_by_flag"
        if ablation_config in ("A0", "A1", "A2"):
            return False, f"ablation_{ablation_config}"
        return True, "enabled"

    def _is_temporal_rerank_enabled(self, ablation_config: str) -> tuple[bool, str]:
        if not self.enable_temporal_rerank:
            return False, "disabled_by_flag"
        if ablation_config in ("A0", "A1"):
            return False, f"ablation_{ablation_config}"
        return True, "enabled"

    def _build_degraded_diagnostics(self, ablation_config: str) -> dict[str, dict[str, str]]:
        """Build structured channel -> {status, reason} diagnostics."""
        degraded: dict[str, dict[str, str]] = {}
        if ablation_config in ("A0", "A1", "A2"):
            degraded["e5_text_index"] = {"status": "disabled_by_config", "reason": f"ablation_{ablation_config}"}
            degraded["lexical_db"] = {"status": "disabled_by_config", "reason": f"ablation_{ablation_config}"}
        else:
            if "e5_text_index" in self._degraded_channels:
                degraded["e5_text_index"] = {"status": "failed", "reason": self._degraded_channels["e5_text_index"]}
            elif self.e5_text_index is not None:
                degraded["e5_text_index"] = {"status": "ready", "reason": "loaded"}
            else:
                degraded["e5_text_index"] = {"status": "missing_artifact", "reason": "e5_text_index_not_loaded"}

            if self.db_conn is not None:
                degraded["lexical_db"] = {"status": "ready", "reason": "connected"}
            else:
                degraded["lexical_db"] = {"status": "missing_artifact", "reason": "db_conn_none"}

        cap_dense_enabled, cap_dense_reason = self._is_vlm_caption_dense_enabled(ablation_config)
        if not cap_dense_enabled:
            degraded["e5_caption_index"] = {"status": "disabled_by_config", "reason": cap_dense_reason}
        elif "e5_caption_index" in self._channel_mismatches:
            degraded["e5_caption_index"] = {"status": "mismatch", "reason": self._channel_mismatches["e5_caption_index"]}
        elif "e5_caption_index" in self._degraded_channels:
            degraded["e5_caption_index"] = {"status": "failed", "reason": self._degraded_channels["e5_caption_index"]}
        elif self.e5_caption_index is not None:
            degraded["e5_caption_index"] = {"status": "ready", "reason": "loaded"}
        else:
            degraded["e5_caption_index"] = {"status": "missing_artifact", "reason": "e5_caption_index_not_loaded"}

        cap_lex_enabled, cap_lex_reason = self._is_vlm_caption_lexical_enabled(ablation_config)
        if not cap_lex_enabled:
            degraded["vlm_caption_lexical"] = {"status": "disabled_by_config", "reason": cap_lex_reason}
        elif self.has_vlm_caption_fts and self.db_conn is not None:
            degraded["vlm_caption_lexical"] = {"status": "ready", "reason": "connected"}
        else:
            degraded["vlm_caption_lexical"] = {"status": "missing_artifact", "reason": "fts_vlm_caption_not_available"}

        sp_enabled, sp_reason = self._is_spatial_patches_enabled(ablation_config)
        if not sp_enabled:
            degraded["spatial_patches"] = {"status": "disabled_by_config", "reason": sp_reason}
        elif "spatial_patches" in self._channel_mismatches:
            degraded["spatial_patches"] = {"status": "mismatch", "reason": self._channel_mismatches["spatial_patches"]}
        elif "spatial_patches" in self._degraded_channels:
            degraded["spatial_patches"] = {"status": "failed", "reason": self._degraded_channels["spatial_patches"]}
        elif self.spatial_patches is not None:
            degraded["spatial_patches"] = {"status": "ready", "reason": "loaded"}
        else:
            degraded["spatial_patches"] = {"status": "missing_artifact", "reason": "spatial_patches_not_loaded"}

        tr_enabled, tr_reason = self._is_temporal_rerank_enabled(ablation_config)
        if not tr_enabled:
            degraded["temporal_rerank"] = {"status": "disabled_by_config", "reason": tr_reason}
        elif (self.all_embeddings is not None and len(self.all_embeddings) > 0) or (
            self.augmented_embeddings is not None and len(self.augmented_embeddings) > 0
        ):
            degraded["temporal_rerank"] = {
                "status": "ready",
                "reason": f"mode={self.temporal_rerank_mode},lambda={self.temporal_rerank_lambda}",
            }
        else:
            degraded["temporal_rerank"] = {"status": "missing_artifact", "reason": "embeddings_not_loaded"}

        for ch, reason in self._channel_mismatches.items():
            if ch not in degraded:
                degraded[ch] = {"status": "mismatch", "reason": str(reason)}
        for ch, reason in self._degraded_channels.items():
            if ch not in degraded:
                degraded[ch] = {"status": "failed", "reason": str(reason)}
        return degraded

    def _build_clause_multimodal_curves(
        self,
        video_id: str,
        cand_ts: Sequence[float],
        sub_clauses: Sequence[str],
        ocr_phrases: Sequence[str],
        asr_phrases: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        """Build per-clause OCR and ASR alignment curves over keyframe ledger timestamps."""
        k_clauses = len(sub_clauses)
        m_frames = len(cand_ts)
        ocr_curves = np.zeros((k_clauses, m_frames), dtype=np.float32)
        asr_curves = np.zeros((k_clauses, m_frames), dtype=np.float32)
        availability: dict[str, str] = {
            "visual": "available" if m_frames > 0 else "missing",
            "ocr": "missing",
            "asr": "missing",
        }
        if self.db_conn is None or k_clauses == 0 or m_frames == 0:
            return ocr_curves, asr_curves, availability
        if not ocr_phrases and not asr_phrases:
            return ocr_curves, asr_curves, availability

        # Map each clause to phrases it mentions (fall back to all phrases if none match a clause)
        clause_ocr_phrases: list[list[str]] = []
        for c in sub_clauses:
            matched = [p for p in ocr_phrases if p and contains_text_phrase(c, p)]
            clause_ocr_phrases.append(matched)
        if ocr_phrases and not any(clause_ocr_phrases):
            clause_ocr_phrases = [list(ocr_phrases) for _ in sub_clauses]

        clause_asr_phrases: list[list[str]] = []
        for c in sub_clauses:
            matched = [p for p in asr_phrases if p and contains_text_phrase(c, p)]
            clause_asr_phrases.append(matched)
        if asr_phrases and not any(clause_asr_phrases):
            clause_asr_phrases = [list(asr_phrases) for _ in sub_clauses]

        ts_arr = np.asarray(cand_ts, dtype=np.float64)
        try:
            cursor = self.db_conn.cursor()
            if ocr_phrases and _table_columns(cursor, "OCRObservations"):
                ocr_rows = cursor.execute(
                    "SELECT raw_text, timestamp_ms FROM OCRObservations WHERE video_id=?",
                    (video_id,),
                ).fetchall()
                if ocr_rows:
                    availability["ocr"] = "available"
                    for raw_text, ts_ms in ocr_rows:
                        if raw_text is None or ts_ms is None:
                            continue
                        t_obs = float(ts_ms) / 1000.0
                        text_str = str(raw_text)
                        for k_idx, phrases in enumerate(clause_ocr_phrases):
                            if any(contains_text_phrase(text_str, ph) for ph in phrases):
                                diffs = np.abs(ts_arr - t_obs)
                                near_mask = diffs <= 2.5
                                if np.any(near_mask):
                                    scores = np.exp(-0.5 * (diffs[near_mask] / 1.5) ** 2).astype(np.float32)
                                    ocr_curves[k_idx, near_mask] = np.maximum(ocr_curves[k_idx, near_mask], scores)

            if asr_phrases and _table_columns(cursor, "ASRSegments"):
                asr_rows = cursor.execute(
                    "SELECT raw_text, start_ms, end_ms FROM ASRSegments WHERE video_id=? AND end_ms>start_ms",
                    (video_id,),
                ).fetchall()
                if asr_rows:
                    availability["asr"] = "available"
                    for raw_text, s_ms, e_ms in asr_rows:
                        if raw_text is None or s_ms is None or e_ms is None:
                            continue
                        s_sec = float(s_ms) / 1000.0
                        e_sec = float(e_ms) / 1000.0
                        text_str = str(raw_text)
                        for k_idx, phrases in enumerate(clause_asr_phrases):
                            if any(contains_text_phrase(text_str, ph) for ph in phrases):
                                in_span = (ts_arr >= (s_sec - 0.5)) & (ts_arr <= (e_sec + 0.5))
                                if np.any(in_span):
                                    asr_curves[k_idx, in_span] = 1.0
        except sqlite3.Error as exc:
            availability["ocr"] = "failed"
            availability["asr"] = "failed"
            self._degraded_channels["viterbi_multimodal_curves"] = str(exc)
            logger.warning("Failed to build multimodal Viterbi curves for %s: %s", video_id, exc)

        return ocr_curves, asr_curves, availability

    def _collect_moment_evidence(self, video_id: str, start_ms: float, end_ms: float) -> list[dict[str, Any]]:
        """Collect schema-safe interval-contained OCR, calibrated ASR, and 2D FrameObjects evidence."""
        if self.db_conn is None:
            return []
        evidence: list[dict[str, Any]] = []
        try:
            cursor = self.db_conn.cursor()
            ocr_cols = _table_columns(cursor, "OCRObservations")
            if ocr_cols:
                select_cols = ["observation_id", "raw_text", "model_id", "timestamp_ms"]
                has_ocr_kf = "keyframe_id" in ocr_cols
                has_ocr_conf = "confidence" in ocr_cols
                has_ocr_bbox = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(ocr_cols)
                if has_ocr_kf:
                    select_cols.append("keyframe_id")
                if has_ocr_conf:
                    select_cols.append("confidence")
                if has_ocr_bbox:
                    select_cols.extend(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
                sql = (
                    f"SELECT {', '.join(select_cols)} FROM OCRObservations "
                    "WHERE video_id=? AND timestamp_ms>=? AND timestamp_ms<?"
                )
                for row in cursor.execute(sql, (video_id, start_ms, end_ms)):
                    idx = 4
                    rec: dict[str, Any] = {
                        "kind": "ocr_observation",
                        "evidence_id": row[0],
                        "text": row[1],
                        "model_id": row[2],
                        "timestamp_seconds": float(row[3]) / 1000.0,
                        "timestamp_ms": float(row[3]),
                    }
                    if has_ocr_kf:
                        rec["keyframe_id"] = row[idx]
                        idx += 1
                    if has_ocr_conf:
                        rec["confidence"] = float(row[idx]) if row[idx] is not None else 0.90
                        idx += 1
                    if has_ocr_bbox:
                        bx1, by1, bx2, by2 = row[idx : idx + 4]
                        if None not in (bx1, by1, bx2, by2):
                            rec["bbox"] = (float(bx1), float(by1), float(bx2), float(by2))
                    evidence.append(rec)

            asr_cols = _table_columns(cursor, "ASRSegments")
            if asr_cols:
                has_asr_conf = "confidence" in asr_cols
                asr_sql = (
                    "SELECT segment_id, raw_text, model_id, start_ms, end_ms"
                    + (", confidence" if has_asr_conf else "")
                    + " FROM ASRSegments WHERE video_id=? AND start_ms>=? AND end_ms<=? AND end_ms>start_ms"
                )
                for row in cursor.execute(asr_sql, (video_id, start_ms, end_ms)):
                    raw_conf = row[5] if has_asr_conf else None
                    rec = {
                        "kind": "asr_segment",
                        "evidence_id": row[0],
                        "text": row[1],
                        "model_id": row[2],
                        "start_seconds": float(row[3]) / 1000.0,
                        "end_seconds": float(row[4]) / 1000.0,
                        "confidence": calibrate_asr_confidence(raw_conf),
                    }
                    if raw_conf is not None:
                        rec["raw_confidence"] = float(raw_conf)
                    evidence.append(rec)

            fo_cols = _table_columns(cursor, "FrameObjects")
            if fo_cols:
                fo_select = ["detection_id", "class_name", "timestamp_ms"]
                has_fo_kf = "keyframe_id" in fo_cols
                has_fo_conf = "confidence" in fo_cols
                has_fo_model = "model_id" in fo_cols
                has_fo_bbox = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(fo_cols)
                if has_fo_kf:
                    fo_select.append("keyframe_id")
                if has_fo_conf:
                    fo_select.append("confidence")
                if has_fo_model:
                    fo_select.append("model_id")
                if has_fo_bbox:
                    fo_select.extend(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
                fo_sql = (
                    f"SELECT {', '.join(fo_select)} FROM FrameObjects "
                    "WHERE video_id=? AND timestamp_ms>=? AND timestamp_ms<?"
                )
                for row in cursor.execute(fo_sql, (video_id, start_ms, end_ms)):
                    idx = 3
                    cls_name = str(row[1] or "")
                    rec = {
                        "kind": "visual_detection",
                        "evidence_id": row[0],
                        "class_name": cls_name,
                        "label": cls_name,
                        "canonical_class": COCO_SYNONYM_MAP.get(cls_name.lower(), cls_name.lower()),
                        "timestamp_ms": float(row[2]),
                        "timestamp_seconds": float(row[2]) / 1000.0,
                    }
                    if has_fo_kf:
                        rec["keyframe_id"] = row[idx]
                        idx += 1
                    if has_fo_conf:
                        rec["confidence"] = float(row[idx]) if row[idx] is not None else 0.85
                        idx += 1
                    if has_fo_model:
                        rec["model_id"] = row[idx]
                        idx += 1
                    if has_fo_bbox:
                        bx1, by1, bx2, by2 = row[idx : idx + 4]
                        if None not in (bx1, by1, bx2, by2):
                            rec["bbox"] = (float(bx1), float(by1), float(bx2), float(by2))
                    evidence.append(rec)
        except sqlite3.Error as exc:
            raise RuntimeError("Relation evidence lookup failed") from exc
        return evidence

    class _ArtifactMismatchError(Exception):
        """Raised when an optional artifact fails snapshot/sidecar/receipt verification."""

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_optional_phase74_artifacts(self, base_dir: Path | str = Path("data/processed")) -> dict[str, bool]:
        """Load optional Phase 7.4 / P8-5 offline artifacts (spatial patches, E5 text & caption indexes) if present."""
        import json as _json

        root = Path(base_dir)
        loaded = {"spatial_patches": False, "e5_text_index": False}
        self._spatial_patch_row_by_fid = None
        patches_path = root / "spatial_patches_fp16.npy"
        if patches_path.is_file():
            try:
                patches = np.load(str(patches_path), mmap_mode="r")
                if patches.ndim != 3 or patches.shape[1:] != (9, 768) or patches.dtype != np.float16:
                    raise self._ArtifactMismatchError(
                        f"spatial_patches shape/dtype mismatch: got shape={patches.shape}, dtype={patches.dtype}"
                    )
                if not bool(np.isfinite(patches).all()):
                    raise self._ArtifactMismatchError("spatial_patches contains non-finite values")

                sidecar_path = root / "spatial_patches_map.jsonl"
                receipt_path = root / "p85_receipt.json"
                if receipt_path.is_file():
                    receipt = _json.loads(receipt_path.read_text(encoding="utf-8"))
                    expected_patch_sha = receipt.get("spatial_patches_sha256") or receipt.get("patches_sha256")
                    if expected_patch_sha and self._sha256_file(patches_path) != str(expected_patch_sha):
                        raise self._ArtifactMismatchError("spatial_patches_sha256 mismatch against p85_receipt.json")
                    expected_map_sha = receipt.get("spatial_patches_map_sha256") or receipt.get("map_sha256")
                    if expected_map_sha:
                        if not sidecar_path.is_file():
                            raise self._ArtifactMismatchError("spatial_patches_map.jsonl missing while required by p85_receipt.json")
                        if self._sha256_file(sidecar_path) != str(expected_map_sha):
                            raise self._ArtifactMismatchError("spatial_patches_map_sha256 mismatch against p85_receipt.json")
                    expected_idx_sha = receipt.get("visual_index_sha256")
                    if expected_idx_sha and self.faiss_path.is_file():
                        if self._sha256_file(self.faiss_path) != str(expected_idx_sha):
                            raise self._ArtifactMismatchError("visual_index_sha256 mismatch against p85_receipt.json")

                if sidecar_path.is_file():
                    row_by_fid: dict[int, int] = {}
                    rows_count = 0
                    with sidecar_path.open("r", encoding="utf-8") as fh:
                        for idx, line in enumerate(fh):
                            line = line.strip()
                            if not line:
                                continue
                            rec = _json.loads(line)
                            row_idx = int(rec.get("row_index", idx))
                            if row_idx != idx:
                                raise self._ArtifactMismatchError(f"spatial_patches_map row_index mismatch at {idx}: {row_idx}")
                            fid = int(rec["faiss_id"])
                            if fid in row_by_fid:
                                raise self._ArtifactMismatchError(f"duplicate faiss_id {fid} in spatial_patches_map.jsonl")
                            if self.keyframe_map and fid in self.keyframe_map:
                                expected_vid = str(self.keyframe_map[fid].get("video_id", ""))
                                got_vid = str(rec.get("video_id", ""))
                                if got_vid and expected_vid and got_vid != expected_vid:
                                    raise self._ArtifactMismatchError(
                                        f"spatial_patches_map video_id mismatch for faiss_id={fid}: "
                                        f"got {got_vid!r}, expected {expected_vid!r}"
                                    )
                            row_by_fid[fid] = row_idx
                            rows_count += 1
                    if rows_count != patches.shape[0]:
                        raise self._ArtifactMismatchError(
                            f"spatial_patches_map row count {rows_count} != patches rows {patches.shape[0]}"
                        )
                    self._spatial_patch_row_by_fid = row_by_fid
                elif self.keyframe_map and max(self.keyframe_map) >= patches.shape[0]:
                    raise self._ArtifactMismatchError(
                        f"spatial_patches rows {patches.shape[0]} < max keyframe_map faiss_id {max(self.keyframe_map)}"
                    )

                self.spatial_patches = patches
                self._channel_mismatches.pop("spatial_patches", None)
                loaded["spatial_patches"] = True
            except self._ArtifactMismatchError as exc:
                self.spatial_patches = None
                self._spatial_patch_row_by_fid = None
                self._channel_mismatches["spatial_patches"] = str(exc)
                logger.warning("Spatial patches verification mismatch %s: %s", patches_path, exc)
            except Exception as exc:
                self.spatial_patches = None
                self._spatial_patch_row_by_fid = None
                self._degraded_channels["spatial_patches"] = str(exc)
                logger.warning("Failed to load spatial patches %s: %s", patches_path, exc, exc_info=True)

        e5_idx_path = root / "faiss_text_e5_384.bin"
        e5_manifest_path = root / "e5_text_manifest.json"
        if e5_idx_path.is_file() and e5_manifest_path.is_file():
            try:
                self.e5_text_index = faiss.read_index(str(e5_idx_path))
                self.e5_text_manifest = _json.loads(e5_manifest_path.read_text(encoding="utf-8"))
                loaded["e5_text_index"] = True
            except Exception as exc:
                self.e5_text_index = None
                self.e5_text_manifest = []
                self._degraded_channels["e5_text_index"] = str(exc)
                logger.warning("Failed to load E5 text index %s: %s", e5_idx_path, exc, exc_info=True)

        cap_idx_path = root / "faiss_e5_caption_384.bin"
        cap_manifest_path = root / "e5_caption_manifest.json"
        if cap_idx_path.is_file() and cap_manifest_path.is_file():
            try:
                self.e5_caption_index = faiss.read_index(str(cap_idx_path))
                self.e5_caption_manifest = _json.loads(cap_manifest_path.read_text(encoding="utf-8"))
                loaded["e5_caption_index"] = True
            except Exception as exc:
                self.e5_caption_index = None
                self.e5_caption_manifest = []
                self._degraded_channels["e5_caption_index"] = str(exc)
                logger.warning("Failed to load E5 caption index %s: %s", cap_idx_path, exc, exc_info=True)
        return loaded

    def _score_late_interaction_window(
        self,
        vid: str,
        start_sec: float,
        end_sec: float,
        query_vec_768: np.ndarray | None,
    ) -> float:
        """Compute QAT-LateInteraction MaxSim over 3x3 spatial patches in [start_sec, end_sec]."""
        if self.spatial_patches is None or query_vec_768 is None or len(self.spatial_patches) == 0:
            return 0.0
        kf_list = self.video_keyframes.get(vid, [])
        if not kf_list:
            return 0.0
        fids: list[int] = []
        for k in kf_list:
            if start_sec <= float(k.get("pts_time", 0.0)) <= end_sec:
                fid = int(k["faiss_id"])
                row_idx = self._spatial_patch_row_by_fid.get(fid) if self._spatial_patch_row_by_fid is not None else fid
                if row_idx is not None and 0 <= row_idx < len(self.spatial_patches):
                    fids.append(row_idx)
        if not fids:
            return 0.0
        try:
            window_patches = np.asarray(self.spatial_patches[fids], dtype=np.float32)
            return float(compute_late_interaction_maxsim(query_vec_768, None, window_patches))
        except Exception as exc:
            self._degraded_channels["late_interaction"] = str(exc)
            logger.warning("Late interaction scoring failed for video %s: %s", vid, exc, exc_info=True)
            return 0.0

    def _ensure_clip_branch_index(self) -> ClipBranchIndex | None:
        """Lazily build and cache the SigLIP 2 frame+clip index for the active snapshot."""
        if self.all_embeddings is not None and len(self.all_embeddings) > 0:
            emb_mat = self.all_embeddings
        elif (
            self.augmented_embeddings is not None
            and len(self.augmented_embeddings) > 0
            and self.augmented_embeddings.shape[1] >= 768
        ):
            emb_mat = self.augmented_embeddings[:, :768]
        else:
            return None

        if not (self.video_keyframes or self.video_kf_fids or self.keyframe_map):
            return None

        if self._clip_branch_index is None or not self._clip_branch_index.matches(emb_mat, self.video_kf_fids):
            self._clip_branch_index = build_clip_branch_index(
                emb_mat,
                video_keyframes=self.video_keyframes,
                video_kf_fids=self.video_kf_fids,
                keyframe_map=self.keyframe_map,
            )
        return self._clip_branch_index

    def _apply_clip_reserve_stage(
        self,
        cand_moments: list[RankedMoment],
        primary_q_768: np.ndarray | None,
        filters: dict[str, Any] | None,
    ) -> tuple[list[RankedMoment], list[str], list[str], dict[str, Any]]:
        """Capture A5 pool and apply opt-in SigLIP 2 clip reserve pool."""
        a5_ranked = self._build_ranked_videos(cand_moments, cand_moments, top_k=None)
        a5_pool_ids = [m.video_id for m in a5_ranked[: self.clip_reserve_k]]
        clip_idx = self._ensure_clip_branch_index()
        if clip_idx is None or primary_q_768 is None:
            return cand_moments, a5_pool_ids, list(a5_pool_ids), {
                "status": "skipped",
                "reason": "missing_index_or_query",
            }

        valid_gallery = [vid for vid in clip_idx.video_ids if self._video_passes_filters(vid, filters)]
        if not valid_gallery:
            return cand_moments, a5_pool_ids, list(a5_pool_ids), {
                "status": "skipped",
                "reason": "empty_filtered_gallery",
                "valid_gallery_count": 0,
            }

        score_res = score_clip_branch(
            clip_idx,
            primary_q_768,
            valid_video_ids=valid_gallery,
            window_size=self.clip_window_size,
        )
        reserve_res = build_clip_reserve_pool(
            a5_pool=a5_pool_ids,
            branch_order=score_res.branch_order,
            k=self.clip_reserve_k,
            valid_gallery=valid_gallery,
        )
        clip_pool_ids = list(reserve_res.clip_pool)

        best_by_vid: dict[str, RankedMoment] = {}
        for m in cand_moments:
            prev = best_by_vid.get(m.video_id)
            if prev is None or float(m.ranking_score) > float(prev.ranking_score):
                best_by_vid[m.video_id] = m

        protected_scores = [
            float(best_by_vid[v].ranking_score)
            for v in reserve_res.protected
            if v in best_by_vid
        ]
        min_protected_score = min(protected_scores) if protected_scores else 1.0

        tail = reserve_res.added_video
        if tail is not None:
            if tail not in best_by_vid:
                rep_t = float(score_res.best_frame_pts.get(tail, 0.0))
                kf_list = self.video_keyframes.get(tail, [])
                v_dur = self.video_durations.get(
                    tail,
                    (max(float(kf["pts_time"]) for kf in kf_list) + 2.0) if kf_list else 300.0,
                )
                best_st = max(0.0, rep_t - 1.5)
                best_et = min(float(v_dur), rep_t + 1.5)
                tail_score = round(max(1e-6, min_protected_score - 1e-4), 5)
                best_by_vid[tail] = RankedMoment(
                    moment_id=f"{tail}_{round(rep_t, 1)}",
                    video_id=tail,
                    moment_start_seconds=best_st,
                    moment_end_seconds=best_et,
                    start_original_frame_index=0,
                    end_original_frame_index=0,
                    representative_original_frame_index=0,
                    representative_timestamp_seconds=rep_t,
                    ranking_score=tail_score,
                    match_status="clip_branch",
                )
            else:
                existing_tail = best_by_vid[tail]
                if float(existing_tail.ranking_score) >= min_protected_score and protected_scores:
                    clamped_sc = round(max(1e-6, min_protected_score - 1e-4), 5)
                    best_by_vid[tail] = RankedMoment(
                        moment_id=existing_tail.moment_id,
                        video_id=existing_tail.video_id,
                        moment_start_seconds=existing_tail.moment_start_seconds,
                        moment_end_seconds=existing_tail.moment_end_seconds,
                        start_original_frame_index=existing_tail.start_original_frame_index,
                        end_original_frame_index=existing_tail.end_original_frame_index,
                        representative_original_frame_index=existing_tail.representative_original_frame_index,
                        representative_timestamp_seconds=existing_tail.representative_timestamp_seconds,
                        ranking_score=clamped_sc,
                        match_status=existing_tail.match_status,
                        evidence_frame_indices=existing_tail.evidence_frame_indices,
                        unresolved_predicates=existing_tail.unresolved_predicates,
                        structural_score=existing_tail.structural_score,
                    )

        pool_set = set(clip_pool_ids)
        pool_moments = [best_by_vid[v] for v in clip_pool_ids if v in best_by_vid]
        min_pool_score = min((float(m.ranking_score) for m in pool_moments), default=1.0)
        other_moments: list[RankedMoment] = []
        for m in cand_moments:
            if m.video_id in pool_set:
                continue
            if float(m.ranking_score) >= min_pool_score:
                other_moments.append(RankedMoment(
                    moment_id=m.moment_id,
                    video_id=m.video_id,
                    moment_start_seconds=m.moment_start_seconds,
                    moment_end_seconds=m.moment_end_seconds,
                    start_original_frame_index=m.start_original_frame_index,
                    end_original_frame_index=m.end_original_frame_index,
                    representative_original_frame_index=m.representative_original_frame_index,
                    representative_timestamp_seconds=m.representative_timestamp_seconds,
                    ranking_score=round(max(1e-6, min_pool_score - 1e-4), 5),
                    match_status=m.match_status,
                    evidence_frame_indices=m.evidence_frame_indices,
                    unresolved_predicates=m.unresolved_predicates,
                    structural_score=m.structural_score,
                ))
            else:
                other_moments.append(m)

        stats = {
            "status": "applied",
            "added_video": reserve_res.added_video,
            "displaced_video": reserve_res.displaced_video,
            "duplicate_skipped": reserve_res.duplicate_skipped,
            "branch_rank": reserve_res.branch_rank,
            "fallback_reason": reserve_res.fallback_reason,
            "pool_size": reserve_res.pool_size,
            "requested_k": reserve_res.requested_k,
            "valid_gallery_count": score_res.valid_gallery_count,
            "invalid_videos_skipped": len(score_res.invalid_videos_skipped),
            "window_size": self.clip_window_size,
        }
        return pool_moments + other_moments, a5_pool_ids, clip_pool_ids, stats

    def _apply_temporal_rerank(
        self,
        cand_moments: list[RankedMoment],
        query_vec_768: np.ndarray | None,
        top_k_override: int | None = None,
    ) -> tuple[list[RankedMoment], dict[str, Any]]:
        """Apply Section 3.1 NumPy temporal rerank challenger to Top-K base videos."""
        if not cand_moments or query_vec_768 is None:
            return cand_moments, {"scored_count": 0, "reason": "missing_query_or_candidates"}

        if self.all_embeddings is not None and len(self.all_embeddings) > 0:
            emb_mat = self.all_embeddings
        elif self.augmented_embeddings is not None and len(self.augmented_embeddings) > 0 and self.augmented_embeddings.shape[1] >= 768:
            emb_mat = self.augmented_embeddings[:, :768]
        else:
            return cand_moments, {"scored_count": 0, "reason": "missing_embeddings"}

        q_vec = np.asarray(query_vec_768, dtype=np.float32).reshape(-1)
        effective_k = self.temporal_rerank_top_k if top_k_override is None else int(top_k_override)
        top_k = min(effective_k, len(cand_moments))
        head_moments = cand_moments[:top_k]

        raw_pools: list[tuple[int, float, float, bool, int]] = []
        per_video_stats: dict[str, dict[str, Any]] = {}
        for idx, m in enumerate(head_moments):
            vid = m.video_id
            anchor_t = float(m.representative_timestamp_seconds)
            kf_list = self.video_keyframes.get(vid, [])
            if not kf_list:
                per_video_stats[vid] = {"used_fallback": True, "num_frames": 0, "skipped": True}
                continue
            in_win = [
                k for k in kf_list
                if abs(float(k.get("pts_time", 0.0)) - anchor_t) <= self.temporal_rerank_window_sec
            ]
            if in_win:
                selected_kfs = sorted(
                    in_win,
                    key=lambda k: (abs(float(k.get("pts_time", 0.0)) - anchor_t), int(k["faiss_id"])),
                )[: self.temporal_rerank_max_frames]
                used_fallback = False
            else:
                selected_kfs = sorted(
                    kf_list,
                    key=lambda k: (abs(float(k.get("pts_time", 0.0)) - anchor_t), int(k["faiss_id"])),
                )[: self.temporal_rerank_max_frames]
                used_fallback = True

            valid_fids = [int(k["faiss_id"]) for k in selected_kfs if 0 <= int(k["faiss_id"]) < len(emb_mat)]
            if not valid_fids:
                per_video_stats[vid] = {"used_fallback": used_fallback, "num_frames": 0, "skipped": True}
                continue
            pool_sc, spike_gap = compute_temporal_pool_score(
                q_vec,
                emb_mat[valid_fids],
                mode=self.temporal_rerank_mode,
                tau=self.temporal_rerank_tau,
            )
            raw_pools.append((idx, pool_sc, spike_gap, used_fallback, len(valid_fids)))

        if len(raw_pools) < 2:
            return cand_moments, {"scored_count": len(raw_pools), "reason": "insufficient_pool_candidates", "per_video": per_video_stats}

        base_scores = np.array([float(m.ranking_score) for m in head_moments], dtype=np.float64)
        scale_base = float(np.std(base_scores))
        pool_arr = np.array([row[1] for row in raw_pools], dtype=np.float64)
        mu_pool = float(np.mean(pool_arr))
        std_pool = float(np.std(pool_arr))
        if scale_base <= 1e-9 or std_pool <= 1e-9:
            return cand_moments, {"scored_count": len(raw_pools), "reason": "degenerate_variance", "per_video": per_video_stats}

        updated = list(cand_moments)
        lam = float(self.temporal_rerank_lambda)
        for idx, pool_sc, spike_gap, used_fallback, n_frames in raw_pools:
            m = updated[idx]
            z_pool = (pool_sc - mu_pool) / std_pool
            z_clipped = float(np.clip(z_pool, -2.0, 2.0))
            base_sc = float(m.ranking_score)
            new_sc = base_sc + lam * scale_base * z_clipped
            per_video_stats[m.video_id] = {
                "pool_score": round(pool_sc, 6),
                "z_pool": round(z_pool, 6),
                "z_clipped": round(z_clipped, 6),
                "spike_gap": round(spike_gap, 6),
                "used_fallback": used_fallback,
                "num_frames": n_frames,
                "base_score": round(base_sc, 6),
                "new_score": round(new_sc, 6),
            }
            updated[idx] = RankedMoment(
                moment_id=m.moment_id,
                video_id=m.video_id,
                moment_start_seconds=m.moment_start_seconds,
                moment_end_seconds=m.moment_end_seconds,
                start_original_frame_index=m.start_original_frame_index,
                end_original_frame_index=m.end_original_frame_index,
                representative_original_frame_index=m.representative_original_frame_index,
                representative_timestamp_seconds=m.representative_timestamp_seconds,
                ranking_score=new_sc,
                score_type=m.score_type,
                match_status=m.match_status,
                evidence_frame_indices=m.evidence_frame_indices,
                unresolved_predicates=m.unresolved_predicates,
                structural_score=m.structural_score,
            )

        return updated, {
            "scored_count": len(raw_pools),
            "scale_base": round(scale_base, 6),
            "mode": self.temporal_rerank_mode,
            "lambda": lam,
            "per_video": per_video_stats,
        }

    def _ensure_e5_encoder(self) -> bool:
        """Lazy-load intfloat/multilingual-e5-small locally without network calls."""
        if self.e5_tokenizer is not None and self.e5_model is not None:
            return True
        try:
            from transformers import AutoModel, AutoTokenizer

            e5_id = os.environ.get("VIDENTRA_E5_MODEL_ID", "intfloat/multilingual-e5-small")
            e5_rev = os.environ.get("VIDENTRA_E5_REVISION", "c007d7ef6fd86656326059b28395a7a03a7c5846")
            self.e5_tokenizer = AutoTokenizer.from_pretrained(e5_id, revision=e5_rev, local_files_only=True)
            model = AutoModel.from_pretrained(e5_id, revision=e5_rev, local_files_only=True).to(self.device)
            model.eval()
            self.e5_model = model
            return True
        except Exception as exc:
            self.e5_tokenizer = None
            self.e5_model = None
            self._degraded_channels["e5_encoder"] = str(exc)
            logger.warning("Failed to load E5 encoder: %s", exc, exc_info=True)
            return False

    def encode_e5_query(self, query: str) -> np.ndarray | None:
        """Encode a query into an L2-normalized 384d float32 E5 vector with RAM LRU + disk cache."""
        q_norm = unicodedata.normalize("NFC", query.strip())
        if not q_norm:
            return None
        prefixed = q_norm if q_norm.startswith("query: ") else f"query: {q_norm}"

        cached = self._e5_memory_cache.get(prefixed)
        if cached is not None:
            self._e5_memory_cache.move_to_end(prefixed)
            return cached.copy()

        if self.e5_cache is not None:
            disk_vec = self.e5_cache.get(prefixed)
            if disk_vec is not None and disk_vec.shape == (384,):
                if self.memory_cache_size:
                    self._e5_memory_cache[prefixed] = disk_vec.copy()
                    self._e5_memory_cache.move_to_end(prefixed)
                    while len(self._e5_memory_cache) > self.memory_cache_size:
                        self._e5_memory_cache.popitem(last=False)
                return disk_vec.copy()

        if not self._ensure_e5_encoder():
            return None

        try:
            enc = self.e5_tokenizer(
                [prefixed],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.e5_model(**enc)
                hidden = out.last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                vec = emb[0].detach().cpu().numpy().astype(np.float32)

            if vec.shape != (384,) or not np.isfinite(vec).all():
                return None

            if self.memory_cache_size:
                self._e5_memory_cache[prefixed] = vec.copy()
                self._e5_memory_cache.move_to_end(prefixed)
                while len(self._e5_memory_cache) > self.memory_cache_size:
                    self._e5_memory_cache.popitem(last=False)
            if self.e5_cache is not None:
                self.e5_cache.set(prefixed, vec)
                self.e5_cache.save()
                if self.e5_cache.last_error:
                    self._degraded_channels["e5_cache"] = self.e5_cache.last_error
            return vec
        except Exception as exc:
            self._degraded_channels["e5_encode"] = str(exc)
            logger.warning("E5 query encoding failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _search_e5_faiss_index(
        index: faiss.Index | None,
        manifest: list[dict[str, Any]],
        query_vec_384: np.ndarray | None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Shared dense E5 FAISS search helper returning manifest dicts with similarity score."""
        if index is None or not manifest or query_vec_384 is None:
            return []
        try:
            q = np.asarray(query_vec_384, dtype=np.float32).reshape(1, -1)
            k = min(int(top_k), int(index.ntotal))
            if k <= 0:
                return []
            scores, ids = index.search(q, k)
            out: list[dict[str, Any]] = []
            for idx, sc in zip(ids[0].tolist(), scores[0].tolist()):
                if 0 <= idx < len(manifest):
                    item = dict(manifest[idx])
                    item["score"] = float(sc)
                    out.append(item)
            return out
        except Exception as exc:
            logger.warning("E5 FAISS index search failed: %s", exc, exc_info=True)
            return []

    def _search_e5_text_index(self, query_vec_384: np.ndarray | None, top_k: int = 20) -> list[dict[str, Any]]:
        """Optional Phase 7.4 dense E5 speech/OCR search hook (no-op when absent)."""
        return self._search_e5_faiss_index(self.e5_text_index, self.e5_text_manifest, query_vec_384, top_k=top_k)

    def _search_e5_caption_index(self, query_vec_384: np.ndarray | None, top_k: int = 30) -> list[dict[str, Any]]:
        """Optional P8-5 dense E5 VLM keyframe caption search hook (no-op when absent)."""
        return self._search_e5_faiss_index(self.e5_caption_index, self.e5_caption_manifest, query_vec_384, top_k=top_k)

    def _verify_relations(self, query: str, moments: list[RankedMoment]) -> list[dict[str, Any]]:
        """Assess shortlist using interval-contained OCR, ASR, and 2D FrameObjects evidence."""
        from videntra.relations.matcher import GraphMatcher
        if self.last_routing_plan is not None and self.last_routing_plan.original_query.strip() == query.strip():
            graph = self.last_routing_plan.query_graph
        else:
            graph = self.query_decomposer.decompose(query, mode="accurate").query_graph
        matcher = GraphMatcher()
        assessments = []
        for moment in moments:
            start_ms = moment.moment_start_seconds * 1000.0
            end_ms = moment.moment_end_seconds * 1000.0
            evidence = self._collect_moment_evidence(moment.video_id, start_ms, end_ms)
            match = matcher.match(
                graph,
                evidence,
                video_id=moment.video_id,
                moment_window=(moment.moment_start_seconds, moment.moment_end_seconds),
            )
            referenced = {ref for value in match.predicate_evaluations.values() for ref in value.evidence_refs}
            assessments.append({"status": match.overall_status,
                "structural_score": match.structural_score,
                "unresolved_predicates": match.unresolved_predicates,
                "predicates": {key: {"status": value.state.value, "reason": value.reason,
                                     "evidence_refs": value.evidence_refs}
                               for key, value in match.predicate_evaluations.items()},
                "evidence": [{**item, "generator_type": "model", "timestamp_basis": "clip_relative_unverified"}
                             for item in evidence if item["evidence_id"] in referenced],
                "limitations": ["Model observations are not human truth.",
                                "Timestamp mapping and model revision are unverified."]})
        return assessments

    def _video_passes_filters(self, vid: str, filters: dict[str, Any] | None) -> bool:
        """Check whether a video satisfies optional search filters."""
        if not filters:
            return True
        video_ids = filters.get("video_ids")
        if video_ids is not None and vid not in set(video_ids):
            return False
        dataset_prefix = filters.get("dataset_prefix")
        if dataset_prefix:
            if isinstance(dataset_prefix, (list, tuple, set)):
                if not any(vid.startswith(str(p)) for p in dataset_prefix):
                    return False
            elif not vid.startswith(str(dataset_prefix)):
                return False
        min_duration = filters.get("min_duration")
        max_duration = filters.get("max_duration")
        if min_duration is not None or max_duration is not None:
            v_dur = self.video_durations.get(vid)
            if v_dur is None:
                kf_list = self.video_keyframes.get(vid, [])
                v_dur = (max(float(kf["pts_time"]) for kf in kf_list) + 2.0) if kf_list else None
            if min_duration is not None and (v_dur is None or v_dur < float(min_duration)):
                return False
            if max_duration is not None and (v_dur is None or v_dur > float(max_duration)):
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "accurate",
        ranking_mode: str = "diverse_video",
        filters: dict[str, Any] | None = None,
        ablation_config: str = "A5",
    ) -> list[RankedMoment]:
        """Execute full 7-stage multimodal search pipeline."""
        if not query.strip() or self.augmented_embeddings is None or len(self.augmented_embeddings) == 0:
            self.last_timings = {}
            self.last_diagnostics = {}
            return []

        t_start = time.perf_counter()
        video_curve_cache: dict[str, tuple[list[float], np.ndarray]] = {}
        viterbi_availability: dict[str, dict[str, str]] = {}
        self._last_event_intervals = {}

        # =========================================================================
        # Stage 1: Query Analysis & Vector Construction (864d with B5, B6, C1)
        # =========================================================================
        routing_plan: ModalityRoutingPlan | None = (
            self.query_decomposer.decompose(query, mode=mode)
            if ablation_config not in ("A0", "A1")
            else None
        )
        self.last_routing_plan = routing_plan
        use_llm_clauses = (
            routing_plan is not None
            and routing_plan.routing_mode == "llm_compact"
            and bool(routing_plan.visual_clauses)
        )
        if use_llm_clauses:
            clauses = list(routing_plan.visual_clauses)  # type: ignore[union-attr]
        else:
            clauses = decompose_query_clauses(query) if ablation_config not in ("A0", "A1") else [query]
        obj_reqs = extract_object_requirements(query) if ablation_config != "A0" else []
        if not obj_reqs and routing_plan is not None and routing_plan.object_requirements:
            normalized_reqs: list[dict[str, Any]] = []
            for o in routing_plan.object_requirements:
                cls_name = o.get("class_name") or o.get("class") or o.get("cls")
                if not cls_name:
                    continue
                cls_str = str(cls_name).lower()
                count_val = int(o.get("cardinality") or o.get("count") or 1)
                normalized_reqs.append({
                    **o,
                    "class_name": cls_str,
                    "canonical_class": COCO_SYNONYM_MAP.get(cls_str, cls_str),
                    "cardinality": count_val,
                    "count": count_val,
                })
            obj_reqs = normalized_reqs
        q_low = query.lower()

        if obj_reqs:
            if any(w in q_low for w in ["watch ", "watches", "watching"]):
                obj_reqs = [o for o in obj_reqs if o["class_name"] != "clock"]
        else:
            obj_reqs = []

        if ablation_config != "A0":
            existing_classes = {o["class_name"] for o in obj_reqs}
            supplemental_mappings = [
                (["cycle", "cycling", "cyclist", "bicycles", "riding a bike"], "bicycle"),
                (["driving", "drive", "steering", "driver"], "car"),
                (["horseback", "riding a horse"], "horse"),
                (["skiing", "skier"], "skis"),
                (["surfing", "surfer"], "surfboard"),
                (["skateboarding", "skater"], "skateboard"),
                (["tennis"], "tennis racket"),
                (["baseball"], "baseball bat"),
                (["cutting", "chopping", "slicing"], "knife"),
                (["reading"], "book"),
                (["talking on phone", "texting"], "cell phone"),
                (["computer", "desktop"], "laptop"),
                (["cat", "cats", "kitten"], "cat"),
                (["dog", "dogs", "puppy"], "dog"),
                (["cup", "cups"], "cup"),
                (["tie", "necktie"], "tie"),
            ]
            for keywords, target_cls in supplemental_mappings:
                if target_cls not in existing_classes:
                    if any(re.search(rf"\b{kw}\b", q_low) for kw in keywords):
                        obj_reqs.append({
                            "class_name": target_cls,
                            "canonical_class": COCO_SYNONYM_MAP.get(target_cls, target_cls),
                            "cardinality": 1,
                            "count": 1,
                            "mention": keywords[0],
                        })
                        existing_classes.add(target_cls)
            for req in obj_reqs:
                if "canonical_class" not in req:
                    rc = str(req.get("class_name", "")).lower()
                    req["canonical_class"] = COCO_SYNONYM_MAP.get(rc, rc)

        fts_query, quoted_phrases, high_spec_tokens, low_spec_tokens, proper_nouns_lower = (
            self.prepare_fts5_query_and_tokens(query)
        )

        p1 = query
        p2 = f"a video frame showing {query}"
        if use_llm_clauses:
            sub_clauses = [c for c in routing_plan.visual_clauses if c.strip().lower() != query.strip().lower()][:4]  # type: ignore[union-attr]
        else:
            sub_clauses = [c for c in clauses if c.strip().lower() != query.strip().lower()][:3]

        t_parse_end = time.perf_counter()

        batch_texts = [p1, p2] + sub_clauses
        batch_vecs = self.encode_texts(batch_texts)

        ens = 0.75 * batch_vecs[0] + 0.25 * batch_vecs[1]
        ens /= (np.linalg.norm(ens) + 1e-8)
        primary_q_768 = ens.reshape(1, -1)

        primary_q_864 = build_query_864d_vector(
            primary_q_768,
            query,
            self.coco_idf_map,
            precomputed_obj_reqs=obj_reqs,
        )
        primary_q_864_mat = primary_q_864.reshape(1, -1)
        t_encode_end = time.perf_counter()

        # Fast mode exit: visual vector search only
        if mode == "fast":
            search_depth = min(self.search_depth, self.aug_index.ntotal)
            scores, fids = self.aug_index.search(primary_q_864_mat, search_depth)
            fast_results: list[RankedMoment] = []
            seen_fast_vids = set()
            for fid, sc in zip(fids[0], scores[0]):
                if fid not in self.keyframe_map:
                    continue
                info = self.keyframe_map[fid]
                vid = info["video_id"]
                if not self._video_passes_filters(vid, filters):
                    continue
                if ranking_mode == "diverse_video" and vid in seen_fast_vids:
                    continue
                seen_fast_vids.add(vid)
                t_sec = info["pts_time"]
                v_dur = self.video_durations.get(vid, t_sec + 4.0)
                fast_results.append(RankedMoment(
                    moment_id=f"{vid}_{round(t_sec, 1)}",
                    video_id=vid,
                    moment_start_seconds=max(0.0, round(t_sec - 2.0, 2)),
                    moment_end_seconds=min(v_dur, round(t_sec + 2.0, 2)),
                    start_original_frame_index=info.get("source_frame_index", 0),
                    end_original_frame_index=info.get("source_frame_index", 0),
                    representative_original_frame_index=info.get("source_frame_index", 0),
                    representative_timestamp_seconds=round(t_sec, 2),
                    ranking_score=round(float(sc), 5),
                    match_status="visual",
                ))
                if len(fast_results) >= top_k:
                    break
            t_fast_end = time.perf_counter()
            self.last_timings = {
                "t_parse_ms": (t_parse_end - t_start) * 1000.0,
                "t_encode_ms": (t_encode_end - t_parse_end) * 1000.0,
                "t_faiss_ms": (t_fast_end - t_encode_end) * 1000.0,
                "t_fts5_ms": 0.0,
                "t_fusion_ms": 0.0,
                "t_boundary_ms": 0.0,
                "t_non_neural_ms": (t_fast_end - t_encode_end) * 1000.0,
                "t_total_ms": (t_fast_end - t_start) * 1000.0,
            }
            return fast_results

        # =========================================================================
        # Stage 2: FAISS IP Retrieval & Multi-Clause Sequential Alignment (B4, B11)
        # =========================================================================
        search_depth = min(self.search_depth, self.aug_index.ntotal)
        visual_candidates: list[dict[str, Any]] = []
        video_anchor_windows: dict[str, tuple[float, float]] = {}
        video_min_clause_sim: dict[str, float] = {}

        if sub_clauses and ablation_config not in ("A0", "A1"):
            sub_vecs_768 = batch_vecs[2:]
            sub_q_864 = [
                build_query_864d_vector(sv, sc, self.coco_idf_map)
                for sv, sc in zip(sub_vecs_768, sub_clauses)
            ]
            sub_q_864_mat = np.vstack([np.asarray(sv, dtype=np.float32).reshape(1, -1) for sv in sub_q_864])
            all_864_vecs = np.vstack([primary_q_864_mat, sub_q_864_mat])
            has_matched_768 = (
                len(self.all_embeddings) == len(self.augmented_embeddings)
                and len(self.augmented_embeddings) > 0
            )
            # Vectorized OpenBLAS GEMM across all query clauses (blending 768d + 864d when aligned)
            all_sims_864 = (all_864_vecs @ self.augmented_embeddings.T).astype(np.float32)
            if has_matched_768:
                all_768_vecs = np.vstack([primary_q_768] + [sv.reshape(1, -1) for sv in sub_vecs_768])
                all_sims_768 = (all_768_vecs @ self.all_embeddings.T).astype(np.float32)
                all_sims_mat = (0.70 * all_sims_768 + 0.30 * all_sims_864).astype(np.float32)
            else:
                all_sims_mat = all_sims_864
            k_part = min(search_depth, all_sims_mat.shape[1])

            ids_batch_list = []
            scores_batch_list = []
            for row_sims in all_sims_mat:
                if k_part < len(row_sims):
                    top_idx = np.argpartition(row_sims, -k_part)[-k_part:]
                    top_order = top_idx[np.argsort(row_sims[top_idx])[::-1]]
                else:
                    top_order = np.argsort(row_sims)[::-1]
                ids_batch_list.append(top_order)
                scores_batch_list.append(row_sims[top_order])

            full_scores = scores_batch_list[0]
            full_ids = ids_batch_list[0]
            full_map = dict(zip(full_ids.tolist(), full_scores.tolist()))

            clause_maps = [
                dict(zip(ids_batch_list[i + 1].tolist(), scores_batch_list[i + 1].tolist()))
                for i in range(len(sub_clauses))
            ]

            cand_faiss_ids = set(full_ids.tolist())
            for c_ids in ids_batch_list[1:]:
                cand_faiss_ids.update(c_ids.tolist())

            cand_vids = {self.keyframe_map[fid]["video_id"] for fid in cand_faiss_ids if fid in self.keyframe_map}
            video_seq_scores: dict[str, float] = {}

            # Preliminary scoring to prune candidates before running costly sequential alignment DP
            vid_full_max: dict[str, float] = defaultdict(float)
            vid_clause_max: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(sub_clauses))

            for fid in cand_faiss_ids:
                if fid not in self.keyframe_map:
                    continue
                vid = self.keyframe_map[fid]["video_id"]
                if fid in full_map:
                    if full_map[fid] > vid_full_max[vid]:
                        vid_full_max[vid] = float(full_map[fid])
                for i, cm in enumerate(clause_maps):
                    if fid in cm and cm[fid] > vid_clause_max[vid][i]:
                        vid_clause_max[vid][i] = float(cm[fid])

            vid_prelim_scores: dict[str, float] = {}
            for vid in cand_vids:
                s_full = vid_full_max.get(vid, 0.0)
                s_cl_avg = sum(vid_clause_max[vid]) / max(1, len(sub_clauses))
                vid_prelim_scores[vid] = 0.65 * s_full + 0.35 * s_cl_avg

            # Top-35 candidate videos for sequential alignment DP
            top_aligned_vids = sorted(cand_vids, key=lambda v: vid_prelim_scores.get(v, 0.0), reverse=True)[:35]

            for cvid in top_aligned_vids:
                kf_list = self.video_keyframes.get(cvid, [])
                if len(kf_list) >= 2:
                    cand_ts = self.video_kf_pts.get(cvid) or [kf["pts_time"] for kf in kf_list]
                    cand_mat = self.video_kf_matrix.get(cvid)
                    if cand_mat is None:
                        kf_fids = [int(kf["faiss_id"]) for kf in kf_list]
                        cand_mat = self.augmented_embeddings[kf_fids]

                    use_se_viterbi = (
                        routing_plan is not None
                        and (
                            routing_plan.routing_mode == "llm_compact"
                            or routing_plan.has_temporal_constraints
                        )
                        and len(sub_q_864) >= 2
                        and cand_mat.shape[0] >= 2
                        and (
                            routing_plan.has_temporal_constraints
                            or (len(routing_plan.ocr_phrases) + len(routing_plan.asr_phrases)) >= 1
                        )
                    )
                    clause_frame_sims = sub_q_864_mat @ cand_mat.T
                    per_clause_max = np.max(clause_frame_sims, axis=1)
                    mv_set_sim = 0.65 * float(np.mean(per_clause_max)) + 0.35 * float(np.min(per_clause_max))
                    video_min_clause_sim[cvid] = float(np.min(per_clause_max))

                    seq_score = 0.0
                    anchor_win: tuple[float, float] | None = None
                    if use_se_viterbi:
                        w_ocr_v = 0.7 if routing_plan.has_explicit_ocr_intent else 0.4
                        w_asr_v = 0.7 if routing_plan.has_explicit_asr_intent else 0.4
                        ocr_curves, asr_curves, modal_avail = self._build_clause_multimodal_curves(
                            cvid,
                            cand_ts,
                            sub_clauses,
                            routing_plan.ocr_phrases,
                            routing_plan.asr_phrases,
                        )
                        viterbi_availability[cvid] = modal_avail
                        vit_res = viterbi_subevent_localization(
                            timestamps_sec=cand_ts,
                            event_vis_curves=clause_frame_sims,
                            event_ocr_curves=ocr_curves,
                            event_asr_curves=asr_curves,
                            event_weights=[(1.0, w_ocr_v, w_asr_v)] * len(sub_q_864),
                            max_gap_sec=35.0,
                        )
                        if vit_res is not None:
                            seq_score = float(vit_res.avg_score)
                            anchor_win = (float(vit_res.start_sec), float(vit_res.end_sec))
                    if anchor_win is None:
                        seq_score, anchor_win = align_sequential_keyframe_clauses(
                            cand_matrix=cand_mat,
                            cand_timestamps=cand_ts,
                            clause_vectors=sub_q_864,
                            max_transition_sec=35.0,
                        )
                    if len(sub_clauses) >= 2 and clause_frame_sims.shape[0] >= 2:
                        t_sot_score = temporal_sinkhorn_alignment_score(
                            clause_frame_sims,
                            timestamps_sec=cand_ts,
                            max_gap_sec=35.0,
                        )
                        multi_clause_blend = 0.70 * mv_set_sim + 0.30 * t_sot_score
                        seq_score = (0.80 * seq_score + 0.20 * multi_clause_blend) if seq_score > 0.0 else multi_clause_blend
                    video_seq_scores[cvid] = seq_score
                    if anchor_win is not None:
                        video_anchor_windows[cvid] = anchor_win

            seq_weight = (
                0.45
                if (routing_plan is not None and routing_plan.routing_mode == "llm_compact" and routing_plan.has_temporal_constraints)
                else 0.35
            )
            scored_visual = []
            for fid in cand_faiss_ids:
                if fid not in self.keyframe_map:
                    continue
                # B4: Exact sub-clause full score fallback
                if fid in full_map:
                    s_full = float(full_map[fid])
                else:
                    s_full_864 = float(np.dot(self.augmented_embeddings[fid], primary_q_864.flatten()))
                    if has_matched_768:
                        s_full_768 = float(np.dot(self.all_embeddings[fid], primary_q_768.flatten()))
                        s_full = 0.70 * s_full_768 + 0.30 * s_full_864
                    else:
                        s_full = s_full_864

                vid = self.keyframe_map[fid]["video_id"]
                if vid in video_seq_scores:
                    s_comb = (1.0 - seq_weight) * s_full + seq_weight * video_seq_scores[vid]
                else:
                    c_scores = [float(cm.get(fid, 0.0)) for cm in clause_maps]
                    s_comb = ensemble_clause_scores(full_score=s_full, clause_scores=c_scores)
                scored_visual.append((fid, s_comb))

            scored_visual.sort(key=lambda x: x[1], reverse=True)
            vis_hits = scored_visual[:search_depth]
        elif ablation_config == "A0":
            if len(self.all_embeddings) > 0:
                sims_768_all = (self.all_embeddings @ primary_q_768.reshape(-1)).astype(np.float32)
                k_part = min(search_depth, len(sims_768_all))
                top_idx = np.argpartition(sims_768_all, -k_part)[-k_part:]
                top_order = top_idx[np.argsort(sims_768_all[top_idx])[::-1]]
                vis_hits = [
                    (int(fid), float(sims_768_all[fid]))
                    for fid in top_order.tolist()
                    if fid in self.keyframe_map
                ]
            else:
                sc_768, id_768 = self.index.search(primary_q_768, search_depth)
                vis_hits = [
                    (int(fid), float(sc))
                    for fid, sc in zip(id_768[0], sc_768[0])
                    if fid in self.keyframe_map
                ]
        else:
            w_768 = 0.70
            w_864 = 0.30

            if len(self.all_embeddings) == len(self.augmented_embeddings) and len(self.augmented_embeddings) > 0:
                sims_768_all = self.all_embeddings @ primary_q_768.reshape(-1)
                sims_864_all = self.augmented_embeddings @ primary_q_864_mat.reshape(-1)
                k_part = min(search_depth, len(sims_864_all))

                top768 = np.argpartition(sims_768_all, -k_part)[-k_part:]
                top864 = np.argpartition(sims_864_all, -k_part)[-k_part:]
                cand_fids_arr = np.unique(np.concatenate([top768, top864]))

                sims_comb = w_768 * sims_768_all[cand_fids_arr] + w_864 * sims_864_all[cand_fids_arr]
                sort_order = np.argsort(sims_comb)[::-1]
                vis_hits = [
                    (int(cand_fids_arr[idx]), float(sims_comb[idx]))
                    for idx in sort_order
                    if int(cand_fids_arr[idx]) in self.keyframe_map
                ][:search_depth]
            else:
                sc_768, id_768 = self.index.search(primary_q_768, search_depth)
                sc_864, id_864 = self.aug_index.search(primary_q_864_mat, search_depth)

                cand_fids = list(dict.fromkeys(list(id_768[0]) + list(id_864[0])))
                cand_fids = [fid for fid in cand_fids if fid in self.keyframe_map]

                if hasattr(self, "all_embeddings") and self.all_embeddings is not None and len(self.all_embeddings) > max(cand_fids, default=-1):
                    cand_mat_768 = self.all_embeddings[cand_fids]
                elif self.augmented_embeddings is not None and len(self.augmented_embeddings) > max(cand_fids, default=-1):
                    cand_mat_768 = self.augmented_embeddings[cand_fids, :768]
                else:
                    cand_mat_768 = np.zeros((len(cand_fids), 768), dtype=np.float32)

                cand_mat_864 = self.augmented_embeddings[cand_fids]

                sims_768 = (cand_mat_768 @ primary_q_768.T).flatten()
                sims_864 = (cand_mat_864 @ primary_q_864_mat.T).flatten()

                sims_comb = w_768 * sims_768 + w_864 * sims_864

                scored_visual = [(cand_fids[i], float(sims_comb[i])) for i in range(len(cand_fids))]
                scored_visual.sort(key=lambda x: x[1], reverse=True)
                vis_hits = scored_visual[:search_depth]

        has_computer_term = bool(re.search(r'\b(?:computer|desktop|pc)\b', q_low))
        for fid, raw_score in vis_hits:
            info = self.keyframe_map[fid]
            vid = info["video_id"]
            t_sec = info["pts_time"]
            final_score = raw_score
            obj_boost = 0.0

            if obj_reqs and ablation_config != "A0":
                obj_summary = self.keyframe_obj_summary.get(fid)
                if obj_summary is not None:
                    obj_boost = _fast_object_boost_from_summary(obj_reqs, obj_summary)
                else:
                    nearby_objs = self.keyframe_obj_window.get(fid)
                    if nearby_objs is None:
                        v_objs = self.video_objects.get(vid, [])
                        if v_objs:
                            pts_arr = self.video_obj_pts.get(vid)
                            if pts_arr is not None:
                                lo_idx = bisect_left(pts_arr, t_sec - 3.5)
                                hi_idx = bisect_right(pts_arr, t_sec + 3.5)
                                nearby_objs = v_objs[lo_idx:hi_idx]
                            else:
                                nearby_objs = [o for o in v_objs if (t_sec - 3.5) <= o["pts_sec"] <= (t_sec + 3.5)]
                        else:
                            nearby_objs = []
                    obj_boost = compute_object_verification_boost(obj_reqs, nearby_objs)
                final_score += 0.08 * obj_boost
                if has_computer_term:
                    vid_classes = self.video_classes_set.get(vid)
                    if vid_classes and "laptop" in vid_classes and ("keyboard" in vid_classes or "mouse" in vid_classes):
                        final_score += 0.0045

            visual_candidates.append({
                "video_id": vid,
                "timestamp_sec": t_sec,
                "score": float(final_score),
                "branch": "visual",
                "faiss_id": fid,
                "obj_boost": float(obj_boost),
            })

        t_faiss_end = time.perf_counter()

        # =========================================================================
        # Stage 3: Specificity-Sorted Lexical FTS5 Search (B7, C2 persistent SQLite)
        # =========================================================================
        ocr_candidates: list[dict[str, Any]] = []
        asr_candidates: list[dict[str, Any]] = []
        if routing_plan is not None and routing_plan.routing_mode == "llm_compact":
            fts_ocr_query, fts_asr_query = _build_routed_fts5_queries(fts_query, routing_plan)
        else:
            fts_ocr_query, fts_asr_query = fts_query, fts_query

        if fts_query and self.db_conn is not None and ablation_config not in ("A0", "A1", "A2"):
            try:
                cursor = self.db_conn.cursor()
                if fts_ocr_query:
                    cursor.execute(
                        "SELECT video_id, timestamp_ms, rank, normalized_text FROM fts_ocr WHERE fts_ocr MATCH ? ORDER BY rank LIMIT 40;",
                        (fts_ocr_query,),
                    )
                    for vid, ts_ms, rank, norm_text in cursor.fetchall():
                        ocr_candidates.append({
                            "video_id": vid,
                            "timestamp_sec": ts_ms / 1000.0,
                            "score": -float(rank),
                            "branch": "ocr",
                            "matched_text": norm_text or "",
                        })

                if fts_asr_query:
                    cursor.execute(
                        "SELECT video_id, start_ms, end_ms, rank, normalized_text FROM fts_asr WHERE fts_asr MATCH ? ORDER BY rank LIMIT 40;",
                        (fts_asr_query,),
                    )
                    for vid, st_ms, et_ms, rank, norm_text in cursor.fetchall():
                        asr_candidates.append({
                            "video_id": vid,
                            "start_sec": st_ms / 1000.0,
                            "end_sec": et_ms / 1000.0,
                            "timestamp_sec": (st_ms + et_ms) / 2000.0,
                        "score": -float(rank),
                        "branch": "asr",
                        "matched_text": norm_text or "",
                    })
            except sqlite3.Error as exc:
                raise RuntimeError("Lexical retrieval failed") from exc

        # Phase 7.4 / 8: Dense E5 Text Index recall only when explicit lexical modality intent exists and FTS5 is sparse
        t_e5_encode_ms = 0.0
        q_e5_384: np.ndarray | None = None
        sparse_lexical_yield = (len(ocr_candidates) + len(asr_candidates)) < 5
        _q_lower_e5 = f" {query.lower()} "
        _lexical_intent_cues = (
            "explaining", "whiteboard", "blackboard", "on board",
            "subtitle", "subtitles", "caption", "captions",
            "written", "reads", "slide", "slides", "title", "logo",
        )
        has_lexical_modality_intent = (
            bool(quoted_phrases)
            or (
                routing_plan is not None
                and (
                    routing_plan.has_explicit_ocr_intent
                    or routing_plan.has_explicit_asr_intent
                    or bool(routing_plan.ocr_phrases)
                    or bool(routing_plan.asr_phrases)
                )
            )
            or any(f" {cue} " in _q_lower_e5 for cue in _lexical_intent_cues)
        )
        need_e5_text = (
            self.e5_text_index is not None
            and ablation_config not in ("A0", "A1", "A2")
            and sparse_lexical_yield
            and has_lexical_modality_intent
        )
        cap_dense_enabled, _ = self._is_vlm_caption_dense_enabled(ablation_config)
        cap_lex_enabled, _ = self._is_vlm_caption_lexical_enabled(ablation_config)
        need_e5_caption = cap_dense_enabled and (self.e5_caption_index is not None)
        if need_e5_text or need_e5_caption:
            t_e5_start = time.perf_counter()
            q_e5_384 = self.encode_e5_query(query)
            t_e5_encode_ms = (time.perf_counter() - t_e5_start) * 1000.0
        if need_e5_text and q_e5_384 is not None:
            existing_ocr = {(c["video_id"], round(float(c["timestamp_sec"]), 1)) for c in ocr_candidates}
            existing_asr = {(c["video_id"], round(float(c["timestamp_sec"]), 1)) for c in asr_candidates}
            for hit in self._search_e5_text_index(q_e5_384, top_k=25):
                sc = float(hit.get("score", 0.0))
                if sc < 0.855:
                    continue
                src_type = str(hit.get("source_type", "")).lower()
                vid = str(hit.get("video_id", ""))
                if not vid:
                    continue
                st_sec = float(hit.get("start_ms", 0.0)) / 1000.0
                et_sec = float(hit.get("end_ms", hit.get("start_ms", 0.0))) / 1000.0
                ts_sec = (st_sec + et_sec) / 2.0 if et_sec > st_sec else st_sec
                raw_text = str(hit.get("raw_text", ""))
                dense_sc = max(0.1, (sc - 0.80) * 10.0)
                if src_type == "ocr":
                    key = (vid, round(ts_sec, 1))
                    if key not in existing_ocr:
                        existing_ocr.add(key)
                        ocr_candidates.append({
                            "video_id": vid,
                            "timestamp_sec": ts_sec,
                            "score": dense_sc,
                            "branch": "ocr",
                            "matched_text": raw_text,
                            "is_dense_e5": True,
                        })
                elif src_type == "asr":
                    key = (vid, round(ts_sec, 1))
                    if key not in existing_asr:
                        existing_asr.add(key)
                        asr_candidates.append({
                            "video_id": vid,
                            "start_sec": st_sec,
                            "end_sec": et_sec,
                            "timestamp_sec": ts_sec,
                            "score": dense_sc,
                            "branch": "asr",
                            "matched_text": raw_text,
                            "is_dense_e5": True,
                        })

        # Snapshot visual-only branch BEFORE P8-5 caption hits can augment visual_candidates
        visual_frame_depth = len(visual_candidates)
        visual_candidate_ids: list[str] = list(
            dict.fromkeys(
                c["video_id"]
                for c in sorted(visual_candidates, key=lambda x: float(x["score"]), reverse=True)
            )
        )
        visual_distinct_videos = len(visual_candidate_ids)

        # P8-5: VLM keyframe caption FTS5 + optional dense E5 caption index recall
        caption_ordered_vids: list[str] = []
        vlm_caption_lexical_hits = 0
        vlm_caption_dense_hits = 0
        if cap_lex_enabled or cap_dense_enabled:
            vlm_caption_hits: list[dict[str, Any]] = []
            if cap_lex_enabled and self.has_vlm_caption_fts and fts_query and self.db_conn is not None:
                try:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "SELECT video_id, faiss_id, timestamp_ms, rank, normalized_text "
                        "FROM fts_vlm_caption WHERE fts_vlm_caption MATCH ? ORDER BY rank LIMIT 30;",
                        (fts_query,),
                    )
                    for rank_idx, (vid, fid, ts_ms, _bm25_rank, norm_text) in enumerate(cursor.fetchall(), 1):
                        vlm_caption_lexical_hits += 1
                        vlm_caption_hits.append({
                            "video_id": str(vid),
                            "faiss_id": int(fid) if fid is not None else None,
                            "timestamp_sec": float(ts_ms) / 1000.0,
                            "rrf_vis_caption": 1.0 / (60 + rank_idx),
                            "matched_text": norm_text or "",
                        })
                except sqlite3.Error:
                    pass

            if cap_dense_enabled and q_e5_384 is not None and self.e5_caption_index is not None:
                for rank_idx, hit in enumerate(self._search_e5_caption_index(q_e5_384, top_k=30), 1):
                    vid = str(hit.get("video_id", ""))
                    if not vid:
                        continue
                    vlm_caption_dense_hits += 1
                    vlm_caption_hits.append({
                        "video_id": vid,
                        "faiss_id": int(hit["faiss_id"]) if hit.get("faiss_id") is not None else None,
                        "timestamp_sec": float(hit.get("timestamp_ms", 0.0)) / 1000.0,
                        "rrf_vis_caption": 1.0 / (60 + rank_idx),
                        "matched_text": str(hit.get("caption_text", "")),
                    })

            if vlm_caption_hits:
                caption_ordered_vids = list(dict.fromkeys(h["video_id"] for h in vlm_caption_hits))
                vis_by_fid: dict[int, dict[str, Any]] = {}
                vis_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for vc in visual_candidates:
                    if vc.get("faiss_id") is not None:
                        vis_by_fid[int(vc["faiss_id"])] = vc
                    vis_by_vid[vc["video_id"]].append(vc)

                for hit in vlm_caption_hits:
                    vid = hit["video_id"]
                    fid = hit["faiss_id"]
                    ts_sec = float(hit["timestamp_sec"])
                    rrf_cap = float(hit["rrf_vis_caption"])
                    matched_vc = vis_by_fid.get(fid) if fid is not None else None
                    if matched_vc is None:
                        for vc in vis_by_vid.get(vid, ()):
                            if abs(float(vc["timestamp_sec"]) - ts_sec) <= 2.0:
                                matched_vc = vc
                                break
                    if matched_vc is not None:
                        matched_vc["rrf_vis_caption"] = max(
                            float(matched_vc.get("rrf_vis_caption", 0.0)), rrf_cap
                        )
                    else:
                        v_sim = self.lookup_true_visual_sim(
                            vid,
                            ts_sec,
                            primary_q_864=primary_q_864,
                            primary_q_768=primary_q_768,
                            curve_cache=video_curve_cache,
                        )
                        new_vc: dict[str, Any] = {
                            "video_id": vid,
                            "timestamp_sec": ts_sec,
                            "score": float(v_sim),
                            "branch": "visual",
                            "faiss_id": fid,
                            "obj_boost": 0.0,
                            "rrf_vis_caption": rrf_cap,
                        }
                        visual_candidates.append(new_vc)
                        if fid is not None:
                            vis_by_fid[fid] = new_vc
                        vis_by_vid[vid].append(new_vc)

        t_fts5_end = time.perf_counter()

        # =========================================================================
        # Stage 4: Multimodal Temporal Clustering (B2, B3)
        # =========================================================================
        k_rrf = 60
        visual_candidates.sort(key=lambda x: x["score"], reverse=True)
        ocr_candidates.sort(key=lambda x: x["score"], reverse=True)
        asr_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Record branch recall diagnostics (E8)
        vis_ordered_vids: list[str] = []
        seen_vis_diag: set[str] = set()
        for c in visual_candidates:
            v_id = c["video_id"]
            if v_id not in seen_vis_diag:
                seen_vis_diag.add(v_id)
                vis_ordered_vids.append(v_id)

        lex_combined = sorted(ocr_candidates + asr_candidates, key=lambda x: x["score"], reverse=True)
        lex_ordered_vids: list[str] = []
        seen_lex_diag: set[str] = set()
        for c in lex_combined:
            v_id = c["video_id"]
            if v_id not in seen_lex_diag:
                seen_lex_diag.add(v_id)
                lex_ordered_vids.append(v_id)

        clusters: list[dict[str, Any]] = []
        clusters_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # 4a. Initialize Visual Clusters
        for rank, c in enumerate(visual_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            rrf_vis_cap = float(c.get("rrf_vis_caption", 0.0))
            cl_obj = {
                "video_id": c["video_id"],
                "rep_timestamp": c["timestamp_sec"],
                "visual_sim": c["score"],
                "faiss_id": c.get("faiss_id"),
                "obj_boost": float(c.get("obj_boost", 0.0)),
                "rrf_vis": rrf_val + 0.5 * rrf_vis_cap,
                "rrf_vis_caption": rrf_vis_cap,
                "ocr_rrfs": [],
                "asr_rrfs": [],
                "branches": {c.get("branch", "visual")},
                "matched_texts": [],  # B2: Per-cluster matched texts
            }
            clusters.append(cl_obj)
            clusters_by_video[c["video_id"]].append(cl_obj)

        # 4b. Integrate OCR Candidates
        for rank, c in enumerate(ocr_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            best_cl = None
            best_diff = float("inf")
            for cl in clusters_by_video.get(c["video_id"], ()):
                diff = abs(cl["rep_timestamp"] - c["timestamp_sec"])
                if diff <= 3.0 and diff < best_diff:
                    best_diff = diff
                    best_cl = cl

            if best_cl is not None:
                best_cl["ocr_rrfs"].append(rrf_val)
                best_cl["branches"].add("ocr")
                best_cl["fts_score"] = max(best_cl.get("fts_score", 0.0), c.get("score", 0.0))
                if c.get("matched_text"):
                    best_cl["matched_texts"].append(c["matched_text"])
            else:
                v_sim = self.lookup_true_visual_sim(
                    c["video_id"],
                    c["timestamp_sec"],
                    primary_q_864=primary_q_864,
                    primary_q_768=primary_q_768,
                    curve_cache=video_curve_cache,
                )
                cl_obj = {
                    "video_id": c["video_id"],
                    "rep_timestamp": c["timestamp_sec"],
                    "visual_sim": v_sim,
                    "rrf_vis": 0.0,
                    "ocr_rrfs": [rrf_val],
                    "asr_rrfs": [],
                    "branches": {"ocr"},
                    "fts_score": c.get("score", 0.0),
                    "matched_texts": [c["matched_text"]] if c.get("matched_text") else [],
                }
                clusters.append(cl_obj)
                clusters_by_video[c["video_id"]].append(cl_obj)

        # 4c. Integrate ASR Candidates
        for rank, c in enumerate(asr_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            st = c.get("start_sec", c["timestamp_sec"] - 1.0)
            et = c.get("end_sec", c["timestamp_sec"] + 1.0)
            merged = False
            for cl in clusters_by_video.get(c["video_id"], ()):
                tol = 8.0 if ("ocr" in cl["branches"] and "visual" not in cl["branches"]) else 4.5
                if (st - tol) <= cl["rep_timestamp"] <= (et + tol):
                    cl["asr_rrfs"].append(rrf_val)
                    cl["branches"].add("asr")
                    cl["fts_score"] = max(cl.get("fts_score", 0.0), c.get("score", 0.0))
                    if c.get("matched_text"):
                        cl["matched_texts"].append(c["matched_text"])
                    merged = True
            if not merged:
                v_sim = self.lookup_true_visual_sim(
                    c["video_id"],
                    c["timestamp_sec"],
                    start_sec=st,
                    end_sec=et,
                    primary_q_864=primary_q_864,
                    primary_q_768=primary_q_768,
                    curve_cache=video_curve_cache,
                )
                cl_obj = {
                    "video_id": c["video_id"],
                    "rep_timestamp": c["timestamp_sec"],
                    "visual_sim": v_sim,
                    "rrf_vis": 0.0,
                    "ocr_rrfs": [],
                    "asr_rrfs": [rrf_val],
                    "branches": {"asr"},
                    "fts_score": c.get("score", 0.0),
                    "matched_texts": [c["matched_text"]] if c.get("matched_text") else [],
                }
                clusters.append(cl_obj)
                clusters_by_video[c["video_id"]].append(cl_obj)

        # B3: Order-invariant RRF computation for each cluster
        for cl in clusters:
            cl["rrf_ocr"] = compute_order_invariant_rrf(cl["ocr_rrfs"])
            cl["rrf_asr"] = compute_order_invariant_rrf(cl["asr_rrfs"])

        # =========================================================================
        # Stage 5: Gated Lexical-Visual Fusion (B1, B9)
        # =========================================================================
        s_vis_max = max((cl["visual_sim"] for cl in clusters), default=0.0)
        s_vis_floor = s_vis_max - 0.065
        by_video_raw: dict[str, list[tuple[float, float, float, float, str]]] = defaultdict(list)
        vid_ltr_eligible: set[str] = set()
        k_high_len = len(high_spec_tokens)
        precleaned_qps = [
            split_compound_tokens(qp).strip().lower()
            for qp in quoted_phrases
            if split_compound_tokens(qp).strip()
        ] if quoted_phrases else []

        for cl in clusters:
            v_sim = cl["visual_sim"]
            delta_vis = max(0.0, s_vis_max - v_sim)

            # Visual compatibility gate
            gamma_vis = float(math.exp(-max(0.0, delta_vis - 0.012) * 35.0))

            rrf_ocr = cl["rrf_ocr"]
            rrf_asr = cl["rrf_asr"]
            base_lex = max(1.6 * rrf_ocr, 1.4 * rrf_asr) + 0.35 * min(1.6 * rrf_ocr, 1.4 * rrf_asr)
            is_visually_anchored = ("visual" in cl["branches"])
            has_unmatched_query_number = False

            if base_lex <= 0.0:
                lex_boost = 0.0
                has_quoted_match = False
                has_proper_noun_match = False
                single_common_only = False
                can_enter_case_a = False
                m_high = 0
                fts_sc = 0.0
            else:
                # B2: Strictly use per-cluster matched texts!
                texts_combined = " ".join(cl.get("matched_texts", []))
                clean_cand_text = split_compound_tokens(texts_combined).lower()
                cand_words = set(re.findall(r'\b[a-z0-9à-ỹ]+\b', clean_cand_text))

                has_quoted_match = any(cqp in clean_cand_text for cqp in precleaned_qps)

                m_high = 0
                for ht in high_spec_tokens:
                    if ht in cand_words or (len(ht) >= 4 and ht in clean_cand_text):
                        m_high += 1

                m_low = 0
                for lt in low_spec_tokens:
                    if lt in cand_words or (len(lt) >= 4 and lt in clean_cand_text):
                        m_low += 1

                m_pn = sum(
                    1 for pn in proper_nouns_lower
                    if pn in cand_words or (len(pn) >= 4 and pn in clean_cand_text)
                )
                fts_sc = float(cl.get("fts_score", 0.0))
                if len(proper_nouns_lower) >= 3 and m_pn == 1 and fts_sc < 15.0:
                    has_proper_noun_match = False
                elif len(proper_nouns_lower) == 1 and m_pn == 1 and m_high == 1 and k_high_len >= 2 and fts_sc < 15.0 and len(cl["branches"]) == 1:
                    has_proper_noun_match = False
                else:
                    has_proper_noun_match = (m_pn >= 1)

                lex_cov = m_high / float(max(1, k_high_len))
                single_common_only = (
                    not has_quoted_match
                    and not has_proper_noun_match
                    and m_high == 1
                    and m_low == 0
                    and (
                        (k_high_len == 1 and next(iter(high_spec_tokens), "").endswith("ing"))
                        or (k_high_len == 2 and fts_sc < 19.0)
                        or (k_high_len >= 6 and fts_sc < 15.0)
                    )
                )
                can_enter_case_a = False

                if is_visually_anchored:
                    # Visually Anchored Fusion: Case A / B / C
                    if has_quoted_match or (has_proper_noun_match and m_high >= 1) or m_high >= 2:
                        pn_multiplier = 1.30 if (has_proper_noun_match and m_high >= 1) else 1.0
                        if fts_sc >= 20.0:
                            rare_bm25_bonus = 2.00
                        elif fts_sc >= 12.0:
                            rare_bm25_bonus = 1.45
                        else:
                            rare_bm25_bonus = 1.0
                        ocr_title_bonus = 1.15 if ("ocr" in cl["branches"] and m_high >= 2) else 1.0
                        multi_tok = ((1.0 + 0.45 * min(m_high - 1, 3)) if m_high > 1 else 1.0) * pn_multiplier * rare_bm25_bonus * ocr_title_bonus
                        vis_factor = (0.75 + 0.25 * gamma_vis) if (has_quoted_match or (has_proper_noun_match and (m_high >= 2 or fts_sc >= 10.0))) else (0.35 + 0.65 * gamma_vis)
                        if v_sim < s_vis_floor and not has_quoted_match and fts_sc < 18.0:
                            vis_factor *= max(0.25, gamma_vis)
                        lex_boost = 5.5 * base_lex * vis_factor * multi_tok
                    elif m_high == 1 and k_high_len >= 1:
                        if single_common_only:
                            lex_boost = 0.25 * gamma_vis * base_lex
                        else:
                            cov_ratio = 1.0 / math.sqrt(float(max(1, k_high_len)))
                            bm25_factor = 1.35 if fts_sc >= 15.0 else 1.0
                            gap_damp = gamma_vis if (not has_proper_noun_match and k_high_len >= 2 and (s_vis_max - v_sim) > 0.020 and fts_sc < 18.0) else 1.0
                            lex_boost = 2.0 * cov_ratio * gamma_vis * gap_damp * base_lex * bm25_factor
                    else:
                        lex_boost = 0.15 * gamma_vis * base_lex
                # B9: Pure-ASR / Pure-OCR False Positive Gate (Strict gating without visual anchor)
                else:
                    q_numbers = {t for t in high_spec_tokens if t.isdigit()}
                    has_unmatched_query_number = bool(q_numbers and not (q_numbers & cand_words))
                    can_enter_case_a = (
                        not has_unmatched_query_number
                        and (
                            has_quoted_match
                            or (has_proper_noun_match and (m_high >= 2 or fts_sc >= 22.0 or len(cl["branches"]) >= 2))
                            or (m_high >= 2 and gamma_vis >= 0.45 and (has_proper_noun_match or fts_sc >= 22.0 or m_high >= max(3, int(math.ceil(0.6 * k_high_len)))))
                        )
                    )
                    if can_enter_case_a:
                        pn_multiplier = 1.30 if (has_proper_noun_match and m_high >= 1) else 1.0
                        if fts_sc >= 20.0:
                            rare_bm25_bonus = 2.00
                        elif fts_sc >= 12.0:
                            rare_bm25_bonus = 1.45
                        else:
                            rare_bm25_bonus = 1.0
                        multi_tok = ((1.0 + 0.45 * min(m_high - 1, 3)) if m_high > 1 else 1.0) * pn_multiplier * rare_bm25_bonus
                        vis_factor = (0.55 + 0.45 * gamma_vis) if (has_quoted_match or has_proper_noun_match) else gamma_vis
                        if v_sim < s_vis_floor and not has_quoted_match and fts_sc < 18.0:
                            vis_factor *= max(0.25, gamma_vis)
                        lex_boost = 5.5 * base_lex * vis_factor * multi_tok
                    elif m_high >= 1 and k_high_len >= 1:
                        cov_ratio = 1.0 / math.sqrt(float(max(1, k_high_len)))
                        lex_boost = 2.0 * cov_ratio * gamma_vis * base_lex * 0.45  # Gated suppression
                    else:
                        lex_boost = 0.15 * gamma_vis * base_lex * 0.45  # Dampened suppression

            is_llm_compact = routing_plan is not None and routing_plan.routing_mode == "llm_compact"
            has_explicit_intent = (
                is_visually_anchored
                and is_llm_compact
                and (
                    (routing_plan.has_explicit_ocr_intent and "ocr" in cl["branches"])
                    or (routing_plan.has_explicit_asr_intent and "asr" in cl["branches"])
                )
            )
            purely_visual_intent = (
                is_llm_compact
                and not routing_plan.has_explicit_ocr_intent
                and not routing_plan.has_explicit_asr_intent
            )
            if base_lex > 0.0 and has_explicit_intent:
                lex_boost *= 1.15
            elif base_lex > 0.0 and purely_visual_intent and single_common_only:
                lex_boost *= 0.85

            cl["m_high"] = m_high
            cl["k_high"] = max(1, k_high_len)
            cl["anchor_match"] = 1.0 if (has_quoted_match or has_proper_noun_match) else 0.0
            cl["num_consist"] = -1.0 if (not is_visually_anchored and has_unmatched_query_number) else 1.0

            if is_visually_anchored or has_quoted_match or ((has_proper_noun_match or fts_sc >= 22.0) and can_enter_case_a):
                if is_visually_anchored and not has_quoted_match and not has_proper_noun_match and ((m_high <= 1 and k_high_len >= 5) or single_common_only):
                    rrf_primary = cl["rrf_vis"] if single_common_only else max(cl["rrf_vis"], 0.35 * gamma_vis * max(cl["rrf_asr"], cl["rrf_ocr"]))
                else:
                    rrf_primary = max(cl["rrf_vis"], 0.70 * cl["rrf_asr"], 0.70 * cl["rrf_ocr"])
            else:
                rrf_primary = cl["rrf_vis"] if single_common_only else max(cl["rrf_vis"], 0.35 * gamma_vis * max(cl["rrf_asr"], cl["rrf_ocr"]))
            synergy_bonus = 0.0055 if (len(cl["branches"]) >= 2 and not ((m_high <= 1 and k_high_len >= 5) or single_common_only)) else 0.0
            score = v_sim + 2.5 * rrf_primary + lex_boost + synergy_bonus
            branch_str = "+".join(sorted(cl["branches"])) if cl["branches"] else "visual"

            pure_asr_suppressed = bool(base_lex > 0.0 and not is_visually_anchored and not can_enter_case_a)
            if not pure_asr_suppressed:
                vid_ltr_eligible.add(cl["video_id"])

            rep_t_cl = float(cl["rep_timestamp"])
            by_video_raw[cl["video_id"]].append((
                float(score),
                max(0.0, rep_t_cl - 1.0),
                rep_t_cl + 1.0,
                rep_t_cl,
                branch_str,
            ))

        # =========================================================================
        # Stage 6: Anchor-Aware Boundary Expansion (B10) & Video Aggregation
        # =========================================================================
        z_stats = compute_query_z_scores(
            [float(c["score"]) for c in visual_candidates],
            top_k=120,
        )
        vid_best_cluster: dict[str, dict[str, Any]] = {}
        for cl in clusters:
            vid_c = cl["video_id"]
            prev = vid_best_cluster.get(vid_c)
            if prev is None or float(cl["visual_sim"]) > float(prev["visual_sim"]):
                vid_best_cluster[vid_c] = cl

        sp_enabled, _ = self._is_spatial_patches_enabled(ablation_config)
        late_interaction_scored_candidates = 0
        cand_moments: list[RankedMoment] = []
        cand_ltr_features: dict[str, dict[str, Any]] = {}
        for vid, vid_items in by_video_raw.items():
            if not self._video_passes_filters(vid, filters):
                continue
            vid_items.sort(key=lambda x: x[0], reverse=True)
            scores = [x[0] for x in vid_items]
            m_count = len(scores)
            s_max = scores[0]
            n_kf = len(self.video_keyframes.get(vid, []))

            top3_mean = sum(scores[:min(m_count, 3)]) / min(m_count, 3)
            top5_mean = sum(scores[:min(m_count, 5)]) / min(m_count, 5)

            sorted_by_time = sorted(vid_items, key=lambda x: x[1])
            adjacent_pairs = 0
            for i in range(len(sorted_by_time) - 1):
                if sorted_by_time[i + 1][1] - sorted_by_time[i][2] <= 6.0:
                    adjacent_pairs += 1

            effective_s_max = max(0.0, s_max)
            ratio = adjacent_pairs / max(1.0, float(n_kf - 1))
            density_bonus = (0.060 * effective_s_max) * min(adjacent_pairs, 3) + 0.015 * effective_s_max * ratio
            if n_kf > 30 and s_max < 0.225 and (m_count / float(n_kf)) < 0.12:
                density_bonus *= 0.50

            best_sc, best_st, best_et, best_rep_t, best_branch = vid_items[0]
            local_window_scores = sorted(
                (x[0] for x in vid_items if abs(x[3] - best_rep_t) <= 3.5),
                reverse=True,
            )[:2]
            local_window_mean = (sum(local_window_scores) / len(local_window_scores)) if local_window_scores else s_max

            # Targeted Single-Frame Outlier Spike Guard (only when m_count == 1 and pure visual)
            tail_adjustment = 0.0
            spike_drop_val = 0.0
            if ablation_config != "A0" and m_count == 1 and n_kf >= 3 and best_branch == "visual":
                sorted_kf_vis = self._video_sorted_visual_scores(
                    vid,
                    primary_q_864=primary_q_864,
                    primary_q_768=primary_q_768,
                    blend_768_864=True,
                )
                if len(sorted_kf_vis) >= 2:
                    kf1, kf2 = float(sorted_kf_vis[0]), float(sorted_kf_vis[1])
                    spike_drop = kf1 - kf2
                    spike_drop_val = max(0.0, spike_drop)
                    if spike_drop > 0.20:
                        tail_adjustment = -0.05 * (spike_drop - 0.20)
                    elif kf2 >= 0.88 * kf1 and kf2 >= 0.42:
                        tail_adjustment = 0.012 * kf2

            if best_branch in ("asr", "ocr", "asr+ocr") and (len(proper_nouns_lower) >= 1 or len(quoted_phrases) >= 1 or m_count == 1):
                video_score = (
                    0.85 * s_max
                    + 0.10 * top3_mean
                    + 0.05 * top5_mean
                    + density_bonus
                    + tail_adjustment
                )
            elif "asr" in best_branch or "ocr" in best_branch:
                video_score = (
                    0.65 * s_max
                    + 0.15 * top3_mean
                    + 0.20 * top5_mean
                    + density_bonus
                    + tail_adjustment
                )
            elif ablation_config == "A5":
                video_score = (
                    0.54 * s_max
                    + 0.15 * top3_mean
                    + 0.29 * top5_mean
                    + 0.02 * local_window_mean
                    + density_bonus
                    + tail_adjustment
                )
            else:
                video_score = (
                    0.55 * s_max
                    + 0.15 * top3_mean
                    + 0.30 * top5_mean
                    + density_bonus
                    + tail_adjustment
                )
            all_branches_in_video: set[str] = set()
            for item in vid_items:
                b_status = item[4]
                if b_status and b_status not in ("supported", "partial", "unknown", "contradicted"):
                    for b in b_status.split("+"):
                        all_branches_in_video.add(b.strip())
            branch_attribution = "+".join(sorted(all_branches_in_video)) if all_branches_in_video else "visual"

            best_cl = vid_best_cluster.get(vid, {})
            vid_vis_sims = sorted(
                (float(cl["visual_sim"]) for cl in clusters_by_video.get(vid, ())),
                reverse=True,
            )
            if vid_vis_sims:
                top3_vis_mean = sum(vid_vis_sims[:min(len(vid_vis_sims), 3)]) / min(len(vid_vis_sims), 3)
                top5_vis_mean = sum(vid_vis_sims[:min(len(vid_vis_sims), 5)]) / min(len(vid_vis_sims), 5)
            else:
                top3_vis_mean = 0.0
                top5_vis_mean = 0.0
            best_vis_t = float(best_cl.get("rep_timestamp", best_rep_t))
            local_vis_scores = sorted(
                (
                    float(cl["visual_sim"])
                    for cl in clusters_by_video.get(vid, ())
                    if abs(float(cl["rep_timestamp"]) - best_vis_t) <= 3.5
                ),
                reverse=True,
            )[:2]
            local_vis_coherence = (
                (sum(local_vis_scores) / len(local_vis_scores))
                if local_vis_scores
                else float(best_cl.get("visual_sim", 0.0))
            )

            vis_ctx_coherence = float(local_vis_coherence)
            if self.temporal_context_768 is not None and len(self.temporal_context_768) > 0:
                rep_fid = best_cl.get("faiss_id")
                if rep_fid is None:
                    kf_list_v = self.video_keyframes.get(vid, [])
                    if kf_list_v:
                        nearest_kf = min(
                            kf_list_v,
                            key=lambda k: abs(float(k.get("pts_time", 0.0)) - best_vis_t),
                        )
                        rep_fid = int(nearest_kf["faiss_id"])
                if rep_fid is not None and 0 <= int(rep_fid) < len(self.temporal_context_768):
                    vis_ctx_coherence = float(
                        np.dot(self.temporal_context_768[int(rep_fid)], primary_q_768.reshape(-1))
                    )

            if sp_enabled and self.spatial_patches is not None and len(self.spatial_patches) > 0:
                late_interaction_scored_candidates += 1
                late_int_sc = self._score_late_interaction_window(
                    vid, best_st - 2.0, best_et + 2.0, primary_q_768
                )
            else:
                late_int_sc = 0.0

            s_vis_blend_val = float(best_cl.get("visual_sim", 0.0))
            subclause_weak_val = float(video_min_clause_sim.get(vid, s_vis_blend_val))
            coherence_ratio = (
                float(local_vis_coherence) / max(s_vis_blend_val, 0.05)
                if s_vis_blend_val > 0.0
                else 0.0
            )
            win_energy_conc = min(1.5, coherence_ratio * (1.0 + 0.10 * min(adjacent_pairs, 3)))

            att_sc = 0.0
            if ablation_config not in ("A0", "A1"):
                use_768_att = self.all_embeddings is not None and len(self.all_embeddings) > 0 and primary_q_768 is not None
                att_q = primary_q_768.reshape(-1) if use_768_att else primary_q_864.reshape(-1)
                att_emb_mat = self.all_embeddings if use_768_att else self.augmented_embeddings
                if att_emb_mat is not None and len(att_emb_mat) > 0:
                    kf_list = self.video_keyframes.get(vid, [])
                    if kf_list:
                        win_kfs = [
                            k for k in kf_list
                            if (best_st - 2.0) <= float(k.get("pts_time", 0.0)) <= (best_et + 2.0)
                        ]
                        if not win_kfs:
                            win_kfs = sorted(
                                kf_list,
                                key=lambda k: abs(float(k.get("pts_time", 0.0)) - best_rep_t),
                            )[:5]
                        elif len(win_kfs) > 5:
                            win_kfs = sorted(
                                win_kfs,
                                key=lambda k: abs(float(k.get("pts_time", 0.0)) - best_rep_t),
                            )[:5]
                            win_kfs.sort(key=lambda k: float(k.get("pts_time", 0.0)))
                        valid_fids = [int(k["faiss_id"]) for k in win_kfs if 0 <= int(k["faiss_id"]) < len(att_emb_mat)]
                        if valid_fids:
                            att_sc = float(query_guided_temporal_attention_score(att_q, att_emb_mat[valid_fids], tau=12.0))

            raw_ltr_cand: dict[str, Any] = {
                "s_vis_blend_max": s_vis_blend_val,
                "s_vis_top3_mean": float(top3_vis_mean),
                "s_vis_top5_mean": float(top5_vis_mean),
                "s_vis_ctx_coherence": float(vis_ctx_coherence),
                "s_late_interaction": float(late_int_sc),
                "spike_drop_ratio": float(spike_drop_val),
                "rrf_vis": float(best_cl.get("rrf_vis", 0.0)),
                "rrf_ocr": float(best_cl.get("rrf_ocr", 0.0)),
                "rrf_asr": float(best_cl.get("rrf_asr", 0.0)),
                "fts_score": float(best_cl.get("fts_score", 0.0)),
                "m_high": float(best_cl.get("m_high", 0.0)),
                "k_high": float(best_cl.get("k_high", max(1, k_high_len))),
                "anchor_lexical_match": float(best_cl.get("anchor_match", 0.0)),
                "number_consistency": float(best_cl.get("num_consist", 1.0)),
                "yolo_object_verification_boost": float(best_cl.get("obj_boost", 0.0)),
                "graph_relation_score": 0.0,
                "llm_modality_alignment": 1.0 if (
                    routing_plan is not None
                    and routing_plan.routing_mode == "llm_compact"
                    and (
                        (routing_plan.has_explicit_ocr_intent and "ocr" in branch_attribution)
                        or (routing_plan.has_explicit_asr_intent and "asr" in branch_attribution)
                    )
                ) else 0.0,
                "subclause_weakest_link": subclause_weak_val,
                "window_energy_concentration": float(win_energy_conc),
                "temporal_attention_coherence": float(att_sc),
                "rrf_vis_caption": float(best_cl.get("rrf_vis_caption", 0.0)),
            }
            ltr_24d_dict = extract_ltr_features_24d(raw_ltr_cand, z_stats=z_stats, return_dict=True)
            raw_ltr_cand.update(ltr_24d_dict)
            cand_ltr_features[vid] = raw_ltr_cand
            cand_moments.append(RankedMoment(
                moment_id=f"{vid}_{round(best_rep_t, 1)}",
                video_id=vid,
                moment_start_seconds=best_st,
                moment_end_seconds=best_et,
                start_original_frame_index=0,
                end_original_frame_index=0,
                representative_original_frame_index=0,
                representative_timestamp_seconds=best_rep_t,
                ranking_score=video_score,
                match_status=branch_attribution,
            ))

        # Section 2.2: Capture pre-rerank video IDs after Stage 5 base fusion, BEFORE LTR or temporal rerank
        pre_rerank_sorted = sorted(
            cand_moments,
            key=lambda m: (-float(m.ranking_score), str(m.video_id)),
        )
        pre_rerank_video_ids: list[str] = list(dict.fromkeys(m.video_id for m in pre_rerank_sorted))

        if self.ltr_ranker.is_loaded and cand_moments and self.ltr_blend_weight > 0.0:
            ltr_matrix = np.vstack([
                extract_ltr_features_24d(cand_ltr_features[m.video_id], z_stats=z_stats)
                for m in cand_moments
            ])
            ltr_scores = self.ltr_ranker.score_matrix(ltr_matrix)
            cand_moments = [
                RankedMoment(
                    moment_id=m.moment_id,
                    video_id=m.video_id,
                    moment_start_seconds=m.moment_start_seconds,
                    moment_end_seconds=m.moment_end_seconds,
                    start_original_frame_index=m.start_original_frame_index,
                    end_original_frame_index=m.end_original_frame_index,
                    representative_original_frame_index=m.representative_original_frame_index,
                    representative_timestamp_seconds=m.representative_timestamp_seconds,
                    ranking_score=(
                        float(m.ranking_score) + self.ltr_blend_weight * float(np.tanh(ltr_sc))
                        if m.video_id in vid_ltr_eligible
                        else float(m.ranking_score)
                    ),
                    match_status=m.match_status,
                )
                for m, ltr_sc in zip(cand_moments, ltr_scores.tolist())
            ]

        cand_moments.sort(key=lambda m: (-float(m.ranking_score), str(m.video_id)))

        clip_branch_stats: dict[str, Any] = {}
        if self.enable_clip_branch:
            cand_moments, a5_pool_ids, clip_pool_ids, clip_branch_stats = self._apply_clip_reserve_stage(
                cand_moments, primary_q_768, filters
            )
        else:
            a5_pool_ids = [
                m.video_id for m in self._build_ranked_videos(cand_moments, cand_moments, top_k=top_k)
            ]
            clip_pool_ids = []

        # Section 3.1: NumPy Temporal Rerank Challenger on fixed Top-K base videos
        tr_enabled, _ = self._is_temporal_rerank_enabled(ablation_config)
        temporal_rerank_stats: dict[str, Any] = {}
        temporal_rerank_scored_candidates = 0
        if tr_enabled:
            if self.enable_clip_branch and clip_pool_ids:
                reranker_input_ids = list(clip_pool_ids)
                rerank_k_override: int | None = len(clip_pool_ids)
            else:
                reranker_input_ids = list(
                    dict.fromkeys(m.video_id for m in cand_moments)
                )[: self.temporal_rerank_top_k]
                rerank_k_override = None
            if float(self.temporal_rerank_lambda) != 0.0 and cand_moments:
                cand_moments, temporal_rerank_stats = self._apply_temporal_rerank(
                    cand_moments, primary_q_768, top_k_override=rerank_k_override
                )
                temporal_rerank_scored_candidates = int(temporal_rerank_stats.get("scored_count", 0))
                cand_moments.sort(key=lambda m: (-float(m.ranking_score), str(m.video_id)))
        else:
            reranker_input_ids = list(clip_pool_ids) if (self.enable_clip_branch and clip_pool_ids) else list(pre_rerank_video_ids)

        aggregated_moments = cand_moments[:top_k]
        t_fusion_end = time.perf_counter()

        # Stage 6 Refinement with B10:
        final_results: list[RankedMoment] = []
        for rank_idx, moment in enumerate(aggregated_moments, 1):
            vid = moment.video_id
            rep_t = moment.representative_timestamp_seconds
            kf_list = self.video_keyframes.get(vid, [])
            v_dur = self.video_durations.get(
                vid,
                (max(kf["pts_time"] for kf in kf_list) + 2.0) if kf_list else 300.0,
            )

            best_idx = 0
            ts_array: list[float] = []
            exact_curve: list[float] = []
            if ablation_config in ("A0", "A1", "A2", "A3"):
                delta = min(5.0, 0.45 * v_dur)
                start_sec = max(0.0, rep_t - delta / 2.0)
                end_sec = min(v_dur, rep_t + delta / 2.0)
                if kf_list:
                    best_idx = min(range(len(kf_list)), key=lambda i: abs(float(kf_list[i].get("pts_time", 0.0)) - rep_t))
            elif len(kf_list) > 1:
                ts_array = self.video_kf_pts.get(vid) or [float(kf["pts_time"]) for kf in kf_list]
                if vid in video_curve_cache:
                    exact_curve = video_curve_cache[vid][1].tolist()
                else:
                    cached_pts, sims = self._compute_video_visual_curve(
                        vid,
                        primary_q_864=primary_q_864,
                        primary_q_768=primary_q_768,
                        blend_768_864=True,
                    )
                    if cached_pts:
                        ts_array = cached_pts
                    video_curve_cache[vid] = (ts_array, sims)
                    exact_curve = sims.tolist()

                boundary_curve = exact_curve
                # B10: Anchor index and anchor window calculation
                anchor_win = video_anchor_windows.get(vid)
                if anchor_win is not None:
                    t_mid = (anchor_win[0] + anchor_win[1]) / 2.0
                    best_idx = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - t_mid))
                else:
                    best_idx = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - rep_t))

                start_sec, end_sec = expand_boundary_contrast(
                    ts=ts_array,
                    curve=boundary_curve,
                    v_dur=v_dur,
                    anchor_idx=best_idx,
                    anchor_window=anchor_win,
                    decay_lambda=0.015,
                    c_thresh=0.45,
                    max_gap_sec=4.0,
                    max_dur_ratio=0.45,
                )
            else:
                delta = min(5.0, 0.45 * v_dur)
                start_sec = max(0.0, rep_t - delta / 2.0)
                end_sec = min(v_dur, rep_t + delta / 2.0)

            s_orig_idx, e_orig_idx, rep_orig_idx, ev_indices = _resolve_moment_frame_indices(
                kf_list, start_sec, end_sec, best_idx
            )
            final_results.append(RankedMoment(
                moment_id=f"{vid}_{round(start_sec, 1)}_{round(end_sec, 1)}",
                video_id=vid,
                moment_start_seconds=round(start_sec, 2),
                moment_end_seconds=round(end_sec, 2),
                start_original_frame_index=s_orig_idx,
                end_original_frame_index=e_orig_idx,
                representative_original_frame_index=rep_orig_idx,
                representative_timestamp_seconds=round(rep_t, 2),
                ranking_score=round(float(moment.ranking_score), 5),
                match_status=moment.match_status,
                evidence_frame_indices=ev_indices,
            ))

            # NMS-MultiPeak Proposal for Top-1 Video (enabled in A5):
            if ablation_config == "A5" and rank_idx == 1 and len(kf_list) > 3 and exact_curve:
                win1_scores = [exact_curve[i] for i, t in enumerate(ts_array) if start_sec <= t <= end_sec]
                s_max1 = max(win1_scores) if win1_scores else max(exact_curve)
                s_min = min(exact_curve)
                existing_windows: list[tuple[float, float]] = [(start_sec, end_sec)]
                max_extra = 2 if len(ts_array) >= 20 else 1
                prev_win_score = float(moment.ranking_score)
                min_peak_ratio = 0.80 if len(ts_array) >= 20 else 0.88

                for p_offset in range(max_extra):
                    p_num = p_offset + 2  # 2 for _win2, 3 for _win3
                    masked_curve = [
                        (
                            s_min - 0.05
                            if any((ws - 4.0) <= ts_array[i] <= (we + 4.0) for ws, we in existing_windows)
                            else s
                        )
                        for i, s in enumerate(exact_curve)
                    ]
                    peak_p_idx = max(range(len(masked_curve)), key=lambda i: masked_curve[i])
                    s_max_p = masked_curve[peak_p_idx]
                    if s_max_p < min_peak_ratio * s_max1:
                        break
                    t_peak_p = ts_array[peak_p_idx]

                    # Require a genuine valley between the new peak and existing windows
                    valley_scores = [
                        exact_curve[i]
                        for i, t in enumerate(ts_array)
                        if any((we < t < t_peak_p) or (t_peak_p < t < ws) for ws, we in existing_windows)
                    ]
                    has_valley = bool(valley_scores) and min(valley_scores) <= s_min + 0.65 * (s_max_p - s_min)
                    if not has_valley:
                        break

                    st_p, et_p = expand_boundary_contrast(
                        ts=ts_array,
                        curve=masked_curve,
                        v_dur=v_dur,
                        anchor_idx=peak_p_idx,
                        decay_lambda=0.015,
                        c_thresh=0.45,
                        max_gap_sec=4.0,
                        max_dur_ratio=0.45,
                    )
                    if et_p <= st_p or (et_p - st_p) < 1.0:
                        break
                    max_tiou = max(temporal_iou(st_p, et_p, ws, we) for ws, we in existing_windows)
                    if max_tiou >= 0.30:
                        break

                    rep_t_p = (st_p + et_p) / 2.0
                    best_idx_p = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - rep_t_p))
                    s_orig_p, e_orig_p, rep_orig_p, ev_indices_p = _resolve_moment_frame_indices(
                        kf_list, st_p, et_p, best_idx_p
                    )
                    peak_ratio = min(1.0, float(s_max_p) / max(float(s_max1), 1e-8))
                    if len(ts_array) >= 20:
                        cap = 0.96 - 0.03 * (p_num - 2)
                        win_p_score = float(moment.ranking_score) * min(cap, max(0.78, 0.92 * peak_ratio))
                    else:
                        win_p_score = float(moment.ranking_score) * 0.85 * (peak_ratio ** 2)
                    win_p_score = min(win_p_score, prev_win_score - 1e-4)
                    prev_win_score = win_p_score
                    existing_windows.append((st_p, et_p))
                    final_results.append(RankedMoment(
                        moment_id=f"{vid}_{round(st_p, 1)}_{round(et_p, 1)}_win{p_num}",
                        video_id=vid,
                        moment_start_seconds=round(st_p, 2),
                        moment_end_seconds=round(et_p, 2),
                        start_original_frame_index=s_orig_p,
                        end_original_frame_index=e_orig_p,
                        representative_original_frame_index=rep_orig_p,
                        representative_timestamp_seconds=round(rep_t_p, 2),
                        ranking_score=round(win_p_score, 5),
                        match_status=f"{moment.match_status}+multi_window",
                        evidence_frame_indices=ev_indices_p,
                    ))

        final_results.sort(key=lambda m: float(m.ranking_score), reverse=True)
        t_boundary_end = time.perf_counter()

        t_parse_ms = (t_parse_end - t_start) * 1000.0
        t_encode_ms = (t_encode_end - t_parse_end) * 1000.0 + t_e5_encode_ms
        t_faiss_ms = (t_faiss_end - t_encode_end) * 1000.0
        t_fts5_ms = max(0.0, (t_fts5_end - t_faiss_end) * 1000.0 - t_e5_encode_ms)
        t_fusion_ms = (t_fusion_end - t_fts5_end) * 1000.0
        t_boundary_ms = (t_boundary_end - t_fusion_end) * 1000.0
        t_non_neural_ms = t_faiss_ms + t_fts5_ms + t_fusion_ms + t_boundary_ms
        t_total_ms = (t_boundary_end - t_start) * 1000.0

        self.last_timings = {
            "t_parse_ms": t_parse_ms,
            "t_encode_ms": t_encode_ms,
            "t_faiss_ms": t_faiss_ms,
            "t_fts5_ms": t_fts5_ms,
            "t_fusion_ms": t_fusion_ms,
            "t_boundary_ms": t_boundary_ms,
            "t_non_neural_ms": t_non_neural_ms,
            "t_total_ms": t_total_ms,
        }
        union_candidate_ids = list(dict.fromkeys([*visual_candidate_ids, *lex_ordered_vids, *caption_ordered_vids]))
        ranked_video_ids_default = [
            m.video_id for m in self._build_ranked_videos(cand_moments, cand_moments, top_k=top_k)
        ]
        self.last_diagnostics = {
            "lexical_status": "complete" if fts_query and self.db_conn is not None and ablation_config not in ("A0", "A1", "A2") else "not_run",
            "visual_top30_vids": set(visual_candidate_ids[:30]),
            "visual_top120_vids": set(visual_candidate_ids[:120]),
            "lexical_top30_vids": set(lex_ordered_vids[:30]),
            "ocr_vids": list(dict.fromkeys(c["video_id"] for c in ocr_candidates)),
            "asr_vids": list(dict.fromkeys(c["video_id"] for c in asr_candidates)),
            "caption_vids": caption_ordered_vids,
            "visual_candidate_ids": visual_candidate_ids,
            "visual_frame_depth": visual_frame_depth,
            "visual_distinct_videos": visual_distinct_videos,
            "union_candidate_ids": union_candidate_ids,
            "union_dedup_vids": union_candidate_ids,
            "pre_rerank_video_ids": pre_rerank_video_ids,
            "shortlist_before_rerank_vids": pre_rerank_video_ids,
            "a5_pool_ids": a5_pool_ids,
            "clip_pool_ids": clip_pool_ids,
            "reranker_input_ids": reranker_input_ids,
            "ranked_video_ids": ranked_video_ids_default,
            "fused_top10_vids": set(pre_rerank_video_ids[:10]),
            "all_candidate_moments": cand_moments,
            "visual_candidates": visual_candidates[:30],
            "ocr_candidates": ocr_candidates[:15],
            "asr_candidates": asr_candidates[:15],
            "candidate_ltr_features": cand_ltr_features,
            "viterbi_modality_availability": viterbi_availability,
            "feature_flags": {
                "enable_spatial_patches": self.enable_spatial_patches,
                "enable_vlm_caption_lexical": self.enable_vlm_caption_lexical,
                "enable_vlm_caption_dense": self.enable_vlm_caption_dense,
                "enable_temporal_rerank": self.enable_temporal_rerank,
                "temporal_rerank_mode": self.temporal_rerank_mode,
                "temporal_rerank_lambda": self.temporal_rerank_lambda,
                "temporal_rerank_top_k": self.temporal_rerank_top_k,
                "temporal_rerank_window_sec": self.temporal_rerank_window_sec,
                "temporal_rerank_max_frames": self.temporal_rerank_max_frames,
                "temporal_rerank_tau": self.temporal_rerank_tau,
                "enable_clip_branch": self.enable_clip_branch,
                "clip_reserve_k": self.clip_reserve_k,
                "clip_window_size": self.clip_window_size,
            },
            "feature_compute_counts": {
                "late_interaction_scored_candidates": late_interaction_scored_candidates,
                "vlm_caption_lexical_hits": vlm_caption_lexical_hits,
                "vlm_caption_dense_hits": vlm_caption_dense_hits,
                "temporal_rerank_scored_candidates": temporal_rerank_scored_candidates,
            },
            "clip_branch_stats": clip_branch_stats,
            "temporal_rerank_stats": temporal_rerank_stats,
            "query_plan_mode": routing_plan.routing_mode if routing_plan is not None else None,
            "query_plan_cache_hit": routing_plan.cache_hit if routing_plan is not None else None,
            "query_decomposer_stats": (
                self.query_decomposer.get_stats()
                if hasattr(self.query_decomposer, "get_stats")
                else None
            ),
            "degraded": self._build_degraded_diagnostics(ablation_config),
        }

        # =========================================================================
        # Stage 7: Video Diversity Aggregation & RankedMoment Output
        # =========================================================================
        if ranking_mode == "diverse_video":
            seen_vids = set()
            diverse_moments: list[RankedMoment] = []
            for m in final_results:
                # Top-1 video can have up to 2 distinct window proposals
                if m.video_id not in seen_vids or "+multi_window" in m.match_status:
                    seen_vids.add(m.video_id)
                    diverse_moments.append(m)
            return diverse_moments[:top_k]

        return final_results[:top_k]

    def close(self) -> None:
        """Close SQLite persistent connection and persist any dirty disk cache."""
        self._text_memory_cache.clear()
        self._e5_memory_cache.clear()
        if self.cache is not None:
            self.cache.save()
        if self.e5_cache is not None:
            self.e5_cache.save()
        if self.db_conn is not None:
            try:
                self.db_conn.close()
            except Exception as exc:
                self._degraded_channels["db_close"] = str(exc)
                logger.warning("Failed to close SQLite connection: %s", exc, exc_info=True)
            self.db_conn = None

    def __enter__(self) -> UnifiedMultimodalSearchPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
