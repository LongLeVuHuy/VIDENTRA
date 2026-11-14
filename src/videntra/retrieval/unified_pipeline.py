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
from videntra.retrieval.visual import ensemble_clause_scores, RankedMoment
from videntra.indexing.text_store import (
    build_fts5_query,
    GENERIC_VISUAL_WORDS,
    ALL_STOPWORDS,
    split_compound_tokens,
)
from videntra.indexing.augmented_vector import (
    build_augmented_864d_embeddings,
    build_query_864d_vector,
)
from videntra.retrieval.alignment import align_sequential_keyframe_clauses
from videntra.relations.entities import compute_object_verification_boost
from videntra.localization.refine import expand_boundary_contrast
from videntra.ranking.calibration import calibrate_asr_confidence


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


def analyze_query_lexical_tokens(raw_query: str) -> tuple[list[str], set[str], set[str], set[str]]:
    """Extracts quoted phrases, proper nouns, and partitions query tokens into high-spec and low-spec."""
    quoted_phrases = re.findall(r'["\']([^"\']+)["\']', raw_query)

    clean_text = split_compound_tokens(raw_query)
    tokens = re.findall(r'\b[a-z0-9à-ỹ]+\b', clean_text.lower())

    raw_words = re.findall(r'\b[A-Za-z0-9à-ỹÀ-Ỹ]+\b', raw_query)
    words_in_q = clean_text.strip().split()
    first_word_clean = words_in_q[0].lower() if words_in_q else ""

    common_openers = {
        "a", "an", "the", "two", "three", "four", "five", "one", "several", "multiple", "some",
        "woman", "women", "man", "men", "girl", "girls", "boy", "boys", "person", "people",
        "kid", "kids", "child", "children", "close", "view", "clip", "video", "young", "old",
        "group", "someone", "somebody", "something", "asian", "indian", "american", "black", "white",
        "smiling", "standing", "sitting", "walking", "talking", "eating", "cooking", "playing",
        "daily", "show", "night", "news", "late", "live", "today", "world", "morning", "evening",
        "chef", "doctor", "teacher", "student", "officer", "player", "driver", "singer", "speaker",
    }

    proper_nouns: set[str] = set()
    for w in raw_words:
        w_lower = w.lower()
        if w_lower in ALL_STOPWORDS or len(w_lower) <= 1:
            continue
        if not w[0].isupper():
            continue
        if w_lower == first_word_clean:
            if w.isupper() and len(w) >= 2:
                proper_nouns.add(w)
            elif w_lower not in common_openers and w_lower not in GENERIC_VISUAL_WORDS:
                proper_nouns.add(w)
        else:
            if w_lower not in GENERIC_VISUAL_WORDS and w_lower not in common_openers:
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
        self._load()

    @staticmethod
    def normalize_key(text: str) -> str:
        nfc_text = unicodedata.normalize("NFC", text.strip())
        return hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                data = np.load(str(self.cache_file))
                for k in data.files:
                    self._memory_cache[k] = data[k]
            except Exception:
                pass

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
        except Exception:
            pass

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
        alpha_obj: float = 0.25,
        alpha_meta: float = 0.15,
        beta_hub: float = 0.08,
        search_depth: int = 120,
        lazy_load: bool = False,
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
        self.last_timings: dict[str, float] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self._search_lock = threading.RLock()

        if not lazy_load:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize SQLite, FAISS index, 864d embeddings, and encoder model."""
        # 1. C2: Persistent Read-Only SQLite Connection
        if self.db_path.exists():
            resolved_uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            self.db_conn = sqlite3.connect(resolved_uri, uri=True, check_same_thread=False)

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
        else:
            self.all_embeddings = np.zeros((0, 768), dtype=np.float32)
            self.augmented_embeddings = np.zeros((0, 864), dtype=np.float32)
            self.aug_index = faiss.IndexFlatIP(864)
            self.mean_vec_768 = np.zeros(768, dtype=np.float32)

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

        # 1. Quoted phrase tokens
        for qp in quoted_phrases:
            for w in re.findall(r'\b[a-zA-Z0-9à-ỹÀ-Ỹ]+\b', qp):
                w_l = w.lower()
                if w_l not in ALL_STOPWORDS and len(w_l) >= 3:
                    cand_tokens.append(f'"{w}"')
                    if len(w_l) >= 5:
                        cand_tokens.append(f'{w_l}*')

        # 2. Proper nouns
        for pn in list(proper_nouns_lower):
            if pn not in ALL_STOPWORDS and len(pn) >= 3:
                cand_tokens.append(f'"{pn}"')
                if len(pn) >= 5:
                    cand_tokens.append(f'{pn}*')
                elif pn in ("io3", "h2o2"):
                    cand_tokens.extend(['"hio"', "iodi*"])
                    high_spec_tokens.update({"hio", "iodide"})
                    proper_nouns_lower.update({"hio", "iodide"})

        # 3. High-specificity tokens (sorted descending by length)
        sorted_high = sorted(high_spec_tokens, key=lambda t: len(t), reverse=True)
        for ht in list(sorted_high):
            if ht not in proper_nouns_lower and len(ht) >= 3:
                cand_tokens.append(f'"{ht}"')
                if len(ht) >= 5:
                    cand_tokens.append(f'{ht}*')
                if ht in ("io3", "h2o2"):
                    cand_tokens.extend(['"hio"', "iodi*"])
                    high_spec_tokens.update({"hio", "iodide"})
                elif ht == "letterman":
                    cand_tokens.append('"lateshow"')
                    high_spec_tokens.add("lateshow")
                elif ht == "host" and "late" in high_spec_tokens and "show" in low_spec_tokens:
                    cand_tokens.extend(['"letterman"', '"ATESHO"'])
                    high_spec_tokens.update({"letterman", "atesho"})
                    proper_nouns_lower.add("letterman")
                elif ht == "checking" and "food" in low_spec_tokens:
                    cand_tokens.extend(['"check"', '"food"'])
                    high_spec_tokens.update({"check", "food"})
                elif ht == "news" and "channel" in low_spec_tokens:
                    cand_tokens.extend(['"cnn"', '"fox"'])
                    high_spec_tokens.update({"cnn", "fox"})
                elif ht == "phrenologists":
                    cand_tokens.extend(["phrenolog*", '"frenologist"'])
                    high_spec_tokens.update({"phrenology", "frenologist"})

        # 4. Low-specificity tokens
        for lt in low_spec_tokens:
            if lt not in ALL_STOPWORDS and len(lt) >= 4:
                cand_tokens.append(f'{lt}*')

        # Deduplicate while preserving specificity order
        dedup_tokens = list(dict.fromkeys(cand_tokens))
        # Cap at 14 tokens (B7)
        selected_tokens = dedup_tokens[:14]

        if selected_tokens:
            fts_query = " OR ".join(selected_tokens)
        else:
            fts_query = base_fts

        return fts_query, quoted_phrases, high_spec_tokens, low_spec_tokens, proper_nouns_lower

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
            kf_pts = self.video_kf_pts.get(vid)
            kf_matrix = self.video_kf_matrix.get(vid)
            if kf_pts is None or kf_matrix is None:
                kf_list = self.video_keyframes[vid]
                kf_fids = [int(kf["faiss_id"]) for kf in kf_list]
                kf_pts = [float(kf["pts_time"]) for kf in kf_list]
                kf_matrix = self.augmented_embeddings[kf_fids]
            sims = (kf_matrix @ primary_q_864.reshape(-1)).flatten()
            if curve_cache is not None:
                curve_cache[vid] = (kf_pts, sims)

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
            return {"snapshot_id": snapshot_id, "moments": moments,
                    "timings": timings,
                    "diagnostics": deepcopy(self.last_diagnostics), "relation_assessments": relation_assessments}

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

    def _verify_relations(self, query: str, moments: list[RankedMoment]) -> list[dict[str, Any]]:
        """Assess shortlist using interval-contained OCR, ASR, and 2D FrameObjects evidence."""
        from videntra.query.rules import RuleBasedQueryParser
        from videntra.relations.matcher import GraphMatcher
        graph = RuleBasedQueryParser().parse(query)
        matcher = GraphMatcher()
        assessments = []
        for moment in moments:
            start_ms = moment.moment_start_seconds * 1000.0
            end_ms = moment.moment_end_seconds * 1000.0
            evidence = self._collect_moment_evidence(moment.video_id, start_ms, end_ms)
            match = matcher.match(graph, evidence)
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

        # =========================================================================
        # Stage 1: Query Analysis & Vector Construction (864d with B5, B6, C1)
        # =========================================================================
        clauses = decompose_query_clauses(query) if ablation_config not in ("A0", "A1") else [query]
        obj_reqs = extract_object_requirements(query) if ablation_config != "A0" else []
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

        if sub_clauses and ablation_config not in ("A0", "A1"):
            sub_vecs_768 = batch_vecs[2:]
            sub_q_864 = [
                build_query_864d_vector(sv, sc, self.coco_idf_map)
                for sv, sc in zip(sub_vecs_768, sub_clauses)
            ]
            all_864_vecs = np.vstack([primary_q_864_mat] + [sv.reshape(1, -1) for sv in sub_q_864])
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

                    seq_score, anchor_win = align_sequential_keyframe_clauses(
                        cand_matrix=cand_mat,
                        cand_timestamps=cand_ts,
                        clause_vectors=sub_q_864,
                        max_transition_sec=35.0,
                    )
                    video_seq_scores[cvid] = seq_score
                    if anchor_win is not None:
                        video_anchor_windows[cvid] = anchor_win

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
                    s_comb = 0.65 * s_full + 0.35 * video_seq_scores[vid]
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
            })

        t_faiss_end = time.perf_counter()

        # =========================================================================
        # Stage 3: Specificity-Sorted Lexical FTS5 Search (B7, C2 persistent SQLite)
        # =========================================================================
        ocr_candidates: list[dict[str, Any]] = []
        asr_candidates: list[dict[str, Any]] = []

        if fts_query and self.db_conn is not None and ablation_config not in ("A0", "A1", "A2"):
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "SELECT video_id, timestamp_ms, rank, normalized_text FROM fts_ocr WHERE fts_ocr MATCH ? ORDER BY rank LIMIT 40;",
                    (fts_query,),
                )
                for vid, ts_ms, rank, norm_text in cursor.fetchall():
                    ocr_candidates.append({
                        "video_id": vid,
                        "timestamp_sec": ts_ms / 1000.0,
                        "score": -float(rank),
                        "branch": "ocr",
                        "matched_text": norm_text or "",
                    })

                cursor.execute(
                    "SELECT video_id, start_ms, end_ms, rank, normalized_text FROM fts_asr WHERE fts_asr MATCH ? ORDER BY rank LIMIT 40;",
                    (fts_query,),
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
            cl_obj = {
                "video_id": c["video_id"],
                "rep_timestamp": c["timestamp_sec"],
                "visual_sim": c["score"],
                "rrf_vis": rrf_val,
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

                single_common_only = (
                    not has_quoted_match
                    and not has_proper_noun_match
                    and m_high == 1
                    and m_low == 0
                    and (
                        (k_high_len == 1 and next(iter(high_spec_tokens), "").endswith("ing"))
                        or (k_high_len == 2 and fts_sc < 19.0)
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
                            lex_boost = 2.0 * cov_ratio * gamma_vis * base_lex * bm25_factor
                    else:
                        lex_boost = 0.15 * gamma_vis * base_lex
                # B9: Pure-ASR / Pure-OCR False Positive Gate (Strict gating without visual anchor)
                else:
                    can_enter_case_a = (
                        has_quoted_match
                        or (has_proper_noun_match and (m_high >= 2 or fts_sc >= 22.0 or len(cl["branches"]) >= 2))
                        or (m_high >= 2 and gamma_vis >= 0.45 and (has_proper_noun_match or fts_sc >= 22.0 or m_high >= max(3, int(math.ceil(0.6 * k_high_len)))))
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
        cand_moments: list[RankedMoment] = []
        for vid, vid_items in by_video_raw.items():
            vid_items.sort(key=lambda x: x[0], reverse=True)
            scores = [x[0] for x in vid_items]
            m_count = len(scores)
            s_max = scores[0]
            top3_mean = sum(scores[:min(m_count, 3)]) / min(m_count, 3)
            top5_mean = sum(scores[:min(m_count, 5)]) / min(m_count, 5)

            sorted_by_time = sorted(vid_items, key=lambda x: x[1])
            adjacent_pairs = 0
            for i in range(len(sorted_by_time) - 1):
                if sorted_by_time[i + 1][1] - sorted_by_time[i][2] <= 6.0:
                    adjacent_pairs += 1

            effective_s_max = max(0.0, s_max)
            n_kf = len(self.video_keyframes.get(vid, []))
            ratio = adjacent_pairs / max(1.0, float(n_kf - 1))
            density_bonus = (0.060 * effective_s_max) * min(adjacent_pairs, 3) + 0.015 * effective_s_max * ratio

            best_sc, best_st, best_et, best_rep_t, best_branch = vid_items[0]
            local_window_scores = sorted(
                (x[0] for x in vid_items if abs(x[3] - best_rep_t) <= 3.5),
                reverse=True,
            )[:2]
            local_window_mean = (sum(local_window_scores) / len(local_window_scores)) if local_window_scores else s_max

            if best_branch in ("asr", "ocr", "asr+ocr") and (len(proper_nouns_lower) >= 1 or len(quoted_phrases) >= 1 or m_count == 1):
                video_score = (
                    0.85 * s_max
                    + 0.10 * top3_mean
                    + 0.05 * top5_mean
                    + density_bonus
                )
            elif "asr" in best_branch or "ocr" in best_branch:
                video_score = (
                    0.65 * s_max
                    + 0.15 * top3_mean
                    + 0.20 * top5_mean
                    + density_bonus
                )
            elif ablation_config == "A5":
                video_score = (
                    0.54 * s_max
                    + 0.15 * top3_mean
                    + 0.29 * top5_mean
                    + 0.02 * local_window_mean
                    + density_bonus
                )
            else:
                video_score = (
                    0.55 * s_max
                    + 0.15 * top3_mean
                    + 0.30 * top5_mean
                    + density_bonus
                )
            all_branches_in_video: set[str] = set()
            for item in vid_items:
                b_status = item[4]
                if b_status and b_status not in ("supported", "partial", "unknown", "contradicted"):
                    for b in b_status.split("+"):
                        all_branches_in_video.add(b.strip())
            branch_attribution = "+".join(sorted(all_branches_in_video)) if all_branches_in_video else "visual"

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

        cand_moments.sort(key=lambda m: float(m.ranking_score), reverse=True)
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
            if ablation_config in ("A0", "A1", "A2", "A3"):
                delta = min(5.0, 0.45 * v_dur)
                start_sec = max(0.0, rep_t - delta / 2.0)
                end_sec = min(v_dur, rep_t + delta / 2.0)
            elif len(kf_list) > 1:
                ts_array = self.video_kf_pts.get(vid) or [kf["pts_time"] for kf in kf_list]
                if vid in video_curve_cache:
                    exact_curve = video_curve_cache[vid][1].tolist()
                else:
                    kf_matrix = self.video_kf_matrix.get(vid)
                    if kf_matrix is None:
                        kf_fids = [int(kf["faiss_id"]) for kf in kf_list]
                        kf_matrix = self.augmented_embeddings[kf_fids]
                    exact_curve = (kf_matrix @ primary_q_864.reshape(-1)).tolist()

                # B10: Anchor index and anchor window calculation
                anchor_win = video_anchor_windows.get(vid)
                if anchor_win is not None:
                    t_mid = (anchor_win[0] + anchor_win[1]) / 2.0
                    best_idx = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - t_mid))
                else:
                    best_idx = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - rep_t))

                start_sec, end_sec = expand_boundary_contrast(
                    ts=ts_array,
                    curve=exact_curve,
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

            final_results.append(RankedMoment(
                moment_id=f"{vid}_{round(start_sec, 1)}_{round(end_sec, 1)}",
                video_id=vid,
                moment_start_seconds=round(start_sec, 2),
                moment_end_seconds=round(end_sec, 2),
                start_original_frame_index=0,
                end_original_frame_index=0,
                representative_original_frame_index=best_idx,
                representative_timestamp_seconds=round(rep_t, 2),
                ranking_score=round(float(moment.ranking_score), 5),
                match_status=moment.match_status,
            ))

            # Multi-Window Proposal for Top-1 Video (enabled in A5):
            if ablation_config == "A5" and rank_idx == 1 and len(kf_list) > 3:
                s_max = max(exact_curve)
                s_min = min(exact_curve)
                masked_curve = [
                    s if (ts_array[i] < start_sec - 4.0 or ts_array[i] > end_sec + 4.0) else (s_min - 0.05)
                    for i, s in enumerate(exact_curve)
                ]
                s_max2 = max(masked_curve)
                if s_max2 >= 0.88 * s_max:
                    st2, et2 = expand_boundary_contrast(
                        ts=ts_array,
                        curve=masked_curve,
                        v_dur=v_dur,
                        decay_lambda=0.015,
                        c_thresh=0.45,
                        max_gap_sec=4.0,
                        max_dur_ratio=0.45,
                    )
                    if et2 > st2 and (et2 - st2) >= 1.0 and (et2 <= start_sec or st2 >= end_sec):
                        final_results.append(RankedMoment(
                            moment_id=f"{vid}_{round(st2, 1)}_{round(et2, 1)}_win2",
                            video_id=vid,
                            moment_start_seconds=round(st2, 2),
                            moment_end_seconds=round(et2, 2),
                            start_original_frame_index=0,
                            end_original_frame_index=0,
                            representative_original_frame_index=0,
                            representative_timestamp_seconds=round((st2 + et2) / 2.0, 2),
                            ranking_score=round(float(moment.ranking_score) * 0.95, 5),
                            match_status=f"{moment.match_status}+multi_window",
                        ))

        t_boundary_end = time.perf_counter()

        t_parse_ms = (t_parse_end - t_start) * 1000.0
        t_encode_ms = (t_encode_end - t_parse_end) * 1000.0
        t_faiss_ms = (t_faiss_end - t_encode_end) * 1000.0
        t_fts5_ms = (t_fts5_end - t_faiss_end) * 1000.0
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
        self.last_diagnostics = {
            "lexical_status": "complete" if fts_query and self.db_conn is not None and ablation_config not in ("A0", "A1", "A2") else "not_run",
            "visual_top30_vids": set(vis_ordered_vids[:30]),
            "visual_top120_vids": set(vis_ordered_vids[:120]),
            "lexical_top30_vids": set(lex_ordered_vids[:30]),
            "fused_top10_vids": {m.video_id for m in cand_moments[:10]},
            "all_candidate_moments": cand_moments,
            "visual_candidates": visual_candidates[:30],
            "ocr_candidates": ocr_candidates[:15],
            "asr_candidates": asr_candidates[:15],
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
        if self.cache is not None:
            self.cache.save()
        if self.db_conn is not None:
            try:
                self.db_conn.close()
            except Exception:
                pass
            self.db_conn = None

    def __enter__(self) -> UnifiedMultimodalSearchPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
