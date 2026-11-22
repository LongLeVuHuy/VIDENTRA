"""864-dimensional multimodal representation builder for SigLIP 2, YOLOv8 IDF object subspace, and context priors."""
from __future__ import annotations

import bisect
import math
from pathlib import Path
import re
from typing import Any, Sequence
import numpy as np

from videntra.query.rules import COCO_SYNONYM_MAP, extract_object_requirements


# Fixed, alphabetically sorted list of 80 COCO classes
COCO_80_CLASSES: list[str] = sorted(set(COCO_SYNONYM_MAP.values()))
COCO_CLASS_TO_IDX: dict[str, int] = {cname: idx for idx, cname in enumerate(COCO_80_CLASSES)}
_SMALL_COCO_CLASSES: frozenset[str] = frozenset({
    "sports ball", "cell phone", "cup", "bottle", "book",
    "remote", "mouse", "keyboard", "fork", "spoon", "knife",
})


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 1e-8 else 0.0


def _extract_obj_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if all(obj.get(k) is not None for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        x1, y1, x2, y2 = float(obj["bbox_x1"]), float(obj["bbox_y1"]), float(obj["bbox_x2"]), float(obj["bbox_y2"])
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    raw_box = obj.get("bbox")
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        x1, y1, x2, y2 = (float(v) for v in raw_box)
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None


def deduplicate_and_filter_frame_objects(
    objs: Sequence[dict[str, Any]],
    *,
    default_conf: float = 0.35,
    small_conf: float = 0.28,
    iou_thresh: float = 0.65,
) -> list[dict[str, Any]]:
    """Filter low-confidence YOLO detections and apply per-(frame, class) IoU NMS."""
    by_frame_cls: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for o in objs:
        cname = str(o.get("canonical_class") or o.get("class_name") or "").lower().strip()
        if not cname:
            continue
        conf = float(o.get("confidence", 0.0))
        min_conf = small_conf if cname in _SMALL_COCO_CLASSES else default_conf
        if conf < min_conf:
            continue
        fk = o.get("keyframe_id")
        if fk is None:
            pts_val = o.get("pts_sec")
            fk = ("pts", round(float(pts_val), 2)) if pts_val is not None else ("noid", id(o))
        by_frame_cls.setdefault((fk, cname), []).append(o)

    kept: list[dict[str, Any]] = []
    for group in by_frame_cls.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(group, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        group_kept: list[tuple[dict[str, Any], tuple[float, float, float, float] | None]] = []
        for cand in group_sorted:
            c_box = _bbox_iou_box = _extract_obj_bbox(cand)
            if c_box is not None and any(
                k_box is not None and _bbox_iou(c_box, k_box) > iou_thresh
                for _, k_box in group_kept
            ):
                continue
            group_kept.append((cand, c_box))
        kept.extend(item for item, _ in group_kept)
    return kept


def compute_coco_global_idf(
    frame_objects_by_video: dict[str, list[dict]],
    total_videos: int,
) -> dict[str, float]:
    """Computes global inverse document frequency (IDF) for 80 COCO classes across indexed videos.
    
    Formula:
        idf_c = ln(1 + N_videos / (1 + df(c)))
    """
    df: dict[str, int] = {c: 0 for c in COCO_80_CLASSES}
    n_vids = max(1, total_videos)

    for vid, objs in frame_objects_by_video.items():
        present_classes = set()
        for o in objs:
            cname = o.get("class_name")
            if cname in df:
                present_classes.add(cname)
        for c in present_classes:
            df[c] += 1

    return {c: float(math.log(1.0 + n_vids / (1.0 + df[c]))) for c in COCO_80_CLASSES}


def build_augmented_864d_embeddings(
    raw_embeddings: np.ndarray,
    keyframe_map: dict[int, Any],
    frame_objects: list[dict] | dict[str, list[dict]],
    videos_duration: dict[str, float],
    ocr_data: dict[str, list[float] | list[dict]] | None = None,
    asr_data: dict[str, list[tuple[float, float]] | list[dict]] | None = None,
    alpha_obj: float = 0.25,
    alpha_meta: float = 0.15,
    beta_hub: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Builds 864d L2-normalized keyframe representations combining:
    - 768d De-hubbed SigLIP 2 semantic embeddings
    - 80d IDF-weighted YOLOv8 high-res object subspace (alpha_obj = 0.25)
    - 16d Context subspace (person cardinality, temporal position, OCR/ASR density, alpha_meta = 0.15)
    
    Returns:
        augmented_embeddings: np.ndarray of shape (N, 864), L2-normalized float32
        mean_vec_768: np.ndarray of shape (768,), L2-normalized
        coco_idf_map: dict mapping COCO class name (str) -> idf weight (float)
    """
    n_total, d_sem = raw_embeddings.shape
    assert d_sem == 768, f"Expected 768d semantic embeddings, got {d_sem}"

    if n_total == 0:
        return (
            np.zeros((0, 864), dtype=np.float32),
            np.zeros(768, dtype=np.float32),
            {c: 1.0 for c in COCO_80_CLASSES},
        )

    # 1. Hubness Reduction on 768d Semantic Subspace
    mean_vec_raw = np.mean(raw_embeddings, axis=0, keepdims=True)
    norm_mean = float(np.linalg.norm(mean_vec_raw)) + 1e-8
    mean_vec_768 = (mean_vec_raw / norm_mean).astype(np.float32)

    # Projection of embeddings onto mean direction
    h_bias = (raw_embeddings @ mean_vec_768.T)  # (N, 1)
    dehubbed_sem = raw_embeddings - beta_hub * (h_bias @ mean_vec_768)
    sem_norms = np.linalg.norm(dehubbed_sem, axis=1, keepdims=True) + 1e-8
    sem_768 = (dehubbed_sem / sem_norms).astype(np.float32)

    # 2. Organize Frame Objects by Video
    if isinstance(frame_objects, dict):
        objs_by_vid = frame_objects
    else:
        objs_by_vid: dict[str, list[dict]] = {}
        for o in frame_objects:
            vid = o.get("video_id")
            if vid not in objs_by_vid:
                objs_by_vid[vid] = []
            pts = o.get("pts_sec")
            if pts is None and "timestamp_ms" in o:
                pts = float(o["timestamp_ms"]) / 1000.0
            objs_by_vid[vid].append({
                "pts_sec": float(pts if pts is not None else 0.0),
                "class_name": str(o.get("class_name", "")),
                "confidence": float(o.get("confidence", 1.0)),
            })

    total_vids = len(videos_duration) if videos_duration else len(objs_by_vid)
    coco_idf_map = compute_coco_global_idf(objs_by_vid, total_vids)

    # Clean/prepare OCR and ASR timestamps lookup
    ocr_ts_map: dict[str, list[float]] = {}
    if ocr_data:
        for vid, entries in ocr_data.items():
            ts_list = []
            for item in entries:
                if isinstance(item, (int, float)):
                    ts_list.append(float(item))
                elif isinstance(item, dict):
                    t = item.get("pts_sec", item.get("timestamp_sec", item.get("start_seconds", 0.0)))
                    ts_list.append(float(t))
            ocr_ts_map[vid] = sorted(ts_list)

    asr_spans_map: dict[str, list[tuple[float, float]]] = {}
    if asr_data:
        for vid, entries in asr_data.items():
            spans = []
            for item in entries:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    spans.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    st = item.get("start_sec", item.get("start_seconds", 0.0))
                    et = item.get("end_sec", item.get("end_seconds", st + 1.0))
                    spans.append((float(st), float(et)))
            asr_spans_map[vid] = sorted(spans, key=lambda x: x[0])

    # Pre-sort frame objects by video and pts_sec for fast bisect retrieval
    objs_by_vid_sorted: dict[str, tuple[list[float], list[dict]]] = {}
    for vid, objs in objs_by_vid.items():
        sorted_objs = sorted(objs, key=lambda x: x["pts_sec"])
        pts_list = [o["pts_sec"] for o in sorted_objs]
        objs_by_vid_sorted[vid] = (pts_list, sorted_objs)

    # Pre-allocate 80d Object Subspace and 16d Context Subspace
    obj_subspace = np.zeros((n_total, 80), dtype=np.float32)
    meta_subspace = np.zeros((n_total, 16), dtype=np.float32)

    for fid in range(n_total):
        kinfo = keyframe_map.get(fid)
        if not kinfo:
            continue

        if isinstance(kinfo, dict):
            vid = kinfo.get("video_id", "")
            t_sec = float(kinfo.get("pts_time", 0.0))
        else:
            vid = getattr(kinfo, "video_id", "")
            t_sec = float(getattr(kinfo, "pts_time", 0.0))

        raw_dur = videos_duration.get(vid) if videos_duration else None
        v_dur = float(raw_dur) if raw_dur is not None else 300.0

        # Build 80d Object feature for keyframe fid using bisect for speed
        vid_entry = objs_by_vid_sorted.get(vid)
        if vid_entry:
            pts_list, sorted_objs = vid_entry
            left = bisect.bisect_left(pts_list, t_sec - 3.5)
            right = bisect.bisect_right(pts_list, t_sec + 3.5)
            nearby_objs = sorted_objs[left:right]
        else:
            nearby_objs = []

        filtered_objs = deduplicate_and_filter_frame_objects(nearby_objs)
        valid_bboxes: list[tuple[float, float, float, float]] = []
        if filtered_objs:
            class_counts: dict[str, int] = {}
            class_max_conf: dict[str, float] = {}
            has_kf_ids = any(o.get("keyframe_id") is not None for o in filtered_objs)
            if has_kf_ids:
                # Group by single keyframe so tracked objects across keyframes are NOT summed
                tight_objs = [o for o in filtered_objs if abs(float(o.get("pts_sec", t_sec)) - t_sec) <= 2.5]
                agg_objs = tight_objs if tight_objs else filtered_objs
                per_kf_counts: dict[Any, dict[str, int]] = {}
                per_kf_person: dict[Any, int] = {}
                for o in agg_objs:
                    cname = o.get("class_name", "")
                    conf = float(o.get("confidence", 0.0))
                    fk = o.get("keyframe_id") if o.get("keyframe_id") is not None else round(float(o.get("pts_sec", t_sec)), 2)
                    cdict = per_kf_counts.setdefault(fk, {})
                    cdict[cname] = cdict.get(cname, 0) + 1
                    if conf > class_max_conf.get(cname, 0.0):
                        class_max_conf[cname] = conf
                    if cname == "person" and conf >= 0.40:
                        per_kf_person[fk] = per_kf_person.get(fk, 0) + 1
                    box = _extract_obj_bbox(o)
                    if box is not None:
                        valid_bboxes.append(box)
                for cdict in per_kf_counts.values():
                    for cname, cnt in cdict.items():
                        if cnt > class_counts.get(cname, 0):
                            class_counts[cname] = cnt
                person_conf_count = max(per_kf_person.values(), default=0)
            else:
                person_conf_count = 0
                for o in filtered_objs:
                    cname = o.get("class_name", "")
                    conf = float(o.get("confidence", 0.0))
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                    if conf > class_max_conf.get(cname, 0.0):
                        class_max_conf[cname] = conf
                    if cname == "person" and conf >= 0.40:
                        person_conf_count += 1
                    box = _extract_obj_bbox(o)
                    if box is not None:
                        valid_bboxes.append(box)

            for cname, count in class_counts.items():
                if cname in COCO_CLASS_TO_IDX:
                    c_idx = COCO_CLASS_TO_IDX[cname]
                    idf = coco_idf_map.get(cname, 1.0)
                    conf = class_max_conf.get(cname, 1.0)
                    obj_subspace[fid, c_idx] = idf * conf * (1.0 + 0.35 * math.log(min(count, 5)))
        else:
            person_conf_count = 0

        # Normalize 80d object vector
        obj_norm = float(np.linalg.norm(obj_subspace[fid]))
        if obj_norm > 1e-8:
            obj_subspace[fid] /= obj_norm

        # Build 16d Context feature for keyframe fid
        # Dim 0..3: Person Cardinality
        if person_conf_count == 1:
            meta_subspace[fid, 0] = 1.0
        elif person_conf_count == 2:
            meta_subspace[fid, 1] = 1.0
        elif person_conf_count >= 3:
            meta_subspace[fid, 2] = 1.0
        else:
            meta_subspace[fid, 3] = 1.0

        # Dim 4..7: Relative Temporal Position Encoding
        p_rel = min(1.0, max(0.0, t_sec / max(1e-4, v_dur)))
        meta_subspace[fid, 4] = math.sin(2.0 * math.pi * 1.0 * p_rel)
        meta_subspace[fid, 5] = math.cos(2.0 * math.pi * 1.0 * p_rel)
        meta_subspace[fid, 6] = math.sin(2.0 * math.pi * 2.0 * p_rel)
        meta_subspace[fid, 7] = math.cos(2.0 * math.pi * 2.0 * p_rel)

        # Dim 8..11: OCR Presence and Token Density (+/- 2.0s)
        vid_ocr = ocr_ts_map.get(vid, [])
        left_ocr = bisect.bisect_left(vid_ocr, t_sec - 2.0)
        right_ocr = bisect.bisect_right(vid_ocr, t_sec + 2.0)
        nearby_ocr = vid_ocr[left_ocr:right_ocr]
        if nearby_ocr:
            meta_subspace[fid, 8] = 1.0
            meta_subspace[fid, 9] = min(1.0, len(nearby_ocr) / 5.0)

        # Dim 12..15: ASR Presence and Speech Duration (+/- 2.0s and +/- 5.0s)
        vid_asr = asr_spans_map.get(vid, [])
        overlap_speech_sec = 0.0
        has_nearby_asr = False
        for st, et in vid_asr:
            if not ((et < t_sec - 2.0) or (st > t_sec + 2.0)):
                has_nearby_asr = True
            # Overlap in 10s window [t_sec - 5.0, t_sec + 5.0]
            w_st, w_et = max(0.0, t_sec - 5.0), t_sec + 5.0
            inter = max(0.0, min(et, w_et) - max(st, w_st))
            overlap_speech_sec += inter
            if st > t_sec + 5.0:
                break

        if has_nearby_asr:
            meta_subspace[fid, 12] = 1.0
        meta_subspace[fid, 13] = min(1.0, overlap_speech_sec / 10.0)

        # Optional spatial centroid & foreground area layout (only when real bboxes exist; preserve 0.0 when empty)
        if valid_bboxes:
            max_x2 = max(b[2] for b in valid_bboxes)
            max_y2 = max(b[3] for b in valid_bboxes)
            norm_w = max(640.0, max_x2) if max_x2 > 2.0 else 1.0
            norm_h = max(360.0, max_y2) if max_y2 > 2.0 else 1.0
            cxs = [0.5 * (b[0] + b[2]) / norm_w for b in valid_bboxes]
            cys = [0.5 * (b[1] + b[3]) / norm_h for b in valid_bboxes]
            areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / (norm_w * norm_h) for b in valid_bboxes]
            meta_subspace[fid, 10] = float(min(1.0, max(0.0, sum(cxs) / len(cxs)))) * 0.25
            meta_subspace[fid, 11] = float(min(1.0, max(0.0, sum(cys) / len(cys)))) * 0.25
            meta_subspace[fid, 15] = float(min(1.0, max(0.0, sum(areas)))) * 0.25

        # Normalize 16d context vector
        meta_norm = float(np.linalg.norm(meta_subspace[fid]))
        if meta_norm > 1e-8:
            meta_subspace[fid] /= meta_norm

        # B5: Auxiliary dummy dimension when no YOLO objects detected
        # If obj_norm <= 1e-8, meta_subspace[fid, 14] is set to 1.0 (with weight alpha_obj in concatenation)
        # to guarantee invariant semantic scaling and exact unit norm across all keyframes.
        if obj_norm <= 1e-8:
            meta_subspace[fid, 14] = 1.0

    # 3. Concatenation and Decoupled Subspace Scaling (B5 fix)
    z_const = float(math.sqrt(1.0 + alpha_obj**2 + alpha_meta**2))

    scaled_meta = alpha_meta * meta_subspace
    no_obj_mask = (meta_subspace[:, 14] == 1.0)
    if np.any(no_obj_mask):
        scaled_meta[no_obj_mask, 14] = alpha_obj

    augmented = np.concatenate(
        [
            sem_768,
            alpha_obj * obj_subspace,
            scaled_meta,
        ],
        axis=1,
    )  # shape (N, 864)

    augmented = (augmented / z_const).astype(np.float32)

    return augmented, mean_vec_768.flatten(), coco_idf_map


def build_query_864d_vector(
    sem_vec: np.ndarray,
    query_text: str,
    coco_idf_map: dict[str, float],
    alpha_obj: float = 0.25,
    alpha_meta: float = 0.15,
    precomputed_obj_reqs: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Builds an 864d L2-normalized query vector matching the 3-subspace representation:
    - 768d Text semantic embedding (L2 normalized)
    - 80d Object IDF vector extracted from query entities (alpha_obj = 0.25)
    - 16d Context priors vector (cardinality and temporal indicators, alpha_meta = 0.15)
    
    Returns:
        np.ndarray of shape (864,), L2-normalized float32
    """
    sem = np.asarray(sem_vec, dtype=np.float32).flatten()
    sem_norm = float(np.linalg.norm(sem)) + 1e-8
    sem_768 = sem / sem_norm

    # 1. 80d Object Subspace from Query
    q_obj = np.zeros(80, dtype=np.float32)
    obj_reqs = precomputed_obj_reqs if precomputed_obj_reqs is not None else extract_object_requirements(query_text)
    person_cardinality = None

    for req in obj_reqs:
        cname = req.get("class_name", "")
        cnt = req.get("cardinality", 1)
        if cname == "person":
            person_cardinality = cnt
        if cname in COCO_CLASS_TO_IDX:
            c_idx = COCO_CLASS_TO_IDX[cname]
            idf = coco_idf_map.get(cname, 1.0)
            q_obj[c_idx] = idf * (1.0 + 0.35 * math.log(min(cnt, 5)))

    q_obj_norm = float(np.linalg.norm(q_obj))
    if q_obj_norm > 1e-8:
        q_obj /= q_obj_norm

    # 2. 16d Context Subspace from Query
    q_meta = np.zeros(16, dtype=np.float32)
    low_q = query_text.lower()

    # Cardinality
    if person_cardinality is not None:
        if person_cardinality == 1:
            q_meta[0] = 1.0
        elif person_cardinality == 2:
            q_meta[1] = 1.0
        elif person_cardinality >= 3:
            q_meta[2] = 1.0
    elif re.search(r'\b(?:no person|no people|nobody|empty room|empty background|không có người)\b', low_q):
        q_meta[3] = 1.0

    # Temporal Priors
    if re.search(r'\b(?:at the beginning|at the start|first|starts with|bắt đầu|mở đầu)\b', low_q):
        p = 0.1
        q_meta[4] = math.sin(2.0 * math.pi * 1.0 * p)
        q_meta[5] = math.cos(2.0 * math.pi * 1.0 * p)
        q_meta[6] = math.sin(2.0 * math.pi * 2.0 * p)
        q_meta[7] = math.cos(2.0 * math.pi * 2.0 * p)
    elif re.search(r'\b(?:at the end|ending|finally|last|concludes|kết thúc|cuối cùng)\b', low_q):
        p = 0.9
        q_meta[4] = math.sin(2.0 * math.pi * 1.0 * p)
        q_meta[5] = math.cos(2.0 * math.pi * 1.0 * p)
        q_meta[6] = math.sin(2.0 * math.pi * 2.0 * p)
        q_meta[7] = math.cos(2.0 * math.pi * 2.0 * p)

    # Text / Speech Modality indicators in query (B6 fix: word boundaries to prevent matching 'context', 'design', etc.)
    if re.search(r'["\']|\b(?:text|words|sign|signs|screen|caption|title|logo|banner|chữ|tiêu đề|biển báo)\b', low_q):
        q_meta[8] = 1.0
    if re.search(r'\b(?:speaking|talking|speech|says|words said|nói|thoại)\b', low_q):
        q_meta[12] = 1.0

    # Ensure auxiliary null-object dimension 14 is never active in query vector
    q_meta[14] = 0.0

    q_meta_norm = float(np.linalg.norm(q_meta))
    if q_meta_norm > 1e-8:
        q_meta /= q_meta_norm

    # 3. Concatenate and Decoupled Subspace Scaling (B5 fix)
    z_const = float(math.sqrt(1.0 + alpha_obj**2 + alpha_meta**2))
    aug = np.concatenate([sem_768, alpha_obj * q_obj, alpha_meta * q_meta])
    return (aug / z_const).astype(np.float32)


def rebuild_augmented_864d_cache(
    faiss_path: str | Path | None = None,
    metadata_db_path: str | Path | None = None,
    output_npy_path: str | Path | None = "data/processed/visual_keyframes/siglip2_yolo_augmented_864d.npy",
    alpha_obj: float = 0.25,
    alpha_meta: float = 0.15,
    beta_hub: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rebuilds decoupled 864d normalized augmented representations directly on CPU in < 0.5s.

    Reads 768d raw vectors from siglip2_keyframes_768d.faiss and metadata from corpus_metadata.db,
    computes de-hubbed semantic + 80d YOLOv8 IDF + 16d context representation,
    and optionally persists to output_npy_path.
    """
    import sqlite3
    from collections import defaultdict
    import faiss
    from videntra.config.paths import (
        resolve_env_db_path,
        resolve_env_faiss_path,
    )

    f_path = resolve_env_faiss_path(faiss_path)
    db_path = resolve_env_db_path(metadata_db_path)

    if not f_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {f_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")

    index = faiss.read_index(str(f_path))
    n_total = index.ntotal
    if n_total > 0:
        all_vecs = index.reconstruct_n(0, n_total)
    else:
        all_vecs = np.zeros((0, 768), dtype=np.float32)

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    cur = conn.cursor()

    cur.execute("SELECT faiss_id, video_id, pts_time FROM KeyframeMap;")
    kf_map = {row[0]: {"video_id": row[1], "pts_time": float(row[2])} for row in cur.fetchall()}

    cur.execute("SELECT video_id, duration_ms FROM Videos;")
    durations = {row[0]: float(row[1]) / 1000.0 for row in cur.fetchall() if row[1] is not None}

    cur.execute("SELECT video_id, timestamp_ms, class_name, confidence FROM FrameObjects;")
    objs: dict[str, list[dict]] = defaultdict(list)
    for vid, ts_ms, cname, conf in cur.fetchall():
        objs[vid].append({
            "pts_sec": float(ts_ms) / 1000.0,
            "class_name": cname,
            "confidence": float(conf),
        })

    cur.execute("SELECT video_id, timestamp_ms FROM OCRObservations;")
    ocr: dict[str, list[float]] = defaultdict(list)
    for vid, ts_ms in cur.fetchall():
        ocr[vid].append(float(ts_ms) / 1000.0)

    cur.execute("SELECT video_id, start_ms, end_ms FROM ASRSegments;")
    asr: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for vid, st_ms, et_ms in cur.fetchall():
        asr[vid].append((float(st_ms) / 1000.0, float(et_ms) / 1000.0))

    conn.close()

    augmented, mean_vec_768, coco_idf_map = build_augmented_864d_embeddings(
        raw_embeddings=all_vecs,
        keyframe_map=kf_map,
        frame_objects=objs,
        videos_duration=durations,
        ocr_data=ocr,
        asr_data=asr,
        alpha_obj=alpha_obj,
        alpha_meta=alpha_meta,
        beta_hub=beta_hub,
    )

    if output_npy_path is not None:
        out_path = Path(output_npy_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, augmented)

    return augmented, mean_vec_768, coco_idf_map


def _extract_kf_video_and_time(kinfo: Any) -> tuple[str, float]:
    """Extract (video_id, pts_time_sec) from a keyframe dict or attribute object."""
    if isinstance(kinfo, dict):
        vid = str(kinfo.get("video_id", ""))
        if "pts_time" in kinfo and kinfo["pts_time"] is not None:
            t_sec = float(kinfo["pts_time"])
        elif "pts_sec" in kinfo and kinfo["pts_sec"] is not None:
            t_sec = float(kinfo["pts_sec"])
        elif "timestamp_ms" in kinfo and kinfo["timestamp_ms"] is not None:
            t_sec = float(kinfo["timestamp_ms"]) / 1000.0
        else:
            t_sec = 0.0
        return vid, t_sec

    vid = str(getattr(kinfo, "video_id", ""))
    pts_time = getattr(kinfo, "pts_time", None)
    if pts_time is None:
        pts_time = getattr(kinfo, "pts_sec", None)
    if pts_time is None:
        ts_ms = getattr(kinfo, "timestamp_ms", None)
        t_sec = float(ts_ms) / 1000.0 if ts_ms is not None else 0.0
    else:
        t_sec = float(pts_time)
    return vid, t_sec


def _group_keyframes_by_video(
    keyframe_map: dict[int, Any] | Sequence[Any],
    n_total: int,
) -> dict[str, list[tuple[int, float]]]:
    """Group keyframe (global_idx, pts_sec) pairs by video_id, sorted by pts_sec."""
    by_vid: dict[str, list[tuple[int, float]]] = {}
    is_dict = isinstance(keyframe_map, dict)
    for fid in range(n_total):
        kinfo = keyframe_map.get(fid) if is_dict else (keyframe_map[fid] if fid < len(keyframe_map) else None)
        if kinfo is None:
            continue
        vid, t_sec = _extract_kf_video_and_time(kinfo)
        by_vid.setdefault(vid, []).append((fid, t_sec))

    for vid in by_vid:
        by_vid[vid].sort(key=lambda item: (item[1], item[0]))
    return by_vid


def compute_temporal_motion_delta(
    embeddings_768: np.ndarray,
    keyframe_map: dict[int, Any] | Sequence[Any],
) -> np.ndarray:
    """Compute CPU-fast Temporal Motion Delta subspace (Delta v_i) per video.

    Formula (Plan §4.1):
        Delta v_i = (v_{i+1} - v_{i-1}) / (||v_{i+1} - v_{i-1}||_2 + 1e-8)
    Boundary keyframes use forward (i=0) or backward (i=M-1) difference;
    single-keyframe videos receive a zero vector.
    """
    vecs = np.asarray(embeddings_768, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {vecs.shape}")
    n_total, dim = vecs.shape
    deltas = np.zeros((n_total, dim), dtype=np.float32)
    if n_total == 0:
        return deltas

    by_vid = _group_keyframes_by_video(keyframe_map, n_total)
    for items in by_vid.values():
        m = len(items)
        if m <= 1:
            continue
        fids = np.array([it[0] for it in items], dtype=np.int64)
        vid_vecs = vecs[fids]

        diffs = np.empty_like(vid_vecs)
        diffs[0] = vid_vecs[1] - vid_vecs[0]
        diffs[-1] = vid_vecs[-1] - vid_vecs[-2]
        if m > 2:
            diffs[1:-1] = vid_vecs[2:] - vid_vecs[:-2]

        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
        deltas[fids] = diffs / norms

    return deltas


def compute_multiscale_temporal_context(
    embeddings_768: np.ndarray,
    keyframe_map: dict[int, Any] | Sequence[Any],
    window_sec: float = 3.0,
    sigma_sec: float = 1.5,
) -> np.ndarray:
    """Compute Gaussian-weighted Multi-Scale Temporal Context vectors within [t_i - window_sec, t_i + window_sec]."""
    vecs = np.asarray(embeddings_768, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {vecs.shape}")
    n_total, dim = vecs.shape
    ctx = np.zeros((n_total, dim), dtype=np.float32)
    if n_total == 0:
        return ctx

    sigma = max(1e-4, float(sigma_sec))
    win = max(0.0, float(window_sec))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    by_vid = _group_keyframes_by_video(keyframe_map, n_total)
    for items in by_vid.values():
        fids = np.array([it[0] for it in items], dtype=np.int64)
        times = np.array([it[1] for it in items], dtype=np.float64)
        vid_vecs = vecs[fids]

        # Pairwise time differences (M, M) within the same video
        dt = times[:, None] - times[None, :]
        in_window = np.abs(dt) <= win
        weights = np.where(in_window, np.exp(-(dt * dt) * inv_2sigma2), 0.0).astype(np.float32)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 1e-8, weight_sums, 1.0)

        blended = (weights @ vid_vecs) / weight_sums
        norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-8
        ctx[fids] = blended / norms

    return ctx


def compute_late_interaction_maxsim(
    query_tokens: np.ndarray,
    query_idf: Sequence[float] | np.ndarray | None,
    window_patches: np.ndarray,
) -> float:
    """Compute QAT-LateInteraction token-to-region MaxSim score (Plan §4.1).

    Args:
        query_tokens: Query content token embeddings of shape (L, D) or (D,).
        query_idf: Optional per-token IDF weights of shape (L,). Defaults to 1.0.
        window_patches: Spatial 3x3 patch tokens in candidate window of shape
            (K, 9, D), (K*9, D), or (D,) (float16 or float32).

    Returns:
        IDF-weighted average of max cosine similarities across all window patches.
    """
    q = np.asarray(query_tokens, dtype=np.float32)
    p = np.asarray(window_patches, dtype=np.float32)

    if q.size == 0 or p.size == 0:
        return 0.0

    if q.ndim == 1:
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    elif p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
    elif p.ndim != 2:
        raise ValueError(f"Expected 2D or 3D window_patches, got shape {p.shape}")

    if q.shape[1] != p.shape[1]:
        raise ValueError(
            f"Dimension mismatch between query_tokens ({q.shape[1]}) and window_patches ({p.shape[1]})"
        )

    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    p_norms = np.linalg.norm(p, axis=1, keepdims=True)
    q_unit = q / np.where(q_norms > 1e-8, q_norms, 1.0)
    p_unit = p / np.where(p_norms > 1e-8, p_norms, 1.0)

    # Single vectorized matrix multiply: (L, K*9)
    sim_matrix = q_unit @ p_unit.T
    max_sims = np.max(sim_matrix, axis=1)  # (L,)

    if query_idf is None:
        return float(np.mean(max_sims))

    idf = np.asarray(query_idf, dtype=np.float32).ravel()
    if idf.shape[0] != q.shape[0]:
        raise ValueError(
            f"query_idf length ({idf.shape[0]}) must match query_tokens ({q.shape[0]})"
        )
    idf = np.maximum(idf, 0.0)
    idf_sum = float(np.sum(idf))
    if idf_sum <= 1e-8:
        return float(np.mean(max_sims))
    return float(np.dot(idf, max_sims) / idf_sum)


if __name__ == "__main__":
    import time
    t0 = time.time()
    aug, _, _ = rebuild_augmented_864d_cache()
    print(f"Successfully rebuilt {aug.shape} augmented embeddings in {time.time() - t0:.3f}s")
