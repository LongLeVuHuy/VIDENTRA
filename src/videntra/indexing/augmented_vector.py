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

        if nearby_objs:
            class_counts: dict[str, int] = {}
            class_max_conf: dict[str, float] = {}
            person_conf_count = 0

            for o in nearby_objs:
                cname = o.get("class_name", "")
                conf = float(o.get("confidence", 0.0))
                class_counts[cname] = class_counts.get(cname, 0) + 1
                if conf > class_max_conf.get(cname, 0.0):
                    class_max_conf[cname] = conf
                if cname == "person" and conf >= 0.40:
                    person_conf_count += 1

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
    faiss_path: str | Path = "data/processed/faiss_index.bin",
    metadata_db_path: str | Path = "data/processed/metadata.db",
    output_npy_path: str | Path | None = "data/processed/augmented_embeddings.npy",
    alpha_obj: float = 0.25,
    alpha_meta: float = 0.15,
    beta_hub: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rebuilds decoupled 864d normalized augmented representations directly on CPU in < 0.5s.

    Reads 768d raw vectors from faiss_index.bin and metadata from metadata.db,
    computes de-hubbed semantic + 80d YOLOv8 IDF + 16d context representation,
    and optionally persists to output_npy_path.
    """
    import sqlite3
    from collections import defaultdict
    import faiss

    f_path = Path(faiss_path)
    db_path = Path(metadata_db_path)

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


if __name__ == "__main__":
    import time
    t0 = time.time()
    aug, _, _ = rebuild_augmented_864d_cache()
    print(f"Successfully rebuilt {aug.shape} augmented embeddings in {time.time() - t0:.3f}s")
