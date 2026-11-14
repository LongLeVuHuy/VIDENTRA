"""Evidence Grouping Algorithms for Multi-Model OCR and ASR.

Dispatches observations and transcripts into consensus groups with explicit status:
- 'agreed': Concordance across models.
- 'partial': Partial concordance or subset overlap.
- 'disputed': Strong temporal/spatial overlap but divergent content.
- 'single_model': Only one model produced output.
"""
from __future__ import annotations

import difflib
from typing import Any, Sequence
from videntra.evidence.normalize import normalize_text, fold_diacritics


def text_similarity(s1: str, s2: str) -> float:
    """Normalized similarity ratio using SequenceMatcher on folded text."""
    f1 = fold_diacritics(normalize_text(s1)).lower().strip()
    f2 = fold_diacritics(normalize_text(s2)).lower().strip()
    if not f1 and not f2:
        return 1.0
    if not f1 or not f2:
        return 0.0
    return difflib.SequenceMatcher(None, f1, f2).ratio()


def box_2d_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class OCREvidenceGrouper:
    """Groups multi-model OCR observations per keyframe based on spatial IoU and text similarity."""

    @staticmethod
    def group_keyframe_observations(
        video_id: str,
        keyframe_id: str,
        timestamp_ms: int,
        observations: Sequence[dict[str, Any]],
        iou_threshold: float = 0.35,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
        """Returns (groups, member_mappings: [(group_id, observation_id, model_id)])."""
        if not observations:
            return [], []

        # Partition by model
        by_model: dict[str, list[dict[str, Any]]] = {}
        for obs in observations:
            by_model.setdefault(obs["model_id"], []).append(obs)

        models = list(by_model.keys())
        groups: list[dict[str, Any]] = []
        member_mappings: list[tuple[str, str, str]] = []

        if len(models) == 1:
            # Single model only
            m_id = models[0]
            for i, obs in enumerate(by_model[m_id]):
                gid = f"{keyframe_id}__ocr_g_{i:04d}"
                bbox_str = f"{obs['bbox_x1']:.4f},{obs['bbox_y1']:.4f},{obs['bbox_x2']:.4f},{obs['bbox_y2']:.4f}"
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "timestamp_ms": timestamp_ms,
                    "consensus_text": obs["raw_text"],
                    "consensus_bbox": bbox_str,
                    "status": "single_model",
                    "models_involved": m_id,
                    "track_id": None,
                    "estimated_start_ms": timestamp_ms,
                    "estimated_end_ms": timestamp_ms,
                })
                member_mappings.append((gid, obs["observation_id"], m_id))
            return groups, member_mappings

        # Two or more models (e.g. OCR_A vs OCR_B)
        m_a = models[0]
        m_b = models[1]
        obs_a = by_model[m_a]
        obs_b = by_model[m_b]

        matched_b_indices = set()
        group_idx = 0

        for a in obs_a:
            bbox_a = (a["bbox_x1"], a["bbox_y1"], a["bbox_x2"], a["bbox_y2"])
            best_iou = 0.0
            best_b_idx = -1

            for j, b in enumerate(obs_b):
                if j in matched_b_indices:
                    continue
                bbox_b = (b["bbox_x1"], b["bbox_y1"], b["bbox_x2"], b["bbox_y2"])
                iou = box_2d_iou(bbox_a, bbox_b)
                if iou > best_iou:
                    best_iou = iou
                    best_b_idx = j

            gid = f"{keyframe_id}__ocr_g_{group_idx:04d}"
            group_idx += 1

            if best_b_idx >= 0 and best_iou >= iou_threshold:
                matched_b_indices.add(best_b_idx)
                b = obs_b[best_b_idx]
                sim = text_similarity(a["raw_text"], b["raw_text"])

                if sim >= 0.85:
                    status = "agreed"
                    # Pick higher confidence or vietnamese-normalized
                    consensus = a["raw_text"] if a.get("confidence", 0) >= b.get("confidence", 0) else b["raw_text"]
                elif sim >= 0.40:
                    status = "partial"
                    consensus = f"{a['raw_text']} | {b['raw_text']}"
                else:
                    status = "disputed"
                    consensus = f"{a['raw_text']} [vs] {b['raw_text']}"

                # Union bbox
                u_bbox = (
                    min(a["bbox_x1"], b["bbox_x1"]),
                    min(a["bbox_y1"], b["bbox_y1"]),
                    max(a["bbox_x2"], b["bbox_x2"]),
                    max(a["bbox_y2"], b["bbox_y2"]),
                )
                bbox_str = f"{u_bbox[0]:.4f},{u_bbox[1]:.4f},{u_bbox[2]:.4f},{u_bbox[3]:.4f}"

                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "timestamp_ms": timestamp_ms,
                    "consensus_text": consensus,
                    "consensus_bbox": bbox_str,
                    "status": status,
                    "models_involved": f"{m_a},{m_b}",
                    "track_id": None,
                    "estimated_start_ms": timestamp_ms,
                    "estimated_end_ms": timestamp_ms,
                })
                member_mappings.append((gid, a["observation_id"], m_a))
                member_mappings.append((gid, b["observation_id"], m_b))
            else:
                # Single model from A
                bbox_str = f"{a['bbox_x1']:.4f},{a['bbox_y1']:.4f},{a['bbox_x2']:.4f},{a['bbox_y2']:.4f}"
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "timestamp_ms": timestamp_ms,
                    "consensus_text": a["raw_text"],
                    "consensus_bbox": bbox_str,
                    "status": "single_model",
                    "models_involved": m_a,
                    "track_id": None,
                    "estimated_start_ms": timestamp_ms,
                    "estimated_end_ms": timestamp_ms,
                })
                member_mappings.append((gid, a["observation_id"], m_a))

        # Unmatched from B
        for j, b in enumerate(obs_b):
            if j not in matched_b_indices:
                gid = f"{keyframe_id}__ocr_g_{group_idx:04d}"
                group_idx += 1
                bbox_str = f"{b['bbox_x1']:.4f},{b['bbox_y1']:.4f},{b['bbox_x2']:.4f},{b['bbox_y2']:.4f}"
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "timestamp_ms": timestamp_ms,
                    "consensus_text": b["raw_text"],
                    "consensus_bbox": bbox_str,
                    "status": "single_model",
                    "models_involved": m_b,
                    "track_id": None,
                    "estimated_start_ms": timestamp_ms,
                    "estimated_end_ms": timestamp_ms,
                })
                member_mappings.append((gid, b["observation_id"], m_b))

        return groups, member_mappings


class ASREvidenceGrouper:
    """Groups multi-model ASR transcript segments based on temporal overlap and text similarity."""

    @staticmethod
    def group_video_segments(
        video_id: str,
        segments: Sequence[dict[str, Any]],
        min_overlap_ratio: float = 0.30,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
        """Returns (groups, member_mappings: [(group_id, segment_id, model_id)])."""
        if not segments:
            return [], []

        by_model: dict[str, list[dict[str, Any]]] = {}
        for s in segments:
            by_model.setdefault(s["model_id"], []).append(s)

        models = list(by_model.keys())
        groups: list[dict[str, Any]] = []
        member_mappings: list[tuple[str, str, str]] = []

        if len(models) == 1:
            m_id = models[0]
            for i, seg in enumerate(by_model[m_id]):
                gid = f"{video_id}__asr_g_{i:04d}"
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "consensus_text": seg["raw_text"],
                    "status": "single_model",
                    "models_involved": m_id,
                    "temporal_overlap_ratio": 1.0,
                    "text_similarity": 1.0,
                })
                member_mappings.append((gid, seg["segment_id"], m_id))
            return groups, member_mappings

        m_a, m_b = models[0], models[1]
        segs_a = sorted(by_model[m_a], key=lambda x: x["start_ms"])
        segs_b = sorted(by_model[m_b], key=lambda x: x["start_ms"])

        matched_b_indices = set()
        group_idx = 0

        for a in segs_a:
            st_a, end_a = a["start_ms"], a["end_ms"]
            dur_a = max(1, end_a - st_a)

            # Find matching B segments (could be multiple if one model sliced more finely)
            overlapping_b: list[tuple[int, dict[str, Any], float]] = []
            for j, b in enumerate(segs_b):
                st_b, end_b = b["start_ms"], b["end_ms"]
                overlap = max(0, min(end_a, end_b) - max(st_a, st_b))
                dur_b = max(1, end_b - st_b)
                overlap_ratio = overlap / float(min(dur_a, dur_b))
                if overlap_ratio >= min_overlap_ratio:
                    overlapping_b.append((j, b, overlap_ratio))

            gid = f"{video_id}__asr_g_{group_idx:04d}"
            group_idx += 1

            if overlapping_b:
                for j, b, o_ratio in overlapping_b:
                    matched_b_indices.add(j)

                b_texts = " ".join(b["raw_text"] for _, b, _ in overlapping_b)
                sim = text_similarity(a["raw_text"], b_texts)

                if sim >= 0.75:
                    status = "agreed"
                    consensus = a["raw_text"] if a.get("confidence", 0) >= overlapping_b[0][1].get("confidence", 0) else b_texts
                elif sim >= 0.35:
                    status = "partial"
                    consensus = f"{a['raw_text']} | {b_texts}"
                else:
                    status = "disputed"
                    consensus = f"{a['raw_text']} [vs] {b_texts}"

                all_st = min(a["start_ms"], min(b["start_ms"] for _, b, _ in overlapping_b))
                all_end = max(a["end_ms"], max(b["end_ms"] for _, b, _ in overlapping_b))
                avg_overlap = sum(r for _, _, r in overlapping_b) / len(overlapping_b)

                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "start_ms": all_st,
                    "end_ms": all_end,
                    "consensus_text": consensus,
                    "status": status,
                    "models_involved": f"{m_a},{m_b}",
                    "temporal_overlap_ratio": round(avg_overlap, 3),
                    "text_similarity": round(sim, 3),
                })
                member_mappings.append((gid, a["segment_id"], m_a))
                for _, b, _ in overlapping_b:
                    member_mappings.append((gid, b["segment_id"], m_b))
            else:
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "start_ms": a["start_ms"],
                    "end_ms": a["end_ms"],
                    "consensus_text": a["raw_text"],
                    "status": "single_model",
                    "models_involved": m_a,
                    "temporal_overlap_ratio": 1.0,
                    "text_similarity": 1.0,
                })
                member_mappings.append((gid, a["segment_id"], m_a))

        # Remaining unmatched B segments
        for j, b in enumerate(segs_b):
            if j not in matched_b_indices:
                gid = f"{video_id}__asr_g_{group_idx:04d}"
                group_idx += 1
                groups.append({
                    "group_id": gid,
                    "video_id": video_id,
                    "start_ms": b["start_ms"],
                    "end_ms": b["end_ms"],
                    "consensus_text": b["raw_text"],
                    "status": "single_model",
                    "models_involved": m_b,
                    "temporal_overlap_ratio": 1.0,
                    "text_similarity": 1.0,
                })
                member_mappings.append((gid, b["segment_id"], m_b))

        return groups, member_mappings
